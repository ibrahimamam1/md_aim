"""
episode_recorder.py
───────────────────
Records per-step episode state so any evaluation episode can later be
recreated frame-by-frame (see render_episode_videos.py).

For every step (including the initial observation after reset) it stores:
  - t              : simulation time of the step (s)
  - step           : RL step index
  - ego            : position (x, y), speed, heading angle (rad),
                     acceleration (m/s^2), distance travelled along route
  - vehicles       : id, (x, y), speed, heading angle, acceleration and edge
                     of every vehicle in the network (the observation contains
                     at most the nearest `max_neighbours` of these; the raw
                     id list is kept so the exact observation subset can be
                     recomputed offline)
  - action         : action applied to the ego vehicle at this step
  - reward         : scalar reward returned by the environment
  - neighbors      : ids of the vehicles that actually entered the
                     observation (info["neighbors"], nearest first)
  - terminated / truncated

On termination the final info dictionaries (mo_telemetry etc.) are stored so
the renderer can filter by outcome (collision / timeout / success) exactly
the way evaluate_mo_sd.py classifies runs.

Output is one JSON file per episode:
    <recording_dir>/<scenario>/run_<nnnn>_coll<0|1>_succ<0|1>.json
plus a manifest.json that aggregates the outcome of every run.
"""

import json
import os
import time


class EpisodeRecorder:
    """Collects per-step ego / neighbor states for one episode and dumps JSON."""

    def __init__(self, env, scenario_id=None, run_index=0, recording_dir=None,
                 metadata=None):
        """
        Parameters
        ----------
        env : AlphaEnv_MO_SD (or any Env_N subclass)
            The live environment. Used only to read positions/states.
        scenario_id : str
            Scenario label (e.g. "S3") used for sub-directories.
        run_index : int
            Run number within the scenario, used in the output filename.
        recording_dir : str
            Base directory for recordings.
        metadata : dict
            Free-form run metadata (checkpoint, weights, seed, ...) stored
            in the header of the episode JSON.
        """
        self.env = env
        self.scenario_id = scenario_id or "unknown"
        self.run_index = int(run_index)
        self.recording_dir = recording_dir
        self.metadata = dict(metadata or {})

        self.frames = []
        self._final_infos = None
        self._stopped = False

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    def _vehicle_states(self):
        """Position/state of every vehicle currently in the network."""
        env = self.env
        states = []
        try:
            veh_ids = env.k.vehicle.get_ids()
        except Exception:
            veh_ids = []

        for veh_id in veh_ids:
            try:
                pos = env.k.vehicle.get_2d_position(veh_id)
                if pos is None or pos == -1001 or pos == (-1001.0, -1001.0):
                    continue
                speed = env.k.vehicle.get_speed(veh_id)
                heading = env.k.vehicle.get_heading(veh_id)
                try:
                    accel = env.k.vehicle.get_accel(veh_id)
                except Exception:
                    accel = 0.0
                try:
                    edge = env.k.vehicle.get_edge(veh_id)
                except Exception:
                    edge = ""
                states.append({
                    "id": str(veh_id),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "speed": float(speed or 0.0),
                    "heading": float(heading or 0.0),
                    "accel": float(accel or 0.0),
                    "edge": str(edge),
                })
            except Exception:
                continue
        return states

    def _ego_state(self):
        """Position/state of the ego (RL) vehicle; None if not in network."""
        env = self.env
        agent_id = getattr(env, "agent_id", None)
        if agent_id is None or agent_id not in env.k.vehicle.get_ids():
            return None
        try:
            pos = env.k.vehicle.get_2d_position(agent_id)
            if pos is None or pos == -1001 or pos == (-1001.0, -1001.0):
                return None
            speed = env.k.vehicle.get_speed(agent_id)
            heading = env.k.vehicle.get_heading(agent_id)
            try:
                accel = env.k.vehicle.get_accel(agent_id)
            except Exception:
                accel = 0.0
            try:
                dist = env.k.vehicle.get_distance(agent_id)
            except Exception:
                dist = None
            if dist is None or dist == -1001:
                dist = getattr(env, "last_valid_distance", 0.0)
            return {
                "id": str(agent_id),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "speed": float(speed or 0.0),
                "heading": float(heading or 0.0),
                "accel": float(accel or 0.0),
                "distance": float(dist or 0.0),
            }
        except Exception:
            return None

    def _neighbor_ids(self, info):
        """Ids of the vehicles that actually entered the observation."""
        neighbors = info.get("neighbors") or []
        ids = []
        for n in neighbors:
            vid = n.get("veh_id") if isinstance(n, dict) else None
            ids.append(str(vid) if vid is not None else None)
        return ids

    def record_reset(self, info=None):
        """Snapshot right after env.reset() (step 0, before any action)."""
        self.frames = []
        self._final_infos = None
        self._stopped = False
        snapshot = {
            "t": float(getattr(self.env, "time_counter", 0.0)),
            "step": 0,
            "ego": self._ego_state(),
            "vehicles": self._vehicle_states(),
            "action": None,
            "reward": 0.0,
            "neighbors": [],
            "terminated": False,
            "truncated": False,
        }
        self.frames.append(snapshot)

    def record_step(self, action, reward, terminated, truncated, info=None):
        """Snapshot after env.step()."""
        info = info or {}
        self.frames.append({
            "t": float(getattr(self.env, "time_counter", 0.0)),
            "step": len(self.frames),
            "ego": self._ego_state(),
            "vehicles": self._vehicle_states(),
            "action": None if action is None else float(np_flatten(action)),
            "reward": float(reward if reward is not None else 0.0),
            "neighbors": self._neighbor_ids(info),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            self._final_infos = dict(info)

    # ------------------------------------------------------------------ #
    # Outcome helpers (match evaluate_mo_sd.py classification)
    # ------------------------------------------------------------------ #
    def outcome(self):
        """Returns (collision, success, timeout) as 0/1 ints.

        Timeout == collision == 0 AND success == 0.
        """
        collision = 0
        success = 0
        src = self._final_infos or {}
        mo = src.get("mo_telemetry") or {}
        if mo:
            collision = int(mo.get("collision", 0) or 0)
            success = int(mo.get("success", 0) or 0)
        else:
            base = src.get("telemetry") or {}
            collision = 1 if base.get("agent_collision", False) else 0
            success = 1 if base.get("agent_success", False) else 0
        timeout = 1 if (collision == 0 and success == 0) else 0
        return collision, success, timeout

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def episode_file_path(self):
        collision, success, _ = self.outcome()
        fname = f"run_{self.run_index:04d}_coll{collision}_succ{success}.json"
        return os.path.join(self.recording_dir or ".",
                            self.scenario_id, fname)

    def save(self):
        """Write the episode JSON. Returns the path or None if disabled."""
        if self._stopped or self.recording_dir is None or not self.frames:
            return None

        out_path = self.episode_file_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        collision, success, timeout = self.outcome()
        episode = {
            "schema": "md_aim_episode_recording_v1",
            "scenario_id": self.scenario_id,
            "run_index": self.run_index,
            "collision": collision,
            "success": success,
            "timeout": timeout,
            "sim_step": float(getattr(self.env, "sim_step", 0.25)),
            "perception_radius": float(getattr(self.env, "perception_radius", 100.0)),
            "max_neighbours": int(getattr(self.env, "max_neighbours", 5)),
            "metadata": self.metadata,
            "final_info": _jsonable(self._final_infos or {}),
            "frames": self.frames,
        }

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(episode, f)
        os.replace(tmp_path, out_path)

        self._stopped = True
        return out_path


def np_flatten(action):
    """Best-effort conversion of a (possibly array) action to float."""
    try:
        import numpy as np
        return float(np.asarray(action).flatten()[0])
    except Exception:
        pass
    if isinstance(action, (list, tuple)) and len(action) > 0:
        try:
            return float(action[0])
        except Exception:
            return 0.0
    try:
        return float(action)
    except Exception:
        return 0.0


def _jsonable(obj, _depth=0):
    """Convert info dicts (numpy, tensors, inf) into JSON-safe structures."""
    if _depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else (
            1e18 if obj > 0 else (-1e18 if obj < 0 else 0.0))
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if v == v and abs(v) != float("inf") else (
                1e18 if v > 0 else (-1e18 if v < 0 else 0.0))
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _jsonable(obj.tolist(), _depth + 1)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in obj]
    return str(obj)


def manifest_path(recording_dir, scenario_id):
    return os.path.join(recording_dir, scenario_id, "manifest.json")


def update_manifest(recording_dir, scenario_id, entry):
    """Append one run entry to the scenario manifest (best-effort)."""
    if recording_dir is None:
        return
    try:
        path = manifest_path(recording_dir, scenario_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"episodes": []}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                pass
        data.setdefault("episodes", []).append(entry)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass
