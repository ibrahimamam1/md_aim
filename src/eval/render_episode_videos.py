"""
render_episode_videos.py
────────────────────────
Renders recorded evaluation episodes (src/eval/episode_recorder.py JSONs) into
videos so any collision / timeout can be inspected after the fact.

Pipeline
  1. Load episode JSON(s) recorded by evaluate_mo_sd.py (EpisodeRecorder).
  2. Render one PNG frame per recorded timestep with matplotlib: the
     intersection is drawn from the network's SUMO net.xml lane shapes, each
     vehicle from its recorded (x, y) position/heading.
  3. Assemble frames into an MP4 with ffmpeg (fallback: mp4 via imageio/FFwriter).

Interpretability (schema v2+ recordings)
  - Vehicles inside the agent's observation are highlighted amber with their
    ID; unobserved vehicles stay red. A red halo marks neighbors inside the
    conflict window (|Δη| < 2.0 s).
  - A side panel decodes the observation vector (4 ego features +
    5 neighbor slots × 5 features + attention mask) into human-readable
    raw values: distances in meters, speeds in m/s, headings in compass
    directions, and the Δη arrival-time gap in raw seconds per slot, plus a
    conflict summary and a legend.
  - Schema v3 recordings store raw units natively; schema v2 (normalized)
    recordings are upgraded automatically so both render identically.
  - v1 recordings (no observation snapshots) still render, but without the
    decoded panel; re-run evaluate_mo_sd.py to capture observations.

Selection
  - Default: only episodes that ended in COLLISION are rendered.
  - --timeouts: render ONLY TIMEOUT episodes (collision == 0 AND success == 0);
    use --all to render everything (collisions, timeouts, successes).
  - --all: render everything including successes.

Examples
    # Render videos for all recorded collisions (default):
    python -m src.eval.render_episode_videos --recordings output/eval_mo_sd/episode_recordings

    # Timeouts only for S3, 8 fps:
    python -m src.eval.render_episode_videos --recordings output/eval_mo_sd/episode_recordings \
        --scenarios S3 --timeouts --fps 8

    # Render from the evaluation output dir root (auto-discovers episode_recordings):
    python -m src.eval.render_episode_videos --eval-output output/eval_mo_sd
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyArrow, Circle
from matplotlib.collections import PatchCollection
import numpy as np

DEFAULT_RECORDINGS_SUBDIR = "episode_recordings"


# --------------------------------------------------------------------------- #
# Episode discovery
# --------------------------------------------------------------------------- #
def discover_episodes(recordings_root, scenarios=None):
    """Yield episode JSON paths grouped by scenario under recordings_root."""
    if not os.path.isdir(recordings_root):
        return []
    wanted = {s.upper() for s in scenarios} if scenarios else None
    episodes = []
    for entry in sorted(os.listdir(recordings_root)):
        scen_dir = os.path.join(recordings_root, entry)
        if not os.path.isdir(scen_dir):
            continue
        if wanted is not None and entry.upper() not in wanted:
            continue
        for fname in sorted(os.listdir(scen_dir)):
            if fname.endswith(".json") and fname != "manifest.json":
                episodes.append(os.path.join(scen_dir, fname))
    return episodes


def filter_outcomes(episodes, include_collisions=True, include_timeouts=False,
                    include_success=False):
    """Filter episode files by their recorded outcome."""
    selected = []
    for path in episodes:
        try:
            with open(path) as f:
                head = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            continue
        collision = int(head.get("collision", 0))
        success = int(head.get("success", 0))
        timeout = int(head.get("timeout", 1 if (collision == 0 and success == 0) else 0))
        if collision and include_collisions:
            selected.append(path)
        elif timeout and include_timeouts:
            selected.append(path)
        elif (not collision and not timeout) and include_success:
            selected.append(path)
    return selected


def resolve_recordings_root(args):
    """Resolve the recordings root from --recordings or --eval-output."""
    if args.recordings:
        root = args.recordings
    elif args.eval_output:
        root = os.path.join(args.eval_output, DEFAULT_RECORDINGS_SUBDIR)
        if not os.path.isdir(root):
            root = args.eval_output
    else:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output", "eval_mo_sd", DEFAULT_RECORDINGS_SUBDIR)
    return root


# --------------------------------------------------------------------------- #
# Network geometry (from SUMO net.xml)
# --------------------------------------------------------------------------- #
def load_network_geometry(net_file):
    """Extract lane shapes (polylines) from a SUMO net.xml.

    Returns dict with:
      lane_shapes   : list of np.ndarray (N, 2) — normal lanes
      internal_lanes: list of np.ndarray (N, 2) — junction-internal lanes
    """
    geom = {"lane_shapes": [], "internal_lanes": []}
    if not net_file or not os.path.exists(net_file):
        return geom
    try:
        root = ET.parse(net_file).getroot()
        for edge in root.findall("edge"):
            internal = edge.get("function") == "internal"
            for lane in edge.findall("lane"):
                shape = lane.get("shape")
                if not shape:
                    continue
                pts = []
                for pair in shape.split():
                    x, y = pair.split(",")[:2]
                    pts.append((float(x), float(y)))
                if internal:
                    geom["internal_lanes"].append(np.array(pts))
                else:
                    geom["lane_shapes"].append(np.array(pts))
    except Exception as e:
        print(f"  WARNING: could not parse net file {net_file}: {e}")
    return geom


def resolve_net_file(episode, recordings_root):
    """Find the net.xml belonging to an episode (repo default as fallback)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net_name = "100m_right_before_left.net.xml"
    scen = str(episode.get("scenario_id", "")).upper()
    if scen == "S7":
        net_name = "100m_allway_stop_fcfs_junction.net.xml"
    candidate = os.path.join(repo_root, "networks", net_name)
    if os.path.exists(candidate):
        return candidate
    # last resort: any net file in networks/
    try:
        for fname in sorted(os.listdir(os.path.join(repo_root, "networks"))):
            if fname.endswith(".net.xml"):
                return os.path.join(repo_root, "networks", fname)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Observation decoding (AlphaEnv_MO_SD observation space)
# --------------------------------------------------------------------------- #
# The observation the agent receives is a flat 34-dim vector:
#   [0:4]    ego   : [d_goal, v, sin(θ), cos(θ)]
#   [4:29]   5 neighbor slots, each 5 features:
#            [ego_dist_to_cp, other_dist_to_cp, other_v, other_sin(θ), other_cos(θ)]
#   [29:34]  attention mask: 1.0 = real neighbor in this slot, 0.0 = padding
# Schema v3+ recordings store raw continuous values (meters, m/s, seconds).
# Schema v2 recordings stored normalized values (goal distance / route_len,
# CP distances / perception_radius, speeds / max_speed, tanh(Δη / 5 s)); they
# are upgraded to raw units below so both render identically.
EGO_FEATURES = 4
NEIGHBOR_FEATURES = 5
PAD_TOL = 1e-3

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# "No conflicting neighbor" sentinel in raw seconds (matches the env's
# D_ETA_SAFE_SENTINEL and the old normalized 1.0 = safe convention).
D_ETA_SAFE_SENTINEL = 5.0


def _upgrade_obs_v2_to_raw(obs, norms):
    """Convert a schema-v2 (normalized) observation vector to raw units.

    v2 conventions: ego d_goal / route_len, CP distances / perception_radius,
    speeds / max_speed, d_eta = tanh(Δη / 5). Padding slots are [1, 0, 1, 0, 0].
    """
    raw = list(obs)
    max_speed = float(norms.get("max_speed") or 0.0)
    perception = float(norms.get("perception_radius") or 0.0)
    route_len = float(norms.get("route_length") or 0.0)

    # Ego: [d_goal, v, sin, cos]
    if route_len > 0:
        raw[0] = obs[0] * route_len
    if max_speed > 0:
        raw[1] = obs[1] * max_speed
    for i in range(max_neighbours_from_len(obs)):
        base = EGO_FEATURES + NEIGHBOR_FEATURES * i
        was_padding = (
            abs(obs[base + 0] - 1.0) < PAD_TOL
            and abs(obs[base + 1] - 0.0) < PAD_TOL
            and abs(obs[base + 2] - 1.0) < PAD_TOL
        )
        if was_padding:
            raw[base + 0] = perception
            raw[base + 1] = perception
            raw[base + 2] = 0.0
            # sin/cos (base+3, base+4) stay 0.0
        else:
            if perception > 0:
                raw[base + 0] = obs[base + 0] * perception
                raw[base + 1] = obs[base + 1] * perception
            if max_speed > 0:
                raw[base + 2] = obs[base + 2] * max_speed
    return raw


def _max_neighbours_from_len(n):
    return (n - EGO_FEATURES) // (NEIGHBOR_FEATURES + 1)


def max_neighbours_from_len(obs):
    return _max_neighbours_from_len(len(obs))


def decode_observation(obs, norms, max_neighbours=5, schema="v3"):
    """Decode a flat observation vector into a readable dict, or None.

    `norms` must contain max_speed, perception_radius and route_length (the
    EpisodeRecorder stores them in the episode's top-level "norms" block).
    schema="v2" upgrades the vector from normalized to raw units first.
    """
    if obs is None:
        return None
    try:
        obs = [float(v) for v in np.asarray(obs).flatten()]
    except Exception:
        return None

    expected = EGO_FEATURES + NEIGHBOR_FEATURES * max_neighbours + max_neighbours
    if len(obs) < expected:
        return None

    if schema == "v2":
        obs = _upgrade_obs_v2_to_raw(obs, norms)

    perception = float(norms.get("perception_radius") or 0.0)
    # Raw continuous units in schema v3 (and upgraded v2): distances in
    # meters, speeds in m/s, sin/cos dimensionless.
    d_goal_m, v_ms, ego_sin, ego_cos = obs[0:4]
    ego = {
        "d_goal_m": d_goal_m,
        "d_goal_norm": (d_goal_m / norms["route_length"]
                        if norms.get("route_length") else None),
        "v_ms": v_ms,
        "v_norm": (v_ms / norms["max_speed"] if norms.get("max_speed") else None),
        "sin": ego_sin,
        "cos": ego_cos,
    }
    if ego_sin != 0.0 or ego_cos != 0.0:
        ego["heading_deg"] = float(np.degrees(np.arctan2(ego_sin, ego_cos)) % 360.0)
    else:
        ego["heading_deg"] = None

    slots = []
    for i in range(max_neighbours):
        base = EGO_FEATURES + NEIGHBOR_FEATURES * i
        mask = obs[EGO_FEATURES + NEIGHBOR_FEATURES * max_neighbours + i]
        # Padding slots: [perception, perception, 0, 0, 0] in raw units
        # ([1, 0, 1, 0, 0] in old v2 normalized units).
        is_padding = (
            mask < 0.5
            and abs(obs[base + 0] - perception) < 1e-6 * max(1.0, perception)
            and abs(obs[base + 1] - perception) < 1e-6 * max(1.0, perception)
            and abs(obs[base + 2] - 0.0) < PAD_TOL
        )
        v_n = obs[base + 2]
        sin_n, cos_n = obs[base + 3], obs[base + 4]
        slots.append({
            "active": bool(mask > 0.5 and not is_padding),
            "padding": bool(is_padding),
            "ego_d_cp_m": obs[base + 0],
            "other_d_cp_m": obs[base + 1],
            "v_ms": v_n,
            "v_norm": (v_n / norms["max_speed"] if norms.get("max_speed") else None),
            "sin": sin_n,
            "cos": cos_n,
            "heading_deg": (float(np.degrees(np.arctan2(sin_n, cos_n)) % 360.0)
                            if (sin_n != 0.0 or cos_n != 0.0) else None),
        })
    return {"ego": ego, "slots": slots}


def _d_eta_seconds(d_eta):
    """Δη in seconds. v3 recordings store raw seconds; v2 normalized values
    (tanh(Δη / 5)) are inverted here."""
    try:
        v = float(d_eta)
    except Exception:
        return None
    if v <= -0.9999:
        return -float("inf")
    if v >= 0.9999:
        return float("inf")
    return v


def _compass(heading_deg):
    """Compass letter for a math-convention heading (0° = east, CCW)."""
    if heading_deg is None:
        return "—"
    sumo_deg = (90.0 - float(heading_deg)) % 360.0  # SUMO convention: 0 = north
    idx = int(((sumo_deg + 22.5) % 360.0) // 45.0)
    return COMPASS[idx]


def _slot_assignments(fr):
    """Map vehicle id -> observation slot index (0-based) for this frame."""
    slots = {}
    for i, n in enumerate(fr.get("neighbors_info") or []):
        if isinstance(n, dict) and n.get("veh_id") is not None:
            slots[str(n["veh_id"])] = i
    if not slots:  # v1 fallback: ids only, nearest-first = slot order
        for i, vid in enumerate(fr.get("neighbors") or []):
            if vid is not None:
                slots[str(vid)] = i
    return slots


def _classify_neighbor(fr, slot_idx, schema="v3"):
    """Human-readable encounter relation for one observation slot.

    Uses the same signal the safety term uses: the arrival-time gap Δη to the
    shared conflict point in raw seconds (window = 2.0 s in raw units, which
    matches the old normalized |Δη| < 0.4 window).
    """
    infos = fr.get("neighbors_info") or []
    info = infos[slot_idx] if slot_idx is not None and slot_idx < len(infos) else None
    if not isinstance(info, dict):
        return None, "no CP"
    if schema == "v2":
        d_eta = _d_eta_seconds_v2(info.get("d_eta"))
    else:
        d_eta = _d_eta_seconds(info.get("d_eta"))
    if d_eta is None or d_eta in (float("inf"), float("-inf")):
        return d_eta, "no CP"      # routes do not share a conflict point
    if abs(d_eta) < D_ETA_REWARD_WINDOW_S:
        return d_eta, "CONFLICT"   # inside the safety-term window
    return d_eta, "lead" if d_eta > 0 else "yield"


D_ETA_REWARD_WINDOW_S = 2.0   # raw-seconds safety-reward window (matches env)


def _d_eta_seconds_v2(d_eta_norm):
    """Inverse of the v2 env's tanh(Δη / 5) normalization, in seconds."""
    try:
        v = float(d_eta_norm)
    except Exception:
        return None
    if v <= -0.9999:
        return -float("inf")
    if v >= 0.9999:
        return float("inf")
    return 5.0 * float(np.arctanh(v))


# --------------------------------------------------------------------------- #
# Frame rendering
# --------------------------------------------------------------------------- #
VEHICLE_LENGTH = 5.0   # m, approx passenger car
VEHICLE_WIDTH = 2.0    # m

EGO_COLOR = "#1f4fd1"        # blue
NON_RL_COLOR = "#c23b3b"     # red
OBSERVED_COLOR = "#e6a817"   # amber for vehicles inside the observation
CONFLICT_COLOR = "#ff0000"   # red halo for neighbors in the conflict window
COLLISION_COLOR = "#000000"  # black flash at collision step


def _draw_vehicle(ax, x, y, heading, color, length=VEHICLE_LENGTH, width=VEHICLE_WIDTH,
                  zorder=5, alpha=1.0):
    """Draw a vehicle as a rotated rectangle centred on (x, y)."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    dx_l, dy_l = (length / 2.0) * cos_h, (length / 2.0) * sin_h
    dx_w, dy_w = (-width / 2.0) * sin_h, (width / 2.0) * cos_h
    corners = [
        (x + dx_l + dx_w, y + dy_l + dy_w),
        (x + dx_l - dx_w, y + dy_l - dy_w),
        (x - dx_l - dx_w, y - dy_l - dy_w),
        (x - dx_l + dx_w, y - dy_l + dy_w),
    ]
    ax.add_patch(MplPolygon(corners, closed=True, facecolor=color,
                            edgecolor="black", linewidth=0.6, alpha=alpha,
                            zorder=zorder))


def _fmt(v, spec=".1f", none="—"):
    if v is None:
        return none
    if v in (float("inf"), float("-inf")):
        return "∞" if v > 0 else "-∞"
    return format(v, spec)


def _conflict_summary(fr, max_neighbours=5, schema="v3"):
    """Derive a compact conflict summary from the recorded neighbor infos."""
    infos = fr.get("neighbors_info") or []
    if not infos:
        return None
    d_etas = []
    gaps = []
    for n in infos[:max_neighbours]:
        if not isinstance(n, dict):
            continue
        if schema == "v2":
            d_etas.append(_d_eta_seconds_v2(n.get("d_eta")))
        else:
            d_etas.append(_d_eta_seconds(n.get("d_eta")))
        try:
            gaps.append(float(n.get("distance")))
        except Exception:
            pass
    d_etas_f = [d for d in d_etas if d is not None]
    if not d_etas_f:
        return None
    finite = [d for d in d_etas_f if d not in (float("inf"), float("-inf"))]
    min_abs = min((abs(d) for d in finite), default=None)
    danger = sum(1 for d in d_etas_f if abs(d) < D_ETA_REWARD_WINDOW_S)
    min_gap = min(gaps) if gaps else None
    return {
        "min_abs_d_eta": min_abs,
        "danger_count": danger,
        "min_gap": min_gap,
        "n_observed": len(d_etas_f),
    }


def _draw_obs_panel(panel_ax, fr, decoded, action_txt, collision, terminated,
                    max_neighbours=5, schema="v3"):
    """Side panel: decoded observation, neighbor table, conflict summary, legend."""
    panel_ax.clear()
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)
    panel_ax.axis("off")

    y = 0.985
    lh = 0.024  # line height (fraction of panel height)

    def line(txt, x=0.03, size=6, color="#222222", weight="normal", mono=True):
        nonlocal y
        panel_ax.text(x, y, txt, fontsize=size, color=color, fontweight=weight,
                      family="monospace" if mono else None, va="top", ha="left")
        y -= lh

    # -- Ego observation -----------------------------------------------------
    ego_id = (fr.get("ego") or {}).get("id", "ego")
    line(f"EGO OBSERVATION  #{ego_id}", weight="bold", color=EGO_COLOR,
         mono=False, size=7)
    if decoded is None:
        line("  (v1 recording — no observation snapshot;")
        line("   re-run evaluate_mo_sd.py to capture it)")
    else:
        e = decoded["ego"]
        d_goal = _fmt(e["d_goal_m"], ".1f") + " m" if e["d_goal_m"] is not None \
            else "n/a"
        v_disp = _fmt(e["v_ms"], ".1f") + " m/s" if e["v_ms"] is not None else "n/a"
        line(f"  d_goal  {d_goal}  (x{(e['d_goal_norm'] or 0):.3f} of route)")
        line(f"  v       {v_disp}")
        line(f"  sinθ    {e['sin']:+.3f}   cosθ {e['cos']:+.3f}  "
             f"(H={_compass(e['heading_deg'])})")

    # -- Action --------------------------------------------------------------
    line(f"action  {action_txt}   (normalized accel)", color=EGO_COLOR)
    y -= 0.012

    # -- Neighbor slots ------------------------------------------------------
    line("NEIGHBOR SLOTS (VEHICLES IN OBSERVATION)",
         weight="bold", color=OBSERVED_COLOR, mono=False, size=7)
    if decoded is None:
        n_obs = len(fr.get("neighbors") or [])
        line(f"  {n_obs} observed vehicle(s) — ids only (v1 recording)")
    else:
        infos = fr.get("neighbors_info") or []
        any_active = False
        for i, slot in enumerate(decoded["slots"]):
            info = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
            vid = info.get("veh_id", "?")
            if slot["padding"]:
                line(f"  slot{i}  — padding (masked out)", color="#999999")
                continue
            any_active = True
            d_eta, relation = _classify_neighbor(fr, i, schema=schema)
            relation_txt = relation if relation != "CONFLICT" else "CONFLICT!"
            color = CONFLICT_COLOR if relation == "CONFLICT" else "#222222"
            weight = "bold" if relation == "CONFLICT" else "normal"
            line(f"  slot{i}  {vid}", color=color, weight=weight)
            ego_d = _fmt(slot["ego_d_cp_m"], ".1f")
            oth_d = _fmt(slot["other_d_cp_m"], ".1f")
            v_disp = _fmt(slot["v_ms"], ".1f")
            line(f"    ego→CP {ego_d} m | other→CP {oth_d} m | "
                 f"v {v_disp} m/s | H {_compass(slot['heading_deg'])}",
                 color=color)
            if d_eta is None:
                line(f"    Δη n/a ({relation_txt})", color=color)
            else:
                line(f"    Δη {d_eta:+.2f} s  → {relation_txt}", color=color,
                     weight=weight)
        if not any_active:
            line("  (no neighbors in observation — all slots padded)")

    # -- Conflict summary ----------------------------------------------------
    summary = _conflict_summary(fr, max_neighbours=max_neighbours, schema=schema)
    if summary is not None:
        y -= 0.012
        line("CONFLICT SUMMARY", weight="bold", color=NON_RL_COLOR,
             mono=False, size=7)
        line(f"  observed neighbors : {summary['n_observed']}")
        line(f"  min |Δη|           : {_fmt(summary['min_abs_d_eta'], '.2f')} s "
             f"(danger window {D_ETA_REWARD_WINDOW_S:.1f} s)")
        line(f"  in danger window   : {summary['danger_count']}")
        line(f"  min vehicle gap    : {_fmt(summary['min_gap'], '.1f')} m")
    y -= 0.012

    # -- Legend --------------------------------------------------------------
    line("LEGEND", weight="bold", color="#444444", mono=False, size=7)
    line("  #id  ego vehicle (RL agent)", color=EGO_COLOR)
    line("  #id  in observation (state in panel)", color=OBSERVED_COLOR)
    line("       red halo = CONFLICT (|Δη| < 2.0 s)", color=CONFLICT_COLOR)
    line("  #id  not observed (background traffic)", color=NON_RL_COLOR)
    line("  observed = within perception radius on conflicting route")
    line("  slot n = position in the observation vector")

    if collision and terminated:
        panel_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                                     linewidth=3, zorder=10))
        panel_ax.text(0.5, 0.5, "COLLISION", fontsize=18, color="black",
                      ha="center", va="center", alpha=0.25, rotation=12,
                      fontweight="bold", family="monospace", zorder=11)


def _even_figsize(w_in, h_in, dpi):
    """Adjust figure inches so width*height pixels are even (libx264/yuv420p
    rejects odd dimensions)."""
    w_px, h_px = int(round(w_in * dpi)), int(round(h_in * dpi))
    if w_px % 2:
        w_in = (w_px + 1) / dpi
    if h_px % 2:
        h_in = (h_px + 1) / dpi
    return w_in, h_in


def compute_frame_bounds(frames, geom, pad=25.0):
    """Fixed axis bounds across all frames so the video does not wobble."""
    xs, ys = [], []
    for fr in frames:
        for v in fr.get("vehicles", []):
            xs.append(v["x"])
            ys.append(v["y"])
        ego = fr.get("ego")
        if ego:
            xs.append(ego["x"])
            ys.append(ego["y"])
    for lane in geom["lane_shapes"] + geom["internal_lanes"]:
        xs.extend(lane[:, 0].tolist())
        ys.extend(lane[:, 1].tolist())
    if not xs:
        return -50, 50, -50, 50
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def render_episode(episode_path, out_root, fps, dpi, keep_frames, net_file=None,
                   show_obs_panel=True):
    """Render one recorded episode to PNGs and assemble them into a video.

    Returns (frames_dir, video_path or None).
    """
    with open(episode_path) as f:
        episode = json.load(f)

    frames = episode.get("frames", [])
    if not frames:
        print(f"  WARNING: no frames in {episode_path}; skipping")
        return None, None

    scen = str(episode.get("scenario_id", "unknown"))
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    collision = int(episode.get("collision", 0))
    success = int(episode.get("success", 0))
    timeout = int(episode.get("timeout", 0))

    # Recording schema: v3 stores raw continuous values (meters, m/s, seconds);
    # v2 stored normalized values and is upgraded for display; v1 has none.
    schema = str(episode.get("schema", "md_aim_episode_recording_v2")).rsplit("_v", 1)[-1]

    # Normalization constants (needed to upgrade v2 recordings; kept for v3)
    norms = dict(episode.get("norms") or {})
    norms.setdefault("perception_radius",
                     float(episode.get("perception_radius", 100.0) or 100.0))
    norms.setdefault("max_neighbours", int(episode.get("max_neighbours", 5) or 5))
    max_neighbours = int(norms.get("max_neighbours", 5))

    frames_dir = os.path.join(out_root, scen, f"{stem}_frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    if net_file is None:
        net_file = resolve_net_file(episode, os.path.dirname(episode_path))
    geom = load_network_geometry(net_file)

    x0, x1, y0, y1 = compute_frame_bounds(frames, geom)

    neighbor_ids_per_frame = [set(fr.get("neighbors") or []) for fr in frames]
    collision_step = None
    for fr in frames:
        if fr.get("terminated") and collision:
            collision_step = fr.get("step")
            break

    if show_obs_panel:
        fig_w, fig_h = _even_figsize(11.5, 8.0, dpi)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.05], wspace=0.10)
        ax = fig.add_subplot(gs[0, 0])
        panel_ax = fig.add_subplot(gs[0, 1])
    else:
        fig_w, fig_h = _even_figsize(8.0, 8.0, dpi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        panel_ax = None

    for fr in frames:
        ax.clear()

        # Static network geometry
        if geom["lane_shapes"]:
            ax.add_collection(PatchCollection(
                [MplPolygon(lane, closed=False, fill=False) for lane in geom["lane_shapes"]],
                edgecolors="#9aa3ad", facecolors="none", linewidths=1.2, zorder=1))
        if geom["internal_lanes"]:
            ax.add_collection(PatchCollection(
                [MplPolygon(lane, closed=False, fill=False) for lane in geom["internal_lanes"]],
                edgecolors="#c9ced4", facecolors="none", linewidths=1.0,
                linestyles="dashed", zorder=1))

        step = fr.get("step", 0)
        neighbor_ids = neighbor_ids_per_frame[step] if step < len(neighbor_ids_per_frame) else set()
        slots_by_id = _slot_assignments(fr)

        # Decode this frame's observation once (v2 recordings are upgraded
        # to raw units so old and new render identically)
        decoded = decode_observation(fr.get("obs"), norms,
                                     max_neighbours=max_neighbours,
                                     schema=schema)

        for veh in fr.get("vehicles", []):
            vid = str(veh["id"])
            in_obs = vid in neighbor_ids or vid in slots_by_id
            slot_idx = slots_by_id.get(vid)
            # Recorder stores SUMO headings (deg, 0 = north); drawing uses math
            # convention (rad, 0 = east).
            heading_math = np.radians(90.0 - veh["heading"])
            if in_obs:
                _, relation = _classify_neighbor(fr, slot_idx, schema=schema)
                if relation == "CONFLICT":
                    # red halo so conflicting neighbors pop out
                    ax.add_patch(Circle((veh["x"], veh["y"]), 4.5, fill=False,
                                        edgecolor=CONFLICT_COLOR, linewidth=1.4,
                                        alpha=0.9, zorder=4))
                _draw_vehicle(ax, veh["x"], veh["y"], heading_math, OBSERVED_COLOR,
                              zorder=5)
            else:
                _draw_vehicle(ax, veh["x"], veh["y"], heading_math, NON_RL_COLOR,
                              zorder=5)
            # Map label: the vehicle id only — every state detail lives in the
            # observation side panel, keyed by this id.
            ax.text(veh["x"], veh["y"] - 3.4, f"#{vid}", fontsize=5.5,
                    ha="center", va="top", color="#333333", zorder=6)

        ego = fr.get("ego")
        if ego:
            color = COLLISION_COLOR if (collision_step is not None and step >= collision_step) else EGO_COLOR
            _draw_vehicle(ax, ego["x"], ego["y"], np.radians(90.0 - ego["heading"]), color,
                          length=VEHICLE_LENGTH * 1.1, width=VEHICLE_WIDTH * 1.1, zorder=7)
            ax.add_patch(Circle((ego["x"], ego["y"]), 25.0, fill=False,
                                edgecolor=EGO_COLOR, linewidth=0.7, alpha=0.5, zorder=2))
            # Map label: id only (details in the side panel).
            ax.text(ego["x"], ego["y"] + 3.6, f"#{ego.get('id', 'ego')}",
                    fontsize=5.5, ha="center", va="bottom", color=EGO_COLOR,
                    zorder=6)

        action = fr.get("action")
        action_txt = "—" if action is None else f"{action:+.2f}"
        # Per-step reward decomposition (progress + safety), persisted by the
        # recorder from info["reward_dict"]. Older recordings without the key
        # show "—".
        rd = fr.get("reward_dict") or {}
        r_prog = _fmt(rd.get("progress_reward"), "+.4f")
        r_safety = _fmt(rd.get("safety_penalty"), "+.4f")
        ax.set_title(
            f"t={fr['t']:.2f}s step={step}  action={action_txt}  "
            f"reward={fr.get('reward', 0.0):+.3f}  "
            f"r_prog={r_prog}  r_safety={r_safety}",
            fontsize=8)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=6)

        if panel_ax is not None:
            _draw_obs_panel(panel_ax, fr, decoded, action_txt, collision,
                            bool(fr.get("terminated")),
                            max_neighbours=max_neighbours, schema=schema)

        fig.canvas.draw()
        frame_path = os.path.join(frames_dir, f"frame_{step:06d}.png")
        fig.savefig(frame_path, dpi=dpi, facecolor="white")

    plt.close(fig)

    video_path = os.path.join(out_root, scen, f"{stem}.mp4")
    ok = assemble_video(frames_dir, video_path, fps)
    if ok and not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return frames_dir, video_path
    return frames_dir, video_path if ok else None


# --------------------------------------------------------------------------- #
# ffmpeg assembly
# --------------------------------------------------------------------------- #
def find_ffmpeg():
    return shutil.which("ffmpeg")


def assemble_video(frames_dir, video_path, fps):
    """Assemble frame_%06d.png files into an MP4 using ffmpeg."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("  ERROR: ffmpeg not found on PATH; frames were kept but no video "
              "was created. Install ffmpeg (e.g. 'sudo apt install ffmpeg').")
        return False

    pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed for {video_path}:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  ERROR: could not run ffmpeg: {e}")
        return False
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_recordings = os.path.join(repo_root, "output", "eval_mo_sd", DEFAULT_RECORDINGS_SUBDIR)

    parser = argparse.ArgumentParser(
        description="Render recorded evaluation episodes into videos (ffmpeg).")
    parser.add_argument("--recordings", type=str, default=default_recordings,
                        help="Episode recordings root (contains <scenario>/run_*.json).")
    parser.add_argument("--eval-output", type=str, default=None,
                        help="Alternative to --recordings: evaluation output dir; "
                             f"uses its '{DEFAULT_RECORDINGS_SUBDIR}' subdirectory.")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Restrict to scenarios (e.g. S3 S5). Default: all found.")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Restrict to run indices (e.g. 4 11). Default: all matching outcome.")
    parser.add_argument("--timeouts", action="store_true", default=False,
                        help="Render ONLY TIMEOUT episodes (collision==0 and success==0) "
                             "instead of collisions. Use --all to render everything.")
    parser.add_argument("--all", dest="render_all", action="store_true", default=False,
                        help="Render every recorded episode including successes.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where videos are written (default: the recordings root).")
    parser.add_argument("--fps", type=int, default=4,
                        help="Video frames per second (recording is at sim_step=0.25s -> default 4 fps = real time).")
    parser.add_argument("--dpi", type=int, default=110, help="Frame resolution (matplotlib dpi).")
    parser.add_argument("--keep_frames", action="store_true", default=False,
                        help="Keep the intermediate PNG frames next to the video.")
    parser.add_argument("--net_file", type=str, default=None,
                        help="Explicit SUMO net.xml for geometry (default: auto per scenario).")
    parser.add_argument("--no-obs-panel", dest="no_obs_panel", action="store_true",
                        default=False,
                        help="Hide the observation side panel (plain map view only).")
    return parser.parse_args()


def main():
    args = parse_args()
    recordings_root = resolve_recordings_root(args)
    out_root = args.output_dir or recordings_root
    os.makedirs(out_root, exist_ok=True)

    episodes = discover_episodes(recordings_root, args.scenarios)
    if args.runs is not None:
        wanted = set(args.runs)

        def _run_of(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(stem.split("_")[1])
            except Exception:
                return -1
        episodes = [p for p in episodes if _run_of(p) in wanted]

    if args.render_all:
        selected = episodes
    elif args.timeouts:
        # --timeouts: timeouts only (no collisions)
        selected = filter_outcomes(episodes, include_collisions=False,
                                   include_timeouts=True,
                                   include_success=False)
    else:
        # default: collisions only
        selected = filter_outcomes(episodes, include_collisions=True,
                                   include_timeouts=False,
                                   include_success=False)

    print("\n" + "=" * 76)
    print(" Episode video rendering")
    print(f" recordings : {recordings_root}")
    print(f" output     : {out_root}")
    print(f" found      : {len(episodes)} episode(s); selected: {len(selected)}")
    print(f" mode       : {'all' if args.render_all else ('timeouts only' if args.timeouts else 'collisions only')}")
    print(f" obs panel  : {'off' if args.no_obs_panel else 'on'}")
    print("=" * 76 + "\n")

    if not selected:
        print("No episodes matched. Recorded outcomes can be listed with:")
        print(f"  ls {recordings_root}/*/")
        return

    made, failed = [], []
    for path in selected:
        print(f"Rendering {os.path.relpath(path, recordings_root)} ...")
        try:
            frames_dir, video = render_episode(path, out_root, args.fps, args.dpi,
                                               args.keep_frames, net_file=args.net_file,
                                               show_obs_panel=not args.no_obs_panel)
            if video:
                made.append(video)
                print(f"  -> {video}")
            else:
                failed.append(path)
        except Exception as e:
            failed.append(path)
            print(f"  ERROR: rendering failed: {e}")

    print(f"\nDone. {len(made)} video(s) created"
          + (f", {len(failed)} failed." if failed else "."))
    for v in made:
        print(f"  video: {v}")


if __name__ == "__main__":
    main()
