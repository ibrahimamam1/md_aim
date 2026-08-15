"""
v0_1_evaluate_deterministic.py
──────────────────────────────
Deterministic scenario evaluation for velocity-profile analysis.

Runs the ego in a fully deterministic environment:
  - Spawns on the WEST lane at t = 20 s
  - Alternates between NORTH and EAST routes across episodes
  - Background vehicles spawn at a fixed period on the SOUTH lane, all going NORTH

Multiple checkpoints (= action-space variants) can be evaluated in a single
invocation so that all profiles are gathered in one output directory, ready
for plot_velocity_profiles.py.

Usage:
    python src/test/v0_1_evaluate_deterministic.py \\
        --checkpoints \\
            attention_continous:checkpoints/attention_continous/final_model \\
            heuristic_continous:checkpoints/heuristic_continous/final_model \\
        --n_sims 20 \\
        [--render] \\
        [--bg_period 4.0] \\
        [--ego_spawn_time 20.0]
"""

import argparse
import os
import sys
import csv
import gc
import random
import resource
import subprocess
import time
from copy import deepcopy
from typing import Tuple

import numpy as np

# ─── Raise OS file-descriptor limit ──────────────────────────────────────────
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.deterministic_scenario import DeterministicSouthNorthNetwork

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from plot_velocity_profiles import plot_velocity_profiles

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Deterministic velocity-profile evaluation.")
parser.add_argument(
    "--checkpoints", nargs="+", required=True,
    metavar="VERSION:PATH",
    help=(
        "One or more 'version_tag:checkpoint_path' pairs, e.g.:\n"
        "  attention_continous:checkpoints/attention_continous/final_model"
    ),
)
parser.add_argument("--n_sims", type=int, default=20,
                    help="Number of episodes per checkpoint (default: 20).")
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--bg_period", type=float, default=4.0,
                    help="Seconds between background vehicle spawns (default: 4.0).")
parser.add_argument("--ego_spawn_time", type=float, default=20.0,
                    help="Simulation time at which the ego spawns (default: 20.0).")
args = parser.parse_args() if __name__ == "__main__" else None

# ─── Sim params (shared) ──────────────────────────────────────────────────────
min_gap    = 2.5
max_accel  = 2.6
max_decel  = 4.5
max_speed  = 13.89      # ≈ 50 km/h — matches the network speed limit
initial_speed = 0.0
speed_factor  = 1.0
speed_dev     = 0.0     # deterministic: no speed deviation
sigma         = 0.0     # deterministic: no Gaussian noise in IDM
tau           = 0.8
horizon       = 300     # generous horizon (s) so the ego always finishes
sim_step      = 0.25
warmup_steps  = 0       # no warmup — env manages spawn timing itself

root_dir   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
net_file   = os.path.join(root_dir, "networks", "100m_right_before_left.net.xml")
output_dir = os.path.join(root_dir, "output_deterministic")
os.makedirs(output_dir, exist_ok=True)

# ─── CSV schema ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "run", "version", "ego_route", "episode_index",
    "collision", "success",
    "avg_speed", "safe_gap", "travel_time", "waiting_time",
    "bg_spawn_period",
    "time_profile", "distance_profile", "velocity_profile",
    "jerk_profile", "acceleration_profile",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_checkpoint_arg(raw: str) -> Tuple[str, str]:
    """Split 'version_tag:path' into (version_tag, path)."""
    if ":" not in raw:
        raise ValueError(
            f"Checkpoint argument must be 'version:path', got: {raw!r}"
        )
    tag, path = raw.split(":", 1)
    return tag.strip(), path.strip()


def _build_inflows(ego_spawn_time: float) -> InFlows:
    """
    Single RL inflow: one ego on the west lane at t = ego_spawn_time.
    Background vehicles are spawned via traci in additional_command(),
    so we do NOT add inflows for them here.
    """
    inf = InFlows()
    inf.add(
        veh_type="RL",
        edge="E#L-X",
        vehs_per_hour=1,           # 1 vehicle total over the episode
        depart_lane=0,
        depart_speed=initial_speed,
        begin=ego_spawn_time,
        end=ego_spawn_time + 1.0,  # only one slot
        color="blue",
    )
    return inf


def _kill_stray_sumo():
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
    except Exception:
        pass


def _open_fd_count():
    try:
        import psutil
        return psutil.Process().num_fds()
    except Exception:
        return -1


def _fmt_profile(arr) -> str:
    return ";".join(f"{float(x):.4f}" for x in arr)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    checkpoint_pairs = [_parse_checkpoint_arg(c) for c in args.checkpoints]
    n_sims        = args.n_sims
    bg_period     = args.bg_period
    ego_spawn_t   = args.ego_spawn_time

    # Append .zip if needed and validate paths
    validated = []
    for tag, path in checkpoint_pairs:
        full = path if path.endswith(".zip") else path + ".zip"
        if not os.path.exists(full):
            raise FileNotFoundError(f"Checkpoint not found: {full}")
        validated.append((tag, full))

    print(f"\n{'='*65}")
    print(" DETERMINISTIC VELOCITY-PROFILE EVALUATION")
    print(f"{'='*65}")
    for tag, path in validated:
        print(f"  {tag:<35} → {path}")
    print(f"\n  Episodes per version : {n_sims}")
    print(f"  BG spawn period      : {bg_period} s")
    print(f"  Ego spawn time       : {ego_spawn_t} s")
    print(f"  Output dir           : {output_dir}")
    print(f"  OS fd limit          : {resource.getrlimit(resource.RLIMIT_NOFILE)}\n")

    # ── Vehicle params ────────────────────────────────────────────────
    vehicles = VehicleParams()

    RL_cfp = SumoCarFollowingParams(
        speed_mode=0,
        accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau,
        min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev,
        impatience=0.0, car_follow_model="IDM",
    )
    NonRL_cfp = SumoCarFollowingParams(
        speed_mode=31,
        accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau,
        min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev,
        impatience=0.0, car_follow_model="IDM",
    )

    vehicles.add(
        veh_id="RL",
        acceleration_controller=(RLController, {}),
        initial_speed=0, num_vehicles=0,
        car_following_params=RL_cfp,
        lane_change_params=None, color="blue",
    )
    vehicles.add(
        veh_id="NonRL",
        acceleration_controller=(IDMController, {}),
        initial_speed=initial_speed, num_vehicles=0,
        car_following_params=NonRL_cfp,
        lane_change_params=None, color="green",
    )

    sim_params = SumoParams(
        port=None, sim_step=sim_step, lateral_resolution=None,
        no_step_log=True, render=args.render, save_render=False,
        sight_radius=25, show_radius=False, pxpm=2,
        force_color_update=False, overtake_right=False,
        seed=42,            # fixed seed for reproducibility
        restart_instance=True, print_warnings=False,
        teleport_time=0, num_clients=1,
        color_by_speed=False, use_ballistic=False,
    )

    env_params = EnvParams(
        additional_params={
            "max_accel": max_accel, "max_decel": max_decel,
            "target_velocity": max_speed, "sort_vehicles": False,
        },
        horizon=horizon,
        warmup_steps=warmup_steps,
        sims_per_step=1, evaluate=False, clip_actions=True,
    )

    initial_config = InitialConfig(
        shuffle=False, spacing="uniform", min_gap=12,
        perturbation=0.0,    # no perturbation — fully deterministic
        x0=5, bunching=0, lanes_distribution=float("inf"),
        edges_distribution=["E#L-X"],   # only west edge for initial placement
    )

    # ── Per-version CSV (one file per version tag) ────────────────────
    csv_paths = {}
    for tag, _ in validated:
        csv_path = os.path.join(output_dir, f"deterministic_{tag}.csv")
        csv_paths[tag] = csv_path
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

    # ── Load all models ONCE ──────────────────────────────────────────
    print("Loading models …")
    models = {}
    for tag, path in validated:
        print(f"  Loading {tag} …")
        models[tag] = PPO.load(path)
    print("All models loaded.\n")

    # ── Resolve env class per version ─────────────────────────────────
    from envs.alpha_env_deterministic import make_deterministic_env
    env_classes = {}
    for tag, _ in validated:
        try:
            env_classes[tag] = make_deterministic_env(tag)
            print(f"  {tag:<35} → {env_classes[tag].__name__}")
        except ValueError as e:
            raise ValueError(str(e))

    # ── Run episodes ──────────────────────────────────────────────────

    for tag, ckpt_path in validated:
        model    = models[tag]
        EnvClass = env_classes[tag]
        csv_path = csv_paths[tag]

        print(f"\n>>> {tag}  ({n_sims} episodes)")

        for run_idx in range(n_sims):
            _net_params = NetParams(
                osm_path=None,
                template=net_file,
                inflows=_build_inflows(ego_spawn_t),
            )

            def _make_env(
                _tag=tag, _bg=bg_period, _ego_t=ego_spawn_t,
                _net_p=_net_params, _EnvClass=EnvClass,
            ):
                _v = deepcopy(vehicles)
                _s = deepcopy(sim_params)
                _s.render = args.render
                net = DeterministicSouthNorthNetwork(
                    name="deterministic_eval",
                    vehicles=_v,
                    net_params=_net_p,
                    initial_config=initial_config,
                    traffic_lights=TrafficLightParams(),
                )
                return _EnvClass(
                    env_params=env_params,
                    sim_params=_s,
                    network=net,
                    simulator="traci",
                    bg_spawn_period=_bg,
                    ego_spawn_time=_ego_t,
                )

            env = None
            row = None
            try:
                env = DummyVecEnv([_make_env])
                obs = env.reset()
                done = False
                info = {}

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = env.step(action)
                    done = dones[0]
                    info = infos[0]

                telemetry = info.get("telemetry") or {}

                row = {
                    "run":                run_idx,
                    "version":            tag,
                    "ego_route":          telemetry.get("ego_route", "unknown"),
                    "episode_index":      telemetry.get("episode_index", run_idx),
                    "collision":          1 if telemetry.get("agent_collision", False) else 0,
                    "success":            1 if telemetry.get("agent_success", False) else 0,
                    "avg_speed":          f"{telemetry.get('agent_avg_speed', 0.0):.4f}",
                    "safe_gap":           f"{telemetry.get('agent_avg_safe_gap', 1.0):.4f}",
                    "travel_time":        f"{telemetry.get('agent_travel_time', 0.0):.4f}",
                    "waiting_time":       f"{telemetry.get('agent_waiting_time', 0.0):.4f}",
                    "bg_spawn_period":    bg_period,
                    "time_profile":       _fmt_profile(telemetry.get("agent_times", [])),
                    "distance_profile":   _fmt_profile(telemetry.get("agent_distances", [])),
                    "velocity_profile":   _fmt_profile(telemetry.get("agent_speeds", [])),
                    "jerk_profile":       _fmt_profile(telemetry.get("agent_jerks", [])),
                    "acceleration_profile": _fmt_profile(
                        telemetry.get("agent_accelerations", [])
                    ),
                }

            except Exception as exc:
                print(f"  Run {run_idx:02d} | ERROR: {exc}")

            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                _kill_stray_sumo()
                del env
                gc.collect()
                time.sleep(0.3)
                if run_idx % 10 == 0:
                    print(f"  [diag] open fds after run {run_idx}: {_open_fd_count()}")

            if row is None:
                print(f"  Run {run_idx:02d} | SKIPPED")
                continue

            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

            print(
                f"  Run {run_idx:02d} | route={row['ego_route']:<5} "
                f"col={row['collision']} suc={row['success']} "
                f"spd={row['avg_speed']} tt={row['travel_time']}"
            )

        print(f"  [CSV] → {csv_path}")

    print("\n--- DETERMINISTIC EVALUATION COMPLETE ---")

    # ── Auto-plot ─────────────────────────────────────────────────────
    try:
        plot_velocity_profiles(
            output_dir=output_dir,
            save_path=os.path.join(output_dir, "velocity_profiles.png"),
            show=False,
        )
    except Exception as e:
        print(f"Could not auto-plot: {e}")


if __name__ == "__main__":
    main()
