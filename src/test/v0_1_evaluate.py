import argparse
import os
import sys
import csv
import random
import gc
import resource
import subprocess
import time
from copy import deepcopy

# ─────────────── Raise OS file-descriptor limit ───────────────────────────────
# Must happen before any other imports that open files/sockets.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.uniform_random import UniformRandomNetwork
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.asymetric_random import AsymmetricRandomNetwork

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

# SB3 Imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from plot_eval_results import plot_eval_results

# ─────────────────────── CLI ───────────────────────
parser = argparse.ArgumentParser(description="Evaluate trained v0.1 agent.")
parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (without .zip).")
parser.add_argument("--version", required=True,
                    choices=["heuristic_continous", "heuristic_discrete",
                             "attention_continous", "attention_discrete",
                             "heuristic_attention_continous", "heuristic_attention_discrete"])
parser.add_argument("--n_sims", type=int, default=42, help="Runs per scenario combo.")
parser.add_argument("--render", action="store_true", default=False)
args = parser.parse_args()

# ─────────────── Sim Params ────────────────────
min_gap=2.5; max_accel=2.6; max_decel=4.5; max_speed=55; initial_speed=0
speed_factor=1.0; speed_dev=0.1; sigma=0; tau=0.8; horizon=180; sim_step=0.25
warmup_steps=50

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
net_file = os.path.join(root_dir, "networks", "100m_right_before_left.net.xml")
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# ─────────────── Scenarios ──────────
scenarios = {"rbl": "100m_right_before_left.net.xml"}

intentions = {
    "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
    "uniform_random": UniformRandomNetwork,
    "asymetric_random": AsymmetricRandomNetwork
}

high_rate=400; medium_rate=275; low_rate=150
traffic_rates = {
    "Sc1_All_low":    [{"N": low_rate, "S": low_rate, "W": low_rate, "E": low_rate}],
    "Sc2_All_high_3H": [
        {"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
        {"N": high_rate, "S": high_rate, "W": high_rate, "E": medium_rate},
        {"N": high_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
        {"N": high_rate, "S": medium_rate, "W": high_rate, "E": high_rate},
        {"N": medium_rate, "S": high_rate, "W": high_rate, "E": high_rate},
    ],
    "Sc3_All_medium": [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}],
    "Sc4_Mixed_2H": [
        {"N": high_rate, "S": high_rate, "W": low_rate, "E": low_rate},
        {"N": low_rate, "S": low_rate, "W": high_rate, "E": high_rate},
        {"N": high_rate, "S": low_rate, "W": high_rate, "E": low_rate},
        {"N": low_rate, "S": high_rate, "W": low_rate, "E": high_rate},
    ],
    "Sc5_Mixed_1H": [
        {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
    ],
    "Sc6_Mixed_ML": [
        {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
        {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
        {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
    ],
}

CSV_HEADER = [
    "run", "collision", "success", "avg_speed", "travel_time", "waiting_time",
]

# ─────────────── Version-specific setup ──────────────
def _get_env_class(version):
    if version == "heuristic_continous":
        from envs.alpha_env_v01 import AlphaEnv_v01
        return AlphaEnv_v01
    elif version == "heuristic_discrete":
        from envs.alpha_env_v01_discrete import AlphaEnv_v01_Discrete
        return AlphaEnv_v01_Discrete
    elif version == "attention_continous":
        from envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
        return AlphaEnv_v01_Attention
    elif version == "attention_discrete":
        from envs.alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete
        return AlphaEnv_v01_AttentionDiscrete
    elif version == "heuristic_attention_continous":
        from envs.alpha_env_v01_heuristic_attention_continous import AlphaEnv_v01_HeuristicAttention
        return AlphaEnv_v01_HeuristicAttention
    elif version == "heuristic_attention_discrete":
        from envs.alpha_env_v01_heuristic_attention_discrete import AlphaEnv_v01_HeuristicAttentionDiscrete
        return AlphaEnv_v01_HeuristicAttentionDiscrete


def _build_inflows(traffic_rate):
    inf = InFlows()
    inf.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="RL", edge="E#L-X", probability=0.8,
            depart_lane=0, depart_speed=initial_speed, begin=warmup_steps, color="green")
    return inf


def _kill_stray_sumo():
    """Kill any orphaned SUMO processes left over from a crashed run."""
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
    except Exception:
        pass


def _open_fd_count():
    """Return the number of open file descriptors for this process (diagnostic)."""
    try:
        import psutil
        return psutil.Process().num_fds()
    except Exception:
        return -1


def main():
    version = args.version
    checkpoint_path = args.checkpoint
    if not checkpoint_path.endswith('.zip'):
        checkpoint_path += '.zip'
    n_sims = args.n_sims

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    EnvClass = _get_env_class(version)

    # ── Build shared vehicle/sim/env params (never mutated per-run) ──────────
    vehicles = VehicleParams()
    RL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev, impatience=0.0,
        car_follow_model="IDM")
    NonRL_cfp = SumoCarFollowingParams(speed_mode=31, accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev, impatience=0.0,
        car_follow_model="IDM")
    vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
        initial_speed=0, num_vehicles=0, car_following_params=RL_cfp,
        lane_change_params=None, color="blue")
    vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
        initial_speed=initial_speed, num_vehicles=0, car_following_params=NonRL_cfp,
        lane_change_params=None, color="red")

    sim_params = SumoParams(
        port=None, sim_step=sim_step, lateral_resolution=None,
        no_step_log=True, render=args.render, save_render=False,
        sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
        overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
        teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False)

    env_params = EnvParams(
        additional_params={"max_accel": max_accel, "max_decel": max_decel,
                           "target_velocity": max_speed, "sort_vehicles": False},
        horizon=horizon, warmup_steps=warmup_steps,
        sims_per_step=1, evaluate=False, clip_actions=True)

    initial_config = InitialConfig(shuffle=False, spacing="uniform", min_gap=12,
        perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
        edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"])

    print(f"\n--- EVALUATION START ---")
    print(f"Version: {version}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sims per scenario: {n_sims}")
    print(f"OS fd limit: {resource.getrlimit(resource.RLIMIT_NOFILE)}\n")

    # ── Load model ONCE — avoids reopening the zip on every run ──────────────
    print("Loading model from checkpoint (once)...")
    base_model = PPO.load(checkpoint_path)
    print("Model loaded.\n")

    for scen_key, scen_net_file in scenarios.items():
        for int_key, int_class in intentions.items():
            for rate_key, rate_list in traffic_rates.items():
                group_name = f"{scen_key}_{int_key}_{rate_key}_{version}"
                csv_path = os.path.join(output_dir, f"{group_name}.csv")

                with open(csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

                print(f"\n>>> {group_name} ({n_sims} runs)")

                for run_idx in range(n_sims):
                    current_flow = random.choice(rate_list)

                    _net_file = os.path.join(root_dir, "networks", scen_net_file)
                    _net_params = NetParams(osm_path=None, template=_net_file,
                                           inflows=_build_inflows(current_flow))

                    flow_params = dict(
                        exp_tag="eval", network=int_class, simulator="traci",
                        sim=sim_params, env=env_params, net=_net_params,
                        veh=vehicles, initial=initial_config,
                    )

                    def _make_env():
                        p = flow_params
                        _v = deepcopy(p["veh"])
                        _n = p["net"]
                        _s = deepcopy(p["sim"])
                        _s.render = args.render
                        net = p["network"](
                            name="eval", vehicles=_v, net_params=_n,
                            initial_config=p.get("initial", InitialConfig()),
                            traffic_lights=p.get("tls", TrafficLightParams()),
                        )
                        return EnvClass(
                            env_params=p["env"], sim_params=_s,
                            network=net, simulator=p["simulator"],
                        )

                    # ── Run with guaranteed teardown ──────────────────────────
                    env = None
                    row = None
                    try:
                        env = DummyVecEnv([_make_env])

                        # Do NOT call set_env() — the model was trained with n_envs>1
                        # and set_env() enforces a matching count. For inference we
                        # only need the policy network; the env is driven manually below.

                        obs = env.reset()
                        done = False
                        info = {}

                        while not done:
                            action, _states = base_model.predict(obs, deterministic=True)
                            obs, reward, dones, infos = env.step(action)
                            done = dones[0]
                            info = infos[0]

                        telemetry = info.get("telemetry", {})
                        row = {
                            "run":          run_idx,
                            "collision":    1 if telemetry.get("agent_collision", False) else 0,
                            "success":      1 if telemetry.get("agent_success",   False) else 0,
                            "avg_speed":    f"{telemetry.get('agent_avg_speed',    0.0):.4f}",
                            "travel_time":  f"{telemetry.get('agent_travel_time',  0.0):.4f}",
                            "waiting_time": f"{telemetry.get('agent_waiting_time', 0.0):.4f}",
                        }

                    except Exception as exc:
                        print(f"  Run {run_idx:02d} | ERROR: {exc}")

                    finally:
                        # 1. Close the vectorised env (shuts down TraCI gracefully)
                        try:
                            if env is not None:
                                env.close()
                        except Exception:
                            pass

                        # 2. Kill any SUMO processes that survived a crash
                        _kill_stray_sumo()

                        # 3. Release the env object and trigger GC
                        del env
                        gc.collect()

                        # 4. Brief pause — gives the OS time to reclaim fds/ports
                        time.sleep(0.3)

                        # 5. Diagnostic: log fd count every 10 runs
                        if run_idx % 10 == 0:
                            print(f"  [diag] open fds after run {run_idx}: {_open_fd_count()}")
                    # ── end guaranteed teardown ───────────────────────────────

                    if row is None:
                        print(f"  Run {run_idx:02d} | SKIPPED (no telemetry — see error above)")
                        continue

                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

                    print(f"  Run {run_idx:02d} | col={row['collision']} suc={row['success']}"
                          f" spd={row['avg_speed']} tt={row['travel_time']}")

                print(f"  [CSV] → {csv_path}")

    print("\n--- EVALUATION COMPLETE ---")

    try:
        plot_eval_results(
            output_dir=output_dir,
            version_filter=version,
            save_path=os.path.join(output_dir, f"heuristic_discrete.png"),
            show=False,
        )
    except Exception as e:
        print(f"Could not plot results automatically: {e}")


if __name__ == "__main__":
    main()
