"""
v0_1_evaluate_all_rl.py
───────────────────────
All-RL-Agents evaluation: every vehicle at the intersection is controlled
by the same trained attention-continuous policy (parameter-sharing).

The environment subclass `AllRLAttentionEnv` overrides `_apply_non_rl_controls()`
so that instead of IDM, each non-RL vehicle's acceleration is computed by:
  1. Building an ego-centric observation via `_get_local_observation(veh_id)`
  2. Querying the shared SB3 policy with `model.predict(obs)`
  3. Denormalizing and applying the resulting acceleration

Usage:
    python src/test/v0_1_evaluate_all_rl.py \
        --checkpoint checkpoints/attention_continous/final_model \
        --n_sims 42 [--render]
"""

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

import numpy as np

# ─────────────── Raise OS file-descriptor limit ───────────────────────────────
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.asymetric_random import AsymmetricRandomNetwork

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from plot_eval_results import plot_eval_results

# ─────────────────────── CLI ───────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate with ALL vehicles controlled by the trained RL policy.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to the attention_continous checkpoint (without .zip).")
parser.add_argument("--n_sims", type=int, default=42, help="Runs per scenario combo.")
parser.add_argument("--render", action="store_true", default=False)
args = parser.parse_args() if __name__ == '__main__' else None

# ─────────────── Sim Params ────────────────────
min_gap = 2.5; max_accel = 2.6; max_decel = 4.5; max_speed = 55; initial_speed = 0
speed_factor = 1.0; speed_dev = 0.1; sigma = 0; tau = 0.8
horizon = 180; sim_step = 0.25; warmup_steps = 50

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
net_file = os.path.join(root_dir, "networks", "100m_right_before_left.net.xml")
output_dir = os.path.join(root_dir, "output_all_rl")
os.makedirs(output_dir, exist_ok=True)

# ─────────────── Scenarios ──────────
scenarios = {"rbl": "100m_right_before_left.net.xml"}
intentions = {"asymetric_random": AsymmetricRandomNetwork}

high_rate = 400; medium_rate = 275; low_rate = 150
traffic_rates = {
    "Sc1_All_low":     [{"N": low_rate,    "S": low_rate,    "W": low_rate,    "E": low_rate}],
    "Sc2_All_high_3H": [
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": medium_rate},
        {"N": high_rate,   "S": high_rate,   "W": medium_rate, "E": high_rate},
        {"N": high_rate,   "S": medium_rate, "W": high_rate,   "E": high_rate},
        {"N": medium_rate, "S": high_rate,   "W": high_rate,   "E": high_rate},
    ],
    "Sc3_All_medium":  [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}],
    "Sc4_Mixed_2H":    [
        {"N": high_rate,   "S": high_rate,   "W": low_rate,    "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": low_rate,    "W": high_rate,   "E": low_rate},
        {"N": low_rate,    "S": high_rate,   "W": low_rate,    "E": high_rate},
    ],
    "Sc5_Mixed_1H":    [
        {"N": high_rate,   "S": medium_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": high_rate,   "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": high_rate,   "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
    ],
    "Sc6_Mixed_ML":    [
        {"N": medium_rate, "S": medium_rate, "W": low_rate,    "E": low_rate},
        {"N": medium_rate, "S": low_rate,    "W": medium_rate, "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": medium_rate, "E": medium_rate},
    ],
}

CSV_HEADER = [
    "run", "collision", "success", "avg_speed", "travel_time", "waiting_time",
    "total_collisions", "num_collided_vehicles", "all_vehicles_avg_travel_time", "total_vehicles_spawned",
    "collision_rate",
    "time_profile", "distance_profile", "velocity_profile", "jerk_profile",
    "acceleration_profile",
]


import gymnasium as gym
from traci.exceptions import FatalTraCIError, TraCIException


# ═══════════════════════════════════════════════════════════════════════════════
# AllRLAttentionEnv — subclass that replaces IDM with policy inference
# ═══════════════════════════════════════════════════════════════════════════════
from envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention


class AllRLAttentionEnv(AlphaEnv_v01_Attention):
    """
    Environment where ALL vehicles are equivalent (no concept of ego vehicle)
    and controlled by a shared SB3 policy.
    The episode runs for the full 180 horizon steps without early termination.

    Set `self.shared_policy` to the loaded SB3 model *after* construction.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.shared_policy = None   # set externally after construction

    def _init_telemetry(self):
        super()._init_telemetry()
        self.all_veh_spawns = {}
        self.all_veh_finishes = {}
        self.all_collided_vehs = set()
        self.all_veh_speeds = []
        self.all_veh_waiting_time = 0.0

    def _update_telemetry_step(self):
        current_time = self.time_counter
        current_ids = set(self.k.vehicle.get_ids())

        for v in current_ids:
            if v not in self.all_veh_spawns:
                self.all_veh_spawns[v] = current_time

            speed = self.k.vehicle.get_speed(v)
            if speed is not None and speed >= 0:
                self.all_veh_speeds.append(float(speed))
                if speed < 0.1:
                    self.all_veh_waiting_time += self.sim_step

        for v in list(self.all_veh_spawns.keys()):
            if v not in current_ids and v not in self.all_veh_finishes:
                self.all_veh_finishes[v] = current_time

        colliding_ids = set(self.k.kernel_api.simulation.getCollidingVehiclesIDList())
        if colliding_ids:
            self.all_collided_vehs.update(colliding_ids)

    def _compute_telemetry_stats(self):
        for v in self.all_veh_spawns:
            if v not in self.all_veh_finishes:
                self.all_veh_finishes[v] = self.time_counter

        all_travel_times = [
            self.all_veh_finishes[v] - self.all_veh_spawns[v]
            for v in self.all_veh_spawns
        ]

        num_collided_vehs = len(self.all_collided_vehs)
        num_collisions = num_collided_vehs // 2 if num_collided_vehs > 0 else 0
        total_vehs = len(self.all_veh_spawns)
        collision_rate = float(num_collisions / total_vehs) if total_vehs > 0 else 0.0
        avg_tt_all = float(np.mean(all_travel_times)) if all_travel_times else 0.0
        avg_speed_all = float(np.mean(self.all_veh_speeds)) if self.all_veh_speeds else 0.0
        avg_waiting_all = float(self.all_veh_waiting_time / max(total_vehs, 1))

        return {
            "agent_success": (num_collisions == 0),
            "agent_collision": (num_collisions > 0),
            "agent_travel_time": avg_tt_all,
            "agent_waiting_time": avg_waiting_all,
            "agent_avg_speed": avg_speed_all,
            "total_collisions": num_collisions,
            "num_collided_vehicles": num_collided_vehs,
            "all_vehicles_avg_travel_time": avg_tt_all,
            "total_vehicles_spawned": total_vehs,
            "collision_rate": collision_rate,
            "agent_times": [],
            "agent_distances": [],
            "agent_speeds": [],
            "agent_accelerations": [],
            "agent_jerks": [],
        }

    def reset(self, *, seed=None, options=None):
        self._init_telemetry()
        gym.Env.reset(self, seed=seed)

        self.last_action = 0.0
        self.last_progress = 0.0
        self.last_neighbors_info = []
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        self.time_counter = 0
        self.rl_agent_spawned = True
        self.agent_id = None

        if self.should_render:
            self.sim_params.render = True
            self.restart_simulation(self.sim_params)

        if self.sim_params.restart_instance or (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            if seed is not None:
                self.sim_params.seed = seed
            else:
                self.sim_params.seed = random.randint(0, 100000)

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            self.k.junction = deepcopy(self.initial_junction)
            self.k.junction.master_kernel = self.k
            self.restart_simulation(self.sim_params)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        if self.simulator == 'traci':
            try:
                for veh_id in self.k.kernel_api.vehicle.getIDList():
                    self.k.vehicle.remove(veh_id)
            except Exception:
                pass

        self.k.vehicle.reset()

        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = self.initial_state[veh_id]
            try:
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)
            except (FatalTraCIError, TraCIException):
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)

        self.k.simulation.simulation_step()
        self.k.update(reset=True)

        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        return self.last_obs, {}

    def step(self, action=None):
        """
        Advance the environment by one step.
        All active vehicles in the simulation are controlled by the shared policy.
        The simulation runs for the full 180 horizon steps without early termination.
        """
        self.step_counter_within_rl_step = 0

        # 1. Apply policy actions to ALL active vehicles currently in the network
        if self.shared_policy is not None:
            max_accel = self.env_params.additional_params["max_accel"]
            max_decel = self.env_params.additional_params["max_decel"]

            self._update_routes()

            active_ids = self.k.vehicle.get_ids()
            veh_ids_to_control = []
            accels = []

            for veh_id in active_ids:
                if veh_id not in self.routes:
                    continue

                try:
                    obs, _ = self._get_local_observation(veh_id)
                    obs_input = obs.reshape(1, -1)
                    act, _ = self.shared_policy.predict(obs_input, deterministic=True)
                    action_val = float(act[0]) if isinstance(act, np.ndarray) else float(act)
                    if np.isnan(action_val) or np.isinf(action_val):
                        action_val = 0.0
                except Exception:
                    action_val = 0.0

                if action_val >= 0:
                    real_accel = action_val * max_accel
                else:
                    real_accel = action_val * max_decel

                veh_ids_to_control.append(veh_id)
                accels.append(real_accel)

            if veh_ids_to_control:
                self.k.vehicle.apply_acceleration(veh_ids_to_control, accels)

        if hasattr(self, "additional_command"):
            self.additional_command()

        # 2. Simulation Step (Inner Loop)
        for inner_step in range(self.env_params.sims_per_step):
            self.time_counter += self.sim_step
            self.step_counter += 1
            self.step_counter_within_rl_step = inner_step

            # Advance Simulator
            self.k.simulation.simulation_step()
            self.k.update(reset=False)

            self._update_telemetry_step()

            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

        # 3. Termination check: No early ego termination, run full 180-step horizon
        terminated = False
        truncated = (self.time_counter >= self.env_params.horizon)

        # 4. Compute telemetry stats at episode end
        telemetry_stats = None
        if truncated:
            telemetry_stats = self._compute_telemetry_stats()

        infos = {}
        if telemetry_stats is not None:
            infos["telemetry"] = telemetry_stats

        return self.last_obs, 0.0, terminated, truncated, infos


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_inflows(traffic_rate):
    inf = InFlows()
    inf.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"] / 3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"] / 3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"] / 3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#L-X", probability=traffic_rate["W"] / 3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    checkpoint_path = args.checkpoint
    if not checkpoint_path.endswith('.zip'):
        checkpoint_path += '.zip'
    n_sims = args.n_sims

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Vehicle params — ALL vehicles get speed_mode=0 (full RL control) ────
    vehicles = VehicleParams()

    # Both RL and NonRL vehicles use speed_mode=0 so the policy has full authority
    shared_cfp = SumoCarFollowingParams(
        speed_mode=0,          # ← no SUMO safety overrides
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
        car_following_params=shared_cfp,
        lane_change_params=None, color="blue",
    )
    vehicles.add(
        veh_id="NonRL",
        acceleration_controller=(IDMController, {}),
        initial_speed=initial_speed, num_vehicles=0,
        car_following_params=shared_cfp,   # ← speed_mode=0 here too
        lane_change_params=None, color="red",
    )

    sim_params = SumoParams(
        port=None, sim_step=sim_step, lateral_resolution=None,
        no_step_log=True, render=args.render, save_render=False,
        sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
        overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
        teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False,
    )

    env_params = EnvParams(
        additional_params={
            "max_accel": max_accel, "max_decel": max_decel,
            "target_velocity": max_speed, "sort_vehicles": False,
        },
        horizon=horizon, warmup_steps=warmup_steps,
        sims_per_step=1, evaluate=False, clip_actions=True,
    )

    initial_config = InitialConfig(
        shuffle=False, spacing="uniform", min_gap=12,
        perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
        edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"],
    )

    print(f"\n{'='*65}")
    print(f" ALL-RL EVALUATION — Every vehicle uses the trained policy")
    print(f"{'='*65}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Sims/scenario: {n_sims}")
    print(f"Output dir : {output_dir}")
    print(f"OS fd limit: {resource.getrlimit(resource.RLIMIT_NOFILE)}\n")

    # ── Load model ONCE ─────────────────────────────────────────────────────
    print("Loading model from checkpoint (once)...")
    base_model = PPO.load(checkpoint_path)
    print("Model loaded.\n")

    version_tag = "attention_continous_all_rl"

    for scen_key, scen_net_file in scenarios.items():
        for int_key, int_class in intentions.items():
            for rate_key, rate_list in traffic_rates.items():
                group_name = f"{scen_key}_{int_key}_{rate_key}_{version_tag}"
                csv_path = os.path.join(output_dir, f"{group_name}.csv")

                with open(csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

                print(f"\n>>> {group_name} ({n_sims} runs)")

                for run_idx in range(n_sims):
                    current_flow = random.choice(rate_list)

                    _net_file = os.path.join(root_dir, "networks", scen_net_file)
                    _net_params = NetParams(
                        osm_path=None, template=_net_file,
                        inflows=_build_inflows(current_flow),
                    )

                    flow_params = dict(
                        exp_tag="eval_all_rl",
                        network=int_class,
                        simulator="traci",
                        sim=sim_params,
                        env=env_params,
                        net=_net_params,
                        veh=vehicles,
                        initial=initial_config,
                    )

                    def _make_env():
                        p = flow_params
                        _v = deepcopy(p["veh"])
                        _n = p["net"]
                        _s = deepcopy(p["sim"])
                        _s.render = args.render
                        net = p["network"](
                            name="eval_all_rl",
                            vehicles=_v,
                            net_params=_n,
                            initial_config=p.get("initial", InitialConfig()),
                            traffic_lights=p.get("tls", TrafficLightParams()),
                        )
                        env = AllRLAttentionEnv(
                            env_params=p["env"],
                            sim_params=_s,
                            network=net,
                            simulator=p["simulator"],
                        )
                        # Attach the shared policy so _apply_non_rl_controls uses it
                        env.shared_policy = base_model
                        return env

                    # ── Run with guaranteed teardown ──────────────────────
                    env = None
                    row = None
                    try:
                        env = DummyVecEnv([_make_env])
                        obs = env.reset()
                        done = False
                        info = {}

                        while not done:
                            action, _states = base_model.predict(obs, deterministic=True)
                            obs, reward, dones, infos = env.step(action)
                            done = dones[0]
                            info = infos[0]

                        telemetry = info.get("telemetry") or {}

                        def _fmt_profile(arr):
                            return ";".join(f"{float(x):.4f}" for x in arr)

                        row = {
                            "run":                          run_idx,
                            "collision":                    1 if telemetry.get("agent_collision", False) else 0,
                            "success":                      1 if telemetry.get("agent_success", False) else 0,
                            "avg_speed":                    f"{telemetry.get('agent_avg_speed', 0.0):.4f}",
                            "travel_time":                  f"{telemetry.get('agent_travel_time', 0.0):.4f}",
                            "waiting_time":                 f"{telemetry.get('agent_waiting_time', 0.0):.4f}",
                            "total_collisions":             telemetry.get("total_collisions", 0),
                            "num_collided_vehicles":        telemetry.get("num_collided_vehicles", 0),
                            "all_vehicles_avg_travel_time": f"{telemetry.get('all_vehicles_avg_travel_time', 0.0):.4f}",
                            "total_vehicles_spawned":       telemetry.get("total_vehicles_spawned", 0),
                            "collision_rate":               f"{telemetry.get('collision_rate', 0.0):.6f}",
                            "time_profile":                 _fmt_profile(telemetry.get("agent_times", [])),
                            "distance_profile":             _fmt_profile(telemetry.get("agent_distances", [])),
                            "velocity_profile":             _fmt_profile(telemetry.get("agent_speeds", [])),
                            "jerk_profile":                 _fmt_profile(telemetry.get("agent_jerks", [])),
                            "acceleration_profile":         _fmt_profile(telemetry.get("agent_accelerations", [])),
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
                        print(f"  Run {run_idx:02d} | SKIPPED (no telemetry — see error above)")
                        continue

                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

                    print(f"  Run {run_idx:02d} | col={row['collision']} (sys_cols={row['total_collisions']}, rate={row['collision_rate']}) | "
                          f"ego_tt={row['travel_time']} all_veh_tt={row['all_vehicles_avg_travel_time']}")

                print(f"  [CSV] → {csv_path}")

    print("\n--- ALL-RL EVALUATION COMPLETE ---")

    try:
        plot_eval_results(
            output_dir=output_dir,
            version_filter=None,
            intention_filter="asymetric_random",
            save_path=os.path.join(output_dir, "eval_results_all_rl.png"),
            show=False,
        )
    except Exception as e:
        print(f"Could not plot results automatically: {e}")


if __name__ == "__main__":
    main()
