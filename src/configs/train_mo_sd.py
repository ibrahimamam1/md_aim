"""
train_mo_sd.py
──────────────
Unified training script for Multi-Objective, State-Dependent Discounting for Autonomous Intersection Management.

Supported Experimental Modes:
  - baseline: Fixed single discount γ_0 ∈ {0.90, 0.95, 0.97, 0.99, 0.995}.
  - exp_a:    State-dependent single discount γ(s).
  - exp_b:    Multi-objective state-dependent discount [γ_l, γ_s(s)] (Core formulation).
  - exp_c:    Learnable discount factors γ_φ(s) with anti-cheating regularized loss.
  - ablation: State-dependent reward weighting λ(s) with fixed discount.

Usage Examples:
  # Baseline with γ_0=0.97:
  python src/configs/train_mo_sd.py --mode baseline --gamma_0 0.97

  # Experiment A (State-dependent single discount):
  python src/configs/train_mo_sd.py --mode exp_a --gamma_0 0.99 --gamma_s_danger 0.0

  # Experiment B (Core multi-objective state-dependent discount):
  python src/configs/train_mo_sd.py --mode exp_b --gamma_l 0.99 --gamma_s_normal 0.95 --gamma_s_danger 0.0 --weight_l 0.5 --weight_s 0.5

  # Experiment C (Learnable discount network):
  python src/configs/train_mo_sd.py --mode exp_c --gamma_l 0.99 --gamma_s_normal 0.95

  # Ablation (State-dependent reward weighting):
  python src/configs/train_mo_sd.py --mode ablation --gamma_0 0.99 --lambda_danger 5.0
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from networks.asymetric_random import AsymmetricRandomNetwork

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.envs.alpha_env_mo_sd import AlphaEnv_MO_SD
from src.models.attention_model import AttentionFeatureExtractor
from src.models.mo_sd_models import MultiObjectiveActorCriticPolicy
from src.models.mo_sd_ppo import MOSDPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Objective State-Dependent RL Training")
    parser.add_argument("--mode", type=str, default="exp_b",
                        choices=["baseline", "exp_a", "exp_b", "exp_c", "ablation"],
                        help="Experimental condition.")
    # Discount parameters
    parser.add_argument("--gamma_0", type=float, default=0.99, help="Fixed baseline discount factor.")
    parser.add_argument("--gamma_l", type=float, default=0.99, help="Long-term efficiency discount.")
    parser.add_argument("--gamma_s_normal", type=float, default=0.95, help="Safety discount in normal conditions.")
    parser.add_argument("--gamma_s_danger", type=float, default=0.0, help="Safety discount during imminent conflict.")

    # Multi-objective Pareto weights
    parser.add_argument("--weight_l", type=float, default=0.5, help="Efficiency Pareto weight w_l.")
    parser.add_argument("--weight_s", type=float, default=0.5, help="Safety Pareto weight w_s.")

    # Calibrated reward parameters
    parser.add_argument("--collision_penalty", type=float, default=20.0, help="Catastrophic collision penalty R_c.")
    parser.add_argument("--gap_penalty_weight", type=float, default=0.25, help="Dangerous proximity gap penalty weight λ_gap.")
    parser.add_argument("--ttc_penalty_weight", type=float, default=0.5, help="Critical TTC penalty weight λ_TTC.")
    parser.add_argument("--progress_weight", type=float, default=10.0, help="Traversal progress reward weight w_p.")
    parser.add_argument("--goal_reward", type=float, default=20.0, help="Terminal goal reward w_g.")
    parser.add_argument("--time_cost", type=float, default=0.01, help="Per-timestep time cost w_t.")

    # Ablation reward weighting parameters
    parser.add_argument("--lambda_danger", type=float, default=5.0, help="Safety reward multiplier in conflict.")
    parser.add_argument("--lambda_normal", type=float, default=1.0, help="Safety reward multiplier when safe.")

    # Training hyperparameters
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel env workers.")
    parser.add_argument("--n_steps", type=int, default=1024, help="Steps per rollout per worker.")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Floor learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Network / Model
    parser.add_argument("--version", type=str, default="attention_continuous",
                        choices=["attention_continuous", "heuristic_continuous"],
                        help="Model architecture variant.")

    # Logging and checkpoints
    parser.add_argument("--note", type=str, default="", help="Experiment description note.")
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="md_aim", help="Weights & Biases project name.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Root folder for saving checkpoints.")

    return parser.parse_args()


class MOTrafficCallback(BaseCallback):
    """
    Callback aggregating Section 15 telemetry across completed episodes and logging to TB/wandb.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episodes = 0
        self._successes = 0
        self._collisions = 0
        self._sum_speed = 0.0
        self._sum_safe_gap = 0.0
        self._sum_min_ttc = 0.0
        self._sum_braking = 0
        self._sum_traversal_time = 0.0
        self._sum_mean_jerk = 0.0
        # Detailed reward component accumulators
        self._sum_progress_reward = 0.0
        self._sum_goal_reward = 0.0
        self._sum_time_penalty = 0.0
        self._sum_gap_penalty = 0.0
        self._sum_collision_penalty = 0.0
        self._sum_long_term_reward = 0.0
        self._sum_safety_reward = 0.0
        self._sum_total_reward = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            mo_tele = info.get("mo_telemetry")
            if mo_tele is None:
                continue

            self._episodes += 1
            if mo_tele.get("success", 0) == 1:
                self._successes += 1
            if mo_tele.get("collision", 0) == 1:
                self._collisions += 1

            self._sum_speed += float(mo_tele.get("average_speed", 0.0))
            self._sum_safe_gap += float(mo_tele.get("min_safe_gap", 1.0))
            self._sum_min_ttc += float(mo_tele.get("min_ttc", 99.0))
            self._sum_braking += int(mo_tele.get("emergency_braking_count", 0))
            self._sum_traversal_time += float(mo_tele.get("traversal_time", 0.0))
            self._sum_mean_jerk += float(mo_tele.get("mean_abs_jerk", 0.0))

            # Reward diagnostics
            self._sum_progress_reward += float(mo_tele.get("progress_reward", 0.0))
            self._sum_goal_reward += float(mo_tele.get("goal_reward", 0.0))
            self._sum_time_penalty += float(mo_tele.get("time_penalty", 0.0))
            self._sum_gap_penalty += float(mo_tele.get("gap_penalty", 0.0))
            self._sum_collision_penalty += float(mo_tele.get("collision_penalty", 0.0))
            self._sum_long_term_reward += float(mo_tele.get("total_long_term_reward", 0.0))
            self._sum_safety_reward += float(mo_tele.get("total_safety_reward", 0.0))
            self._sum_total_reward += float(mo_tele.get("total_reward", 0.0))
        return True

    def _on_rollout_end(self) -> bool:
        n = max(self._episodes, 1)
        if self._episodes > 0:
            self.logger.record("traffic/completed_episodes", self._episodes)
            self.logger.record("traffic/success_rate", self._successes / n)
            self.logger.record("traffic/collision_rate", self._collisions / n)
            self.logger.record("traffic/average_speed", self._sum_speed / n)
            self.logger.record("traffic/min_safe_gap", self._sum_safe_gap / n)
            self.logger.record("traffic/min_ttc", self._sum_min_ttc / n)
            self.logger.record("traffic/emergency_braking_per_ep", self._sum_braking / n)
            self.logger.record("traffic/traversal_time", self._sum_traversal_time / n)
            self.logger.record("traffic/mean_abs_jerk", self._sum_mean_jerk / n)
            # Detailed episode reward decomposition logs
            self.logger.record("rewards/progress_reward", self._sum_progress_reward / n)
            self.logger.record("rewards/goal_reward", self._sum_goal_reward / n)
            self.logger.record("rewards/time_penalty", self._sum_time_penalty / n)
            self.logger.record("rewards/gap_penalty", self._sum_gap_penalty / n)
            self.logger.record("rewards/collision_penalty", self._sum_collision_penalty / n)
            self.logger.record("rewards/total_long_term_reward", self._sum_long_term_reward / n)
            self.logger.record("rewards/total_safety_reward", self._sum_safety_reward / n)
            self.logger.record("rewards/total_reward", self._sum_total_reward / n)

        # Reset accumulators for next rollout
        self._episodes = 0
        self._successes = 0
        self._collisions = 0
        self._sum_speed = 0.0
        self._sum_safe_gap = 0.0
        self._sum_min_ttc = 0.0
        self._sum_braking = 0
        self._sum_traversal_time = 0.0
        self._sum_mean_jerk = 0.0
        self._sum_progress_reward = 0.0
        self._sum_goal_reward = 0.0
        self._sum_time_penalty = 0.0
        self._sum_gap_penalty = 0.0
        self._sum_collision_penalty = 0.0
        self._sum_long_term_reward = 0.0
        self._sum_safety_reward = 0.0
        self._sum_total_reward = 0.0
        return True


def linear_schedule(initial_value: float, min_value: float):
    def func(progress_remaining: float) -> float:
        return max(min_value, progress_remaining * initial_value)
    return func


def create_env_factory(args, render=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net_file = os.path.join(root_dir, "networks", "100m_skewed_right_before_left.net.xml")

    # Inflows for training (nominal high traffic)
    inflow = InFlows()
    inflow.add(veh_type="NonRL", edge="E#T-X", probability=400.0 / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#R-X", probability=400.0 / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#D-X", probability=400.0 / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    # No background traffic on the RL agent's spawn edge (West / E#L-X): rate 0.
    inflow.add(veh_type="NonRL", edge="E#L-X", probability=0.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="RL", edge="E#L-X", probability=0.3,
               depart_lane=0, depart_speed=0, begin=50, color="blue")

    vehicles = VehicleParams()
    rl_cfp = SumoCarFollowingParams(
        speed_mode=0, accel=2.6, decel=4.5, sigma=0.0, tau=0.8,
        min_gap=2.5, max_speed=55.0, speed_factor=1.0, speed_dev=0.1,
        impatience=0.0, car_follow_model="IDM",
    )
    non_rl_cfp = SumoCarFollowingParams(
        speed_mode=0, accel=2.6, decel=4.5, sigma=0.0, tau=0.8,
        min_gap=2.5, max_speed=55.0, speed_factor=1.0, speed_dev=0.1,
        impatience=0.0, car_follow_model="IDM",
    )
    vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
                 initial_speed=0, num_vehicles=0, car_following_params=rl_cfp,
                 lane_change_params=None, color="blue")
    vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
                 initial_speed=0, num_vehicles=0, car_following_params=non_rl_cfp,
                 lane_change_params=None, color="red")

    net_params = NetParams(osm_path=None, template=net_file, inflows=inflow)
    initial_config = InitialConfig(
        shuffle=False, spacing="uniform", min_gap=12, perturbation=5.0,
        x0=5, bunching=0, lanes_distribution=float("inf"),
        edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"],
    )
    env_params = EnvParams(
        additional_params={"max_accel": 2.6, "max_decel": 4.5,
                           "target_velocity": 55.0, "sort_vehicles": False},
        horizon=180, warmup_steps=5, sims_per_step=1, evaluate=False, clip_actions=True,
    )
    sim_params = SumoParams(
        port=None, sim_step=0.25, lateral_resolution=None, no_step_log=True,
        render=render, save_render=False, sight_radius=25, show_radius=False,
        pxpm=2, force_color_update=False, overtake_right=False, seed=args.seed,
        restart_instance=True, print_warnings=False, teleport_time=0, num_clients=1,
        color_by_speed=False, use_ballistic=False,
    )

    env_mode = "ablation_reward_adaptation" if args.mode == "ablation" else (
        "baseline" if args.mode == "baseline" else "multi_objective"
        "baseline" if args.mode in ("baseline", "exp_a") else "multi_objective"
    )

    def _make():
        net = AsymmetricRandomNetwork(
            name="AlphaEnv-Train",
            vehicles=deepcopy(vehicles),
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=TrafficLightParams(),
        )
        env = AlphaEnv_MO_SD(
            env_params=env_params,
            sim_params=deepcopy(sim_params),
            network=net,
            simulator="traci",
            mode=env_mode,
            weight_l=args.weight_l,
            weight_s=args.weight_s,
            lambda_danger=args.lambda_danger,
            lambda_normal=args.lambda_normal,
            collision_penalty=args.collision_penalty,
            gap_penalty_weight=args.gap_penalty_weight,
            ttc_penalty_weight=args.ttc_penalty_weight,
            progress_weight=args.progress_weight,
            goal_reward=args.goal_reward,
            time_cost=args.time_cost,
        )
        return Monitor(env)

    return _make


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.mode}_g0_{args.gamma_0}_gl_{args.gamma_l}_gs_{args.gamma_s_normal}_wl_{args.weight_l}_{timestamp}"

    checkpoint_root = args.checkpoint_dir or os.path.join(root_dir, "checkpoints", "mo_sd", run_name)
    tensorboard_dir = os.path.join(root_dir, "tensorboard_logs", "mo_sd", run_name)
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Save experiment configuration metadata
    config_path = os.path.join(checkpoint_root, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 76)
    print(f" Multi-Objective, State-Dependent RL Training [{args.mode.upper()}]")
    print(f" Run name        : {run_name}")
    print(f" Checkpoint root : {checkpoint_root}")
    print(f" Tensorboard dir : {tensorboard_dir}")
    print(f" Mode            : {args.mode}")
    print(f" Discounts       : gamma_0={args.gamma_0}, gamma_l={args.gamma_l}, gamma_s_normal={args.gamma_s_normal}, gamma_s_danger={args.gamma_s_danger}")
    print(f" Pareto weights  : weight_l={args.weight_l}, weight_s={args.weight_s}")
    print(f" Timesteps       : {args.timesteps} across {args.num_workers} workers")
    print("=" * 76 + "\n")

    # W&B Initialization
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                notes=args.note,
                config=vars(args),
                sync_tensorboard=True,
                save_code=True,
            )
            print("[wandb] initialized successfully.")
        except Exception as e:
            print(f"[wandb] init skipped or failed ({e}); continuing with TensorBoard.")

    # Vectorized environments
    env_fn = create_env_factory(args)
    if args.num_workers > 1:
        vec_env = SubprocVecEnv([env_fn for _ in range(args.num_workers)])
    else:
        vec_env = DummyVecEnv([env_fn])

    # Policy kwargs
    if args.version == "attention_continuous":
        policy_kwargs = dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                ego_features=4,
                neighbor_features=5,
                max_neighbors=5,
                embed_dim=64,
                num_heads=4,
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
    else:
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Model instantiation
    model = MOSDPPO(
        policy=MultiObjectiveActorCriticPolicy,
        env=vec_env,
        mode=args.mode,
        gamma_0=args.gamma_0,
        gamma_l=args.gamma_l,
        gamma_s_normal=args.gamma_s_normal,
        gamma_s_danger=args.gamma_s_danger,
        weight_l=args.weight_l,
        weight_s=args.weight_s,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(args.learning_rate, args.min_learning_rate),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.25,
        max_grad_norm=0.5,
        ent_coef=0.01,
        tensorboard_log=tensorboard_dir,
        verbose=1,
    )

    callbacks = [MOTrafficCallback()]
    if wandb_run is not None:
        try:
            from wandb.integration.sb3 import WandbCallback
            callbacks.append(WandbCallback(model_save_path=checkpoint_root))
        except Exception:
            pass

    # Execute training
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    finally:
        final_model_path = os.path.join(checkpoint_root, "final_model")
        model.save(final_model_path)
        print(f"\nModel checkpoint saved → {final_model_path}.zip")
        vec_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print("\n--- TRAINING COMPLETE ---\n")


if __name__ == "__main__":
    main()

