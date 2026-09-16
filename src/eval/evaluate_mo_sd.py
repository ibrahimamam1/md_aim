"""
evaluate_mo_sd.py
─────────────────
Comprehensive evaluation pipeline for Multi-Objective, State-Dependent RL agents.

Evaluates trained policies across:
  - Scenarios S1 to S4 (free flow, moderate, dense, sudden conflict)
  - Multi-objective Pareto weight sweep (w_l, w_s)
  - Comprehensive Section 15 evaluation metrics:
      Safety: collision rate, near-collision rate, min TTC, min safe gap, emergency braking frequency, unsafe interactions
      Efficiency: traversal time, waiting/delay time, average speed, stops count
      Comfort: average acceleration, max deceleration, mean jerk, jerk variance
"""

import argparse
import os
import sys
import csv
import json
import random
import time
import gc
import subprocess
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scenarios.traffic_scenarios import get_scenario_definition, create_scenario_env
from src.models.mo_sd_ppo import MOSDPPO
from stable_baselines3.common.vec_env import DummyVecEnv

CSV_HEADER = [
    "run", "scenario", "weight_l", "weight_s", "collision", "success",
    "avg_speed", "min_safe_gap", "min_ttc", "emergency_braking_count", "near_collision_count",
    "traversal_time", "waiting_time", "stops_count", "max_deceleration", "mean_abs_jerk", "jerk_variance",
    "progress_reward", "goal_reward", "waiting_penalty", "time_penalty", "gap_penalty", "collision_penalty",
    "total_long_term_reward", "total_safety_reward", "total_reward",
    "time_profile", "distance_profile", "velocity_profile", "jerk_profile", "acceleration_profile"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOSDPPO policies across scenarios.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint zip file.")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4"],
                        help="List of scenarios to evaluate (S1..S4).")
    parser.add_argument("--n_sims", type=int, default=20, help="Number of simulation runs per scenario.")
    parser.add_argument("--weights_l", nargs="+", type=float, default=[0.5],
                        help="List of efficiency weights w_l to test.")
    parser.add_argument("--render", action="store_true", default=False, help="Render SUMO simulation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Log evaluation metrics to wandb.")
    parser.add_argument("--wandb_project", type=str, default="md_aim", help="Wandb project.")
    return parser.parse_args()


def kill_stray_sumo():
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
    except Exception:
        pass


def evaluate():
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    checkpoint_path = args.checkpoint
    if not checkpoint_path.endswith(".zip"):
        checkpoint_path += ".zip"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    out_dir = args.output_dir or os.path.join(root_dir, "output", "eval_mo_sd")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print(" Multi-Objective Autonomous Intersection Management Evaluation")
    print(f" Checkpoint : {checkpoint_path}")
    print(f" Scenarios  : {args.scenarios}")
    print(f" Runs/scen  : {args.n_sims}")
    print(f" Output dir : {out_dir}")
    print("=" * 76 + "\n")

    # Load model once
    print("Loading MOSDPPO model...")
    model = MOSDPPO.load(checkpoint_path)
    print("Model loaded successfully.\n")

    all_scenario_summaries = {}

    for scen_id in args.scenarios:
        print(f"\n>>> Running Scenario {scen_id} ({args.n_sims} runs) <<<")
        csv_path = os.path.join(out_dir, f"eval_scenario_{scen_id}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

        scenario_runs = []

        for w_l in args.weights_l:
            w_s = max(0.0, 1.0 - w_l)

            for run_idx in range(args.n_sims):
                env = None
                row = None
                try:
                    # Create isolated scenario environment
                    env = DummyVecEnv([
                        lambda: create_scenario_env(
                            scenario_id=scen_id,
                            root_dir=root_dir,
                            render=args.render,
                            weight_l=w_l,
                            weight_s=w_s,
                        )
                    ])

                    obs = env.reset()
                    done = False
                    final_info = {}

                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, dones, infos = env.step(action)
                        done = dones[0]
                        final_info = infos[0]

                    mo_telemetry = final_info.get("mo_telemetry", {})
                    base_telemetry = final_info.get("telemetry", {})

                    def fmt_arr(arr):
                        return ";".join(f"{float(x):.3f}" for x in (arr or []))

                    row = {
                        "run": run_idx,
                        "scenario": scen_id,
                        "weight_l": f"{w_l:.2f}",
                        "weight_s": f"{w_s:.2f}",
                        "collision": int(mo_telemetry.get("collision", 0)),
                        "success": int(mo_telemetry.get("success", 0)),
                        "avg_speed": f"{mo_telemetry.get('average_speed', 0.0):.4f}",
                        "min_safe_gap": f"{mo_telemetry.get('min_safe_gap', 1.0):.4f}",
                        "min_ttc": f"{mo_telemetry.get('min_ttc', 99.0):.4f}",
                        "emergency_braking_count": int(mo_telemetry.get("emergency_braking_count", 0)),
                        "near_collision_count": int(mo_telemetry.get("near_collision_count", 0)),
                        "traversal_time": f"{mo_telemetry.get('traversal_time', 0.0):.4f}",
                        "waiting_time": f"{mo_telemetry.get('waiting_time', 0.0):.4f}",
                        "stops_count": int(mo_telemetry.get("stops_count", 0)),
                        "max_deceleration": f"{mo_telemetry.get('max_deceleration', 0.0):.4f}",
                        "mean_abs_jerk": f"{mo_telemetry.get('mean_abs_jerk', 0.0):.4f}",
                        "jerk_variance": f"{mo_telemetry.get('jerk_variance', 0.0):.4f}",
                        "progress_reward": f"{mo_telemetry.get('progress_reward', 0.0):.4f}",
                        "goal_reward": f"{mo_telemetry.get('goal_reward', 0.0):.4f}",
                        "waiting_penalty": f"{mo_telemetry.get('waiting_penalty', mo_telemetry.get('time_penalty', 0.0)):.4f}",
                        "time_penalty": f"{mo_telemetry.get('time_penalty', mo_telemetry.get('waiting_penalty', 0.0)):.4f}",
                        "gap_penalty": f"{mo_telemetry.get('gap_penalty', 0.0):.4f}",
                        "collision_penalty": f"{mo_telemetry.get('collision_penalty', 0.0):.4f}",
                        "total_long_term_reward": f"{mo_telemetry.get('total_long_term_reward', 0.0):.4f}",
                        "total_safety_reward": f"{mo_telemetry.get('total_safety_reward', 0.0):.4f}",
                        "total_reward": f"{mo_telemetry.get('total_reward', 0.0):.4f}",
                        "time_profile": fmt_arr(base_telemetry.get("agent_times", [])),
                        "distance_profile": fmt_arr(base_telemetry.get("agent_distances", [])),
                        "velocity_profile": fmt_arr(base_telemetry.get("agent_speeds", [])),
                        "jerk_profile": fmt_arr(base_telemetry.get("agent_jerks", [])),
                        "acceleration_profile": fmt_arr(base_telemetry.get("agent_accelerations", [])),
                    }

                except Exception as e:
                    print(f"  Run {run_idx:02d} | ERROR: {e}")

                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    kill_stray_sumo()
                    del env
                    gc.collect()
                    time.sleep(0.1)

                if row is not None:
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

                    scenario_runs.append({
                        "collision": float(row["collision"]),
                        "success": float(row["success"]),
                        "avg_speed": float(row["avg_speed"]),
                        "min_safe_gap": float(row["min_safe_gap"]),
                        "min_ttc": float(row["min_ttc"]),
                        "traversal_time": float(row["traversal_time"]),
                        "waiting_time": float(row["waiting_time"]),
                        "near_collision_count": float(row["near_collision_count"]),
                        "emergency_braking_count": float(row["emergency_braking_count"]),
                        "mean_abs_jerk": float(row["mean_abs_jerk"]),
                    })

                    print(f"  Run {run_idx:02d} | Col={row['collision']} Suc={row['success']} "
                          f"Spd={row['avg_speed']} m/s TT={row['traversal_time']} s "
                          f"MinTTC={row['min_ttc']} s SafeGap={row['min_safe_gap']}")

        # Compute scenario aggregates
        if scenario_runs:
            summary = {
                "collision_rate": float(np.mean([r["collision"] for r in scenario_runs])),
                "success_rate": float(np.mean([r["success"] for r in scenario_runs])),
                "average_speed": float(np.mean([r["avg_speed"] for r in scenario_runs])),
                "min_safe_gap": float(np.mean([r["min_safe_gap"] for r in scenario_runs])),
                "min_ttc": float(np.mean([r["min_ttc"] for r in scenario_runs])),
                "traversal_time": float(np.mean([r["traversal_time"] for r in scenario_runs])),
                "waiting_time": float(np.mean([r["waiting_time"] for r in scenario_runs])),
                "near_collision_mean": float(np.mean([r["near_collision_count"] for r in scenario_runs])),
                "emergency_braking_mean": float(np.mean([r["emergency_braking_count"] for r in scenario_runs])),
                "mean_abs_jerk": float(np.mean([r["mean_abs_jerk"] for r in scenario_runs])),
                "runs_count": len(scenario_runs),
            }
            all_scenario_summaries[scen_id] = summary

    # Save summary JSON
    summary_file = os.path.join(out_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_scenario_summaries, f, indent=2)

    print(f"\nEvaluation summary saved → {summary_file}")
    print("\n--- EVALUATION COMPLETE ---\n")


if __name__ == "__main__":
    evaluate()

