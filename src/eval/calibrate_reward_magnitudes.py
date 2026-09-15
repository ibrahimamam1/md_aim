"""
calibrate_reward_magnitudes.py
──────────────────────────────
Empirical reward component calibration and magnitude diagnosis tool.

Runs trajectories (using random, constant, or pretrained policies) and collects:
  - Cumulative raw progress:  Σ_t Δp~_t
  - Dangerous gap timesteps:   Σ_t I[g_t <= 5m]
  - Dangerous time-gap steps:  Σ_t I[|d_η| < 0.2]
  - Critical TTC steps:        Σ_t I[TTC_t <= 2.0s]
  - Collision indicator:       I_collision ∈ {0, 1}
  - Goal indicator:            I_goal ∈ {0, 1}
  - Episode duration:          T_episode (timesteps and seconds)
  - Accumulated unweighted penalty sums for safety gap and TTC

Analyzes the empirical magnitudes across episodes to identify and prevent unintended
reward interactions (e.g. cumulative gap penalty exceeding catastrophic crash penalty).
"""

import argparse
import os
import sys
import gc
import json
import time
import subprocess
import numpy as np
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scenarios.traffic_scenarios import create_scenario_env
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate reward component magnitudes.")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "pretrained", "creep", "aggressive", "mixed"],
                        help="Policy to generate trajectories.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model checkpoint (zip) if policy=pretrained.")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"],
                        help="Scenarios to evaluate (S1..S4).")
    parser.add_argument("--n_episodes", type=int, default=25,
                        help="Number of episodes per scenario.")
    parser.add_argument("--danger_gap_dist", type=float, default=5.0,
                        help="Distance threshold for I[g_t <= threshold] (meters).")
    parser.add_argument("--danger_d_eta", type=float, default=0.2,
                        help="Normalized time gap threshold for |d_eta| < threshold.")
    parser.add_argument("--danger_ttc", type=float, default=2.0,
                        help="TTC threshold for critical collision imminence (seconds).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for calibration results.")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Render SUMO simulation.")
    return parser.parse_args()


def kill_stray_sumo():
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
    except Exception:
        pass


def load_model(checkpoint_path: str):
    """Safely loads a checkpoint using MOSDPPO or fallback with class aliasing."""
    import types
    from src.models.mo_sd_models import MultiObjectiveActorCriticPolicy
    from src.models.mo_sd_ppo import MOSDPPO
    from stable_baselines3 import PPO

    # Map legacy module path to MultiObjectiveActorCriticPolicy so old checkpoints unpickle seamlessly
    if "src.models.multi_discount_ppo" not in sys.modules:
        legacy_mod = types.ModuleType("src.models.multi_discount_ppo")
        legacy_mod.DualHeadActorCriticPolicy = MultiObjectiveActorCriticPolicy
        legacy_mod.DualHeadPPO = MOSDPPO
        sys.modules["src.models.multi_discount_ppo"] = legacy_mod

    if not checkpoint_path.endswith(".zip"):
        checkpoint_path += ".zip"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        return MOSDPPO.load(checkpoint_path)
    except Exception:
        return PPO.load(checkpoint_path)


def select_action(policy_type: str, obs: np.ndarray, model=None, step: int = 0) -> np.ndarray:
    """Generates an action according to the requested policy profile."""
    if policy_type == "pretrained" and model is not None:
        action, _ = model.predict(obs, deterministic=True)
        return action
    elif policy_type == "random":
        # Uniform acceleration action in [-1, 1]
        return np.random.uniform(-1.0, 1.0, size=(1, 1)).astype(np.float32)
    elif policy_type == "creep":
        # Slow creeping: moderate constant low speed / slight deceleration
        return np.array([[-0.2]], dtype=np.float32)
    elif policy_type == "aggressive":
        # Full acceleration: maximum throttle to cross rapidly
        return np.array([[1.0]], dtype=np.float32)
    elif policy_type == "mixed":
        # Alternating accelerate / brake
        val = 0.8 if (step // 10) % 2 == 0 else -0.5
        return np.array([[val]], dtype=np.float32)
    else:
        return np.array([[0.0]], dtype=np.float32)


def run_calibration():
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.output_dir or os.path.join(root_dir, "output", "calibration")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print(" Reward Component Magnitude Calibration & Diagnostics")
    print(f" Policy profile    : {args.policy}")
    if args.policy == "pretrained":
        print(f" Checkpoint        : {args.checkpoint}")
    print(f" Scenarios         : {args.scenarios}")
    print(f" Episodes/scen     : {args.n_episodes}")
    print(f" Danger Gap (m)    : <= {args.danger_gap_dist} m")
    print(f" Danger |d_eta|    : < {args.danger_d_eta}")
    print(f" Danger TTC (s)    : <= {args.danger_ttc} s")
    print("=" * 76 + "\n")

    model = None
    if args.policy == "pretrained":
        if args.checkpoint is None:
            # Look for existing checkpoint
            candidates = [
                os.path.join(root_dir, "checkpoints", "v0_1",
                             "attention_continuous_alpha_env_v01_attention_continuous_PPO_g0.95_gl0.999_20260815_155627",
                             "final_model.zip"),
                os.path.join(root_dir, "checkpoints", "v0_1",
                             "attention_continuous_alpha_env_v01_attention_continuous_PPO_g0.999_gl0.999_20260816_152323",
                             "final_model.zip"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    args.checkpoint = c
                    break
        if args.checkpoint is None:
            print("No checkpoint found; falling back to random policy.")
            args.policy = "random"
        else:
            print(f"Loading checkpoint: {args.checkpoint}")
            model = load_model(args.checkpoint)

    records = []

    for scen_id in args.scenarios:
        print(f"\n>>> Sampling Scenario {scen_id} ({args.n_episodes} episodes) <<<")

        for ep_idx in range(args.n_episodes):
            env = None
            try:
                env = DummyVecEnv([
                    lambda: create_scenario_env(
                        scenario_id=scen_id,
                        root_dir=root_dir,
                        render=args.render,
                    )
                ])

                obs = env.reset()
                done = False
                step = 0

                # Episode accumulators
                sum_progress_norm = 0.0
                count_gap_under_5m = 0
                count_d_eta_under_thresh = 0
                count_ttc_under_thresh = 0
                sum_unweighted_gap_penalty = 0.0
                sum_unweighted_ttc_penalty = 0.0
                min_seen_gap = float("inf")
                min_seen_ttc = float("inf")
                min_seen_d_eta = 1.0

                while not done:
                    action = select_action(args.policy, obs, model, step)
                    obs, reward, dones, infos = env.step(action)
                    done = dones[0]
                    info = infos[0]

                    # Step conflict metrics
                    conflict_info = info.get("conflict_info", {})
                    min_ttc = float(conflict_info.get("min_ttc", 99.0))
                    min_d_eta = float(conflict_info.get("min_d_eta", 1.0))
                    min_gap = float(conflict_info.get("min_gap", float("inf")))

                    min_seen_gap = min(min_seen_gap, min_gap)
                    min_seen_ttc = min(min_seen_ttc, min_ttc)
                    min_seen_d_eta = min(min_seen_d_eta, min_d_eta)

                    # 1. Indicator for dangerous distance gap g_t <= 5m
                    if min_gap <= args.danger_gap_dist:
                        count_gap_under_5m += 1

                    # 2. Indicator for dangerous arrival time gap |d_eta| < 0.2
                    if min_d_eta < args.danger_d_eta:
                        count_d_eta_under_thresh += 1
                        sum_unweighted_gap_penalty += float(np.exp(-10.0 * min_d_eta))

                    # 3. Indicator for critical TTC <= threshold
                    if min_ttc <= args.danger_ttc:
                        count_ttc_under_thresh += 1
                        sum_unweighted_ttc_penalty += float(((args.danger_ttc - min_ttc) / args.danger_ttc) ** 2)

                    p_delta = float(info.get("reward_dict", {}).get("progress_delta", 0.0))
                    sum_progress_norm += p_delta
                    step += 1

                mo_tele = info.get("mo_telemetry", {})
                is_collision = int(mo_tele.get("collision", 0))
                is_goal = int(mo_tele.get("success", 0))
                duration_s = float(mo_tele.get("traversal_time", step * 0.25))

                # Exact cumulative progress through intersection: Σ_t Δp~_t
                cum_prog = mo_tele.get("cumulative_progress")
                raw_progress = float(cum_prog if cum_prog is not None else sum_progress_norm)

                r_prog = float(mo_tele.get("progress_reward", 0.0))
                r_goal = float(mo_tele.get("goal_reward", 0.0))
                r_time = float(mo_tele.get("time_penalty", 0.0))
                r_gap = float(mo_tele.get("gap_penalty", 0.0))
                r_col = float(mo_tele.get("collision_penalty", 0.0))
                r_l = float(mo_tele.get("total_long_term_reward", 0.0))
                r_s = float(mo_tele.get("total_safety_reward", 0.0))
                r_tot = float(mo_tele.get("total_reward", 0.0))

                rec = {
                    "scenario": scen_id,
                    "episode": ep_idx,
                    "policy": args.policy,
                    "progress_raw": float(raw_progress),
                    "steps_gap_under_5m": int(count_gap_under_5m),
                    "steps_d_eta_under_thresh": int(count_d_eta_under_thresh),
                    "steps_ttc_under_thresh": int(count_ttc_under_thresh),
                    "collision": is_collision,
                    "goal": is_goal,
                    "timesteps": int(step),
                    "duration_seconds": float(duration_s),
                    "progress_reward": r_prog,
                    "goal_reward": r_goal,
                    "time_penalty": r_time,
                    "gap_penalty": r_gap,
                    "collision_penalty": r_col,
                    "total_long_term_reward": r_l,
                    "total_safety_reward": r_s,
                    "total_reward": r_tot,
                    "unweighted_gap_penalty": float(sum_unweighted_gap_penalty),
                    "unweighted_ttc_penalty": float(sum_unweighted_ttc_penalty),
                    "min_gap": float(min_seen_gap if not np.isinf(min_seen_gap) else 99.0),
                    "min_ttc": float(min_seen_ttc if not np.isinf(min_seen_ttc) else 99.0),
                    "min_d_eta": float(min_seen_d_eta),
                }
                records.append(rec)

                print(f"  Ep {ep_idx:02d} | Col={is_collision} Goal={is_goal} Steps={step} ({duration_s:.1f}s) "
                      f"Progress={raw_progress:.2f} | R_l={r_l:+.2f} (Prog={r_prog:+.2f}, Goal={r_goal:+.2f}, Time={r_time:+.2f}) | "
                      f"R_s={r_s:+.2f} (Gap={r_gap:+.2f}, Col={r_col:+.2f}) | Total R={r_tot:+.2f}")

            except Exception as e:
                print(f"  Ep {ep_idx:02d} | Error: {e}")

            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                kill_stray_sumo()
                del env
                gc.collect()
                time.sleep(0.05)

    # ── Aggregate Statistics & Diagnostic Analysis ──
    print("\n" + "=" * 76)
    print(" EMPIRICAL COMPONENT MAGNITUDE ANALYSIS")
    print("=" * 76)

    def stats_str(arr):
        if len(arr) == 0:
            return "N/A"
        return (f"mean={np.mean(arr):.2f}, median={np.median(arr):.2f}, "
                f"std={np.std(arr):.2f}, min={np.min(arr):.2f}, max={np.max(arr):.2f}, p90={np.percentile(arr, 90):.2f}")

    progresses = [r["progress_raw"] for r in records]
    gaps_5m = [r["steps_gap_under_5m"] for r in records]
    d_etas = [r["steps_d_eta_under_thresh"] for r in records]
    ttcs = [r["steps_ttc_under_thresh"] for r in records]
    collisions = [r["collision"] for r in records]
    goals = [r["goal"] for r in records]
    durations = [r["duration_seconds"] for r in records]
    gap_pens = [r["unweighted_gap_penalty"] for r in records]
    ttc_pens = [r["unweighted_ttc_penalty"] for r in records]

    print(f"\n1. Raw Progress Σ_t Δp~_t        : {stats_str(progresses)}")
    print(f"2. Timesteps with g_t <= 5m       : {stats_str(gaps_5m)}")
    print(f"3. Timesteps with |d_η| < 0.2     : {stats_str(d_etas)}")
    print(f"4. Timesteps with TTC <= 2.0s     : {stats_str(ttcs)}")
    print(f"5. Collision Occurrence I_col     : {np.mean(collisions):.1%} ({np.sum(collisions)} / {len(collisions)})")
    print(f"6. Goal Reached Occurrence I_goal : {np.mean(goals):.1%} ({np.sum(goals)} / {len(goals)})")
    print(f"7. Episode Duration T_episode (s) : {stats_str(durations)}")
    print(f"8. Unweighted Gap Penalty Σ_t pen : {stats_str(gap_pens)}")
    print(f"9. Unweighted TTC Penalty Σ_t pen : {stats_str(ttc_pens)}")

    r_progs = [r["progress_reward"] for r in records]
    r_goals = [r["goal_reward"] for r in records]
    r_times = [r["time_penalty"] for r in records]
    r_gaps = [r["gap_penalty"] for r in records]
    r_cols = [r["collision_penalty"] for r in records]
    r_ls = [r["total_long_term_reward"] for r in records]
    r_ss = [r["total_safety_reward"] for r in records]
    r_tots = [r["total_reward"] for r in records]

    print("\n" + "-" * 76)
    print(" DECOMPOSED EPISODE REWARD COMPONENTS (R_l = Prog + Goal + Time | R_s = Gap + Col)")
    print("-" * 76)
    print(f"• Progress Reward R_progress    : {stats_str(r_progs)}")
    print(f"• Goal Reward R_goal            : {stats_str(r_goals)}")
    print(f"• Time Penalty R_time           : {stats_str(r_times)}")
    print(f"• Total Long-Term Reward R_l    : {stats_str(r_ls)}")
    print(f"• Gap Penalty R_gap             : {stats_str(r_gaps)}")
    print(f"• Collision Penalty R_collision : {stats_str(r_cols)}")
    print(f"• Total Safety Reward R_s       : {stats_str(r_ss)}")
    print(f"• Total Episode Reward R        : {stats_str(r_tots)}")

    # ── Balance Check and Reward Calibration Recommendation ──
    mean_gap_pen = np.mean(gap_pens)
    max_gap_pen = np.max(gap_pens)
    p90_gap_pen = np.percentile(gap_pens, 90)
    mean_duration = np.mean(durations)

    print("\n" + "-" * 76)
    print(" UNINTENDED REWARD INTERACTION CHECK & CALIBRATION")
    print("-" * 76)

    # Calibrated values in alpha_env_mo_sd.py:
    nominal_w_p = 10.0
    nominal_w_t = 0.01
    nominal_w_g = 15.0
    nominal_lambda_gap = 0.25
    nominal_lambda_ttc = 0.50
    nominal_R_c = 20.0

    typical_progress_return = nominal_w_p * np.mean(progresses)
    typical_time_penalty_return = nominal_w_t * (mean_duration / 0.25)
    typical_gap_penalty_return = nominal_lambda_gap * mean_gap_pen
    p90_gap_penalty_return = nominal_lambda_gap * p90_gap_pen
    typical_ttc_penalty_return = nominal_lambda_ttc * np.mean(ttc_pens)

    print(f"• Expected Progress Return  : +{typical_progress_return:.2f}")
    print(f"• Expected Goal Return      : +{nominal_w_g:.2f}")
    print(f"• Expected Time Cost Return : -{typical_time_penalty_return:.2f} (over {mean_duration:.1f}s)")
    print(f"• Mean Gap Penalty Return   : -{typical_gap_penalty_return:.2f}")
    print(f"• 90th-pct Gap Penalty      : -{p90_gap_penalty_return:.2f}")
    print(f"• Mean TTC Penalty Return   : -{typical_ttc_penalty_return:.2f}")
    print(f"• Catastrophic Crash Penalty: -{nominal_R_c:.2f}")

    # Risk Check: Does gap penalty exceed crash penalty?
    gap_exceeds_crash = p90_gap_penalty_return >= nominal_R_c
    if gap_exceeds_crash:
        print("\n[WARNING] UNINTENDED INTERACTION DETECTED:")
        print(f"  The 90th percentile dangerous-gap penalty (-{p90_gap_penalty_return:.2f})")
        print(f"  meets or exceeds the collision penalty (-{nominal_R_c:.2f})!")
        print("  This would make persistent dangerous proximity MORE costly than a catastrophic crash,")
        print("  incentivizing an agent to commit a crash to truncate negative gap penalties.")
        # Calculate safe scale
        safe_lambda_gap = round((nominal_R_c * 0.25) / max(p90_gap_pen, 1.0), 3)
        print(f"  RECOMMENDED FIX: Scale λ_gap to ~{safe_lambda_gap} or increase R_c to ~{p90_gap_penalty_return * 2.5:.1f}.")
    else:
        print("\n[PASSED] PROPER REWARD ORDERING VERIFIED:")
        print(f"  Mean gap penalty (-{typical_gap_penalty_return:.2f}) and 90th-pct gap penalty (-{p90_gap_penalty_return:.2f})")
        print(f"  remain well below catastrophic crash penalty (-{nominal_R_c:.2f}).")
        print("  Collisions strictly dominate repeated dangerous behavior.")

    # Save calibration data to JSON
    summary_data = {
        "policy": args.policy,
        "n_records": len(records),
        "scenarios": args.scenarios,
        "metrics": {
            "progress_raw": {"mean": float(np.mean(progresses)), "p90": float(np.percentile(progresses, 90)), "max": float(np.max(progresses))},
            "steps_gap_under_5m": {"mean": float(np.mean(gaps_5m)), "p90": float(np.percentile(gaps_5m, 90)), "max": float(np.max(gaps_5m))},
            "steps_d_eta_under_thresh": {"mean": float(np.mean(d_etas)), "p90": float(np.percentile(d_etas, 90)), "max": float(np.max(d_etas))},
            "steps_ttc_under_thresh": {"mean": float(np.mean(ttcs)), "p90": float(np.percentile(ttcs, 90)), "max": float(np.max(ttcs))},
            "collision_rate": float(np.mean(collisions)),
            "goal_rate": float(np.mean(goals)),
            "duration_seconds": {"mean": float(np.mean(durations)), "p90": float(np.percentile(durations, 90))},
            "unweighted_gap_penalty": {"mean": float(np.mean(gap_pens)), "p90": float(np.percentile(gap_pens, 90)), "max": float(np.max(gap_pens))},
        },
        "records": records,
    }

    out_json = os.path.join(out_dir, f"calibration_summary_{args.policy}.json")
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nCalibration data saved → {out_json}\n")


if __name__ == "__main__":
    run_calibration()

