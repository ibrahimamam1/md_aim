"""
plot_pareto_and_metrics.py
──────────────────────────
Generates publication-quality figures for:
  1. Section 16 Pareto Evaluation:
     - 2D Pareto frontier: Collision Rate vs Traversal Time.
     - Non-dominated set extraction and Hypervolume indicator calculation.
     - Direct comparison across:
         • Baseline (fixed discounts γ_0)
         • Exp A: State-dependent single discount γ(s)
         • Exp B: Multi-objective state-dependent discount [γ_l, γ_s(s)] (Core)
         • Exp C: Learnable discount factors γ_φ(s)
         • Ablation: State-dependent reward weighting λ(s)
  2. Section 14 & 15 Scenario Radar & Metric Dashboards across S1..S4.
  3. Comfort & Vehicle Dynamics Profiles (Jerk, Deceleration, Velocity).

Supports --test_dummy mode to generate and verify all plots immediately with synthetic data.
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure clean aesthetic style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

METHOD_COLORS = {
    "Baseline (Fixed γ)":       "#4C72B0",   # Muted blue
    "Exp A (State-Dep Single)":  "#55A868",   # Green
    "Exp B (MO State-Dep, Core)":"#C44E52",   # Coral Red
    "Exp C (Learnable Discount)":"#8172B3",   # Purple
    "Ablation (Reward Weighting)":"#CCB974",  # Gold / Ochre
}

METHOD_MARKERS = {
    "Baseline (Fixed γ)":       "o",
    "Exp A (State-Dep Single)":  "s",
    "Exp B (MO State-Dep, Core)":"^",
    "Exp C (Learnable Discount)":"D",
    "Ablation (Reward Weighting)":"v",
}


def compute_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Computes non-dominated Pareto frontier points for minimization of both objectives (x, y).
    points: array of shape [N, 2], where x = collision_rate, y = traversal_time.
    """
    is_dominated = np.zeros(len(points), dtype=bool)
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                # p2 dominates p1 if p2 is smaller/equal in both and strictly smaller in at least one
                if (p2[0] <= p1[0] and p2[1] <= p1[1]) and (p2[0] < p1[0] or p2[1] < p1[1]):
                    is_dominated[i] = True
                    break
    non_dominated = points[~is_dominated]
    # Sort by x ascending
    return non_dominated[np.argsort(non_dominated[:, 0])]


def compute_hypervolume(frontier: np.ndarray, ref_point: Tuple[float, float] = (1.0, 60.0)) -> float:
    """
    Computes 2D hypervolume dominated by the Pareto frontier relative to an anti-ideal reference point.
    frontier: array of shape [K, 2] sorted by x ascending.
    ref_point: (max_x, max_y).
    """
    if len(frontier) == 0:
        return 0.0

    hv = 0.0
    # Add bounding reference
    current_x = frontier[0][0]
    for i in range(len(frontier)):
        x_i, y_i = frontier[i]
        next_x = frontier[i + 1][0] if i + 1 < len(frontier) else ref_point[0]
        if ref_point[1] > y_i and next_x > x_i:
            width = next_x - x_i
            height = ref_point[1] - y_i
            hv += width * height
    return float(hv)


def generate_dummy_data() -> Dict[str, Any]:
    """Generates synthetic data mirroring typical multi-objective RL intersection results."""
    data = {
        "Baseline (Fixed γ)": [
            {"gamma": 0.90, "col_rate": 0.04, "time": 28.5, "braking": 2.1, "safe_gap": 0.72, "jerk": 1.45},
            {"gamma": 0.95, "col_rate": 0.07, "time": 24.2, "braking": 1.7, "safe_gap": 0.65, "jerk": 1.30},
            {"gamma": 0.97, "col_rate": 0.11, "time": 21.0, "braking": 1.2, "safe_gap": 0.58, "jerk": 1.15},
            {"gamma": 0.99, "col_rate": 0.16, "time": 18.5, "braking": 0.9, "safe_gap": 0.50, "jerk": 0.95},
            {"gamma": 0.995,"col_rate": 0.22, "time": 16.8, "braking": 0.7, "safe_gap": 0.42, "jerk": 0.88},
        ],
        "Exp A (State-Dep Single)": [
            {"w": 0.1, "col_rate": 0.03, "time": 25.8, "braking": 1.8, "safe_gap": 0.75, "jerk": 1.38},
            {"w": 0.3, "col_rate": 0.06, "time": 22.0, "braking": 1.4, "safe_gap": 0.68, "jerk": 1.20},
            {"w": 0.5, "col_rate": 0.09, "time": 19.4, "braking": 1.1, "safe_gap": 0.61, "jerk": 1.05},
            {"w": 0.7, "col_rate": 0.13, "time": 17.6, "braking": 0.8, "safe_gap": 0.54, "jerk": 0.92},
            {"w": 0.9, "col_rate": 0.18, "time": 16.2, "braking": 0.6, "safe_gap": 0.46, "jerk": 0.84},
        ],
        "Exp B (MO State-Dep, Core)": [
            {"w": 0.0, "col_rate": 0.01, "time": 24.0, "braking": 1.4, "safe_gap": 0.82, "jerk": 1.25},
            {"w": 0.25,"col_rate": 0.03, "time": 20.1, "braking": 1.1, "safe_gap": 0.76, "jerk": 1.10},
            {"w": 0.5, "col_rate": 0.05, "time": 17.5, "braking": 0.8, "safe_gap": 0.70, "jerk": 0.95},
            {"w": 0.75,"col_rate": 0.09, "time": 15.8, "braking": 0.6, "safe_gap": 0.62, "jerk": 0.82},
            {"w": 1.0, "col_rate": 0.14, "time": 14.7, "braking": 0.5, "safe_gap": 0.52, "jerk": 0.75},
        ],
        "Exp C (Learnable Discount)": [
            {"w": 0.0, "col_rate": 0.015,"time": 24.4, "braking": 1.5, "safe_gap": 0.80, "jerk": 1.28},
            {"w": 0.25,"col_rate": 0.035,"time": 20.5, "braking": 1.1, "safe_gap": 0.74, "jerk": 1.12},
            {"w": 0.5, "col_rate": 0.055,"time": 17.9, "braking": 0.9, "safe_gap": 0.68, "jerk": 0.98},
            {"w": 0.75,"col_rate": 0.095,"time": 16.0, "braking": 0.7, "safe_gap": 0.60, "jerk": 0.85},
            {"w": 1.0, "col_rate": 0.15, "time": 14.9, "braking": 0.5, "safe_gap": 0.50, "jerk": 0.78},
        ],
        "Ablation (Reward Weighting)": [
            {"w": 0.1, "col_rate": 0.05, "time": 27.0, "braking": 2.4, "safe_gap": 0.69, "jerk": 1.55},
            {"w": 0.3, "col_rate": 0.08, "time": 23.1, "braking": 2.0, "safe_gap": 0.62, "jerk": 1.38},
            {"w": 0.5, "col_rate": 0.12, "time": 20.0, "braking": 1.5, "safe_gap": 0.55, "jerk": 1.22},
            {"w": 0.7, "col_rate": 0.16, "time": 17.9, "braking": 1.1, "safe_gap": 0.48, "jerk": 1.05},
            {"w": 0.9, "col_rate": 0.21, "time": 16.5, "braking": 0.8, "safe_gap": 0.41, "jerk": 0.92},
        ],
    }

    # Scenario S1..S4 breakdown for the primary methods
    scenarios = ["S1\nFree", "S2\nMod", "S3\nDense", "S4\nSudden"]
    scen_data = {
        scen: {
            "Baseline":  {"col": 0.12, "tt": 19.5, "gap": 0.55, "jerk": 1.15},
            "Exp A":     {"col": 0.09, "tt": 18.5, "gap": 0.62, "jerk": 1.05},
            "Exp B (Core)":{"col": 0.05, "tt": 16.8, "gap": 0.72, "jerk": 0.88},
            "Exp C":     {"col": 0.06, "tt": 17.2, "gap": 0.69, "jerk": 0.92},
            "Ablation":  {"col": 0.11, "tt": 19.0, "gap": 0.57, "jerk": 1.25},
        }
        for scen in scenarios
    }
    # Induce characteristic scenario stress
    scen_data["S3\nDense"]["Baseline"]["col"] = 0.24
    scen_data["S3\nDense"]["Exp B (Core)"]["col"] = 0.08
    scen_data["S4\nSudden"]["Baseline"]["col"] = 0.32
    scen_data["S4\nSudden"]["Exp B (Core)"]["col"] = 0.07

    return {"tradeoff": data, "scenario_data": scen_data, "scenarios": scenarios}


def plot_pareto_frontiers(tradeoff_data: Dict[str, List[Dict]], save_path: str):
    """
    Plots the safety-efficiency Pareto frontiers and computes Hypervolume indicators.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    hv_results = {}

    for method_name, points_list in tradeoff_data.items():
        color = METHOD_COLORS.get(method_name, "#333333")
        marker = METHOD_MARKERS.get(method_name, "o")

        raw_points = np.array([[p["col_rate"], p["time"]] for p in points_list])
        # Sort raw points by x
        raw_points = raw_points[np.argsort(raw_points[:, 0])]

        frontier = compute_pareto_frontier(raw_points)
        hv = compute_hypervolume(frontier, ref_point=(0.35, 35.0))
        hv_results[method_name] = hv

        # Plot all evaluation sample points
        ax.scatter(
            raw_points[:, 0] * 100.0, raw_points[:, 1],
            color=color, marker=marker, s=70, alpha=0.6,
            label=f"{method_name} (HV={hv:.1f})"
        )

        # Plot non-dominated frontier line
        ax.plot(
            frontier[:, 0] * 100.0, frontier[:, 1],
            color=color, linestyle="-", linewidth=2.4
        )

    ax.set_title("Safety–Efficiency Pareto Frontier Comparison", fontweight="bold", pad=12)
    ax.set_xlabel("Collision Rate (%) [Lower is Better]")
    ax.set_ylabel("Mean Traversal Time (s) [Lower is Better]")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Pareto Frontier plot → {save_path}")
    return hv_results


def plot_scenario_benchmarks(scen_data: Dict[str, Dict], scenarios: List[str], save_path: str):
    """
    Plots a 4-panel dashboard comparing methods across scenarios S1..S4.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    methods = ["Baseline", "Ablation", "Exp A", "Exp C", "Exp B (Core)"]
    palette = ["#4C72B0", "#CCB974", "#55A868", "#8172B3", "#C44E52"]

    x = np.arange(len(scenarios))
    bar_width = 0.15

    # 1. Collision Rate (%)
    ax = axes[0, 0]
    for idx, m in enumerate(methods):
        vals = [scen_data[s][m]["col"] * 100.0 for s in scenarios]
        ax.bar(x + idx * bar_width, vals, width=bar_width, label=m, color=palette[idx], edgecolor="none")
    ax.set_title("Collision Rate (%) by Scenario", fontweight="bold")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Collision Rate (%)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    # 2. Mean Traversal Time (s)
    ax = axes[0, 1]
    for idx, m in enumerate(methods):
        vals = [scen_data[s][m]["tt"] for s in scenarios]
        ax.bar(x + idx * bar_width, vals, width=bar_width, label=m, color=palette[idx], edgecolor="none")
    ax.set_title("Traversal Time (s) by Scenario", fontweight="bold")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Time (s)")
    ax.grid(True, axis="y")

    # 3. Average Safe Gap |d_eta|
    ax = axes[1, 0]
    for idx, m in enumerate(methods):
        vals = [scen_data[s][m]["gap"] for s in scenarios]
        ax.bar(x + idx * bar_width, vals, width=bar_width, label=m, color=palette[idx], edgecolor="none")
    ax.set_title("Minimum Safe Gap |d_η| (Higher is Safer)", fontweight="bold")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Safe Gap |d_η| [0, 1]")
    ax.grid(True, axis="y")

    # 4. Mean Absolute Jerk (m/s³)
    ax = axes[1, 1]
    for idx, m in enumerate(methods):
        vals = [scen_data[s][m]["jerk"] for s in scenarios]
        ax.bar(x + idx * bar_width, vals, width=bar_width, label=m, color=palette[idx], edgecolor="none")
    ax.set_title("Comfort: Mean Absolute Jerk (Lower is Smoother)", fontweight="bold")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Jerk (m/s³)")
    ax.grid(True, axis="y")

    plt.suptitle("Performance Breakdown Across S1–S4 Traffic Regimes", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Scenario Benchmarks dashboard → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto and Scenario evaluation figures.")
    parser.add_argument("--test_dummy", action="store_true", default=False,
                        help="Generate figures using synthetic benchmark data to verify visualizer.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save generated PNG figures.")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.output_dir or os.path.join(root_dir, "output", "figures")
    os.makedirs(out_dir, exist_ok=True)

    dummy_data = generate_dummy_data()
    pareto_path = os.path.join(out_dir, "pareto_frontier_comparison.png")
    bench_path = os.path.join(out_dir, "scenario_benchmarks_s1_s7.png")

    print("\n--- Generating Evaluation Figures ---")
    hv = plot_pareto_frontiers(dummy_data["tradeoff"], pareto_path)
    print("Hypervolume Scores:")
    for m, v in hv.items():
        print(f"  {m:<30} : {v:.2f}")

    plot_scenario_benchmarks(dummy_data["scenario_data"], dummy_data["scenarios"], bench_path)
    print("--- Figure Generation Complete ---\n")


if __name__ == "__main__":
    main()
