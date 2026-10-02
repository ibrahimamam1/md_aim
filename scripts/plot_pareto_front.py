"""Plot the collision-rate vs traversal-time Pareto front for the baseline gamma sweep.

Reads every output/**/evaluation_summary.json, averages the metrics across scenarios
(S1-S4) to get one point per run, and highlights the Pareto front (lower collision
rate and lower traversal time is better). Points from non-baseline runs
(eval_exp_*) are overlaid as distinct markers.
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "output"
SUMMARY_NAME = "evaluation_summary.json"
FIG_PATH = os.path.join(OUTPUT_DIR, "pareto_front.png")

BASELINE_RE = re.compile(r"^eval_baseline_g0_([\d.]+)$")
EXP_RE = re.compile(r"^eval_exp_(.+)$")


def load_points():
    """Return (baselines, experiments).

    baselines: list of (gamma, mean_coll, mean_time, coll_per_scenario, time_per_scenario)
    experiments: list of (name, mean_coll, mean_time, coll_per_scenario, time_per_scenario)
    """
    baselines, experiments = [], []
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", SUMMARY_NAME), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), OUTPUT_DIR)
        top = rel.split(os.sep)[0]
        m = BASELINE_RE.match(top)
        if m:
            name, gamma = None, float(m.group(1))
        else:
            m = EXP_RE.match(top)
            if not m:
                continue
            name, gamma = m.group(1), None
        with open(path) as f:
            summary = json.load(f)
        coll = np.array([s["collision_rate"] for s in summary.values()])
        time = np.array([s["traversal_time"] for s in summary.values()])
        entry = (gamma if gamma is not None else name, coll.mean(), time.mean(), coll, time)
        (baselines if gamma is not None else experiments).append(entry)
    baselines.sort(key=lambda p: p[0])
    return baselines, experiments


def pareto_mask(x, y):
    """Indices of non-dominated points (both objectives minimized)."""
    idx = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        dominated = np.any((x <= xi) & (y <= yi) & ((x < xi) | (y < yi)))
        if not dominated:
            idx.append(i)
    return np.array(idx, dtype=int)


def main():
    baselines, experiments = load_points()
    if not baselines:
        raise SystemExit(f"No baseline summaries found in {OUTPUT_DIR}/eval_baseline_g0_*/")

    gammas = np.array([b[0] for b in baselines])
    coll = np.array([b[1] for b in baselines])
    time = np.array([b[2] for b in baselines])

    # Pareto front over baseline + experiment points
    all_coll = np.concatenate([coll] + [np.array([e[1]]) for e in experiments])
    all_time = np.concatenate([time] + [np.array([e[2]]) for e in experiments])
    front = pareto_mask(all_coll, all_time)
    front = front[np.argsort(all_coll[front])]  # order by collision rate for the step line

    fig, ax = plt.subplots(figsize=(8, 6))

    # Faint per-scenario points to show spread within each run
    for _, _, _, c_sc, t_sc in baselines + experiments:
        ax.scatter(c_sc, t_sc, s=14, color="tab:gray", alpha=0.25, zorder=1)

    # Baseline gamma points
    ax.scatter(coll, time, s=70, color="tab:blue", zorder=3, label="baseline gamma (scenario avg)")
    for g, c, t in zip(gammas, coll, time):
        ax.annotate(f"{g:g}", (c, t), textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Experiment points as distinct markers
    exp_markers = [("o", "tab:orange"), ("s", "tab:green"), ("^", "tab:purple"), ("D", "tab:brown")]
    for i, (name, c, t, _, _) in enumerate(experiments):
        marker, color = exp_markers[i % len(exp_markers)]
        ax.scatter([c], [t], s=140, marker=marker, color=color, zorder=4, label=f"exp: {name}")
        ax.annotate(name, (c, t), textcoords="offset points", xytext=(8, -12), fontsize=8)

    # Pareto front as a step line through the non-dominated points
    ax.step(all_coll[front], all_time[front], where="post", color="tab:red", lw=2,
            zorder=2, label="Pareto front")
    ax.scatter(all_coll[front], all_time[front], s=110, facecolors="none",
               edgecolors="tab:red", lw=2, zorder=5, label="non-dominated")

    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Traversal time (s)")
    ax.set_title("Collision rate vs traversal time (averaged over scenarios S1-S4)\n"
                 "baseline gamma sweep + experiments")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved {FIG_PATH}")
    front_labels = []
    for i in front:
        if i < len(baselines):
            front_labels.append(f"gamma {gammas[i]:g}")
        else:
            front_labels.append(f"exp {experiments[i - len(baselines)][0]}")
    print("Pareto-optimal points:", ", ".join(front_labels))
    for j, (name, c, t, _, _) in enumerate(experiments):
        dominating = [
            f"gamma {gammas[i]:g}" for i in range(len(baselines))
            if coll[i] <= c and time[i] <= t and (coll[i] < c or time[i] < t)
        ]
        status = f"dominated by {', '.join(dominating)}" if dominating else "non-dominated (on front)"
        print(f"exp {name}: avg collision={c:.4f}, avg traversal={t:.2f}s -> {status}")


if __name__ == "__main__":
    main()
