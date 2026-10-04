"""Plot the collision-rate vs traversal-time Pareto front for the baseline gamma sweep.

Reads every output/**/evaluation_summary.json, averages the metrics across scenarios
(S1-S4) to get one point per run, and highlights the Pareto front (lower collision
rate and lower traversal time is better).

Point classes:
  - output/eval_baseline_g0_<gamma>/: baseline fixed-discount sweep (one point per gamma)
  - output/eval_exp_b_danger_gsd_<gmin>_g0_<g0>/: exp B gamma_min sweep; the best arm
    (balanced knee score) is added to the plot and the Pareto front computation
  - other output/eval_exp_*: overlaid as distinct markers
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
EXP_B_SWEEP_RE = re.compile(r"^eval_exp_b_danger_gsd_([\d.]+)_g0_([\d.]+)$")
EXP_RE = re.compile(r"^eval_exp_(.+)$")


def load_points():
    """Return (baselines, experiments, sweep_arms).

    baselines:   [(gamma, mean_coll, mean_time, coll_scen, time_scen), ...]
    experiments: [(name, mean_coll, mean_time, coll_scen, time_scen), ...]
    sweep_arms:  [(gmin, mean_coll, mean_time, coll_scen, time_scen), ...]
    """
    baselines, experiments, sweep_arms = [], [], []
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", SUMMARY_NAME), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), OUTPUT_DIR)
        top = rel.split(os.sep)[0]
        with open(path) as f:
            summary = json.load(f)
        coll = np.array([s["collision_rate"] for s in summary.values()])
        time = np.array([s["traversal_time"] for s in summary.values()])
        if BASELINE_RE.match(top):
            gamma = float(BASELINE_RE.match(top).group(1))
            baselines.append((gamma, coll.mean(), time.mean(), coll, time))
        elif EXP_B_SWEEP_RE.match(top):
            gmin = float(EXP_B_SWEEP_RE.match(top).group(1))
            sweep_arms.append((gmin, coll.mean(), time.mean(), coll, time))
        elif EXP_RE.match(top):
            experiments.append((EXP_RE.match(top).group(1), coll.mean(), time.mean(), coll, time))
    baselines.sort(key=lambda p: p[0])
    sweep_arms.sort(key=lambda p: p[0])
    return baselines, experiments, sweep_arms


def pareto_mask(x, y):
    """Indices of non-dominated points (both objectives minimized)."""
    idx = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        dominated = np.any((x <= xi) & (y <= yi) & ((x < xi) | (y < yi)))
        if not dominated:
            idx.append(i)
    return np.array(idx, dtype=int)


def pick_best_arm(arms, all_coll, all_time):
    """Pick the sweep arm with the best balanced knee score (normalized 50/50)."""
    cn = (all_coll - all_coll.min()) / max(all_coll.max() - all_coll.min(), 1e-9)
    tn = (all_time - all_time.min()) / max(all_time.max() - all_time.min(), 1e-9)
    scores = (cn + tn) / 2.0
    n_base = len(all_coll) - len(arms)
    arm_scores = scores[n_base:]
    best = int(np.argmin(arm_scores))
    ranking = sorted(range(len(arms)), key=lambda j: arm_scores[j])
    return best, ranking


def main():
    baselines, experiments, sweep_arms = load_points()
    if not baselines:
        raise SystemExit(f"No baseline summaries found in {OUTPUT_DIR}/eval_baseline_g0_*/")

    gammas = np.array([b[0] for b in baselines])
    coll = np.array([b[1] for b in baselines])
    time = np.array([b[2] for b in baselines])

    # Sweep arms + best arm selection (balanced knee score over all plotted points)
    all_coll = np.concatenate([coll] + [np.array([e[1]]) for e in experiments + sweep_arms])
    all_time = np.concatenate([time] + [np.array([e[2]]) for e in experiments + sweep_arms])
    best_arm, ranking = pick_best_arm(sweep_arms, all_coll, all_time)
    best = sweep_arms[best_arm]

    print("Exp B gamma_min sweep ranking (knee score, lower is better):")
    for rank, j in enumerate(ranking, 1):
        g, c, t, _, _ = sweep_arms[j]
        print(f"  {rank}. gmin={g:g}  collision={c:.4f}  traversal={t:.2f}s")
    print(f"Best arm: gmin={best[0]:g}")

    # Points entering the front: baselines + experiments + best sweep arm
    front_coll = np.concatenate([coll] + [np.array([e[1]]) for e in experiments] + [np.array([best[1]])])
    front_time = np.concatenate([time] + [np.array([e[2]]) for e in experiments] + [np.array([best[2]])])
    front = pareto_mask(front_coll, front_time)
    front = front[np.argsort(front_coll[front])]  # order by collision rate for the step line

    fig, ax = plt.subplots(figsize=(8, 6))

    # Faint per-scenario points to show spread within each run
    for _, _, _, c_sc, t_sc in baselines + experiments + [best]:
        ax.scatter(c_sc, t_sc, s=14, color="tab:gray", alpha=0.25, zorder=1)

    # Baseline gamma points
    ax.scatter(coll, time, s=70, color="tab:blue", zorder=3, label="baseline gamma (scenario avg)")
    for g, c, t in zip(gammas, coll, time):
        ax.annotate(f"{g:g}", (c, t), textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Best exp B sweep arm as a prominent star
    ax.scatter([best[1]], [best[2]], s=260, marker="*", color="tab:red", zorder=6,
               label=f"exp B best (gmin={best[0]:g})")
    ax.annotate(f"exp B best\ngmin={best[0]:g}", (best[1], best[2]),
                textcoords="offset points", xytext=(10, -22), fontsize=8)

    # Other experiment runs as distinct markers
    exp_markers = [("o", "tab:orange"), ("s", "tab:green"), ("^", "tab:purple"), ("D", "tab:brown")]
    for i, (name, c, t, _, _) in enumerate(experiments):
        marker, color = exp_markers[i % len(exp_markers)]
        ax.scatter([c], [t], s=140, marker=marker, color=color, zorder=4, label=f"exp: {name}")
        ax.annotate(name, (c, t), textcoords="offset points", xytext=(8, -12), fontsize=8)

    # Pareto front as a step line through the non-dominated points
    ax.step(front_coll[front], front_time[front], where="post", color="tab:red", lw=2,
            zorder=2, label="Pareto front")
    ax.scatter(front_coll[front], front_time[front], s=110, facecolors="none",
               edgecolors="tab:red", lw=2, zorder=5, label="non-dominated")

    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Traversal time (s)")
    ax.set_title("Collision rate vs traversal time (averaged over scenarios S1-S4)\n"
                 "baseline gamma sweep, experiments, best exp B gamma_min arm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved {FIG_PATH}")

    front_labels = []
    for i in front:
        if i < len(baselines):
            front_labels.append(f"gamma {gammas[i]:g}")
        elif i < len(baselines) + len(experiments):
            front_labels.append(f"exp {experiments[i - len(baselines)][0]}")
        else:
            front_labels.append(f"exp B best gmin={best[0]:g}")
    print("Pareto-optimal points:", ", ".join(front_labels))


if __name__ == "__main__":
    main()
