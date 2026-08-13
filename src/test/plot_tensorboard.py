"""
plot_tensorboard.py — Publication-quality training curves from TensorBoard logs.

Produces two figures:
  1. Mean Episode Reward   (all agents on one plot)
  2. Average Speed          (all agents on one plot)

Auto-discovers agent runs under:
    <project_root>/tensorboard_logs/v0_1_/<agent_name>/<run>/PPO_1/

Usage:
    python src/test/plot_tensorboard.py
    python src/test/plot_tensorboard.py --logdir tensorboard_logs/v0_1_
    python src/test/plot_tensorboard.py --output_dir plots/training
    python src/test/plot_tensorboard.py --smooth 0.90
"""

import argparse
import glob
import os
import sys
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# TensorBoard reader (uses the public EventAccumulator API)
# ---------------------------------------------------------------------------
try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    sys.exit(
        "ERROR: tensorboard is required.  Install with:\n"
        "  pip install tensorboard"
    )


# ---------------------------------------------------------------------------
# Agent display names, colours, and styles
# ---------------------------------------------------------------------------
AGENT_META = {
    # key (folder name)         : (display_label,                  colour,     linestyle, marker)
    "heuristic_discrete":        ("Heuristic + Discrete",          "#1f77b4",  "-",       "o"),
    "heuristic_discrete_3":      ("Heuristic + Discrete (3)",      "#6baed6",  "-",       "o"),
    "heuristic_discrete_5":      ("Heuristic + Discrete (5)",      "#3182bd",  "-",       "s"),
    "heuristic_discrete_10":     ("Heuristic + Discrete (10)",     "#08519c",  "-",       "D"),
    "heuristic_continuous":      ("Heuristic + Continuous",        "#d62728",  "--",      "^"),
    "heuristic_continous":       ("Heuristic + Continuous",        "#d62728",  "--",      "^"),
    "attention_discrete":        ("Attention + Discrete",          "#2ca02c",  "-.",      "v"),
    "attention_discrete_3":      ("Attention + Discrete (3)",      "#74c476",  "-.",      "v"),
    "attention_discrete_5":      ("Attention + Discrete (5)",      "#31a354",  "-.",      "p"),
    "attention_discrete_10":     ("Attention + Discrete (10)",     "#006d2c",  "-.",      "h"),
    "attention_continuous":      ("Attention + Continuous",        "#9467bd",  ":",       "P"),
    "attention_continous":       ("Attention + Continuous",        "#9467bd",  ":",       "P"),
    "heuristic_attention_discrete":  ("Heur.+Attn. + Discrete",   "#17becf",  "-",       "X"),
    "heuristic_attention_continous": ("Heur.+Attn. + Continuous",  "#8c564b",  "--",      "*"),
}

# Fallback palette for agents not listed above
_FALLBACK_COLORS = [
    "#e377c2", "#7f7f7f", "#bcbd22", "#ff7f0e",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "X"]


def _agent_style(name: str, idx: int):
    """Return (label, color, linestyle, marker) for an agent folder name."""
    if name in AGENT_META:
        return AGENT_META[name]
    color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    marker = _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)]
    if name.startswith("gamma_"):
        val = name.replace("gamma_", "").replace("_", ".")
        label = f"\u03b3 = {val}"
    else:
        label = name.replace("_", " ").title()
        
    return label, color, "-", marker


# ---------------------------------------------------------------------------
# Matplotlib global style  (IEEE / Nature compatible)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9.5,
    "legend.title_fontsize": 10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "--",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_scalars(logdir: str, tag: str):
    """Return (steps, values) arrays for *tag* from the newest run in *logdir*."""
    # logdir points to e.g.  tensorboard_logs/v0_1_/heuristic_discrete_10
    # Inside there may be one or more run dirs.  Pick the latest by name.
    run_dirs = sorted(glob.glob(os.path.join(logdir, "*", "PPO_1")))
    if not run_dirs:
        # try one level up in case PPO_1 is directly under logdir
        run_dirs = sorted(glob.glob(os.path.join(logdir, "PPO_1")))
    if not run_dirs:
        return None, None

    # Use the latest run
    event_dir = run_dirs[-1]

    ea = EventAccumulator(event_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    order = np.argsort(steps)
    return steps[order], values[order]


def ema_smooth(values: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Exponential moving average (weight = alpha)."""
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def discover_agents(base_dir: str) -> List[str]:
    """Return sorted list of agent folder names under *base_dir*."""
    agents = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full):
            agents.append(entry)
    return agents


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_metric(
    base_dir: str,
    agents: List[str],
    tag: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str,
    smooth_alpha: float = 0.9,
    cumulative: bool = False,
):
    """Create a single figure with one curve per agent for *tag*.

    If *cumulative* is True, the values are cumulatively summed before
    smoothing and plotting.
    """

    fig, ax = plt.subplots(figsize=(8, 4.8))

    plotted = 0
    for idx, agent in enumerate(agents):
        agent_dir = os.path.join(base_dir, agent)
        steps, values = read_scalars(agent_dir, tag)
        if steps is None or len(steps) == 0:
            print(f"  [skip] No data for tag '{tag}' in {agent}")
            continue

        if cumulative:
            values = np.cumsum(values)

        label, color, ls, marker = _agent_style(agent, idx)
        smoothed = ema_smooth(values, smooth_alpha)

        # Raw values as a light shaded band
        ax.fill_between(
            steps, values, smoothed,
            color=color, alpha=0.08, linewidth=0,
        )
        # Raw curve — very thin, low alpha
        ax.plot(
            steps, values,
            color=color, alpha=0.18, linewidth=0.7,
        )
        # Smoothed curve — prominent
        marker_every = max(1, len(steps) // 12)
        ax.plot(
            steps, smoothed,
            color=color, linestyle=ls, linewidth=2.0,
            marker=marker, markevery=marker_every, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.6,
            label=label, zorder=4,
        )
        plotted += 1

    if plotted == 0:
        print(f"  [!] Nothing plotted for {tag}")
        plt.close(fig)
        return

    # --- Axes formatting ---
    ax.set_xlabel("Environment Steps", fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=8)
    ax.set_title(title, fontweight="bold", pad=12)

    # Thousands / millions formatter for x-axis
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else (f"{x / 1e3:.0f}k" if x >= 1e3 else f"{x:.0f}")
        )
    )

    # Legend — outside right if many agents, else upper-left
    ncol = 1 if plotted <= 6 else 2
    legend = ax.legend(
        loc="best",
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        shadow=False,
        borderpad=0.8,
        handlelength=2.5,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=1.5)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {filename}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard training curves for the paper."
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Root directory containing agent subdirectories "
             "(default: <project>/tensorboard_logs).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save figures (default: <project>/plots/training).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="EMA smoothing factor (0 = none, 0.99 = very smooth). Default: 0.9.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    base_dir = args.logdir or os.path.join(project_root, "tensorboard_logs")
    output_dir = args.output_dir or os.path.join(project_root, "plots", "training")

    if not os.path.isdir(base_dir):
        sys.exit(f"ERROR: Log directory not found: {base_dir}")

    agents = discover_agents(base_dir)
    if not agents:
        sys.exit(f"ERROR: No agent subdirectories found in: {base_dir}")

    print(f"Log directory : {base_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Agents found  : {agents}")
    print(f"EMA smoothing : α = {args.smooth}")
    print()

    # --- Figure 1: Mean Episode Reward ---
    print("[1/3] Plotting Mean Episode Reward ...")
    plot_metric(
        base_dir=base_dir,
        agents=agents,
        tag="rollout/ep_rew_mean",
        ylabel="Mean Episode Reward",
        title="Training Reward",
        filename="training_reward",
        output_dir=output_dir,
        smooth_alpha=args.smooth,
    )

    # --- Figure 2: Cumulative Reward ---
    print("[2/3] Plotting Cumulative Reward ...")
    plot_metric(
        base_dir=base_dir,
        agents=agents,
        tag="rollout/ep_rew_mean",
        ylabel="Cumulative Reward",
        title="Cumulative Reward",
        filename="training_cumulative_reward",
        output_dir=output_dir,
        smooth_alpha=args.smooth,
        cumulative=True,
    )

    # --- Figure 3: Average Speed ---
    print("[3/3] Plotting Average Speed ...")
    plot_metric(
        base_dir=base_dir,
        agents=agents,
        tag="custom_metrics/avg_speed",
        ylabel="Average Speed (m/s)",
        title="Average Speed During Training",
        filename="training_avg_speed",
        output_dir=output_dir,
        smooth_alpha=args.smooth,
    )

    print(f"\nDone. Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
