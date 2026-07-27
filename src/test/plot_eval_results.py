"""
plot_eval_results.py
────────────────────
Reads all CSV files produced by v0_1_evaluate.py from the output directory
and generates two grouped bar charts:

  1. Collision Rate      (%) — grouped by intention, one bar-group per scenario
  2. Average Travel Time (s) — grouped by intention, one bar-group per scenario

Usage (standalone):
    python plot_eval_results.py                         # uses default output/ dir
    python plot_eval_results.py --output_dir /path/to/output --version heuristic_discrete

Or call plot_eval_results() directly from your eval script after main() finishes.
"""

from __future__ import annotations

import os
import re
import csv
import argparse
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── Aesthetic config ──────────────────────────────────────────────────────────
INTENTION_COLORS = {
    "all_straight":   "#2E86AB",   # steel blue
    "all_left":       "#E84855",   # vivid red
    "uniform_random": "#F4A261",   # warm amber
}

INTENTION_LABELS = {
    "all_straight":   "All Straight",
    "all_left":       "All Left",
    "uniform_random": "Uniform Random",
}

BAR_WIDTH      = 0.25          # width of a single bar
GROUP_SPACING  = 0.10          # extra gap between scenario groups
FIGSIZE        = (14, 6)
TITLE_FONTSIZE = 14
AXIS_FONTSIZE  = 11
TICK_FONTSIZE  = 9
CAPSIZE        = 4
ALPHA_BAR      = 0.88
ALPHA_ERR      = 1.0
GRID_ALPHA     = 0.25

# ── CSV filename pattern ──────────────────────────────────────────────────────
# Expected: {scen}_{intention}_{rate_key}_{version}.csv
# Both rate_key (e.g. Sc2_All_high_3H) and version (e.g. heuristic_attention_discrete)
# contain underscores, so we anchor on the known fixed-vocabulary strings.
_VERSIONS   = [
    "heuristic_continous", "heuristic_discrete",
    "attention_continous", "attention_discrete",
    "heuristic_attention_continous", "heuristic_attention_discrete",
]
_INTENTIONS = ["all_straight", "all_left", "uniform_random"]
_FNAME_RE = re.compile(
    r"^(?P<scen>[^_]+)_"
    r"(?P<intention>" + "|".join(re.escape(i) for i in _INTENTIONS) + r")_"
    r"(?P<rate_key>Sc.+?)_"   # non-greedy — stops before the version token
    r"(?P<version>" + "|".join(re.escape(v) for v in _VERSIONS) + r")\.csv$"
)

# ─────────────────────────────────────────────────────────────────────────────

def _load_csvs(output_dir: str, version_filter: Optional[str] = None) -> dict:
    """
    Returns a nested dict:
        data[rate_key][intention] = {"collision_rate": float, "collision_se": float,
                                     "avg_travel_time": float, "travel_time_se": float,
                                     "n": int}
    """
    data: dict = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".csv"):
            continue
        m = _FNAME_RE.match(fname)
        if m is None:
            continue
        if version_filter and m.group("version") != version_filter:
            continue

        rate_key  = m.group("rate_key")
        intention = m.group("intention")
        fpath     = os.path.join(output_dir, fname)

        collisions, travel_times = [], []
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    collisions.append(int(row["collision"]))
                    travel_times.append(float(row["travel_time"]))
                except (KeyError, ValueError):
                    continue

        if not collisions:
            continue

        data[rate_key][intention].extend(
            [{"collision": c, "travel_time": t}
             for c, t in zip(collisions, travel_times)]
        )

    # Aggregate: mean ± standard error
    aggregated: dict = defaultdict(dict)
    for rate_key, intentions in data.items():
        for intention, rows in intentions.items():
            n          = len(rows)
            col_vals   = np.array([r["collision"]    for r in rows], dtype=float)
            tt_vals    = np.array([r["travel_time"]  for r in rows], dtype=float)
            se         = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            aggregated[rate_key][intention] = {
                "collision_rate":  col_vals.mean() * 100,   # → percentage
                "collision_se":    se(col_vals)    * 100,
                "avg_travel_time": tt_vals.mean(),
                "travel_time_se":  se(tt_vals),
                "n":               n,
            }

    return aggregated


def _make_grouped_bar(
    ax: plt.Axes,
    aggregated: dict,
    rate_keys: list[str],
    intentions: list[str],
    value_key: str,
    error_key: str,
    ylabel: str,
    title: str,
) -> None:
    """Draw grouped bars on *ax* for the given metric."""
    n_groups     = len(rate_keys)
    n_intentions = len(intentions)
    total_width  = n_intentions * BAR_WIDTH + GROUP_SPACING
    group_centers = np.arange(n_groups) * total_width

    for i, intention in enumerate(intentions):
        offset = (i - (n_intentions - 1) / 2) * BAR_WIDTH
        values = []
        errors = []
        for rate_key in rate_keys:
            entry = aggregated.get(rate_key, {}).get(intention)
            if entry:
                values.append(entry[value_key])
                errors.append(entry[error_key])
            else:
                values.append(0.0)
                errors.append(0.0)

        color = INTENTION_COLORS.get(intention, "#999999")
        ax.bar(
            group_centers + offset,
            values,
            width=BAR_WIDTH,
            color=color,
            alpha=ALPHA_BAR,
            label=INTENTION_LABELS.get(intention, intention),
            zorder=3,
        )
        ax.errorbar(
            group_centers + offset,
            values,
            yerr=errors,
            fmt="none",
            color="black",
            capsize=CAPSIZE,
            linewidth=1.2,
            alpha=ALPHA_ERR,
            zorder=4,
        )

    # Formatting
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [k.replace("_", "\n") for k in rate_keys],
        fontsize=TICK_FONTSIZE,
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_eval_results(
    output_dir: str = "output",
    version_filter: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Parse all CSVs in *output_dir* and produce two grouped bar charts side-by-side.

    Parameters
    ----------
    output_dir    : Directory containing the CSV files from v0_1_evaluate.py.
    version_filter: If given, only CSVs whose version tag matches are loaded
                    (e.g. "heuristic_discrete").
    save_path     : If given, the figure is saved to this path (PNG/PDF/SVG).
    show          : Whether to call plt.show().
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    aggregated = _load_csvs(output_dir, version_filter)

    if not aggregated:
        raise ValueError(
            f"No matching CSV files found in '{output_dir}'"
            + (f" for version='{version_filter}'" if version_filter else "")
            + ".\nCheck that the eval script has been run and files follow the "
              "expected naming convention."
        )

    # Canonical ordering: preserve insertion order but sort scenario keys
    rate_keys  = sorted(aggregated.keys())
    intentions = list(INTENTION_COLORS.keys())   # fixed display order

    title_suffix = f" — {version_filter}" if version_filter else ""

    fig, (ax_col, ax_tt) = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(
        f"Evaluation Results{title_suffix}",
        fontsize=TITLE_FONTSIZE + 2,
        fontweight="bold",
        y=1.01,
    )

    _make_grouped_bar(
        ax=ax_col,
        aggregated=aggregated,
        rate_keys=rate_keys,
        intentions=intentions,
        value_key="collision_rate",
        error_key="collision_se",
        ylabel="Collision Rate (%)",
        title="Collision Rate by Scenario & Intention",
    )

    _make_grouped_bar(
        ax=ax_tt,
        aggregated=aggregated,
        rate_keys=rate_keys,
        intentions=intentions,
        value_key="avg_travel_time",
        error_key="travel_time_se",
        ylabel="Avg Travel Time (s)",
        title="Average Travel Time by Scenario & Intention",
    )

    # Shared legend below both charts
    handles = [
        mpatches.Patch(
            color=INTENTION_COLORS[i],
            alpha=ALPHA_BAR,
            label=INTENTION_LABELS[i],
        )
        for i in intentions
        if i in INTENTION_COLORS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(intentions),
        fontsize=AXIS_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot v0.1 evaluation results.")
    ap.add_argument("--output_dir", default="output",
                    help="Directory containing eval CSV files (default: output/)")
    ap.add_argument("--version", default=None,
                    help="Filter to a single version tag, e.g. heuristic_discrete")
    ap.add_argument("--save", default=None,
                    help="Save figure to this path, e.g. figures/results.png")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not call plt.show() (useful for headless servers)")
    args = ap.parse_args()

    plot_eval_results(
        output_dir=args.output_dir,
        version_filter=args.version,
        save_path=args.save,
        show=not args.no_show,
    )
