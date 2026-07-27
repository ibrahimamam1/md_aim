"""
plot.py  —  Academic-quality figures for intersection-controller evaluation.

Produces four figures:
  Fig 1 — Collision Rate curves  (4 subplots, one per intention)
  Fig 2 — Collision Rate heatmap (controller × traffic scenario, avg over intentions)
  Fig 3 — Travel Time curves     (4 subplots, one per intention)
  Fig 4 — Travel Time heatmap    (controller × traffic scenario, avg over intentions)

CSV naming convention:
  rbl_{intention}_{traffic_scenario}_{controller}.csv
  e.g.  rbl_uniform_random_Sc6_Mixed_ML_heuristic_discrete.csv
"""

import os
import re
import warnings
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  — adjust results_dir / output_dir to match your folder layout
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(_HERE, "output")   # ← folder that holds the CSVs
output_dir  = os.path.join(_HERE, "plots")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------

# Traffic-scenario definitions (kept for total-flow computation)
high_rate   = 400
medium_rate = 275
low_rate    = 150

TRAFFIC_SCENARIOS = {
    "Sc1_All_low":      [{"N": low_rate,    "S": low_rate,    "W": low_rate,    "E": low_rate}],
    "Sc2_All_high_3H":  [
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": medium_rate},
        {"N": high_rate,   "S": high_rate,   "W": medium_rate, "E": high_rate},
        {"N": high_rate,   "S": medium_rate, "W": high_rate,   "E": high_rate},
        {"N": medium_rate, "S": high_rate,   "W": high_rate,   "E": high_rate},
    ],
    "Sc3_All_medium":   [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}],
    "Sc4_Mixed_2H":     [
        {"N": high_rate,   "S": high_rate,   "W": low_rate,    "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": low_rate,    "W": high_rate,   "E": low_rate},
        {"N": low_rate,    "S": high_rate,   "W": low_rate,    "E": high_rate},
    ],
    "Sc5_Mixed_1H":     [
        {"N": high_rate,   "S": medium_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": high_rate,   "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": high_rate,   "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
    ],
    "Sc6_Mixed_ML":     [
        {"N": medium_rate, "S": medium_rate, "W": low_rate,    "E": low_rate},
        {"N": medium_rate, "S": low_rate,    "W": medium_rate, "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": medium_rate, "E": medium_rate},
    ],
}

# Mean total inflow per scenario (used for x-axis ordering)
SCENARIO_MEAN_FLOW = {
    sc: np.mean([sum(cfg.values()) for cfg in cfgs])
    for sc, cfgs in TRAFFIC_SCENARIOS.items()
}

# Canonical x-axis order: increasing mean total flow
SCENARIO_ORDER = sorted(TRAFFIC_SCENARIOS.keys(), key=lambda s: SCENARIO_MEAN_FLOW[s])

# Human-readable labels for traffic scenarios
SCENARIO_LABELS = {
    "Sc1_All_low":     "Sc1\nAll Low",
    "Sc2_All_high_3H": "Sc2\nAll High",
    "Sc3_All_medium":  "Sc3\nAll Med.",
    "Sc4_Mixed_2H":    "Sc4\nMixed 2H",
    "Sc5_Mixed_1H":    "Sc5\nMixed 1H",
    "Sc6_Mixed_ML":    "Sc6\nMixed ML",
}

# Controllers present in the filenames
CONTROLLERS = {
    "heuristic_discrete":   "Heuristic + Discrete",
    "heuristic_continous":  "Heuristic + Continuous",
    "attention_discrete":   "Attention + Discrete",
    "attention_continous":  "Attention + Continuous",
}

# Intentions present in the filenames
INTENTIONS = {
    "all_straight":      "All Straight",
    "all_left":          "All Left",
    "uniform_random":    "Uniform Random",
    "asymetric_random":  "Asymmetric Random",
}

# ---------------------------------------------------------------------------
# Academic colour / style palette
# ---------------------------------------------------------------------------
CONTROLLER_COLORS = {
    "heuristic_discrete":  "#1f77b4",   # muted blue
    "heuristic_continous": "#d62728",   # brick red
    "attention_discrete":  "#2ca02c",   # green
    "attention_continous": "#9467bd",   # purple
}

CONTROLLER_MARKERS = {
    "heuristic_discrete":  "o",
    "heuristic_continous": "s",
    "attention_discrete":  "^",
    "attention_continous": "D",
}

CONTROLLER_LINESTYLES = {
    "heuristic_discrete":  "-",
    "heuristic_continous": "--",
    "attention_discrete":  "-.",
    "attention_continous": ":",
}

# ---------------------------------------------------------------------------
# Matplotlib / Seaborn global style  (IEEE / Nature-compatible)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.title_fontsize": 9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})

# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
def parse_filename(path: str):
    """
    Extract (intention, traffic_scenario, controller) from a CSV filename.

    Expected pattern (prefix 'rbl_' is optional):
        [rbl_]{intention}_{Sc#_...traffic_scenario...}_{controller}.csv
    """
    name = os.path.basename(path).replace(".csv", "")

    # strip optional leading 'rbl_'
    if name.startswith("rbl_"):
        name = name[4:]

    # Find which controller key matches (longest-first to avoid partial hits)
    controller = None
    for key in sorted(CONTROLLERS.keys(), key=len, reverse=True):
        if name.endswith(key):
            controller = key
            name = name[: -(len(key) + 1)]   # remove '_controller'
            break

    if controller is None:
        return None

    # Find which traffic scenario key matches
    scenario = None
    for key in TRAFFIC_SCENARIOS:
        # use a word-boundary-style check so 'Sc2' doesn't match 'Sc2_...' partially
        if key in name:
            scenario = key
            name = name.replace(key, "").strip("_")
            break

    if scenario is None:
        return None

    # What remains should be the intention
    intention = name.strip("_") if name.strip("_") in INTENTIONS else None

    return {
        "intention":  intention,
        "scenario":   scenario,
        "controller": controller,
    }


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(data_dir: str) -> pd.DataFrame:
    """
    Scan *data_dir* for CSV files matching the naming convention.
    Returns a DataFrame with columns:
        intention, scenario, controller, collision_rate, travel_time_mean
    """
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    records = []
    for path in csv_files:
        meta = parse_filename(path)
        if meta is None:
            print(f"  [skip] Could not parse: {os.path.basename(path)}")
            continue

        df = pd.read_csv(path)

        # --- collision rate  (fraction of runs with collision == 1) ---
        if "collision" not in df.columns:
            print(f"  [skip] No 'collision' column in: {os.path.basename(path)}")
            continue

        collision_rate = df["collision"].mean()          # ∈ [0, 1]

        # --- mean travel time (exclude rows with collision so we measure
        #     completed trips only; fall back to all rows if needed) ---
        if "travel_time" in df.columns:
            mask = df["collision"] == 0 if "collision" in df.columns else slice(None)
            tt_series = df.loc[mask, "travel_time"] if hasattr(mask, "__iter__") else df["travel_time"]
            travel_time_mean = tt_series.mean()
        else:
            travel_time_mean = np.nan

        records.append({
            "intention":        meta["intention"],
            "scenario":         meta["scenario"],
            "controller":       meta["controller"],
            "collision_rate":   collision_rate,
            "travel_time_mean": travel_time_mean,
        })

    if not records:
        raise ValueError("No valid CSV files could be parsed.")

    df_all = pd.DataFrame(records)
    print(
        f"Loaded {len(df_all)} file(s) | "
        f"intentions: {sorted(df_all['intention'].dropna().unique())} | "
        f"scenarios: {sorted(df_all['scenario'].dropna().unique())} | "
        f"controllers: {sorted(df_all['controller'].unique())}"
    )
    return df_all


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _scenario_x_positions(present_scenarios):
    """Return x-tick positions for the ordered subset of scenarios present."""
    ordered = [s for s in SCENARIO_ORDER if s in present_scenarios]
    return ordered, {s: i for i, s in enumerate(ordered)}


def _add_legend(ax, handles_labels=None, title="Controller", ncol=1):
    if handles_labels:
        handles, labels = handles_labels
    else:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        title=title,
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        loc="best",
    )


# ---------------------------------------------------------------------------
# FIGURE 1 & 3 — Line-curve subplots (one per intention)
# ---------------------------------------------------------------------------
def plot_curves(df: pd.DataFrame, metric: str, output_dir: str):
    """
    metric: 'collision_rate' or 'travel_time_mean'
    Produces a 2×2 grid of subplots, one per intention.
    Each subplot shows 4 curves — one per controller — across 6 traffic scenarios.
    """
    assert metric in ("collision_rate", "travel_time_mean")

    if metric == "collision_rate":
        ylabel     = "Collision Rate"
        fig_title  = "Collision Rate vs. Traffic Scenario"
        fname      = "fig1_collision_rate_curves.pdf"
        y_format   = lambda v: f"{v:.3f}"   # noqa: E731
        y_pct      = True
    else:
        ylabel     = "Mean Travel Time (s)"
        fig_title  = "Mean Travel Time vs. Traffic Scenario"
        fname      = "fig3_travel_time_curves.pdf"
        y_format   = lambda v: f"{v:.1f}"   # noqa: E731
        y_pct      = False

    df_plot = df.dropna(subset=["intention", metric])
    intentions_present = [i for i in INTENTIONS if i in df_plot["intention"].unique()]
    controllers_present = [c for c in CONTROLLERS if c in df_plot["controller"].unique()]
    scenarios_present   = df_plot["scenario"].dropna().unique()
    ordered_sc, sc_pos  = _scenario_x_positions(scenarios_present)
    x_vals = np.arange(len(ordered_sc))

    n_int = len(intentions_present)
    ncols = 2
    nrows = int(np.ceil(n_int / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 3.8 * nrows),
                             sharex=False, sharey=False)
    axes_flat = np.array(axes).flatten()

    for idx, intention in enumerate(intentions_present):
        ax = axes_flat[idx]
        sub = df_plot[df_plot["intention"] == intention]

        # Group: mean across multiple sub-scenarios (e.g. Sc2 has 5 rows)
        agg = (
            sub.groupby(["scenario", "controller"])[metric]
            .mean()
            .reset_index()
        )

        for ctrl in controllers_present:
            ctrl_df = agg[agg["controller"] == ctrl].set_index("scenario")

            ys = np.full(len(ordered_sc), np.nan)
            for j, sc in enumerate(ordered_sc):
                if sc in ctrl_df.index:
                    ys[j] = ctrl_df.loc[sc, metric]

            ax.plot(
                x_vals,
                ys * 100 if y_pct else ys,
                color=CONTROLLER_COLORS.get(ctrl, "grey"),
                linestyle=CONTROLLER_LINESTYLES.get(ctrl, "-"),
                marker=CONTROLLER_MARKERS.get(ctrl, "o"),
                label=CONTROLLERS.get(ctrl, ctrl),
                zorder=3,
            )

        ax.set_title(INTENTIONS.get(intention, intention), fontweight="bold")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            [SCENARIO_LABELS.get(s, s) for s in ordered_sc],
            rotation=0,
            ha="center",
            fontsize=8,
        )
        ax.set_ylabel("Collision Rate (%)" if y_pct else ylabel)
        ax.set_xlabel("Traffic Scenario (increasing total flow →)")

        if y_pct:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        _add_legend(ax, title="Controller")

    # hide unused panels
    for idx in range(n_int, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.5, w_pad=2.5, h_pad=2.5)

    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# FIGURE 2 & 4 — Heatmap (controller × scenario, averaged over intentions)
# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: str):
    """
    Rows    = controllers  (4)
    Columns = traffic scenarios  (6, ordered by increasing flow)
    Values  = metric averaged over all intentions and all sub-scenario variants.
    """
    assert metric in ("collision_rate", "travel_time_mean")

    if metric == "collision_rate":
        cmap      = "YlOrRd"
        fmt       = ".3f"
        cbar_label = "Collision Rate"
        title      = "Collision Rate — Controller × Traffic Scenario\n(averaged over intentions)"
        fname      = "fig2_collision_rate_heatmap.pdf"
        scale      = 100.0           # show as %
        fmt_annot  = ".2f"
        cbar_label = "Collision Rate (%)"
    else:
        cmap      = "YlGnBu"
        fmt       = ".1f"
        cbar_label = "Mean Travel Time (s)"
        title      = "Mean Travel Time — Controller × Traffic Scenario\n(averaged over intentions)"
        fname      = "fig4_travel_time_heatmap.pdf"
        scale      = 1.0
        fmt_annot  = ".1f"

    df_plot = df.dropna(subset=["intention", metric])
    controllers_present = [c for c in CONTROLLERS if c in df_plot["controller"].unique()]
    scenarios_present   = df_plot["scenario"].dropna().unique()
    ordered_sc, _       = _scenario_x_positions(scenarios_present)

    # Aggregate: mean over all intentions and all sub-scenario repetitions
    agg = (
        df_plot.groupby(["controller", "scenario"])[metric]
        .mean()
        .reset_index()
    )
    agg[metric] = agg[metric] * scale

    pivot = agg.pivot(index="controller", columns="scenario", values=metric)
    # re-order rows (controllers) and columns (scenarios)
    pivot = pivot.reindex(index=controllers_present, columns=ordered_sc)

    row_labels = [CONTROLLERS.get(c, c) for c in pivot.index]
    col_labels = [SCENARIO_LABELS.get(s, s).replace("\n", " ") for s in pivot.columns]

    vmin = pivot.values[~np.isnan(pivot.values)].min() if not pivot.empty else 0
    vmax = pivot.values[~np.isnan(pivot.values)].max() if not pivot.empty else 1

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=fmt_annot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="0.85",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )

    ax.set_xlabel("Traffic Scenario (increasing total flow →)", labelpad=6)
    ax.set_ylabel("Controller", labelpad=6)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.5)

    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Intersection Controller Analysis — Academic Figures")
    print("=" * 65)
    print(f"Results dir : {results_dir}")
    print(f"Output  dir : {output_dir}\n")

    df = load_data(results_dir)

    # ── Fig 1 — Collision rate curves ────────────────────────────────────────
    print("\n[Fig 1] Collision-rate line curves …")
    plot_curves(df, "collision_rate", output_dir)

    # ── Fig 2 — Collision rate heatmap ───────────────────────────────────────
    print("[Fig 2] Collision-rate heatmap …")
    plot_heatmap(df, "collision_rate", output_dir)

    # ── Fig 3 — Travel time curves ───────────────────────────────────────────
    print("[Fig 3] Travel-time line curves …")
    plot_curves(df, "travel_time_mean", output_dir)

    # ── Fig 4 — Travel time heatmap ──────────────────────────────────────────
    print("[Fig 4] Travel-time heatmap …")
    plot_heatmap(df, "travel_time_mean", output_dir)

    print(f"\nDone! All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
