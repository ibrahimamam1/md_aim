"""
plot.py  —  Academic-quality figures for intersection-controller evaluation.

Produces figures:
  Fig 0 — Summary bar charts     (Collision Rate & Travel Time across all scenarios)
  Fig 1 — Collision Rate curves  (4 subplots, one per intention)
  Fig 2 — Collision Rate heatmap (controller × traffic scenario, avg over intentions)
  Fig 3 — Travel Time curves     (4 subplots, one per intention)
  Fig 4 — Travel Time heatmap    (controller × traffic scenario, avg over intentions)

Usage:
  python src/test/plot.py /path/to/results
  python src/test/plot.py --results_dir /path/to/results --output_dir /path/to/save_plots

CSV naming convention:
  rbl_{intention}_{traffic_scenario}_{controller}.csv
  e.g.  rbl_uniform_random_Sc6_Mixed_ML_heuristic_discrete.csv
"""


import argparse
import os
import re
import warnings
from glob import glob

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  — default output_dir
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(_HERE, "plots")
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
    "heuristic_discrete_3":          "Heuristic + Discrete (3)",
    "heuristic_discrete_5":          "Heuristic + Discrete (5)",
    "heuristic_discrete_10":         "Heuristic + Discrete (10)",
    "heuristic_discrete":            "Heuristic + Discrete",
    "heuristic_continuous":          "Heuristic + Continuous",
    "heuristic_continous":           "Heuristic + Continuous",
    "attention_discrete_3":          "Attention + Discrete (3)",
    "attention_discrete_5":          "Attention + Discrete (5)",
    "attention_discrete_10":         "Attention + Discrete (10)",
    "attention_discrete":            "Attention + Discrete",
    "attention_continuous":          "Attention + Continuous",
    "attention_continous":           "Attention + Continuous",
    "heuristic_attention_discrete":  "Heuristic+Attention + Discrete",
    "heuristic_attention_continous": "Heuristic+Attention + Continuous",
}

# Intentions present in the filenames
INTENTIONS = {
    "all_straight":      "All Straight",
    "all_left":          "All Left",
    "uniform_random":    "Uniform Random",
    "asymetric_random":  "Asymmetric Random",
    "asymmetric_random": "Asymmetric Random",
}

# ---------------------------------------------------------------------------
# Academic colour / style palette
# ---------------------------------------------------------------------------
CONTROLLER_COLORS = {
    "heuristic_discrete":            "#1f77b4",   # muted blue
    "heuristic_discrete_3":          "#6baed6",   # light blue
    "heuristic_discrete_5":          "#1f77b4",   # muted blue
    "heuristic_discrete_10":         "#08519c",   # dark blue
    "heuristic_continuous":          "#d62728",   # brick red
    "heuristic_continous":           "#d62728",   # brick red
    "attention_discrete":            "#2ca02c",   # green
    "attention_discrete_3":          "#74c476",   # light green
    "attention_discrete_5":          "#2ca02c",   # green
    "attention_discrete_10":         "#006d2c",   # dark green
    "attention_continuous":          "#9467bd",   # purple
    "attention_continous":           "#9467bd",   # purple
    "heuristic_attention_discrete":  "#17becf",   # cyan/teal
    "heuristic_attention_continous": "#8c564b",   # brown/burgundy
}

CONTROLLER_MARKERS = {
    "heuristic_discrete":            "o",
    "heuristic_discrete_3":          "o",
    "heuristic_discrete_5":          "o",
    "heuristic_discrete_10":         "o",
    "heuristic_continuous":          "s",
    "heuristic_continous":           "s",
    "attention_discrete":            "^",
    "attention_discrete_3":          "^",
    "attention_discrete_5":          "^",
    "attention_discrete_10":         "^",
    "attention_continuous":          "D",
    "attention_continous":           "D",
    "heuristic_attention_discrete":  "v",
    "heuristic_attention_continous": "P",
}

CONTROLLER_LINESTYLES = {
    "heuristic_discrete":            "-",
    "heuristic_discrete_3":          "-",
    "heuristic_discrete_5":          "-",
    "heuristic_discrete_10":         "-",
    "heuristic_continuous":          "--",
    "heuristic_continous":           "--",
    "attention_discrete":            "-.",
    "attention_discrete_3":          "-.",
    "attention_discrete_5":          "-.",
    "attention_discrete_10":         "-.",
    "attention_continuous":          ":",
    "attention_continous":           ":",
    "heuristic_attention_discrete":  "-",
    "heuristic_attention_continous": "--",
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


def parse_profile_str(val: str) -> np.ndarray:
    if not val or pd.isna(val):
        return np.array([], dtype=float)
    val = str(val).strip()
    if not val:
        return np.array([], dtype=float)
    if val.startswith("[") and val.endswith("]"):
        import json
        try:
            return np.array(json.loads(val), dtype=float)
        except Exception:
            pass
    for sep in [";", ",", " "]:
        if sep in val:
            try:
                return np.array([float(x) for x in val.split(sep) if x.strip()], dtype=float)
            except Exception:
                continue
    try:
        return np.array([float(val)], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def load_profiles(data_dir: str) -> dict:
    """
    Load profile telemetry (times, distances, velocities, jerks) from CSV files in *data_dir*,
    grouped by controller.
    """
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    profiles = {ctrl: {"times": [], "distances": [], "velocities": [], "jerks": [], "jerk_times": [], "accelerations": []} for ctrl in CONTROLLERS}

    for path in csv_files:
        meta = parse_filename(path)
        if meta is None or meta["controller"] not in profiles:
            continue
        ctrl = meta["controller"]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        for _, row in df.iterrows():
            t_arr = parse_profile_str(row.get("time_profile", ""))
            d_arr = parse_profile_str(row.get("distance_profile", ""))
            v_arr = parse_profile_str(row.get("velocity_profile", ""))
            a_arr = parse_profile_str(row.get("acceleration_profile", ""))

            if len(t_arr) > 0 and len(v_arr) == len(t_arr):
                profiles[ctrl]["times"].extend(t_arr)
                profiles[ctrl]["velocities"].extend(v_arr)
                if len(v_arr) > 1:
                    profiles[ctrl]["jerk_times"].extend(t_arr[1:])
                    profiles[ctrl]["jerks"].extend(np.diff(v_arr))
            if len(d_arr) > 0 and len(v_arr) == len(d_arr):
                profiles[ctrl]["distances"].extend(d_arr)
            if len(t_arr) > 0 and len(a_arr) == len(t_arr):
                profiles[ctrl]["accelerations"].extend(a_arr)

    return profiles


def plot_profile_scatters(profiles: dict, output_dir: str):
    """
    Plot academic-quality scatter plots of velocity profile with time, velocity profile with distance,
    and jerk profile with time across various controllers.
    """
    os.makedirs(output_dir, exist_ok=True)
    controllers = [c for c in CONTROLLERS.keys() if c in profiles and any(len(profiles[c][k]) > 0 for k in profiles[c])]

    def _draw_scatter(ax, x_key, y_key, xlabel, ylabel, title, max_pts=2500):
        has_data = False
        for c in controllers:
            x_data = np.asarray(profiles[c].get(x_key, []), dtype=float)
            y_data = np.asarray(profiles[c].get(y_key, []), dtype=float)
            if len(x_data) == 0 or len(y_data) == 0 or len(x_data) != len(y_data):
                continue
            valid = np.isfinite(x_data) & np.isfinite(y_data)
            x_data, y_data = x_data[valid], y_data[valid]
            if len(x_data) == 0:
                continue
            has_data = True
            color = CONTROLLER_COLORS.get(c, "#333333")
            label = CONTROLLERS.get(c, c)
            if len(x_data) > max_pts:
                idx = np.random.choice(len(x_data), size=max_pts, replace=False)
                x_sc, y_sc = x_data[idx], y_data[idx]
            else:
                x_sc, y_sc = x_data, y_data
            ax.scatter(x_sc, y_sc, color=color, alpha=0.35, s=12, edgecolors="none", label=label, zorder=2)

            x_min, x_max = np.min(x_data), np.max(x_data)
            if x_max > x_min:
                n_bins = min(40, max(10, int(np.sqrt(len(x_data)))))
                bins = np.linspace(x_min, x_max, n_bins + 1)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                bin_means = []
                valid_centers = []
                for b_idx in range(n_bins):
                    in_bin = (x_data >= bins[b_idx]) & (x_data < bins[b_idx + 1])
                    if b_idx == n_bins - 1:
                        in_bin = (x_data >= bins[b_idx]) & (x_data <= bins[b_idx + 1])
                    if np.any(in_bin):
                        bin_means.append(np.mean(y_data[in_bin]))
                        valid_centers.append(bin_centers[b_idx])
                if len(valid_centers) > 1:
                    ax.plot(valid_centers, bin_means, color=color, linewidth=2.2, linestyle="-", zorder=4)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not has_data:
            ax.text(0.5, 0.5, "No profile telemetry found\n(Re-run evaluation to record profiles)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray", style="italic")

    # 1. Individual figures
    configs = [
        ("times", "velocities", "Time (s)", "Velocity (m/s)", "Velocity Profile vs. Time", "fig5_velocity_vs_time_scatter"),
        ("distances", "velocities", "Distance (m)", "Velocity (m/s)", "Velocity Profile vs. Distance", "fig6_velocity_vs_distance_scatter"),
        ("jerk_times", "jerks", "Time (s)", "ΔVelocity (m/s)", "Velocity Difference vs. Time", "fig7_jerk_vs_time_scatter"),
    ]
    for x_key, y_key, xlabel, ylabel, title, fname in configs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        _draw_scatter(ax, x_key, y_key, xlabel, ylabel, title)
        handles = [mpatches.Patch(color=CONTROLLER_COLORS[c], alpha=0.88, label=CONTROLLERS[c]) for c in controllers if c in CONTROLLER_COLORS]
        if handles:
            ax.legend(handles=handles, loc="best", frameon=False, fontsize=9)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved {fname}.png/.pdf")

    # 2. Combined 1x3 panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (x_key, y_key, xlabel, ylabel, title, _) in zip(axes, configs):
        _draw_scatter(ax, x_key, y_key, xlabel, ylabel, title)
    handles = [mpatches.Patch(color=CONTROLLER_COLORS[c], alpha=0.88, label=CONTROLLERS[c]) for c in controllers if c in CONTROLLER_COLORS]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(max(1, len(handles)), 3), frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fig8_all_profiles_scatter.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved fig8_all_profiles_scatter.png/.pdf")


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
# FIGURE 0 — Summary Bar Charts (Collision Rate & Travel Time)
# ---------------------------------------------------------------------------
def plot_summary_bars(df: pd.DataFrame, output_dir: str):
    """
    Produces a 1x2 summary panel figure of horizontal bar charts comparing
    all present controllers across all traffic scenarios & intentions:
      - Left: Mean Collision Rate (%) with SEM error bars & text labels
      - Right: Mean Travel Time (s) with SEM error bars & text labels
    """
    os.makedirs(output_dir, exist_ok=True)
    controllers_present = [c for c in CONTROLLERS if c in df["controller"].unique()]
    if not controllers_present:
        controllers_present = sorted(df["controller"].unique())

    # Reverse order so the first controller in CONTROLLERS is plotted at the top of barh
    controllers_plot = controllers_present[::-1]
    n_ctrl = len(controllers_plot)
    y_pos = np.arange(n_ctrl)

    fig_height = max(4.2, n_ctrl * 0.78 + 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, fig_height))

    def _draw_summary_barh(ax, metric: str, xlabel: str, title: str, is_pct: bool):
        means = []
        sems = []
        colors = []
        labels = []

        for c in controllers_plot:
            sub = df[df["controller"] == c][metric].dropna()
            if len(sub) == 0:
                m_val = 0.0
                s_val = 0.0
            else:
                m_val = float(sub.mean())
                s_val = float(sub.sem()) if len(sub) > 1 else 0.0

            if is_pct:
                m_val *= 100.0
                s_val *= 100.0

            if np.isnan(s_val):
                s_val = 0.0

            means.append(m_val)
            sems.append(s_val)
            colors.append(CONTROLLER_COLORS.get(c, "#333333"))
            labels.append(CONTROLLERS.get(c, c))

        bars = ax.barh(
            y_pos,
            means,
            height=0.58,
            xerr=sems,
            color=colors,
            alpha=0.90,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
            zorder=3,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)

        ax.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=0)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        max_x = max((m + s for m, s in zip(means, sems)), default=1.0)
        max_x = max(max_x, 1.0)
        ax.set_xlim(0, max_x * 1.25)

        offset = max_x * 0.025
        for i in range(n_ctrl):
            if np.isnan(means[i]):
                continue
            x_pos = means[i] + sems[i] + offset
            txt = f"{means[i]:.1f}%" if is_pct else f"{means[i]:.1f}s"
            ax.text(
                x_pos,
                y_pos[i],
                txt,
                va="center",
                ha="left",
                fontsize=10.5,
                fontweight="bold",
                color="#111111",
                zorder=5,
            )

    _draw_summary_barh(
        ax1,
        "collision_rate",
        "Mean Collision Rate (%)",
        "Collision Rate (all scenarios)",
        is_pct=True,
    )

    _draw_summary_barh(
        ax2,
        "travel_time_mean",
        "Mean Travel Time (s)",
        "Travel Time (all scenarios)",
        is_pct=False,
    )

    fig.tight_layout(pad=2.2, w_pad=3.8)

    for ext in ["pdf", "png"]:
        save_path = os.path.join(output_dir, f"fig0_summary_bars.{ext}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig0_summary_bars.pdf / .png")


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
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate academic-quality figures for intersection-controller evaluation."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to the directory containing result CSV files.",
    )
    parser.add_argument(
        "-r", "--results_dir", "--results-dir",
        dest="results_dir_flag",
        default=None,
        help="Path to the directory containing result CSV files (flag alternative).",
    )
    parser.add_argument(
        "-o", "--output_dir", "--output-dir",
        dest="output_dir",
        default=output_dir,
        help="Directory to save generated plots (default: plots/)",
    )
    args = parser.parse_args()
    results_dir_val = args.results_dir_flag or args.results_dir
    if not results_dir_val:
        parser.error(
            "The results directory argument is required (e.g. 'python plot.py /path/to/results' "
            "or '--results_dir /path/to/results')."
        )
    return results_dir_val, args.output_dir


def main(results_dir: str = None, out_dir: str = None):
    if results_dir is None or out_dir is None:
        cli_results_dir, cli_output_dir = parse_args()
        if results_dir is None:
            results_dir = cli_results_dir
        if out_dir is None:
            out_dir = cli_output_dir

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("Intersection Controller Analysis — Academic Figures")
    print("=" * 65)
    print(f"Results dir : {results_dir}")
    print(f"Output  dir : {out_dir}\n")

    df = load_data(results_dir)

    # ── Fig 0 — Summary bar charts ───────────────────────────────────────────
    print("\n[Fig 0] Summary bar charts (Collision Rate & Travel Time) …")
    plot_summary_bars(df, out_dir)

    # ── Fig 1 — Collision rate curves ────────────────────────────────────────
    print("\n[Fig 1] Collision-rate line curves …")
    plot_curves(df, "collision_rate", out_dir)

    # ── Fig 2 — Collision rate heatmap ───────────────────────────────────────
    print("[Fig 2] Collision-rate heatmap …")
    plot_heatmap(df, "collision_rate", out_dir)

    # ── Fig 3 — Travel time curves ───────────────────────────────────────────
    print("[Fig 3] Travel-time line curves …")
    plot_curves(df, "travel_time_mean", out_dir)

    # ── Fig 4 — Travel time heatmap ──────────────────────────────────────────
    print("[Fig 4] Travel-time heatmap …")
    plot_heatmap(df, "travel_time_mean", out_dir)

    # ── Fig 5-8 — Profile scatter plots ──────────────────────────────────────
    print("\n[Fig 5-8] Velocity & Velocity Difference profile scatter plots …")
    profiles = load_profiles(results_dir)
    plot_profile_scatters(profiles, out_dir)

    print(f"\nDone! All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
