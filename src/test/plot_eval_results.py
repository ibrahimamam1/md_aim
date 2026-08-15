"""
plot_eval_results.py
────────────────────
Reads all CSV files produced by v0_1_evaluate.py from the output directory
and generates a 3x3 dashboard with grouped bar charts:

  1. Collision Rate  (%), 2. Average Travel Time (s), 3. Average Waiting Time (s)
  4. Success Rate    (%), 5. Average Speed (m/s),     6. Average Safe Gap (0–1)
  7–9. Velocity / jerk profile scatter plots + worst-case velocity profile

Each bar group is a scenario, with one bar per controller version.

Usage (standalone):
    python plot_eval_results.py                         # uses default output/ dir
    python plot_eval_results.py --output_dir /path/to/output --intention asymetric_random
    python plot_eval_results.py --version attention_continous

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
VERSION_COLORS = {
    "heuristic_continuous":          "#2E86AB",   # steel blue
    "heuristic_continous":           "#2E86AB",   # steel blue
    "heuristic_discrete":            "#A23B72",   # deep magenta/purple
    "heuristic_discrete_3":          "#C56E90",   # lighter magenta
    "heuristic_discrete_5":          "#A23B72",   # deep magenta/purple
    "heuristic_discrete_10":         "#6F1D4A",   # darker magenta
    "attention_continuous":          "#F18F01",   # warm amber/orange
    "attention_continous":           "#F18F01",   # warm amber/orange
    "attention_discrete":            "#C73E1D",   # vivid red/terracotta
    "attention_discrete_3":          "#E26D52",   # lighter terracotta
    "attention_discrete_5":          "#C73E1D",   # vivid red/terracotta
    "attention_discrete_10":         "#8C260E",   # darker red
    "heuristic_attention_continous": "#3B1F2B",   # dark burgundy
    "heuristic_attention_discrete":  "#2F9599",   # teal
}

VERSION_LABELS = {
    "heuristic_continuous":          "Heuristic (Continuous)",
    "heuristic_continous":           "Heuristic (Continuous)",
    "heuristic_discrete":            "Heuristic (Discrete)",
    "heuristic_discrete_3":          "Heuristic (Discrete 3)",
    "heuristic_discrete_5":          "Heuristic (Discrete 5)",
    "heuristic_discrete_10":         "Heuristic (Discrete 10)",
    "attention_continuous":          "Attention (Continuous)",
    "attention_continous":           "Attention (Continuous)",
    "attention_discrete":            "Attention (Discrete)",
    "attention_discrete_3":          "Attention (Discrete 3)",
    "attention_discrete_5":          "Attention (Discrete 5)",
    "attention_discrete_10":         "Attention (Discrete 10)",
    "heuristic_attention_continous": "Heur.+Att. (Continuous)",
    "heuristic_attention_discrete":  "Heur.+Att. (Discrete)",
}

_FALLBACK_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

INTENTION_LABELS = {
    "asymetric_random":  "Asymmetric Random",
    "asymmetric_random": "Asymmetric Random",
    "all_straight":      "All Straight",
    "all_left":          "All Left",
    "uniform_random":    "Uniform Random",
}

SCENARIO_ORDER = [
    "Sc1_All_low",
    "Sc6_Mixed_ML",
    "Sc3_All_medium",
    "Sc5_Mixed_1H",
    "Sc4_Mixed_2H",
    "Sc2_All_high_3H",
]

SCENARIO_LABELS = {
    "Sc1_All_low":     "Sc1\nAll Low",
    "Sc2_All_high_3H": "Sc2\nAll High",
    "Sc3_All_medium":  "Sc3\nAll Medium",
    "Sc4_Mixed_2H":    "Sc4\nMixed 2H",
    "Sc5_Mixed_1H":    "Sc5\nMixed 1H",
    "Sc6_Mixed_ML":    "Sc6\nMixed ML",
}

BAR_WIDTH      = 0.25          # default width of a single bar
GROUP_SPACING  = 0.12          # extra gap between scenario groups
FIGSIZE        = (18, 10)
TITLE_FONTSIZE = 14
AXIS_FONTSIZE  = 11
TICK_FONTSIZE  = 9
CAPSIZE        = 4
ALPHA_BAR      = 0.88
ALPHA_ERR      = 1.0
GRID_ALPHA     = 0.25

# ── CSV filename pattern ──────────────────────────────────────────────────────
# Expected: {scen}_{intention}_{rate_key}_{version}.csv
_VERSIONS   = [
    "heuristic_attention_continous", "heuristic_attention_discrete",
    "heuristic_continuous", "heuristic_continous", "heuristic_discrete_10", "heuristic_discrete_5", "heuristic_discrete_3", "heuristic_discrete",
    "attention_continuous", "attention_continous", "attention_discrete_10", "attention_discrete_5", "attention_discrete_3", "attention_discrete",
]
_INTENTIONS = [
    "asymetric_random", "asymmetric_random",
    "all_straight", "all_left", "uniform_random",
]
_RATE_KEYS = [
    "Sc2_All_high_3H",
    "Sc1_All_low",
    "Sc3_All_medium",
    "Sc4_Mixed_2H",
    "Sc5_Mixed_1H",
    "Sc6_Mixed_ML",
]
_FNAME_RE = re.compile(
    r"^(?P<scen>[^_]+)_"
    r"(?P<intention>" + "|".join(re.escape(i) for i in _INTENTIONS) + r")_"
    r"(?P<rate_key>Sc.+?)_"   # non-greedy — stops before the version token
    r"(?P<version>" + "|".join(re.escape(v) for v in _VERSIONS) + r")\.csv$"
)

# ─────────────────────────────────────────────────────────────────────────────

def get_version_style(version: str, idx: int = 0) -> tuple[str, str]:
    """Return (color, label) for a given controller version."""
    color = VERSION_COLORS.get(version)
    if not color:
        color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    label = VERSION_LABELS.get(version, version.replace("_", " ").title())
    return color, label


def parse_eval_filename(fname: str) -> Optional[dict]:
    """
    Parse {scen}_{intention}_{rate_key}_{version}.csv robustly.
    Works for both known and custom intentions/versions.
    """
    if not fname.endswith(".csv"):
        return None
    name = fname[:-4]  # remove ".csv"

    # Try matching known _FNAME_RE first
    m = _FNAME_RE.match(fname)
    if m:
        return {
            "scen": m.group("scen"),
            "intention": m.group("intention"),
            "rate_key": m.group("rate_key"),
            "version": m.group("version"),
        }

    # Robust fallback using known rate_keys
    for rk in _RATE_KEYS:
        token = f"_{rk}_"
        if token in name:
            left, right = name.split(token, 1)
            parts = left.split("_", 1)
            if len(parts) == 2:
                return {
                    "scen": parts[0],
                    "intention": parts[1],
                    "rate_key": rk,
                    "version": right,
                }
    return None


def _parse_profile_str(val: str) -> np.ndarray:
    if not val:
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


def _load_csvs(
    output_dir: str,
    version_filter: Optional[str] = None,
    intention_filter: Optional[str] = None,
) -> tuple[dict, Optional[str], dict]:
    """
    Returns:
        (aggregated, used_intention, profiles_by_version)
    where:
        aggregated[rate_key][version] = {"collision_rate": float, "collision_se": float,
                                         "avg_travel_time": float, "travel_time_se": float,
                                         "avg_waiting_time": float, "waiting_time_se": float,
                                         "success_rate": float, "success_se": float,
                                         "avg_speed": float, "speed_se": float,
                                         "avg_safe_gap": float, "safe_gap_se": float,
                                         "n": int}
    """
    data: dict = defaultdict(lambda: defaultdict(list))
    profiles_by_version: dict = defaultdict(lambda: {
        "distances": [], "velocities": [], "jerks": [], "accelerations": [],
        "episodes": [],
    })
    parsed_files = []

    for fname in sorted(os.listdir(output_dir)):
        meta = parse_eval_filename(fname)
        if meta:
            parsed_files.append((fname, meta))

    if not parsed_files:
        return {}, None, {}

    # Determine default intention if none specified
    all_intentions = {meta["intention"] for _, meta in parsed_files}
    used_intention = intention_filter
    if not used_intention:
        if len(all_intentions) == 1:
            used_intention = all_intentions.pop()
        elif "asymetric_random" in all_intentions:
            used_intention = "asymetric_random"
        elif "asymmetric_random" in all_intentions:
            used_intention = "asymmetric_random"
        elif all_intentions:
            used_intention = sorted(all_intentions)[0]

    for fname, meta in parsed_files:
        if used_intention and meta["intention"] != used_intention:
            continue
        if version_filter and meta["version"] != version_filter:
            continue

        rate_key = meta["rate_key"]
        version = meta["version"]
        fpath = os.path.join(output_dir, fname)

        collisions, travel_times, waiting_times = [], [], []
        successes, avg_speeds, safe_gaps = [], [], []
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if "collision_rate" in row and row["collision_rate"] != "":
                        col = float(row["collision_rate"])
                    elif "total_vehicles_spawned" in row and int(row.get("total_vehicles_spawned", 0)) > 0:
                        tot_col = float(row.get("total_collisions", row.get("collision", 0)))
                        tot_veh = float(row["total_vehicles_spawned"])
                        col = tot_col / tot_veh
                    else:
                        col = float(row.get("collision", 0))

                    tt = float(row.get("all_vehicles_avg_travel_time", row["travel_time"]))
                    collisions.append(col)
                    travel_times.append(tt)
                    waiting_times.append(float(row.get("waiting_time", 0.0)))
                    successes.append(float(row.get("success", 0)))
                    avg_speeds.append(float(row.get("avg_speed", 0.0)))
                    # safe_gap is a newer column; older CSVs without it contribute NaN
                    # (those cells are simply skipped in the plots)
                    try:
                        safe_gaps.append(float(row.get("safe_gap", "")))
                    except (TypeError, ValueError):
                        safe_gaps.append(float("nan"))
                except (KeyError, ValueError):
                    continue

                t_arr = _parse_profile_str(row.get("time_profile", ""))
                d_arr = _parse_profile_str(row.get("distance_profile", ""))
                v_arr = _parse_profile_str(row.get("velocity_profile", ""))
                j_arr = _parse_profile_str(row.get("jerk_profile", ""))
                a_arr = _parse_profile_str(row.get("acceleration_profile", ""))
                ep_collision = int(row.get("collision", 0))

                if len(d_arr) > 0 and len(v_arr) == len(d_arr):
                    profiles_by_version[version]["distances"].extend(d_arr)
                    profiles_by_version[version]["velocities"].extend(v_arr)
                if len(j_arr) > 0:
                    profiles_by_version[version]["jerks"].extend(j_arr)
                if len(t_arr) > 0 and len(a_arr) == len(t_arr):
                    profiles_by_version[version]["accelerations"].extend(a_arr)

                # Per-episode data for sampled/worst-case plots
                if len(d_arr) > 1 and len(v_arr) == len(d_arr):
                    profiles_by_version[version]["episodes"].append({
                        "distances": d_arr,
                        "velocities": v_arr,
                        "jerks": j_arr if len(j_arr) == len(d_arr) else np.zeros_like(d_arr),
                        "scenario": rate_key,
                        "collision": ep_collision,
                    })

        if not collisions:
            continue

        data[rate_key][version].extend(
            [{"collision": c, "travel_time": t, "waiting_time": w,
              "success": s, "avg_speed": v, "safe_gap": g}
             for c, t, w, s, v, g in
             zip(collisions, travel_times, waiting_times, successes, avg_speeds, safe_gaps)]
        )

    def _se(vals: np.ndarray) -> float:
        """Standard error of the mean, NaN-aware."""
        vals = np.asarray(vals, dtype=float)
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            return float("nan")
        if len(valid) == 1:
            return 0.0
        return float(valid.std(ddof=1) / np.sqrt(len(valid)))

    # Aggregate: mean ± standard error
    aggregated: dict = defaultdict(dict)
    for rate_key, versions in data.items():
        for version, rows in versions.items():
            n          = len(rows)
            col_vals   = np.array([r["collision"]    for r in rows], dtype=float)
            tt_vals    = np.array([r["travel_time"]  for r in rows], dtype=float)
            wt_vals    = np.array([r["waiting_time"] for r in rows], dtype=float)
            suc_vals   = np.array([r["success"]      for r in rows], dtype=float)
            spd_vals   = np.array([r["avg_speed"]    for r in rows], dtype=float)
            sg_vals    = np.array([r["safe_gap"]     for r in rows], dtype=float)
            aggregated[rate_key][version] = {
                "collision_rate":    col_vals.mean() * 100,   # → percentage
                "collision_se":      _se(col_vals)    * 100,
                "avg_travel_time":   tt_vals.mean(),
                "travel_time_se":    _se(tt_vals),
                "avg_waiting_time":  wt_vals.mean(),
                "waiting_time_se":   _se(wt_vals),
                "success_rate":      suc_vals.mean() * 100,   # → percentage
                "success_se":        _se(suc_vals)    * 100,
                "avg_speed":         float(np.nanmean(spd_vals)) if np.isfinite(spd_vals).any() else float("nan"),
                "speed_se":          _se(spd_vals),
                "avg_safe_gap":      float(np.nanmean(sg_vals)) if np.isfinite(sg_vals).any() else float("nan"),
                "safe_gap_se":       _se(sg_vals),
                "n":                 n,
            }

    return aggregated, used_intention, profiles_by_version


def _make_grouped_bar(
    ax: plt.Axes,
    aggregated: dict,
    rate_keys: list[str],
    versions: list[str],
    value_key: str,
    error_key: str,
    ylabel: str,
    title: str,
) -> None:
    """Draw grouped bars on *ax* for the given metric."""
    n_groups   = len(rate_keys)
    n_versions = len(versions)
    bar_width  = min(BAR_WIDTH, 0.72 / max(1, n_versions))
    total_width = n_versions * bar_width + GROUP_SPACING
    group_centers = np.arange(n_groups) * total_width

    for i, ver in enumerate(versions):
        offset = (i - (n_versions - 1) / 2) * bar_width
        values = []
        errors = []
        for rate_key in rate_keys:
            entry = aggregated.get(rate_key, {}).get(ver)
            if entry is not None and np.isfinite(entry[value_key]):
                values.append(entry[value_key])
                errors.append(entry[error_key])
            else:
                # Missing / non-finite data (e.g. older CSVs without the metric)
                values.append(np.nan)
                errors.append(0.0)

        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        color, label = get_version_style(ver, i)

        valid = np.isfinite(values)
        if not valid.any():
            continue

        ax.bar(
            group_centers[valid] + offset,
            values[valid],
            width=bar_width,
            color=color,
            alpha=ALPHA_BAR,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            group_centers[valid] + offset,
            values[valid],
            yerr=errors[valid],
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
        [SCENARIO_LABELS.get(k, k.replace("_", "\n")) for k in rate_keys],
        fontsize=TICK_FONTSIZE,
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_profile_scatter(
    ax: plt.Axes,
    profiles_by_version: dict,
    versions: list[str],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    max_scatter_points: int = 2500,
) -> None:
    """Draw an academic-quality scatter plot of profile data with semi-transparent scatter and binned mean curve."""
    has_any_data = False
    for i, ver in enumerate(versions):
        x_data = np.asarray(profiles_by_version.get(ver, {}).get(x_key, []), dtype=float)
        y_data = np.asarray(profiles_by_version.get(ver, {}).get(y_key, []), dtype=float)

        n = min(len(x_data), len(y_data))
        if n == 0:
            continue
        x_data, y_data = x_data[:n], y_data[:n]

        valid = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid]
        y_data = y_data[valid]
        if len(x_data) == 0:
            continue

        has_any_data = True
        color, label = get_version_style(ver, i)

        if len(x_data) > max_scatter_points:
            indices = np.random.choice(len(x_data), size=max_scatter_points, replace=False)
            x_scatter = x_data[indices]
            y_scatter = y_data[indices]
        else:
            x_scatter = x_data
            y_scatter = y_data

        ax.scatter(
            x_scatter,
            y_scatter,
            color=color,
            alpha=0.35,
            s=12,
            edgecolors="none",
            label=label,
            zorder=2,
        )

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
                ax.plot(
                    valid_centers,
                    bin_means,
                    color=color,
                    linewidth=2.2,
                    linestyle="-",
                    zorder=4,
                )

    ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not has_any_data:
        ax.text(
            0.5,
            0.5,
            "No profile data available\n(Re-run evaluation script to record profiles)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=AXIS_FONTSIZE,
            color="gray",
            style="italic",
        )


def _plot_worst_case_on_ax(
    ax: plt.Axes,
    profiles_by_version: dict,
    versions: list[str],
) -> None:
    """
    Identify the worst-case episode per version and plot its velocity vs distance.
    Worst-case = collision episode with lowest avg speed, or lowest avg speed overall.
    """
    has_data = False
    for i, ver in enumerate(versions):
        episodes = profiles_by_version.get(ver, {}).get("episodes", [])
        if not episodes:
            continue

        # Worst-case: lowest avg speed among non-collision episodes only
        safe_eps = [ep for ep in episodes if ep.get("collision", 0) == 0]
        if not safe_eps:
            continue
        worst = min(safe_eps, key=lambda ep: np.mean(ep["velocities"]))

        color, label = get_version_style(ver, i)
        sc_label = SCENARIO_LABELS.get(worst["scenario"], worst["scenario"]).replace("\n", " ")

        ax.plot(worst["distances"], worst["velocities"], color=color, linewidth=1.8,
                alpha=0.85, label=f"{label} [{sc_label}]", zorder=3)
        has_data = True

    ax.set_xlabel("Distance (m)", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Velocity (m/s)", fontsize=AXIS_FONTSIZE)
    ax.set_title("Worst-Case Velocity Profile", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if has_data:
        ax.legend(loc="best", frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, "No episode data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=AXIS_FONTSIZE, color="gray", style="italic")




def plot_eval_results(
    output_dir: str = "output",
    version_filter: Optional[str] = None,
    intention_filter: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Parse all CSVs in *output_dir* and produce a comprehensive 3x3 dashboard
    with grouped bar charts (collision/success rate, travel/waiting time,
    average speed, average safe gap) and velocity/jerk profile scatter plots.

    Parameters
    ----------
    output_dir       : Directory containing the CSV files from v0_1_evaluate.py.
    version_filter   : If given, only CSVs whose version tag matches are loaded.
    intention_filter : If given, only CSVs whose intention tag matches are loaded.
    save_path        : If given, the figure is saved to this path (PNG/PDF/SVG).
    show             : Whether to call plt.show().
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    aggregated, used_intention, profiles_by_version = _load_csvs(
        output_dir, version_filter=version_filter, intention_filter=intention_filter
    )

    if not aggregated:
        raise ValueError(
            f"No matching CSV files found in '{output_dir}'"
            + (f" for intention='{intention_filter}'" if intention_filter else "")
            + (f" and version='{version_filter}'" if version_filter else "")
            + ".\nCheck that the eval script has been run and files follow the "
              "expected naming convention."
        )

    # Sort scenario keys by canonical traffic flow order
    def _scen_sort_key(k):
        try:
            return (0, SCENARIO_ORDER.index(k))
        except ValueError:
            return (1, k)

    rate_keys = sorted(aggregated.keys(), key=_scen_sort_key)

    # Collect all versions present in aggregated data
    versions_set = set()
    for rk_dict in aggregated.values():
        versions_set.update(rk_dict.keys())

    # Canonical order for known versions, custom versions sorted alphabetically at end
    known_order = list(VERSION_COLORS.keys())
    versions = sorted(
        versions_set,
        key=lambda v: (known_order.index(v) if v in known_order else len(known_order), v),
    )

    int_label = (
        INTENTION_LABELS.get(used_intention, used_intention.replace("_", " ").title())
        if used_intention
        else "All Intentions"
    )
    title_suffix = f" — {int_label}"
    if version_filter:
        title_suffix += f" ({version_filter})"

    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE)
    fig.suptitle(
        f"Evaluation Results & Profile Analysis{title_suffix}",
        fontsize=TITLE_FONTSIZE + 2,
        fontweight="bold",
        y=0.99,
    )

    ax_col, ax_tt, ax_wt  = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_sr, ax_spd, ax_sg  = axes[1, 0], axes[1, 1], axes[1, 2]
    ax_vd, ax_jd, ax_worst = axes[2, 0], axes[2, 1], axes[2, 2]

    _make_grouped_bar(
        ax=ax_col,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="collision_rate",
        error_key="collision_se",
        ylabel="Collision Rate (%)",
        title="Collision Rate by Scenario & Controller",
    )

    _make_grouped_bar(
        ax=ax_tt,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="avg_travel_time",
        error_key="travel_time_se",
        ylabel="Avg Travel Time (s)",
        title="Average Travel Time by Scenario & Controller",
    )

    _make_grouped_bar(
        ax=ax_wt,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="avg_waiting_time",
        error_key="waiting_time_se",
        ylabel="Avg Waiting Time (s)",
        title="Average Waiting Time by Scenario & Controller",
    )

    _make_grouped_bar(
        ax=ax_sr,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="success_rate",
        error_key="success_se",
        ylabel="Success Rate (%)",
        title="Success Rate by Scenario & Controller",
    )

    _make_grouped_bar(
        ax=ax_spd,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="avg_speed",
        error_key="speed_se",
        ylabel="Avg Speed (m/s)",
        title="Average Speed by Scenario & Controller",
    )

    _make_grouped_bar(
        ax=ax_sg,
        aggregated=aggregated,
        rate_keys=rate_keys,
        versions=versions,
        value_key="avg_safe_gap",
        error_key="safe_gap_se",
        ylabel="Avg Safe Gap (0\u20131)",
        title="Average Safe Gap by Scenario & Controller",
    )

    # Plot profile scatter charts on bottom row — distance-based only
    _make_profile_scatter(
        ax_vd,
        profiles_by_version,
        versions,
        "distances",
        "velocities",
        "Distance (m)",
        "Velocity (m/s)",
        "Velocity Profile vs. Distance",
    )
    _make_profile_scatter(
        ax_jd,
        profiles_by_version,
        versions,
        "distances",
        "jerks",
        "Distance (m)",
        "Jerk (m/s\u00b3)",
        "Jerk Profile vs. Distance",
    )

    # Worst-case velocity episode on the third bottom panel
    _plot_worst_case_on_ax(ax_worst, profiles_by_version, versions)

    # Shared legend below all charts
    handles = [
        mpatches.Patch(
            color=get_version_style(ver, i)[0],
            alpha=ALPHA_BAR,
            label=get_version_style(ver, i)[1],
        )
        for i, ver in enumerate(versions)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(max(1, len(versions)), 3),
        fontsize=AXIS_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved combined dashboard → {save_path}")

        # Also save a dedicated standalone figure for the profile scatter plots
        base, ext = os.path.splitext(save_path)
        scatter_save_path = f"{base}_profiles_scatter{ext}"
        fig_scat, ax_scat = plt.subplots(1, 3, figsize=(18, 5))
        fig_scat.suptitle(
            f"Velocity & Jerk Profile Analysis{title_suffix}",
            fontsize=TITLE_FONTSIZE + 2,
            fontweight="bold",
            y=1.03,
        )
        _make_profile_scatter(
            ax_scat[0], profiles_by_version, versions, "distances", "velocities",
            "Distance (m)", "Velocity (m/s)", "Velocity Profile vs. Distance"
        )
        _make_profile_scatter(
            ax_scat[1], profiles_by_version, versions, "distances", "jerks",
            "Distance (m)", "Jerk (m/s\u00b3)", "Jerk Profile vs. Distance"
        )
        _plot_worst_case_on_ax(ax_scat[2], profiles_by_version, versions)
        fig_scat.legend(
            handles=handles,
            loc="lower center",
            ncol=min(max(1, len(versions)), 3),
            fontsize=AXIS_FONTSIZE,
            frameon=False,
            bbox_to_anchor=(0.5, -0.1),
        )
        fig_scat.tight_layout()
        fig_scat.savefig(scatter_save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved standalone profile scatter plots → {scatter_save_path}")
        plt.close(fig_scat)

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
    ap.add_argument("--intention", default=None,
                    help="Filter to a single intention tag, e.g. asymetric_random")
    ap.add_argument("--save", default=None,
                    help="Save figure to this path, e.g. figures/results.png")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not call plt.show() (useful for headless servers)")
    args = ap.parse_args()

    plot_eval_results(
        output_dir=args.output_dir,
        version_filter=args.version,
        intention_filter=args.intention,
        save_path=args.save,
        show=not args.no_show,
    )

