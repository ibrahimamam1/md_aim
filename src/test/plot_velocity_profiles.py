"""
plot_velocity_profiles.py
─────────────────────────
Reads CSVs produced by v0_1_evaluate_deterministic.py and generates
focused velocity-profile figures for comparing action-space variants.

Figure layout
─────────────
Row 1: Velocity vs. Distance  (ego route = north | ego route = east)
Row 2: Velocity vs. Time      (ego route = north | ego route = east)
Row 3: Acceleration vs. Time  (both routes combined)
       Jerk vs. Time          (both routes combined)

Each panel shows:
  - Thin semi-transparent lines for every individual episode
  - Thick solid line = mean across episodes for that version
  - Shaded band = ±1 standard deviation

Usage (standalone):
    python src/test/plot_velocity_profiles.py \\
        --output_dir output_deterministic \\
        --save figures/velocity_profiles.png

Or call plot_velocity_profiles() from your eval script.
"""

from __future__ import annotations

import os
import csv
import argparse
from collections import defaultdict
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe default; overridden if show=True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colour palette (one colour per version tag) ────────────────────────────────
_VERSION_COLORS = {
    "heuristic_continous":           "#2E86AB",   # steel blue
    "heuristic_continuous":          "#2E86AB",
    "heuristic_discrete":            "#A23B72",
    "heuristic_discrete_3":          "#C56E90",
    "heuristic_discrete_5":          "#A23B72",
    "heuristic_discrete_10":         "#6F1D4A",
    "attention_continous":           "#F18F01",   # warm amber
    "attention_continuous":          "#F18F01",
    "attention_discrete":            "#C73E1D",   # terracotta
    "attention_discrete_3":          "#E26D52",
    "attention_discrete_5":          "#C73E1D",
    "attention_discrete_10":         "#8C260E",
    "attention_continous_all_rl":    "#3BB273",   # green
    "heuristic_attention_continous": "#3B1F2B",
}
_VERSION_LABELS = {
    "heuristic_continous":           "Heuristic (Continuous)",
    "heuristic_continuous":          "Heuristic (Continuous)",
    "heuristic_discrete":            "Heuristic (Discrete)",
    "heuristic_discrete_3":          "Heuristic (Discrete-3)",
    "heuristic_discrete_5":          "Heuristic (Discrete-5)",
    "heuristic_discrete_10":         "Heuristic (Discrete-10)",
    "attention_continous":           "Attention (Continuous)",
    "attention_continuous":          "Attention (Continuous)",
    "attention_discrete":            "Attention (Discrete)",
    "attention_discrete_3":          "Attention (Discrete-3)",
    "attention_discrete_5":          "Attention (Discrete-5)",
    "attention_discrete_10":         "Attention (Discrete-10)",
    "attention_continous_all_rl":    "Attention All-RL",
    "heuristic_attention_continous": "Heur.+Att. (Continuous)",
}
_FALLBACK_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

FIGSIZE     = (16, 14)
ALPHA_THIN  = 0.18    # individual episode lines
ALPHA_BAND  = 0.20    # ±1-std shaded band
LW_THIN     = 0.8
LW_MEAN     = 2.2
GRID_ALPHA  = 0.25
TITLE_FS    = 13
AXIS_FS     = 11
TICK_FS     = 9


# ── Helpers ────────────────────────────────────────────────────────────────────

def _color(version: str, idx: int) -> str:
    return _VERSION_COLORS.get(version, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _label(version: str) -> str:
    return _VERSION_LABELS.get(version, version.replace("_", " ").title())


def _parse_profile(val: str) -> np.ndarray:
    if not val:
        return np.array([], dtype=float)
    val = str(val).strip()
    if not val:
        return np.array([], dtype=float)
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


def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample *arr* to *n* points for averaging across episodes."""
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.full(n, arr[0] if len(arr) == 1 else 0.0)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, arr)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data(output_dir: str) -> dict:
    """
    Returns
    -------
    data[version][route] = list of episode dicts with keys:
        distances, velocities, times, accelerations, jerks
    """
    data: dict = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(output_dir)):
        if not fname.startswith("deterministic_") or not fname.endswith(".csv"):
            continue
        fpath = os.path.join(output_dir, fname)

        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                version = row.get("version", "").strip()
                if not version:
                    # Try to infer from filename: deterministic_{version}.csv
                    stem = fname[len("deterministic_"):-len(".csv")]
                    version = stem

                route = row.get("ego_route", "unknown").strip()

                t_arr = _parse_profile(row.get("time_profile", ""))
                d_arr = _parse_profile(row.get("distance_profile", ""))
                v_arr = _parse_profile(row.get("velocity_profile", ""))
                a_arr = _parse_profile(row.get("acceleration_profile", ""))
                j_arr = _parse_profile(row.get("jerk_profile", ""))

                if len(v_arr) < 2:
                    continue

                episode = {
                    "times":         t_arr,
                    "distances":     d_arr,
                    "velocities":    v_arr,
                    "accelerations": a_arr,
                    "jerks":         j_arr,
                    "collision":     int(row.get("collision", 0)),
                }
                data[version][route].append(episode)

    return data


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _plot_profile_panel(
    ax: plt.Axes,
    data: dict,
    versions: List[str],
    route: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    n_resample: int = 200,
) -> None:
    """
    Plot individual episode traces + mean ± std band for each version.
    x_key and y_key are keys in the episode dict.
    """
    has_data = False

    for idx, version in enumerate(versions):
        episodes = data.get(version, {}).get(route, [])
        # Filter out colliding episodes for clean profile comparison
        safe_eps = [ep for ep in episodes if ep.get("collision", 0) == 0]
        if not safe_eps:
            continue

        color = _color(version, idx)
        label = _label(version)

        # Gather arrays
        x_arrays, y_arrays = [], []
        for ep in safe_eps:
            x_arr = ep.get(x_key, np.array([]))
            y_arr = ep.get(y_key, np.array([]))
            n = min(len(x_arr), len(y_arr))
            if n < 2:
                continue
            x_arr, y_arr = x_arr[:n], y_arr[:n]
            if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
                continue
            x_arrays.append(x_arr)
            y_arrays.append(y_arr)

        if not x_arrays:
            continue

        has_data = True

        # --- Draw individual episodes (thin, semi-transparent) ---
        for x_arr, y_arr in zip(x_arrays, y_arrays):
            ax.plot(x_arr, y_arr, color=color, alpha=ALPHA_THIN,
                    linewidth=LW_THIN, zorder=2)

        # --- Compute mean/std over a common x-axis (resampled) ---
        # Use the median episode length as the reference x-axis
        median_len = int(np.median([len(a) for a in x_arrays]))
        n_pts = max(n_resample, median_len)

        # Build a common x-axis spanning the union of all x-ranges
        x_min = np.min([a[0] for a in x_arrays])
        x_max = np.max([a[-1] for a in x_arrays])
        x_common = np.linspace(x_min, x_max, n_pts)

        y_resampled = np.array([
            np.interp(x_common, xa, ya, left=np.nan, right=np.nan)
            for xa, ya in zip(x_arrays, y_arrays)
        ])  # shape (n_episodes, n_pts)

        y_mean = np.nanmean(y_resampled, axis=0)
        y_std  = np.nanstd(y_resampled, axis=0)

        # Mask regions where fewer than 2 episodes contributed
        n_valid = np.sum(~np.isnan(y_resampled), axis=0)
        mask = n_valid >= 2
        x_masked = x_common[mask]
        y_mean_m = y_mean[mask]
        y_std_m  = y_std[mask]

        if len(x_masked) > 1:
            ax.fill_between(
                x_masked,
                y_mean_m - y_std_m,
                y_mean_m + y_std_m,
                color=color, alpha=ALPHA_BAND, zorder=3,
            )
            ax.plot(x_masked, y_mean_m, color=color, linewidth=LW_MEAN,
                    label=f"{label} (n={len(x_arrays)})", zorder=4)

    ax.set_xlabel(xlabel, fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if has_data:
        ax.legend(loc="best", fontsize=TICK_FS, frameon=False)
    else:
        ax.text(0.5, 0.5,
                f"No data for route='{route}'",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=AXIS_FS, color="gray", style="italic")


def _plot_combined_panel(
    ax: plt.Axes,
    data: dict,
    versions: List[str],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    n_resample: int = 200,
) -> None:
    """Like _plot_profile_panel but pools both routes together."""
    # Merge all routes into a single episode list per version
    merged_data: dict = {}
    for version in versions:
        all_eps = []
        for route_eps in data.get(version, {}).values():
            all_eps.extend(route_eps)
        merged_data[version] = {"all": all_eps}

    _plot_profile_panel(
        ax, merged_data, versions, "all",
        x_key, y_key, xlabel, ylabel, title, n_resample,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_velocity_profiles(
    output_dir: str = "output_deterministic",
    save_path: Optional[str] = None,
    show: bool = True,
    routes: Optional[list[str]] = None,
) -> None:
    """
    Generate the full velocity-profile dashboard.

    Parameters
    ----------
    output_dir : Directory containing deterministic_*.csv files.
    save_path  : If given, save the figure here (PNG/PDF/SVG).
    show       : Whether to call plt.show().
    routes     : Route names to plot (default: ['north', 'east']).
                 Pass a custom list if your env uses different names.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if routes is None:
        routes = ["north", "east"]

    data = _load_data(output_dir)
    if not data:
        raise ValueError(
            f"No 'deterministic_*.csv' files found in '{output_dir}'.\n"
            "Run v0_1_evaluate_deterministic.py first."
        )

    versions = sorted(data.keys())

    # ── Figure layout: 4 rows × 2 cols ────────────────────────────────
    # Row 0: velocity vs distance  (north | east)
    # Row 1: velocity vs time      (north | east)
    # Row 2: acceleration vs time  (all routes combined) | jerk vs time
    # Row 3: summary stats bar     (avg speed per route  | avg travel time)
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE)
    fig.suptitle(
        "Velocity Profile Comparison — Deterministic Scenario",
        fontsize=TITLE_FS + 2, fontweight="bold", y=0.99,
    )

    route_labels = {
        "north": "West → North",
        "east":  "West → East",
        "unknown": "Unknown Route",
    }

    # Row 0: velocity vs distance
    for col, route in enumerate(routes[:2]):
        rl = route_labels.get(route, route.title())
        _plot_profile_panel(
            axes[0, col], data, versions, route,
            x_key="distances", y_key="velocities",
            xlabel="Distance travelled (m)",
            ylabel="Velocity (m/s)",
            title=f"Velocity vs. Distance — {rl}",
        )

    # Row 1: velocity vs time
    for col, route in enumerate(routes[:2]):
        rl = route_labels.get(route, route.title())
        _plot_profile_panel(
            axes[1, col], data, versions, route,
            x_key="times", y_key="velocities",
            xlabel="Time since spawn (s)",
            ylabel="Velocity (m/s)",
            title=f"Velocity vs. Time — {rl}",
        )

    # Row 2: acceleration and jerk (both routes pooled)
    _plot_combined_panel(
        axes[2, 0], data, versions,
        x_key="times", y_key="accelerations",
        xlabel="Time since spawn (s)",
        ylabel="Acceleration (m/s²)",
        title="Acceleration vs. Time (all routes)",
    )
    _plot_combined_panel(
        axes[2, 1], data, versions,
        x_key="times", y_key="jerks",
        xlabel="Time since spawn (s)",
        ylabel="Jerk (m/s³)",
        title="Jerk vs. Time (all routes)",
    )

    # ── Shared legend ──────────────────────────────────────────────────
    handles = [
        mpatches.Patch(
            color=_color(ver, i),
            alpha=0.85,
            label=_label(ver),
        )
        for i, ver in enumerate(versions)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(versions), 4),
        fontsize=TICK_FS,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")

    if show:
        matplotlib.use("TkAgg")  # switch to interactive backend if showing
        plt.show()

    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot velocity profiles from the deterministic scenario.")
    ap.add_argument("--output_dir", default="output_deterministic",
                    help="Directory with deterministic_*.csv files (default: output_deterministic/)")
    ap.add_argument("--save", default=None,
                    help="Save figure to this path, e.g. figures/velocity_profiles.png")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not call plt.show() (useful for headless servers)")
    ap.add_argument("--routes", nargs="+", default=["north", "east"],
                    help="Route names to plot (default: north east)")
    cli_args = ap.parse_args()

    plot_velocity_profiles(
        output_dir=cli_args.output_dir,
        save_path=cli_args.save,
        show=not cli_args.no_show,
        routes=cli_args.routes,
    )
