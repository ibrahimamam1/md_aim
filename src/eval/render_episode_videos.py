"""
render_episode_videos.py
────────────────────────
Renders recorded evaluation episodes (src/eval/episode_recorder.py JSONs) into
videos so any collision / timeout can be inspected after the fact.

Pipeline
  1. Load episode JSON(s) recorded by evaluate_mo_sd.py (EpisodeRecorder).
  2. Render one PNG frame per recorded timestep with matplotlib: the
     intersection is drawn from the network's SUMO net.xml lane shapes, each
     vehicle from its recorded (x, y) position/heading.
  3. Assemble frames into an MP4 with ffmpeg (fallback: mp4 via imageio/FFwriter).

Interpretability (schema v2 recordings)
  - Vehicles inside the agent's observation are highlighted amber and labeled
    with their observation slot, ID, speed, acceleration, heading and edge;
    unobserved vehicles stay red with a compact label.
  - The ego vehicle is labeled with the exact features the agent sees:
    normalized / physical goal distance, speed and heading.
  - A side panel decodes the raw observation vector (4 ego features +
    5 neighbor slots × 5 features + attention mask) into human-readable
    values: distances to the conflict point in meters, speeds in m/s,
    headings in compass directions, and the Δη (arrival-time gap) per slot,
    plus a conflict summary and a legend.
  - v1 recordings (no observation snapshots) still render, but without the
    decoded panel; re-run evaluate_mo_sd.py to capture observations.

Selection
  - Default: only episodes that ended in COLLISION are rendered.
  - --timeouts: render ONLY TIMEOUT episodes (collision == 0 AND success == 0);
    use --all to render everything (collisions, timeouts, successes).
  - --all: render everything including successes.

Examples
    # Render videos for all recorded collisions (default):
    python -m src.eval.render_episode_videos --recordings output/eval_mo_sd/episode_recordings

    # Timeouts only for S3, 8 fps:
    python -m src.eval.render_episode_videos --recordings output/eval_mo_sd/episode_recordings \
        --scenarios S3 --timeouts --fps 8

    # Render from the evaluation output dir root (auto-discovers episode_recordings):
    python -m src.eval.render_episode_videos --eval-output output/eval_mo_sd
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyArrow, Circle
from matplotlib.collections import PatchCollection
import numpy as np

DEFAULT_RECORDINGS_SUBDIR = "episode_recordings"


# --------------------------------------------------------------------------- #
# Episode discovery
# --------------------------------------------------------------------------- #
def discover_episodes(recordings_root, scenarios=None):
    """Yield episode JSON paths grouped by scenario under recordings_root."""
    if not os.path.isdir(recordings_root):
        return []
    wanted = {s.upper() for s in scenarios} if scenarios else None
    episodes = []
    for entry in sorted(os.listdir(recordings_root)):
        scen_dir = os.path.join(recordings_root, entry)
        if not os.path.isdir(scen_dir):
            continue
        if wanted is not None and entry.upper() not in wanted:
            continue
        for fname in sorted(os.listdir(scen_dir)):
            if fname.endswith(".json") and fname != "manifest.json":
                episodes.append(os.path.join(scen_dir, fname))
    return episodes


def filter_outcomes(episodes, include_collisions=True, include_timeouts=False,
                    include_success=False):
    """Filter episode files by their recorded outcome."""
    selected = []
    for path in episodes:
        try:
            with open(path) as f:
                head = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            continue
        collision = int(head.get("collision", 0))
        success = int(head.get("success", 0))
        timeout = int(head.get("timeout", 1 if (collision == 0 and success == 0) else 0))
        if collision and include_collisions:
            selected.append(path)
        elif timeout and include_timeouts:
            selected.append(path)
        elif (not collision and not timeout) and include_success:
            selected.append(path)
    return selected


def resolve_recordings_root(args):
    """Resolve the recordings root from --recordings or --eval-output."""
    if args.recordings:
        root = args.recordings
    elif args.eval_output:
        root = os.path.join(args.eval_output, DEFAULT_RECORDINGS_SUBDIR)
        if not os.path.isdir(root):
            root = args.eval_output
    else:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output", "eval_mo_sd", DEFAULT_RECORDINGS_SUBDIR)
    return root


# --------------------------------------------------------------------------- #
# Network geometry (from SUMO net.xml)
# --------------------------------------------------------------------------- #
def load_network_geometry(net_file):
    """Extract lane shapes (polylines) from a SUMO net.xml.

    Returns dict with:
      lane_shapes   : list of np.ndarray (N, 2) — normal lanes
      internal_lanes: list of np.ndarray (N, 2) — junction-internal lanes
    """
    geom = {"lane_shapes": [], "internal_lanes": []}
    if not net_file or not os.path.exists(net_file):
        return geom
    try:
        root = ET.parse(net_file).getroot()
        for edge in root.findall("edge"):
            internal = edge.get("function") == "internal"
            for lane in edge.findall("lane"):
                shape = lane.get("shape")
                if not shape:
                    continue
                pts = []
                for pair in shape.split():
                    x, y = pair.split(",")[:2]
                    pts.append((float(x), float(y)))
                if internal:
                    geom["internal_lanes"].append(np.array(pts))
                else:
                    geom["lane_shapes"].append(np.array(pts))
    except Exception as e:
        print(f"  WARNING: could not parse net file {net_file}: {e}")
    return geom


def resolve_net_file(episode, recordings_root):
    """Find the net.xml belonging to an episode (repo default as fallback)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net_name = "100m_right_before_left.net.xml"
    scen = str(episode.get("scenario_id", "")).upper()
    if scen == "S7":
        net_name = "100m_allway_stop_fcfs_junction.net.xml"
    candidate = os.path.join(repo_root, "networks", net_name)
    if os.path.exists(candidate):
        return candidate
    # last resort: any net file in networks/
    try:
        for fname in sorted(os.listdir(os.path.join(repo_root, "networks"))):
            if fname.endswith(".net.xml"):
                return os.path.join(repo_root, "networks", fname)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Observation decoding (AlphaEnv_MO_SD observation space)
# --------------------------------------------------------------------------- #
# The observation the agent receives is a flat 34-dim vector:
#   [0:4]    ego   : [d_goal_norm, v_norm, sin(θ), cos(θ)]
#   [4:29]   5 neighbor slots, each 5 features:
#            [ego_dist_to_cp_norm, other_dist_to_cp_norm, other_v_norm,
#             other_sin(θ), other_cos(θ)]
#   [29:34]  attention mask: 1.0 = real neighbor in this slot, 0.0 = padding
# All distances are normalized by the perception radius (100 m), speeds by the
# network max speed, and the goal distance by the total route length.
EGO_FEATURES = 4
NEIGHBOR_FEATURES = 5
PAD_TOL = 1e-3

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def decode_observation(obs, norms, max_neighbours=5):
    """Decode a flat observation vector into a readable dict, or None.

    `norms` must contain max_speed, perception_radius and route_length (the
    EpisodeRecorder stores them in the episode's top-level "norms" block).
    """
    if obs is None:
        return None
    try:
        obs = [float(v) for v in np.asarray(obs).flatten()]
    except Exception:
        return None

    expected = EGO_FEATURES + NEIGHBOR_FEATURES * max_neighbours + max_neighbours
    if len(obs) < expected:
        return None

    max_speed = float(norms.get("max_speed") or 0.0)
    perception = float(norms.get("perception_radius") or 0.0)
    route_len = float(norms.get("route_length") or 0.0)

    d_goal_norm, v_norm, ego_sin, ego_cos = obs[0:4]
    ego = {
        "d_goal_norm": d_goal_norm,
        "d_goal_m": d_goal_norm * route_len if route_len > 0 else None,
        "v_norm": v_norm,
        "v_ms": v_norm * max_speed if max_speed > 0 else None,
        "sin": ego_sin,
        "cos": ego_cos,
    }
    if ego_sin != 0.0 or ego_cos != 0.0:
        ego["heading_deg"] = float(np.degrees(np.arctan2(ego_sin, ego_cos)) % 360.0)
    else:
        ego["heading_deg"] = None

    slots = []
    for i in range(max_neighbours):
        base = EGO_FEATURES + NEIGHBOR_FEATURES * i
        mask = obs[EGO_FEATURES + NEIGHBOR_FEATURES * max_neighbours + i]
        # Padding slots are exactly [1, 0, 1, 0, 0]
        is_padding = (
            mask < 0.5
            and abs(obs[base + 0] - 1.0) < PAD_TOL
            and abs(obs[base + 1] - 0.0) < PAD_TOL
            and abs(obs[base + 2] - 1.0) < PAD_TOL
        )
        v_norm_n = obs[base + 2]
        sin_n, cos_n = obs[base + 3], obs[base + 4]
        slots.append({
            "active": bool(mask > 0.5 and not is_padding),
            "padding": bool(is_padding),
            "ego_d_cp_norm": obs[base + 0],
            "other_d_cp_norm": obs[base + 1],
            "v_norm": v_norm_n,
            "v_ms": v_norm_n * max_speed if max_speed > 0 else None,
            "sin": sin_n,
            "cos": cos_n,
            "heading_deg": (float(np.degrees(np.arctan2(sin_n, cos_n)) % 360.0)
                            if (sin_n != 0.0 or cos_n != 0.0) else None),
            # distances to the conflict point in meters
            "ego_d_cp_m": obs[base + 0] * perception if perception > 0 else None,
            "other_d_cp_m": obs[base + 1] * perception if perception > 0 else None,
        })
    return {"ego": ego, "slots": slots}


def _d_eta_seconds(d_eta_norm):
    """Inverse of the env's tanh(Δη / 5) normalization, in seconds."""
    try:
        v = float(d_eta_norm)
    except Exception:
        return None
    if v <= -0.9999:
        return -float("inf")
    if v >= 0.9999:
        return float("inf")
    return 5.0 * float(np.arctanh(v))


def _compass(heading_deg):
    """Compass letter for a math-convention heading (0° = east, CCW)."""
    if heading_deg is None:
        return "—"
    sumo_deg = (90.0 - float(heading_deg)) % 360.0  # SUMO convention: 0 = north
    idx = int(((sumo_deg + 22.5) % 360.0) // 45.0)
    return COMPASS[idx]


def _slot_assignments(fr):
    """Map vehicle id -> observation slot index (0-based) for this frame."""
    slots = {}
    for i, n in enumerate(fr.get("neighbors_info") or []):
        if isinstance(n, dict) and n.get("veh_id") is not None:
            slots[str(n["veh_id"])] = i
    if not slots:  # v1 fallback: ids only, nearest-first = slot order
        for i, vid in enumerate(fr.get("neighbors") or []):
            if vid is not None:
                slots[str(vid)] = i
    return slots


def _classify_neighbor(fr, slot_idx):
    """Human-readable encounter relation for one observation slot.

    Uses the same signal the safety term uses: the normalized arrival-time gap
    Δη to the shared conflict point (|Δη| < 0.4 ≈ conflicting approach).
    """
    infos = fr.get("neighbors_info") or []
    info = infos[slot_idx] if slot_idx is not None and slot_idx < len(infos) else None
    d_eta = _d_eta_seconds(info.get("d_eta")) if isinstance(info, dict) else None
    if d_eta is None or d_eta in (float("inf"), float("-inf")):
        return d_eta, "no CP"      # routes do not share a conflict point
    if abs(d_eta) < 0.4:
        return d_eta, "CONFLICT"   # inside the safety-term window
    return d_eta, "lead" if d_eta > 0 else "yield"


# --------------------------------------------------------------------------- #
# Frame rendering
# --------------------------------------------------------------------------- #
VEHICLE_LENGTH = 5.0   # m, approx passenger car
VEHICLE_WIDTH = 2.0    # m

EGO_COLOR = "#1f4fd1"        # blue
NON_RL_COLOR = "#c23b3b"     # red
OBSERVED_COLOR = "#e6a817"   # amber for vehicles inside the observation
CONFLICT_COLOR = "#ff0000"   # red halo for neighbors in the conflict window
COLLISION_COLOR = "#000000"  # black flash at collision step


def _draw_vehicle(ax, x, y, heading, color, length=VEHICLE_LENGTH, width=VEHICLE_WIDTH,
                  zorder=5, alpha=1.0):
    """Draw a vehicle as a rotated rectangle centred on (x, y)."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    dx_l, dy_l = (length / 2.0) * cos_h, (length / 2.0) * sin_h
    dx_w, dy_w = (-width / 2.0) * sin_h, (width / 2.0) * cos_h
    corners = [
        (x + dx_l + dx_w, y + dy_l + dy_w),
        (x + dx_l - dx_w, y + dy_l - dy_w),
        (x - dx_l - dx_w, y - dy_l - dy_w),
        (x - dx_l + dx_w, y - dy_l + dy_w),
    ]
    ax.add_patch(MplPolygon(corners, closed=True, facecolor=color,
                            edgecolor="black", linewidth=0.6, alpha=alpha,
                            zorder=zorder))


def _fmt(v, spec=".1f", none="—"):
    if v is None:
        return none
    if v in (float("inf"), float("-inf")):
        return "∞" if v > 0 else "-∞"
    return format(v, spec)


def _vehicle_label(veh, slot_idx, relation):
    """Compact interpretable label for a non-ego vehicle."""
    parts = [f"#{veh['id']}"]
    if slot_idx is not None:
        parts.insert(0, f"slot{slot_idx}")
    parts.append(f"v={veh['speed']:.1f}")
    acc = veh.get("accel")
    if acc is not None and abs(acc) > 0.05:
        parts.append(f"a={acc:+.1f}")
    parts.append(f"H={_compass((90.0 - float(veh['heading'])) % 360.0)}")
    edge = veh.get("edge") or ""
    if edge:
        parts.append(edge.replace("E#", ""))
    label = " ".join(parts)
    if relation is not None:
        label += f"  {relation}"
    return label


def _ego_label(ego, decoded):
    """Interpretable label for the ego vehicle."""
    lines = [f"EGO  v={ego['speed']:.1f} m/s  a={ego.get('accel', 0.0):+.1f}  "
             f"H={ego['heading']:.0f}°"]
    if decoded is not None:
        e = decoded["ego"]
        d_goal = _fmt(e["d_goal_m"], ".0f") + "m" if e["d_goal_m"] is not None \
            else f"{e['d_goal_norm']:.2f} (norm)"
        v_disp = f"{e['v_ms']:.1f} m/s" if e["v_ms"] is not None else f"{e['v_norm']:.2f} (norm)"
        lines.append(f"obs: d_goal={d_goal}  v={v_disp}  "
                     f"θ={_compass(e['heading_deg'])} (sin={e['sin']:+.2f} cos={e['cos']:+.2f})")
    return lines


def _conflict_summary(fr, max_neighbours=5):
    """Derive a compact conflict summary from the recorded neighbor infos."""
    infos = fr.get("neighbors_info") or []
    if not infos:
        return None
    d_etas = []
    gaps = []
    for n in infos[:max_neighbours]:
        if not isinstance(n, dict):
            continue
        d_etas.append(_d_eta_seconds(n.get("d_eta")))
        try:
            gaps.append(float(n.get("distance")))
        except Exception:
            pass
    d_etas_f = [d for d in d_etas if d is not None]
    if not d_etas_f:
        return None
    finite = [d for d in d_etas_f if d not in (float("inf"), float("-inf"))]
    min_abs = min((abs(d) for d in finite), default=None)
    danger = sum(1 for d in d_etas_f if abs(d) < 0.4)
    min_gap = min(gaps) if gaps else None
    return {
        "min_abs_d_eta": min_abs,
        "danger_count": danger,
        "min_gap": min_gap,
        "n_observed": len(d_etas_f),
    }


def _draw_obs_panel(panel_ax, fr, decoded, action_txt, collision, terminated,
                    max_neighbours=5):
    """Side panel: decoded observation, neighbor table, conflict summary, legend."""
    panel_ax.clear()
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)
    panel_ax.axis("off")

    y = 0.985
    lh = 0.024  # line height (fraction of panel height)

    def line(txt, x=0.03, size=6, color="#222222", weight="normal", mono=True):
        nonlocal y
        panel_ax.text(x, y, txt, fontsize=size, color=color, fontweight=weight,
                      family="monospace" if mono else None, va="top", ha="left")
        y -= lh

    # -- Ego observation -----------------------------------------------------
    line("EGO OBSERVATION", weight="bold", color=EGO_COLOR, mono=False, size=7)
    if decoded is None:
        line("  (v1 recording — no observation snapshot;")
        line("   re-run evaluate_mo_sd.py to capture it)")
    else:
        e = decoded["ego"]
        d_goal = _fmt(e["d_goal_m"], ".1f") + " m" if e["d_goal_m"] is not None \
            else "n/a"
        v_disp = _fmt(e["v_ms"], ".1f") + " m/s" if e["v_ms"] is not None else "n/a"
        line(f"  d_goal  {e['d_goal_norm']:+.3f}  ({d_goal})")
        line(f"  v       {e['v_norm']:+.3f}  ({v_disp})")
        line(f"  sinθ    {e['sin']:+.3f}   cosθ {e['cos']:+.3f}  "
             f"(H={_compass(e['heading_deg'])})")

    # -- Action --------------------------------------------------------------
    line(f"action  {action_txt}   (normalized accel)", color=EGO_COLOR)
    y -= 0.012

    # -- Neighbor slots ------------------------------------------------------
    line("NEIGHBOR SLOTS (VEHICLES IN OBSERVATION)",
         weight="bold", color=OBSERVED_COLOR, mono=False, size=7)
    if decoded is None:
        n_obs = len(fr.get("neighbors") or [])
        line(f"  {n_obs} observed vehicle(s) — ids only (v1 recording)")
    else:
        infos = fr.get("neighbors_info") or []
        any_active = False
        for i, slot in enumerate(decoded["slots"]):
            info = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
            vid = info.get("veh_id", "?")
            if slot["padding"]:
                line(f"  slot{i}  — padding (masked out)", color="#999999")
                continue
            any_active = True
            d_eta, relation = _classify_neighbor(fr, i)
            relation_txt = relation if relation != "CONFLICT" else "CONFLICT!"
            color = CONFLICT_COLOR if relation == "CONFLICT" else "#222222"
            weight = "bold" if relation == "CONFLICT" else "normal"
            line(f"  slot{i}  {vid}", color=color, weight=weight)
            ego_d = _fmt(slot["ego_d_cp_m"], ".1f")
            oth_d = _fmt(slot["other_d_cp_m"], ".1f")
            v_disp = _fmt(slot["v_ms"], ".1f")
            line(f"    ego→CP {ego_d} m | other→CP {oth_d} m | "
                 f"v {v_disp} m/s | H {_compass(slot['heading_deg'])}",
                 color=color)
            if d_eta is None:
                line(f"    Δη n/a ({relation_txt})", color=color)
            else:
                line(f"    Δη {d_eta:+.2f} s  → {relation_txt}", color=color,
                     weight=weight)
        if not any_active:
            line("  (no neighbors in observation — all slots padded)")

    # -- Conflict summary ----------------------------------------------------
    summary = _conflict_summary(fr, max_neighbours=max_neighbours)
    if summary is not None:
        y -= 0.012
        line("CONFLICT SUMMARY", weight="bold", color=NON_RL_COLOR,
             mono=False, size=7)
        line(f"  observed neighbors : {summary['n_observed']}")
        line(f"  min |Δη|           : {_fmt(summary['min_abs_d_eta'], '.2f')} s "
             f"(danger window 0.4 s)")
        line(f"  in danger window   : {summary['danger_count']}")
        line(f"  min vehicle gap    : {_fmt(summary['min_gap'], '.1f')} m")
    y -= 0.012

    # -- Legend --------------------------------------------------------------
    line("LEGEND", weight="bold", color="#444444", mono=False, size=7)
    line("  ego vehicle (RL agent)", color=EGO_COLOR)
    line("  in observation (labeled, slot id)", color=OBSERVED_COLOR)
    line("  CONFLICT = |Δη| < 0.4 (safety term active)", color=CONFLICT_COLOR)
    line("  not observed (background traffic)", color=NON_RL_COLOR)
    line("  observed = within perception radius on conflicting route")
    line("  slot n = position in the observation vector")

    if collision and terminated:
        panel_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                                     linewidth=3, zorder=10))
        panel_ax.text(0.5, 0.5, "COLLISION", fontsize=18, color="black",
                      ha="center", va="center", alpha=0.25, rotation=12,
                      fontweight="bold", family="monospace", zorder=11)


def _even_figsize(w_in, h_in, dpi):
    """Adjust figure inches so width*height pixels are even (libx264/yuv420p
    rejects odd dimensions)."""
    w_px, h_px = int(round(w_in * dpi)), int(round(h_in * dpi))
    if w_px % 2:
        w_in = (w_px + 1) / dpi
    if h_px % 2:
        h_in = (h_px + 1) / dpi
    return w_in, h_in


def compute_frame_bounds(frames, geom, pad=25.0):
    """Fixed axis bounds across all frames so the video does not wobble."""
    xs, ys = [], []
    for fr in frames:
        for v in fr.get("vehicles", []):
            xs.append(v["x"])
            ys.append(v["y"])
        ego = fr.get("ego")
        if ego:
            xs.append(ego["x"])
            ys.append(ego["y"])
    for lane in geom["lane_shapes"] + geom["internal_lanes"]:
        xs.extend(lane[:, 0].tolist())
        ys.extend(lane[:, 1].tolist())
    if not xs:
        return -50, 50, -50, 50
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def render_episode(episode_path, out_root, fps, dpi, keep_frames, net_file=None,
                   show_obs_panel=True):
    """Render one recorded episode to PNGs and assemble them into a video.

    Returns (frames_dir, video_path or None).
    """
    with open(episode_path) as f:
        episode = json.load(f)

    frames = episode.get("frames", [])
    if not frames:
        print(f"  WARNING: no frames in {episode_path}; skipping")
        return None, None

    scen = str(episode.get("scenario_id", "unknown"))
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    collision = int(episode.get("collision", 0))
    success = int(episode.get("success", 0))
    timeout = int(episode.get("timeout", 0))
    outcome_tag = "collision" if collision else ("success" if success else "timeout")

    # Normalization constants for denormalizing the observation (schema v2+)
    norms = dict(episode.get("norms") or {})
    norms.setdefault("perception_radius",
                     float(episode.get("perception_radius", 100.0) or 100.0))
    norms.setdefault("max_neighbours", int(episode.get("max_neighbours", 5) or 5))
    max_neighbours = int(norms.get("max_neighbours", 5))

    frames_dir = os.path.join(out_root, scen, f"{stem}_frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    if net_file is None:
        net_file = resolve_net_file(episode, os.path.dirname(episode_path))
    geom = load_network_geometry(net_file)

    x0, x1, y0, y1 = compute_frame_bounds(frames, geom)

    neighbor_ids_per_frame = [set(fr.get("neighbors") or []) for fr in frames]
    collision_step = None
    for fr in frames:
        if fr.get("terminated") and collision:
            collision_step = fr.get("step")
            break

    if show_obs_panel:
        fig_w, fig_h = _even_figsize(11.5, 8.0, dpi)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.05], wspace=0.10)
        ax = fig.add_subplot(gs[0, 0])
        panel_ax = fig.add_subplot(gs[0, 1])
    else:
        fig_w, fig_h = _even_figsize(8.0, 8.0, dpi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        panel_ax = None

    for fr in frames:
        ax.clear()

        # Static network geometry
        if geom["lane_shapes"]:
            ax.add_collection(PatchCollection(
                [MplPolygon(lane, closed=False, fill=False) for lane in geom["lane_shapes"]],
                edgecolors="#9aa3ad", facecolors="none", linewidths=1.2, zorder=1))
        if geom["internal_lanes"]:
            ax.add_collection(PatchCollection(
                [MplPolygon(lane, closed=False, fill=False) for lane in geom["internal_lanes"]],
                edgecolors="#c9ced4", facecolors="none", linewidths=1.0,
                linestyles="dashed", zorder=1))

        step = fr.get("step", 0)
        neighbor_ids = neighbor_ids_per_frame[step] if step < len(neighbor_ids_per_frame) else set()
        slots_by_id = _slot_assignments(fr)

        # Decode this frame's observation once
        decoded = decode_observation(fr.get("obs"), norms,
                                     max_neighbours=max_neighbours)

        for veh in fr.get("vehicles", []):
            vid = str(veh["id"])
            in_obs = vid in neighbor_ids or vid in slots_by_id
            slot_idx = slots_by_id.get(vid)
            # Recorder stores SUMO headings (deg, 0 = north); drawing uses math
            # convention (rad, 0 = east).
            heading_math = np.radians(90.0 - veh["heading"])
            if in_obs:
                _, relation = _classify_neighbor(fr, slot_idx)
                if relation == "CONFLICT":
                    # red halo so conflicting neighbors pop out
                    ax.add_patch(Circle((veh["x"], veh["y"]), 4.5, fill=False,
                                        edgecolor=CONFLICT_COLOR, linewidth=1.4,
                                        alpha=0.9, zorder=4))
                _draw_vehicle(ax, veh["x"], veh["y"], heading_math, OBSERVED_COLOR,
                              zorder=5)
                ax.text(veh["x"], veh["y"] - 3.4, _vehicle_label(veh, slot_idx, relation),
                        fontsize=5.5, ha="center", va="top", color="#222222",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.75, linewidth=0.3))
            else:
                _draw_vehicle(ax, veh["x"], veh["y"], heading_math, NON_RL_COLOR,
                              zorder=5)
                ax.text(veh["x"], veh["y"] - 3.2, f"#{vid} v={veh['speed']:.1f}",
                        fontsize=5.0, ha="center", va="top", color="#777777",
                        zorder=6)

        ego = fr.get("ego")
        if ego:
            color = COLLISION_COLOR if (collision_step is not None and step >= collision_step) else EGO_COLOR
            _draw_vehicle(ax, ego["x"], ego["y"], np.radians(90.0 - ego["heading"]), color,
                          length=VEHICLE_LENGTH * 1.1, width=VEHICLE_WIDTH * 1.1, zorder=7)
            ax.add_patch(Circle((ego["x"], ego["y"]), 25.0, fill=False,
                                edgecolor=EGO_COLOR, linewidth=0.7, alpha=0.5, zorder=2))
            # Flip the ego label to the side away from nearby observed vehicles
            # so the two labels never overlap during close encounters.
            dy = 1.0
            for veh in fr.get("vehicles", []):
                if str(veh["id"]) in slots_by_id:
                    dy = 1.0 if veh["y"] <= ego["y"] else -1.0
                    break
            ego_lines = _ego_label(ego, decoded)
            for k, ln in enumerate(ego_lines):
                ax.text(ego["x"], ego["y"] + dy * (4.0 + 2.6 * k), ln,
                        fontsize=6.0 if k else 6.5, ha="center",
                        va="bottom" if dy > 0 else "top",
                        color=EGO_COLOR, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=EGO_COLOR, alpha=0.6, linewidth=0.4)
                        if k == 0 else None)

        action = fr.get("action")
        action_txt = "—" if action is None else f"{action:+.2f}"
        colliding = "YES" if (collision and fr.get("terminated")) else "no"
        n_obs = len(slots_by_id)
        ax.set_title(
            f"Scenario {scen} | {stem} | {outcome_tag.upper()}  "
            f"t={fr['t']:.2f}s step={step}  action={action_txt}  "
            f"reward={fr.get('reward', 0.0):+.3f}  collision={colliding}  "
            f"obs_vehicles={n_obs}",
            fontsize=8)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=6)

        if panel_ax is not None:
            _draw_obs_panel(panel_ax, fr, decoded, action_txt, collision,
                            bool(fr.get("terminated")),
                            max_neighbours=max_neighbours)

        fig.canvas.draw()
        frame_path = os.path.join(frames_dir, f"frame_{step:06d}.png")
        fig.savefig(frame_path, dpi=dpi, facecolor="white")

    plt.close(fig)

    video_path = os.path.join(out_root, scen, f"{stem}.mp4")
    ok = assemble_video(frames_dir, video_path, fps)
    if ok and not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return frames_dir, video_path
    return frames_dir, video_path if ok else None


# --------------------------------------------------------------------------- #
# ffmpeg assembly
# --------------------------------------------------------------------------- #
def find_ffmpeg():
    return shutil.which("ffmpeg")


def assemble_video(frames_dir, video_path, fps):
    """Assemble frame_%06d.png files into an MP4 using ffmpeg."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("  ERROR: ffmpeg not found on PATH; frames were kept but no video "
              "was created. Install ffmpeg (e.g. 'sudo apt install ffmpeg').")
        return False

    pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed for {video_path}:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  ERROR: could not run ffmpeg: {e}")
        return False
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_recordings = os.path.join(repo_root, "output", "eval_mo_sd", DEFAULT_RECORDINGS_SUBDIR)

    parser = argparse.ArgumentParser(
        description="Render recorded evaluation episodes into videos (ffmpeg).")
    parser.add_argument("--recordings", type=str, default=default_recordings,
                        help="Episode recordings root (contains <scenario>/run_*.json).")
    parser.add_argument("--eval-output", type=str, default=None,
                        help="Alternative to --recordings: evaluation output dir; "
                             f"uses its '{DEFAULT_RECORDINGS_SUBDIR}' subdirectory.")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Restrict to scenarios (e.g. S3 S5). Default: all found.")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Restrict to run indices (e.g. 4 11). Default: all matching outcome.")
    parser.add_argument("--timeouts", action="store_true", default=False,
                        help="Render ONLY TIMEOUT episodes (collision==0 and success==0) "
                             "instead of collisions. Use --all to render everything.")
    parser.add_argument("--all", dest="render_all", action="store_true", default=False,
                        help="Render every recorded episode including successes.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where videos are written (default: the recordings root).")
    parser.add_argument("--fps", type=int, default=4,
                        help="Video frames per second (recording is at sim_step=0.25s -> default 4 fps = real time).")
    parser.add_argument("--dpi", type=int, default=110, help="Frame resolution (matplotlib dpi).")
    parser.add_argument("--keep_frames", action="store_true", default=False,
                        help="Keep the intermediate PNG frames next to the video.")
    parser.add_argument("--net_file", type=str, default=None,
                        help="Explicit SUMO net.xml for geometry (default: auto per scenario).")
    parser.add_argument("--no-obs-panel", dest="no_obs_panel", action="store_true",
                        default=False,
                        help="Hide the observation side panel (plain map view only).")
    return parser.parse_args()


def main():
    args = parse_args()
    recordings_root = resolve_recordings_root(args)
    out_root = args.output_dir or recordings_root
    os.makedirs(out_root, exist_ok=True)

    episodes = discover_episodes(recordings_root, args.scenarios)
    if args.runs is not None:
        wanted = set(args.runs)

        def _run_of(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(stem.split("_")[1])
            except Exception:
                return -1
        episodes = [p for p in episodes if _run_of(p) in wanted]

    if args.render_all:
        selected = episodes
    elif args.timeouts:
        # --timeouts: timeouts only (no collisions)
        selected = filter_outcomes(episodes, include_collisions=False,
                                   include_timeouts=True,
                                   include_success=False)
    else:
        # default: collisions only
        selected = filter_outcomes(episodes, include_collisions=True,
                                   include_timeouts=False,
                                   include_success=False)

    print("\n" + "=" * 76)
    print(" Episode video rendering")
    print(f" recordings : {recordings_root}")
    print(f" output     : {out_root}")
    print(f" found      : {len(episodes)} episode(s); selected: {len(selected)}")
    print(f" mode       : {'all' if args.render_all else ('timeouts only' if args.timeouts else 'collisions only')}")
    print(f" obs panel  : {'off' if args.no_obs_panel else 'on'}")
    print("=" * 76 + "\n")

    if not selected:
        print("No episodes matched. Recorded outcomes can be listed with:")
        print(f"  ls {recordings_root}/*/")
        return

    made, failed = [], []
    for path in selected:
        print(f"Rendering {os.path.relpath(path, recordings_root)} ...")
        try:
            frames_dir, video = render_episode(path, out_root, args.fps, args.dpi,
                                               args.keep_frames, net_file=args.net_file,
                                               show_obs_panel=not args.no_obs_panel)
            if video:
                made.append(video)
                print(f"  -> {video}")
            else:
                failed.append(path)
        except Exception as e:
            failed.append(path)
            print(f"  ERROR: rendering failed: {e}")

    print(f"\nDone. {len(made)} video(s) created"
          + (f", {len(failed)} failed." if failed else "."))
    for v in made:
        print(f"  video: {v}")


if __name__ == "__main__":
    main()
