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

Selection
  - Default: only episodes that ended in COLLISION are rendered.
  - --timeouts: also render TIMEOUT episodes (collision == 0 AND success == 0);
    when set, collisions AND timeouts are both rendered.
  - --all: render everything including successes.

Examples
    # Render videos for all recorded collisions (default):
    python -m src.eval.render_episode_videos --recordings output/eval_mo_sd/episode_recordings

    # Collisions + timeouts for S3 only, 8 fps:
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
# Frame rendering
# --------------------------------------------------------------------------- #
VEHICLE_LENGTH = 5.0   # m, approx passenger car
VEHICLE_WIDTH = 2.0    # m

EGO_COLOR = "#1f4fd1"        # blue
NON_RL_COLOR = "#c23b3b"     # red
OBSERVED_COLOR = "#e6a817"   # amber for vehicles inside the observation
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


def render_episode(episode_path, out_root, fps, dpi, keep_frames, net_file=None):
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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

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

        neighbor_ids = neighbor_ids_per_frame[fr["step"]] if fr["step"] < len(neighbor_ids_per_frame) else set()

        for veh in fr.get("vehicles", []):
            color = OBSERVED_COLOR if veh["id"] in neighbor_ids else NON_RL_COLOR
            # Recorder stores SUMO headings (deg, 0 = north); drawing uses math
            # convention (rad, 0 = east).
            heading_math = np.radians(90.0 - veh["heading"])
            _draw_vehicle(ax, veh["x"], veh["y"], heading_math, color)
            speed_lbl = f"{veh['speed']:.1f}"
            ax.text(veh["x"], veh["y"] - 3.2, speed_lbl, fontsize=5.5,
                    ha="center", va="top", color="#333333", zorder=6)

        ego = fr.get("ego")
        if ego:
            color = COLLISION_COLOR if (collision_step is not None and fr["step"] >= collision_step) else EGO_COLOR
            _draw_vehicle(ax, ego["x"], ego["y"], np.radians(90.0 - ego["heading"]), color,
                          length=VEHICLE_LENGTH * 1.1, width=VEHICLE_WIDTH * 1.1, zorder=7)
            ax.add_patch(Circle((ego["x"], ego["y"]), 25.0, fill=False,
                                edgecolor=EGO_COLOR, linewidth=0.7, alpha=0.5, zorder=2))
            ax.text(ego["x"], ego["y"] + 4.0, f"ego  v={ego['speed']:.1f} m/s",
                    fontsize=6.5, ha="center", color=EGO_COLOR, zorder=6)

        action = fr.get("action")
        action_txt = "—" if action is None else f"{action:+.2f}"
        colliding = "YES" if (collision and fr.get("terminated")) else "no"
        ax.set_title(
            f"Scenario {scen} | {stem} | {outcome_tag.upper()}  "
            f"t={fr['t']:.2f}s step={fr['step']}  action={action_txt}  "
            f"reward={fr.get('reward', 0.0):+.3f}  collision={colliding}",
            fontsize=8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=6)

        fig.canvas.draw()
        frame_path = os.path.join(frames_dir, f"frame_{fr['step']:06d}.png")
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
                        help="Also render TIMEOUT episodes (collision==0 and success==0). "
                             "Collisions are always rendered when selected.")
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
    else:
        selected = filter_outcomes(episodes, include_collisions=True,
                                   include_timeouts=args.timeouts,
                                   include_success=False)

    print("\n" + "=" * 76)
    print(" Episode video rendering")
    print(f" recordings : {recordings_root}")
    print(f" output     : {out_root}")
    print(f" found      : {len(episodes)} episode(s); selected: {len(selected)}")
    print(f" mode       : {'all' if args.render_all else ('collisions + timeouts' if args.timeouts else 'collisions only')}")
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
                                               args.keep_frames, net_file=args.net_file)
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
