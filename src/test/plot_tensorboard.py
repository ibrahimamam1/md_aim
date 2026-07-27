"""
Plot cumulative reward for all four PPO variants from TensorBoard event files.
Looks for events in:
  <base_dir>/{att_cont,att_dis,heu_cont,heu_dis}/PPO_1/
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = "/home/rgb/Desktop/research/alpha_model/tensorboard_logs/final"
VARIANTS = ["att_cont", "att_dis", "heu_cont", "heu_dis"]
REWARD_TAG = "rollout/ep_rew_mean"          # standard SB3 tag
COLORS     = ["#e05252", "#4a90d9", "#5cb85c", "#f0a500"]
LABELS     = {
    "att_cont": "Attention – Continuous",
    "att_dis":  "Attention – Discrete",
    "heu_cont": "Heuristic – Continuous",
    "heu_dis":  "Heuristic – Discrete",
}
SMOOTH_WINDOW = 10          # set to 1 to disable smoothing
# ─────────────────────────────────────────────────────────────────────────────


def read_tb_reward(event_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for *tag* found in *event_dir*."""
    steps, values = [], []
    pattern = os.path.join(event_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No event files found in {event_dir}")

    for path in sorted(files):
        for record in tf_record.tf_record_iterator(path):
            event = event_pb2.Event.FromString(record)
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.tag == tag:
                    steps.append(event.step)
                    values.append(value.simple_value)

    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'valid' mode shrinks the array; use 'same' to keep length
    return np.convolve(values, kernel, mode="same")


def cumulative_reward(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values)


# ── Main ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for variant, color in zip(VARIANTS, COLORS):
    event_dir = os.path.join(BASE_DIR, variant, "PPO_1")
    try:
        steps, values = read_tb_reward(event_dir, REWARD_TAG)
    except FileNotFoundError as e:
        print(f"[WARN] {e} – skipping {variant}")
        continue

    cum_reward = cumulative_reward(values)
    cum_smoothed = smooth(cum_reward, SMOOTH_WINDOW)

    ax.plot(steps, cum_smoothed, label=LABELS[variant], color=color, linewidth=2)
    # light shaded raw curve behind the smooth one
    if SMOOTH_WINDOW > 1:
        ax.plot(steps, cum_reward, color=color, linewidth=0.6, alpha=0.25)

ax.set_xlabel("Environment Steps", fontsize=12)
ax.set_ylabel("Cumulative Reward", fontsize=12)
ax.set_title("Cumulative Reward – PPO Variants", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "cumulative_reward.png")
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
