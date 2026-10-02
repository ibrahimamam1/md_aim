#!/usr/bin/env bash
# ==============================================================================
# run_danger_horizon_sweep.sh
# ───────────────────────────
# Danger Horizon Floor (gamma_s_danger / gamma_min) sweep for state-dependent
# discounting agents (primarily Experiment B: Learnable Single Discounting,
# or Experiment A: Rule-Based Single Discounting).
#
# Sweeps gamma_s_danger across:
#   {0.0, 0.50, 0.70, 0.80, 0.85}
#
# Planning Horizon Physics (at sim_step = 0.25 s):
#   - gamma_min = 0.00 -> H_eff = 1.0 step  = 0.25 s (extreme myopic, blind to crashes > 0.25 s away)
#   - gamma_min = 0.50 -> H_eff = 2.0 steps = 0.50 s
#   - gamma_min = 0.70 -> H_eff = 3.3 steps = 0.83 s
#   - gamma_min = 0.80 -> H_eff = 5.0 steps = 1.25 s (calibrated braking reaction horizon)
#   - gamma_min = 0.85 -> H_eff = 6.7 steps = 1.67 s (early decelerative margin)
#
# For each value of gamma_s_danger:
#   1. Trains MOSDPPO in the specified mode (default: exp_b).
#   2. Evaluates the resulting checkpoint across scenarios (default: S1..S4).
#   3. Generates evaluation_summary.json and logs per arm.
#
# Usage:
#   bash scripts/run_danger_horizon_sweep.sh
#   MODE=exp_b DANGER_GAMMAS="0.70 0.80 0.85" bash scripts/run_danger_horizon_sweep.sh
#   TIMESTEPS=400000 N_SIMS=100 bash scripts/run_danger_horizon_sweep.sh
# ==============================================================================
set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

export PYTHONPATH="${PROJECT_ROOT}:/home/rgb/flow:${PYTHONPATH}"

if [ -f "/home/rgb/miniconda3/envs/flow/bin/python" ]; then
    PYTHON_CMD="/home/rgb/miniconda3/envs/flow/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Experiment Mode: default to exp_b (Learnable Single Discounting)
MODE=${MODE:-"exp_b"}

# Sweep configuration (override via environment):
GAMMA_0=${GAMMA_0:-"0.99"}
DANGER_GAMMAS=${DANGER_GAMMAS:-"0.0 0.50 0.70 0.80 0.85"}
TIMESTEPS=${TIMESTEPS:-800000}
N_SIMS=${N_SIMS:-200}
SCENARIOS=${SCENARIOS:-"S1 S2 S3 S4"}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-24}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
GAP_PENALTY_WEIGHT=${GAP_PENALTY_WEIGHT:-0.25}

RECORD_FLAG=""
if [ "${RECORD_EPISODES:-0}" = "1" ]; then
    RECORD_FLAG="--record_episodes"
else
    RECORD_FLAG="--no_record_episodes"
fi

echo "========================================================================"
echo " MD-AIM Danger Horizon Floor Sweep"
echo " Python        : ${PYTHON_CMD}"
echo " Mode          : ${MODE}"
echo " Base gamma_0  : ${GAMMA_0}"
echo " Danger floors : ${DANGER_GAMMAS}"
echo " Timesteps     : ${TIMESTEPS} per arm | eval n_sims: ${N_SIMS} | seed: ${SEED}"
echo " Scenarios     : ${SCENARIOS}"
echo "========================================================================"

for gsd in ${DANGER_GAMMAS}; do
    TAG="${MODE}_danger_gsd_${gsd}_g0_${GAMMA_0}"
    CKPT_DIR="checkpoints/mo_sd/${TAG}"
    EVAL_DIR="output/eval_${TAG}"

    echo ""
    echo "------------------------------------------------------------------------"
    echo " Arm: Mode=${MODE} | gamma_0=${GAMMA_0} | gamma_s_danger=${gsd}"
    echo "   checkpoints : ${CKPT_DIR}"
    echo "   eval output : ${EVAL_DIR}"
    echo "------------------------------------------------------------------------"

    # --- 1. Train -------------------------------------------------------------
    "$PYTHON_CMD" src/configs/train_mo_sd.py \
        --mode "${MODE}" \
        --gamma_0 "${GAMMA_0}" \
        --gamma_s_danger "${gsd}" \
        --timesteps "${TIMESTEPS}" \
        --num_workers "${NUM_WORKERS}" \
        --learning_rate "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --gap_penalty_weight "${GAP_PENALTY_WEIGHT}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --note "danger_horizon_sweep mode=${MODE} g0=${GAMMA_0} gsd=${gsd} seed=${SEED} steps=${TIMESTEPS}"

    if [ ! -f "${CKPT_DIR}/final_model.zip" ]; then
        echo "ERROR: training finished but ${CKPT_DIR}/final_model.zip is missing."
        exit 1
    fi

    # --- 2. Evaluate the checkpoint -------------------------------------------
    "$PYTHON_CMD" src/eval/evaluate_mo_sd.py \
        --checkpoint "${CKPT_DIR}/final_model.zip" \
        --scenarios ${SCENARIOS} \
        --n_sims "${N_SIMS}" \
        --output_dir "${EVAL_DIR}" \
        ${RECORD_FLAG}
done

echo ""
echo "========================================================================"
echo " Danger Horizon Floor Sweep Complete!"
echo " Results per arm:"
for gsd in ${DANGER_GAMMAS}; do
    TAG="${MODE}_danger_gsd_${gsd}_g0_${GAMMA_0}"
    echo "   gamma_s_danger=${gsd} -> output/eval_${TAG}/"
done
echo "========================================================================"

# --- 3. Print Aggregate Summary Table -----------------------------------------
"$PYTHON_CMD" - << EOF
import os, json

scenarios = "${SCENARIOS}".split()
danger_gammas = "${DANGER_GAMMAS}".split()
mode = "${MODE}"
gamma_0 = "${GAMMA_0}"

print("\n" + "=" * 92)
print(f" DANGER HORIZON FLOOR SWEEP SUMMARY [{mode.upper()} | gamma_0={gamma_0}]")
print("=" * 92)
header = f"{'gamma_s_danger':<16} | {'Scenario':<8} | {'Col Rate':<10} | {'Success':<10} | {'Avg Spd (m/s)':<14} | {'Min Safe Gap (s)':<16} | {'Min TTC (s)':<12}"
print(header)
print("-" * 92)

for gsd in danger_gammas:
    tag = f"{mode}_danger_gsd_{gsd}_g0_{gamma_0}"
    summary_path = os.path.join("output", f"eval_{tag}", "evaluation_summary.json")
    if not os.path.exists(summary_path):
        print(f"{gsd:<16} | {'MISSING':<8} |")
        continue

    with open(summary_path, "r") as f:
        data = json.load(f)

    for scen in scenarios:
        s = data.get(scen, {})
        col = f"{s.get('collision_rate', 0.0):.2%}"
        suc = f"{s.get('success_rate', 0.0):.2%}"
        spd = f"{s.get('average_speed', 0.0):.2f}"
        gap = f"{s.get('min_safe_gap', 0.0):.2f}"
        ttc = f"{s.get('min_ttc', 0.0):.2f}"
        print(f"{gsd:<16} | {scen:<8} | {col:<10} | {suc:<10} | {spd:<14} | {gap:<16} | {ttc:<12}")
    print("-" * 92)
print("=" * 92 + "\n")
EOF

