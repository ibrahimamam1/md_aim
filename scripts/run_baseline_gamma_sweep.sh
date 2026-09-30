#!/usr/bin/env bash
# ==============================================================================
# run_baseline_gamma_sweep.sh
# Baseline-only discount-factor sweep with per-checkpoint evaluation.
#
# For each gamma in {0.90, 0.925, 0.95, 0.975, 0.99, 0.999}:
#   1. Train the baseline (single critic, fixed discount, mode=baseline).
#   2. Evaluate the final checkpoint on S1..S4 with n_sims sims per scenario,
#      writing results into a DEDICATED folder per gamma:
#        output/eval_baseline_g0_<gamma>/
#
# Usage:
#   bash scripts/run_baseline_gamma_sweep.sh
#   TIMESTEPS=200000 N_SIMS=100 bash scripts/run_baseline_gamma_sweep.sh
#   GAMMAS="0.95 0.99" bash scripts/run_baseline_gamma_sweep.sh   # subset / rerun
#
# Notes:
#   - Fixed seed across arms so only the discount factor varies.
#   - Episode recording during eval is OFF by default (200 sims x 4 scenarios
#     per arm produces a lot of JSON); enable with RECORD_EPISODES=1.
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

# Sweep configuration (override via environment):
GAMMAS=${GAMMAS:-"0.90 0.925 0.95 0.975 0.99 0.999"}
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
echo " MD-AIM baseline gamma sweep"
echo " Python    : ${PYTHON_CMD}"
echo " Gammas    : ${GAMMAS}"
echo " Timesteps : ${TIMESTEPS} per arm | eval n_sims: ${N_SIMS} | seed: ${SEED}"
echo " Scenarios : ${SCENARIOS}"
echo "========================================================================"

for gamma in ${GAMMAS}; do
    TAG="baseline_g0_${gamma}"
    CKPT_DIR="checkpoints/mo_sd/${TAG}"
    EVAL_DIR="output/eval_${TAG}"

    echo ""
    echo "------------------------------------------------------------------------"
    echo " Arm: gamma_0=${gamma}"
    echo "   checkpoints : ${CKPT_DIR}"
    echo "   eval output : ${EVAL_DIR}"
    echo "------------------------------------------------------------------------"

    # --- 1. Train -------------------------------------------------------------
    "$PYTHON_CMD" src/configs/train_mo_sd.py \
        --mode baseline \
        --gamma_0 "${gamma}" \
        --timesteps "${TIMESTEPS}" \
        --num_workers "${NUM_WORKERS}" \
        --learning_rate "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --gap_penalty_weight "${GAP_PENALTY_WEIGHT}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --note "baseline_gamma_sweep g0=${gamma} seed=${SEED} steps=${TIMESTEPS}"

    if [ ! -f "${CKPT_DIR}/final_model.zip" ]; then
        echo "ERROR: training finished but ${CKPT_DIR}/final_model.zip is missing."
        exit 1
    fi

    # --- 2. Evaluate the checkpoint (dedicated folder per gamma) ---------------
    "$PYTHON_CMD" src/eval/evaluate_mo_sd.py \
        --checkpoint "${CKPT_DIR}/final_model.zip" \
        --scenarios ${SCENARIOS} \
        --n_sims "${N_SIMS}" \
        --output_dir "${EVAL_DIR}" \
        ${RECORD_FLAG}
done

echo ""
echo "========================================================================"
echo " Baseline gamma sweep complete. Results per arm:"
for gamma in ${GAMMAS}; do
    echo "   gamma_0=${gamma}  ->  output/eval_baseline_g0_${gamma}/"
done
echo "========================================================================"
