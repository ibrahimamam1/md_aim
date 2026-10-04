#!/usr/bin/env bash
# ==============================================================================
# run_exp_a_gamma_grid_sweep.sh
# ───────────────────────────
# 2D gamma grid sweep for Experiment A (Rule-Based State-Dependent Single
# Discounting):  gamma(s) = gamma_s_danger while in conflict, else gamma_0.
#
# Unlike Experiment B (where gamma_s_danger only bounds a learnable net), both
# parameters here act DIRECTLY on the discount, so both are swept
# simultaneously over a grid:
#
#   gamma_0        (normal-state horizon)  x  gamma_s_danger (danger-state horizon)
#
# Default grid (3 x 3 = 9 arms), anchored to prior results:
#   gamma_0        ∈ {0.90, 0.95, 0.99}   <- safe / mid / knee of the baseline sweep
#   gamma_s_danger ∈ {0.0,  0.50, 0.80}    <- myopic / medium / danger-sweep winner
#
# For each arm:
#   1. Trains MOSDPPO (mode=exp_a) with (gamma_0, gamma_s_danger), fixed seed.
#   2. Evaluates the checkpoint across scenarios (default: S1..S4).
#   3. Writes output/eval_exp_a_g0_<g0>_g_s_danger_<gsd>/evaluation_summary.json
#
# Usage:
#   bash scripts/run_exp_a_gamma_grid_sweep.sh
#   GAMMA_0S="0.95 0.99" DANGER_GAMMAS="0.0 0.80" bash scripts/run_exp_a_gamma_grid_sweep.sh
#   TIMESTEPS=400000 N_SIMS=100 bash scripts/run_exp_a_gamma_grid_sweep.sh
#   GAMMA_0S="0.925" DANGER_GAMMAS="0.0" bash scripts/run_exp_a_gamma_grid_sweep.sh  # single arm
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
GAMMA_0S=${GAMMA_0S:-"0.90 0.95 0.99"}
DANGER_GAMMAS=${DANGER_GAMMAS:-"0.0 0.50 0.80"}
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

N_ARMS=$(( $(echo ${GAMMA_0S} | wc -w) * $(echo ${DANGER_GAMMAS} | wc -w) ))

echo "========================================================================"
echo " MD-AIM Experiment A: 2D gamma grid sweep (gamma_0 x gamma_s_danger)"
echo " Python        : ${PYTHON_CMD}"
echo " gamma_0 grid  : ${GAMMA_0S}"
echo " g_s_danger    : ${DANGER_GAMMAS}"
echo " Arms          : ${N_ARMS}"
echo " Timesteps     : ${TIMESTEPS} per arm | eval n_sims: ${N_SIMS} | seed: ${SEED}"
echo " Scenarios     : ${SCENARIOS}"
echo "========================================================================"

ARM_IDX=0
for g0 in ${GAMMA_0S}; do
    for gsd in ${DANGER_GAMMAS}; do
        ARM_IDX=$((ARM_IDX + 1))
        TAG="exp_a_g0_${g0}_g_s_danger_${gsd}"
        CKPT_DIR="checkpoints/mo_sd/${TAG}"
        EVAL_DIR="output/eval_${TAG}"

        echo ""
        echo "------------------------------------------------------------------------"
        echo " Arm ${ARM_IDX}/${N_ARMS}: gamma_0=${g0} | gamma_s_danger=${gsd}"
        echo "   checkpoints : ${CKPT_DIR}"
        echo "   eval output : ${EVAL_DIR}"
        echo "------------------------------------------------------------------------"

        # --- 1. Train ---------------------------------------------------------
        "$PYTHON_CMD" src/configs/train_mo_sd.py \
            --mode exp_a \
            --gamma_0 "${g0}" \
            --gamma_s_danger "${gsd}" \
            --timesteps "${TIMESTEPS}" \
            --num_workers "${NUM_WORKERS}" \
            --learning_rate "${LEARNING_RATE}" \
            --seed "${SEED}" \
            --gap_penalty_weight "${GAP_PENALTY_WEIGHT}" \
            --checkpoint_dir "${CKPT_DIR}" \
            --note "exp_a_grid_sweep g0=${g0} gsd=${gsd} seed=${SEED} steps=${TIMESTEPS}"

        if [ ! -f "${CKPT_DIR}/final_model.zip" ]; then
            echo "ERROR: training finished but ${CKPT_DIR}/final_model.zip is missing."
            exit 1
        fi

        # --- 2. Evaluate the checkpoint ---------------------------------------
        "$PYTHON_CMD" src/eval/evaluate_mo_sd.py \
            --checkpoint "${CKPT_DIR}/final_model.zip" \
            --scenarios ${SCENARIOS} \
            --n_sims "${N_SIMS}" \
            --output_dir "${EVAL_DIR}" \
            ${RECORD_FLAG}
    done
done

echo ""
echo "========================================================================"
echo " Experiment A grid sweep complete. Results per arm:"
for g0 in ${GAMMA_0S}; do
    for gsd in ${DANGER_GAMMAS}; do
        echo "   gamma_0=${g0} gamma_s_danger=${gsd} -> output/eval_exp_a_g0_${g0}_g_s_danger_${gsd}/"
    done
done
echo "========================================================================"

# --- 3. Print Aggregate Summary Grids (macro-averaged over scenarios) ---------
"$PYTHON_CMD" - "${GAMMA_0S}" "${DANGER_GAMMAS}" "${SCENARIOS}" << 'PYEOF'
import json
import os
import sys

gamma_0s = sys.argv[1].split()
danger_gammas = sys.argv[2].split()
scenarios = sys.argv[3].split()
W = 12


def load(g0, gsd):
    path = os.path.join("output", f"eval_exp_a_g0_{g0}_g_s_danger_{gsd}",
                        "evaluation_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def macro(data, key):
    return sum(data[s].get(key, 0.0) for s in scenarios) / len(scenarios)


def print_grid(title, key, fmt):
    print(f"\n{title} (avg over {', '.join(scenarios)}):")
    print("gamma_0 \\ gsd |" + "".join(f"{g:>{W}}" for g in danger_gammas))
    print("-" * (14 + W * len(danger_gammas)))
    for g0 in gamma_0s:
        cells = []
        for gsd in danger_gammas:
            data = load(g0, gsd)
            cells.append("MISSING".center(W) if data is None else fmt.format(macro(data, key)).center(W))
        print(f"{g0:<14}|" + "".join(cells))


print("\n" + "=" * (14 + W * len(danger_gammas)))
print(" EXPERIMENT A GRID SWEEP SUMMARY [mode=exp_a]")
print("=" * (14 + W * len(danger_gammas)))
print_grid("Collision rate", "collision_rate", "{:.2%}")
print_grid("Traversal time (s)", "traversal_time", "{:.2f}")
print_grid("Success rate", "success_rate", "{:.2%}")
print_grid("Waiting time (s)", "waiting_time", "{:.2f}")
print("=" * (14 + W * len(danger_gammas)) + "\n")
PYEOF
