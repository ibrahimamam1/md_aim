#!/usr/bin/env bash
# ==============================================================================
# evaluate_baseline_checkpoints.sh
# ────────────────────────────────
# Evaluates a lineup of trained baseline checkpoints across scenarios S1..S5.
#
# Configuration:
#   - Checkpoints directory: checkpoints/mo_sd/baseline (override via CHECKPOINTS_DIR or $1)
#   - Simulations per scenario: 200 (override via N_SIMS)
#   - Evaluation seed: 42 (override via SEED)
#   - Scenarios: S1 S2 S3 S4 S5 (override via SCENARIOS)
#   - Dedicated output folder per checkpoint: output/eval_<checkpoint_name>/
#
# Usage:
#   bash scripts/evaluate_baseline_checkpoints.sh
#   N_SIMS=100 SEED=42 bash scripts/evaluate_baseline_checkpoints.sh
#   CHECKPOINTS_DIR="checkpoints/mo_sd/baseline" bash scripts/evaluate_baseline_checkpoints.sh
#   FORCE_RERUN=1 bash scripts/evaluate_baseline_checkpoints.sh
# ==============================================================================
set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

export PYTHONPATH="${PROJECT_ROOT}:/home/rgb/flow:${PYTHONPATH}"

# Pick Python environment (supporting conda flow env across different hosts)
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_CMD="${CONDA_PREFIX}/bin/python"
elif [ -f "/home/cse/miniconda3/envs/flow/bin/python" ]; then
    PYTHON_CMD="/home/cse/miniconda3/envs/flow/bin/python"
elif [ -f "/home/rgb/miniconda3/envs/flow/bin/python" ]; then
    PYTHON_CMD="/home/rgb/miniconda3/envs/flow/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Configuration overrides
if [ -n "$1" ]; then
    CHECKPOINTS_DIR="$1"
else
    CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-"checkpoints/mo_sd/baseline"}"
    # Fallback: if checkpoints/mo_sd/baseline doesn't exist, try checkpoints/mo_sd
    if [ ! -d "${CHECKPOINTS_DIR}" ] && [ -d "checkpoints/mo_sd" ]; then
        CHECKPOINTS_DIR="checkpoints/mo_sd"
    fi
fi

N_SIMS=${N_SIMS:-200}
SEED=${SEED:-42}
SCENARIOS=${SCENARIOS:-"S1 S2 S3 S4 S5"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"output"}
FORCE_RERUN=${FORCE_RERUN:-0}
DRY_RUN=${DRY_RUN:-0}

# Episode recording during eval is OFF by default to avoid huge disk I/O (enable with RECORD_EPISODES=1)
RECORD_FLAG=""
if [ "${RECORD_EPISODES:-0}" = "1" ]; then
    RECORD_FLAG="--record_episodes"
else
    RECORD_FLAG="--no_record_episodes"
fi

if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "ERROR: Checkpoints directory not found at: ${CHECKPOINTS_DIR}"
    echo "Please specify a valid directory via argument or CHECKPOINTS_DIR env variable."
    exit 1
fi

# Target baseline arms (default order matching gamma sweep)
DEFAULT_ARMS="baseline_g0_0.90 baseline_g0_0.925 baseline_g0_0.95 baseline_g0_0.975 baseline_g0_0.99 baseline_g0_0.999"

# Discover checkpoint directories
ARMS_TO_EVAL=()
if [ -n "${ARMS}" ]; then
    for arm in ${ARMS}; do
        ARMS_TO_EVAL+=("${arm}")
    done
else
    # Check if default arms exist in CHECKPOINTS_DIR
    found_default=0
    for arm in ${DEFAULT_ARMS}; do
        if [ -d "${CHECKPOINTS_DIR}/${arm}" ]; then
            ARMS_TO_EVAL+=("${arm}")
            found_default=1
        fi
    done

    # If none of default arms matched, discover all subdirectories or zip files
    if [ ${found_default} -eq 0 ]; then
        for entry in "${CHECKPOINTS_DIR}"/*; do
            if [ -d "${entry}" ]; then
                ARMS_TO_EVAL+=("$(basename "${entry}")")
            elif [[ "${entry}" == *.zip ]]; then
                arm_name="$(basename "${entry}" .zip)"
                ARMS_TO_EVAL+=("${arm_name}")
            fi
        done
    fi
fi

if [ ${#ARMS_TO_EVAL[@]} -eq 0 ]; then
    echo "ERROR: No checkpoint folders found in ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "========================================================================"
echo " MD-AIM Baseline Checkpoints Batch Evaluation"
echo " Python          : ${PYTHON_CMD}"
echo " Checkpoints Dir : ${CHECKPOINTS_DIR}"
echo " Arms to evaluate: ${#ARMS_TO_EVAL[@]} (${ARMS_TO_EVAL[*]})"
echo " Eval parameters : n_sims=${N_SIMS} | seed=${SEED}"
echo " Scenarios       : ${SCENARIOS}"
echo " Output Base     : ${OUTPUT_BASE_DIR}"
echo " Force Rerun     : ${FORCE_RERUN}"
echo "========================================================================"

TOTAL_ARMS=${#ARMS_TO_EVAL[@]}
ARM_INDEX=0

for arm in "${ARMS_TO_EVAL[@]}"; do
    ARM_INDEX=$((ARM_INDEX + 1))
    
    # Locate checkpoint file
    arm_path="${CHECKPOINTS_DIR}/${arm}"
    ckpt_file=""
    if [ -f "${arm_path}/final_model.zip" ]; then
        ckpt_file="${arm_path}/final_model.zip"
    elif [ -f "${arm_path}/final_model" ]; then
        ckpt_file="${arm_path}/final_model"
    elif [ -f "${arm_path}.zip" ]; then
        ckpt_file="${arm_path}.zip"
    elif [ -f "${arm_path}" ] && [[ "${arm_path}" == *.zip ]]; then
        ckpt_file="${arm_path}"
    elif [ -d "${arm_path}" ]; then
        # Pick latest .zip inside
        ckpt_file=$(ls -t "${arm_path}"/*.zip 2>/dev/null | head -n 1 || true)
    fi

    if [ -z "${ckpt_file}" ] || [ ! -f "${ckpt_file}" ]; then
        echo ""
        echo ">>> [Arm ${ARM_INDEX}/${TOTAL_ARMS}] Skipping ${arm}: no checkpoint .zip found in ${arm_path}"
        continue
    fi

    # Output folder for this checkpoint (e.g. output/eval_baseline_g0_0.90)
    if [[ "${arm}" == eval_* ]]; then
        EVAL_DIR="${OUTPUT_BASE_DIR}/${arm}"
    else
        EVAL_DIR="${OUTPUT_BASE_DIR}/eval_${arm}"
    fi

    SUMMARY_JSON="${EVAL_DIR}/evaluation_summary.json"

    echo ""
    echo "------------------------------------------------------------------------"
    echo " [Arm ${ARM_INDEX}/${TOTAL_ARMS}] Evaluating: ${arm}"
    echo "   Checkpoint : ${ckpt_file}"
    echo "   Output Dir : ${EVAL_DIR}"
    echo "   Seed       : ${SEED} | n_sims: ${N_SIMS}"
    echo "------------------------------------------------------------------------"

    # Skip if already evaluated unless FORCE_RERUN is enabled
    if [ "${FORCE_RERUN}" = "0" ] && [ -f "${SUMMARY_JSON}" ]; then
        echo "  [SKIP] ${SUMMARY_JSON} already exists. Set FORCE_RERUN=1 to re-evaluate."
        continue
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY RUN] Would run: ${PYTHON_CMD} src/eval/evaluate_mo_sd.py --checkpoint ${ckpt_file} --scenarios ${SCENARIOS} --n_sims ${N_SIMS} --seed ${SEED} --output_dir ${EVAL_DIR} ${RECORD_FLAG}"
        continue
    fi

    mkdir -p "${EVAL_DIR}"

    "$PYTHON_CMD" src/eval/evaluate_mo_sd.py \
        --checkpoint "${ckpt_file}" \
        --scenarios ${SCENARIOS} \
        --n_sims "${N_SIMS}" \
        --seed "${SEED}" \
        --output_dir "${EVAL_DIR}" \
        ${RECORD_FLAG}

    echo "  [DONE] Results saved to: ${EVAL_DIR}/"
done

echo ""
echo "========================================================================"
echo " All checkpoint evaluations complete!"
echo " Results per arm:"
for arm in "${ARMS_TO_EVAL[@]}"; do
    if [[ "${arm}" == eval_* ]]; then
        out="${OUTPUT_BASE_DIR}/${arm}"
    else
        out="${OUTPUT_BASE_DIR}/eval_${arm}"
    fi
    if [ -f "${out}/evaluation_summary.json" ]; then
        echo "   [✓] ${arm} -> ${out}/"
    else
        echo "   [✗] ${arm} -> missing or incomplete"
    fi
done
echo "========================================================================"

# Optionally update Pareto front plot if script exists
if [ "${DRY_RUN}" != "1" ] && [ -f "scripts/plot_pareto_front.py" ]; then
    echo ""
    echo ">>> Updating Pareto front plot via scripts/plot_pareto_front.py..."
    "$PYTHON_CMD" scripts/plot_pareto_front.py || true
fi
