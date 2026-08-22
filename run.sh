#!/usr/bin/env bash
set -e

# Ensure we run from the project root directory
cd "$(dirname "$0")"

# Set up PYTHONPATH so flow and local src imports work
export PYTHONPATH="${PWD}:/home/rgb/flow:${PYTHONPATH}"

# Pick the appropriate Python executable (preferring the conda 'flow' env python if available)
if [ -f "/home/rgb/miniconda3/envs/flow/bin/python" ]; then
    PYTHON_CMD="/home/rgb/miniconda3/envs/flow/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

function run_experiment() {
    local gamma_short=$1
    local gamma_long=$2
    local version="attention_continuous"
    local eval_version="attention_continous"

    echo "========================================================================"
    echo " Starting training: gamma_short=${gamma_short}, gamma_long=${gamma_long}, version=${version}"
    echo "========================================================================"

    "$PYTHON_CMD" src/configs/v0_1_single_agent.py --train --version "${version}" \
        --gamma "${gamma_short}" --gamma_long "${gamma_long}" \
        --note "gamma_short=${gamma_short} gamma_long=${gamma_long}"

    # Find the most recently created checkpoint directory for this version
    LATEST_DIR=$(ls -td checkpoints/v0_1/"${version}"_* 2>/dev/null | head -n 1)
    if [ -z "${LATEST_DIR}" ]; then
        echo "ERROR: No checkpoint directory found in checkpoints/v0_1/ matching prefix '${version}_*'"
        exit 1
    fi

    CHECKPOINT_PATH="${LATEST_DIR}/final_model"
    if [ ! -f "${CHECKPOINT_PATH}.zip" ] && [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "ERROR: Final model checkpoint not found at ${CHECKPOINT_PATH}.zip"
        exit 1
    fi

    echo "========================================================================"
    echo " Running evaluation: gamma_short=${gamma_short}, gamma_long=${gamma_long}, version=${eval_version}"
    echo " Checkpoint: ${CHECKPOINT_PATH}"
    echo "========================================================================"

    "$PYTHON_CMD" src/test/v0_1_evaluate.py --checkpoint "${CHECKPOINT_PATH}" --version "${eval_version}" --n_sims 50 --wandb

    echo " Finished gamma_short=${gamma_short} gamma_long=${gamma_long}"
    echo ""
}

# run_experiment <gamma_short> <gamma_long>
run_experiment 0.90  0.999
run_experiment 0.95  0.999
run_experiment 0.99  0.999
run_experiment 0.999 0.999
