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

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <version1> [version2 ...]"
    echo "Example: $0 heuristic_discrete_3 heuristic_discrete_5"
    exit 1
fi

# Iterate through each version passed (handles multiple arguments or space-separated lists)
for arg in "$@"; do
    for VERSION in $arg; do
        echo "========================================================================"
        echo " [1/2] Starting v0_1 training for version: ${VERSION}"
        echo "========================================================================"

        "$PYTHON_CMD" src/configs/v0_1_single_agent.py --train --version "${VERSION}"

        # Find the most recently created checkpoint directory for this version
        LATEST_DIR=$(ls -td checkpoints/v0_1/"${VERSION}"_* 2>/dev/null | head -n 1)
        if [ -z "${LATEST_DIR}" ]; then
            echo "ERROR: No checkpoint directory found in checkpoints/v0_1/ matching prefix '${VERSION}_*'"
            exit 1
        fi

        CHECKPOINT_PATH="${LATEST_DIR}/final_model"
        if [ ! -f "${CHECKPOINT_PATH}.zip" ] && [ ! -f "${CHECKPOINT_PATH}" ]; then
            echo "ERROR: Final model checkpoint not found at ${CHECKPOINT_PATH}.zip"
            exit 1
        fi

        echo "========================================================================"
        echo " [2/2] Running evaluation for version: ${VERSION}"
        echo "       Checkpoint: ${CHECKPOINT_PATH}"
        echo "       n_sims: 400"
        echo "========================================================================"

        "$PYTHON_CMD" src/test/v0_1_evaluate.py --checkpoint "${CHECKPOINT_PATH}" --version "${VERSION}" --n_sims 400

        echo "========================================================================"
        echo " Finished training and evaluation for version: ${VERSION}"
        echo "========================================================================"
        echo ""
    done
done
