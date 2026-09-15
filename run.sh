#!/usr/bin/env bash
# ==============================================================================
# run.sh
# Entry point delegating to scripts/run_all_experiments.sh
# ==============================================================================

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

exec bash scripts/run_all_experiments.sh "$@"
