#!/usr/bin/env bash
# ==============================================================================
# wandb_lr_sweep.sh
# Launch the W&B learning-rate sweep defined in scripts/sweep_lr.yaml.
#
#   bash scripts/wandb_lr_sweep.sh              # create sweep + 1 agent
#   AGENTS=4 bash scripts/wandb_lr_sweep.sh     # create sweep + 4 parallel agents
#
# Prerequisites (checked below):
#   1. wandb actually importable (a local ./wandb artifacts dir can shadow the
#      real package — this script fails fast with instructions instead of
#      silently running without logging).
#   2. WANDB_API_KEY exported (wandb login) — sweeps need the cloud service.
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

echo "========================================================================"
echo " MD-AIM W&B learning-rate sweep"
echo " Python    : ${PYTHON_CMD}"
echo " Sweep cfg : scripts/sweep_lr.yaml"
echo "========================================================================"

# --- Preflight 1: real wandb importable (not shadowed by ./wandb dir) --------
WANDB_STATUS=$("$PYTHON_CMD" - <<'EOF'
try:
    import wandb
except Exception:
    print("missing")
else:
    if not hasattr(wandb, "sweep"):
        print("shadowed")
    else:
        print("ok")
EOF
)
if [ "${WANDB_STATUS}" = "shadowed" ]; then
    echo ""
    echo "ERROR: 'import wandb' resolves to the local ./wandb artifacts directory,"
    echo "not the installed package (wandb.sweep is missing). Fix with:"
    echo "  pip uninstall -y wandb && pip install wandb"
    echo "(or move/rename the ./wandb directory while sweeping)"
    exit 1
elif [ "${WANDB_STATUS}" = "missing" ]; then
    echo ""
    echo "ERROR: wandb is not installed in this environment. Install it with:"
    echo "  pip install wandb"
    exit 1
fi

# --- Preflight 2: API key / login state --------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    if ! "$PYTHON_CMD" -c "import wandb; assert wandb.setup().settings.api_key" 2>/dev/null; then
        echo ""
        echo "ERROR: no WANDB_API_KEY found and no saved login."
        echo "Sweeps run on the W&B service, so an API key is required:"
        echo "  export WANDB_API_KEY=...   # https://wandb.ai/authorize"
        echo "  # or: wandb login"
        exit 1
    fi
fi
echo "[preflight] wandb import + credentials OK"

# --- Create the sweep ---------------------------------------------------------
SWEEP_ARGS=(sweep --project "${WANDB_PROJECT:-md_aim}")
if [ -n "${WANDB_ENTITY:-}" ]; then
    SWEEP_ARGS+=(--entity "${WANDB_ENTITY}")
fi
SWEEP_OUTPUT=$("$PYTHON_CMD" -m wandb "${SWEEP_ARGS[@]}" scripts/sweep_lr.yaml 2>&1) || {
    echo "${SWEEP_OUTPUT}"
    echo ""
    echo "ERROR: failed to create the sweep (see output above)."
    exit 1
}

# wandb prints e.g. "View sweep at: https://wandb.ai/<entity>/<project>/sweeps/<id>"
SWEEP_URL=$(echo "${SWEEP_OUTPUT}" | grep -oE "https://wandb\.ai/[^[:space:]]+/sweeps/[^[:space:]]+" | head -1)
# agent wants <entity>/<project>/<sweep_id>
SWEEP_PATH=$(echo "${SWEEP_URL}" | awk -F/ '{print $4"/"$5"/"$7}')
SWEEP_ID=$(echo "${SWEEP_URL}" | awk -F/ '{print $NF}')

if [ -z "${SWEEP_ID}" ]; then
    echo "${SWEEP_OUTPUT}"
    echo ""
    echo "ERROR: could not parse the sweep id / URL from wandb output above."
    exit 1
fi

echo ""
echo " Sweep created: ${SWEEP_URL}"
echo ""

# --- Run agents ---------------------------------------------------------------
AGENTS=${AGENTS:-1}
ARMS=${ARMS:-5}   # grid size in sweep_lr.yaml
echo " Starting ${AGENTS} agent(s), ${ARMS} arms in the grid"
echo " (Ctrl-C stops the local agents; the sweep survives on the service)."
for i in $(seq 1 "${AGENTS}"); do
    "$PYTHON_CMD" -m wandb agent --count "${ARMS}" "${SWEEP_PATH}" &
    AGENT_PID=$!
    echo "   agent ${i}: pid ${AGENT_PID}"
    LAST_AGENT_PID=${AGENT_PID}
done
wait

echo ""
echo "========================================================================"
echo " Sweep agents finished. Compare arms on the W&B dashboard:"
echo "   ${SWEEP_URL}"
echo "========================================================================"
