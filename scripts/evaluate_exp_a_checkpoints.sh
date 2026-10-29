#!/usr/bin/env bash
# ==============================================================================
# evaluate_exp_a_checkpoints.sh
# ─────────────────────────────
# Evaluates trained Experiment A (State-Dependent Single Discounting) checkpoints
# across scenarios S1..S5.
#
# Configuration:
#   - Checkpoints directory: checkpoints/mo_sd (or checkpoints/mo_sd/exp_a)
#   - Simulations per scenario: 200 (override via N_SIMS)
#   - Evaluation seed: 42 (override via SEED)
#   - Scenarios: S1 S2 S3 S4 S5 (override via SCENARIOS)
#   - Dedicated output folder per checkpoint: output/eval_<arm_name>/
#
# Usage:
#   bash scripts/evaluate_exp_a_checkpoints.sh
#   DRY_RUN=1 bash scripts/evaluate_exp_a_checkpoints.sh
#   N_SIMS=100 SEED=42 bash scripts/evaluate_exp_a_checkpoints.sh
#   FORCE_RERUN=1 bash scripts/evaluate_exp_a_checkpoints.sh
#   ARMS="exp_a_g0_0.90_g_s_danger_0.0 exp_a_g0_0.95_g_s_danger_0.80" bash scripts/evaluate_exp_a_checkpoints.sh
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
    if [ -d "checkpoints/mo_sd/exp_a" ]; then
        CHECKPOINTS_DIR="checkpoints/mo_sd/exp_a"
    else
        CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-"checkpoints/mo_sd"}"
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

# Discover Experiment A checkpoint directories
ARMS_TO_EVAL=()
if [ -n "${ARMS}" ]; then
    for arm in ${ARMS}; do
        ARMS_TO_EVAL+=("${arm}")
    done
else
    # Auto-discover all exp_a_* directories or zip files in CHECKPOINTS_DIR
    for entry in "${CHECKPOINTS_DIR}"/exp_a_*; do
        if [ -d "${entry}" ]; then
            ARMS_TO_EVAL+=("$(basename "${entry}")")
        elif [[ "${entry}" == *.zip ]]; then
            ARMS_TO_EVAL+=("$(basename "${entry}" .zip)")
        fi
    done
fi

if [ ${#ARMS_TO_EVAL[@]} -eq 0 ]; then
    echo "ERROR: No 'exp_a_*' checkpoint folders found in ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "========================================================================"
echo " MD-AIM Experiment A Checkpoints Batch Evaluation"
echo " Python          : ${PYTHON_CMD}"
echo " Checkpoints Dir : ${CHECKPOINTS_DIR}"
echo " Arms to evaluate: ${#ARMS_TO_EVAL[@]}"
for arm in "${ARMS_TO_EVAL[@]}"; do
    echo "   - ${arm}"
done
echo " Eval parameters : n_sims=${N_SIMS} | seed=${SEED}"
echo " Scenarios       : ${SCENARIOS}"
echo " Output Base     : ${OUTPUT_BASE_DIR}"
echo " Force Rerun     : ${FORCE_RERUN}"
if [ "${DRY_RUN}" = "1" ]; then
    echo " Mode            : DRY RUN (no simulations executed)"
fi
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

    # Output folder for this checkpoint (e.g. output/eval_exp_a_g0_0.90_g_s_danger_0.0)
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
echo " Experiment A evaluation complete! Status per arm:"
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

# Display tabular summary across all evaluated arms
"$PYTHON_CMD" - "${OUTPUT_BASE_DIR}" "${SCENARIOS}" "${ARMS_TO_EVAL[*]}" << 'PYEOF'
import json
import os
import sys

out_base = sys.argv[1]
scenarios = sys.argv[2].split()
arms = sys.argv[3].split()

results = []
for arm in arms:
    dirname = arm if arm.startswith("eval_") else f"eval_{arm}"
    summary_path = os.path.join(out_base, dirname, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        continue
    try:
        with open(summary_path) as f:
            data = json.load(f)
        
        valid_scens = [s for s in scenarios if s in data]
        if not valid_scens:
            continue
        n = len(valid_scens)
        mean_coll = sum(data[s].get("collision_rate", 0.0) for s in valid_scens) / n
        mean_succ = sum(data[s].get("success_rate", 0.0) for s in valid_scens) / n
        mean_time = sum(data[s].get("traversal_time", 0.0) for s in valid_scens) / n
        mean_spd = sum(data[s].get("average_speed", 0.0) for s in valid_scens) / n
        mean_ttc = sum(data[s].get("min_ttc", 0.0) for s in valid_scens) / n
        mean_gap = sum(data[s].get("min_safe_gap", 0.0) for s in valid_scens) / n

        results.append({
            "arm": arm,
            "collision": mean_coll,
            "success": mean_succ,
            "time": mean_time,
            "speed": mean_spd,
            "ttc": mean_ttc,
            "gap": mean_gap,
        })
    except Exception:
        pass

if results:
    print("\n" + "=" * 94)
    print(" EXPERIMENT A AGGREGATE SUMMARY (Macro-averaged over " + ", ".join(scenarios) + ")")
    print("=" * 94)
    header = f"{'Arm Name':<38} | {'Coll Rate':<10} | {'Success':<8} | {'Time (s)':<9} | {'Speed':<8} | {'Min TTC':<8}"
    print(header)
    print("-" * 94)
    for r in sorted(results, key=lambda x: (x["collision"], x["time"])):
        print(f"{r['arm']:<38} | {r['collision']:>9.2%} | {r['success']:>7.2%} | {r['time']:>9.2f} | {r['speed']:>7.2f} | {r['ttc']:>7.2f}")
    print("=" * 94 + "\n")
PYEOF

# Optionally update Pareto front plot if script exists
if [ "${DRY_RUN}" != "1" ] && [ -f "scripts/plot_pareto_front.py" ]; then
    echo ">>> Updating Pareto front plot via scripts/plot_pareto_front.py..."
    "$PYTHON_CMD" scripts/plot_pareto_front.py || true
fi
