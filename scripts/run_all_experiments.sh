#!/usr/bin/env bash
# ==============================================================================
# run_all_experiments.sh
# Multi-Objective, State-Dependent Discounting for Autonomous Intersection Management
#
# Automates:
#   1. Framework verification unit tests
#   2. Baseline fixed discount sweep (γ_0 ∈ {0.90, 0.95, 0.97, 0.99, 0.995})
#   3. Experiment A: State-Dependent Single Discount γ(s)
#   4. Experiment B: Multi-Objective State-Dependent Discount [γ_l, γ_s(s)] (Core)
#   5. Experiment C: Learnable Discount Factors γ_φ(s)
#   6. Ablation: State-Dependent Reward Weighting λ(s)
#   7. Full evaluation across scenarios S1..S4 and Pareto weight sweep
#   8. Generation of publication-grade Pareto frontiers and Oracle diagnostics
# ==============================================================================

set -e

# Run from project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

export PYTHONPATH="${PROJECT_ROOT}:/home/rgb/flow:${PYTHONPATH}"

# Pick Python environment
if [ -f "/home/rgb/miniconda3/envs/flow/bin/python" ]; then
    PYTHON_CMD="/home/rgb/miniconda3/envs/flow/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "========================================================================"
echo " MD-AIM: Multi-Objective, State-Dependent Discounting Framework"
echo " Python: ${PYTHON_CMD}"
echo " Root:   ${PROJECT_ROOT}"
echo "========================================================================"

# Step 1: Run verification test suite
echo ""
echo ">>> Step 1: Running unit tests on models, buffer, GAE, and scenarios..."
"$PYTHON_CMD" src/test/test_framework.py

# Step 2: Generate initial Oracle and Pareto diagnostic visualizations
echo ""
echo ">>> Step 2: Generating diagnostic visualizations..."
"$PYTHON_CMD" src/eval/plot_pareto_and_metrics.py --test_dummy
"$PYTHON_CMD" src/eval/plot_oracle_interpretability.py --test_dummy

# Helper function to train and evaluate an experiment condition
run_pipeline() {
    local mode=$1
    local extra_args=$2
    local tag=$3
    local timesteps=${4:-500000}

    echo "------------------------------------------------------------------------"
    echo " Running Pipeline: Mode=${mode}, Tag=${tag}, Timesteps=${timesteps}"
    echo "------------------------------------------------------------------------"

    "$PYTHON_CMD" src/configs/train_mo_sd.py \
        --mode "${mode}" \
        --timesteps "${timesteps}" \
        ${extra_args} \
        --note "${tag}"

    # Locate the most recently saved checkpoint for this run
    LATEST_DIR=$(ls -td checkpoints/mo_sd/"${mode}"_* 2>/dev/null | head -n 1)
    if [ -n "${LATEST_DIR}" ] && [ -f "${LATEST_DIR}/final_model.zip" ]; then
        echo "Evaluating checkpoint: ${LATEST_DIR}/final_model.zip across scenarios S1..S4..."
        "$PYTHON_CMD" src/eval/evaluate_mo_sd.py \
            --checkpoint "${LATEST_DIR}/final_model.zip" \
            --scenarios S1 S2 S3 S4 \
            --n_sims 15 \
            --output_dir "output/eval_${mode}_${tag}"
    fi
}

# Check if user requested training run
if [ "$1" == "--train-all" ]; then
    echo ""
    echo ">>> Step 3: Starting Full Experimental Campaign..."

    # 1. Baseline sweep
    for g0 in 0.90 0.95 0.97 0.99 0.995; do
        run_pipeline "baseline" "--gamma_0 ${g0}" "baseline_g0_${g0}" 500000
    done

    # 2. Experiment A: State-Dependent Single Discount
    run_pipeline "exp_a" "--gamma_0 0.99 --gamma_s_danger 0.0" "exp_a_adaptive_single" 500000

    # 3. Experiment B: Multi-Objective State-Dependent Discount (Pareto Sweep)
    for wl in 0.1 0.3 0.5 0.7 0.9; do
        ws=$(echo "1.0 - $wl" | bc -l 2>/dev/null || python -c "print(1.0 - $wl)")
        run_pipeline "exp_b" "--gamma_l 0.99 --gamma_s_normal 0.95 --gamma_s_danger 0.0 --weight_l ${wl} --weight_s ${ws}" "exp_b_mo_sd_wl_${wl}" 500000
    done

    # 4. Experiment C: Learnable Discount Factors
    run_pipeline "exp_c" "--gamma_l 0.99 --gamma_s_normal 0.95" "exp_c_learned_discount" 500000

    # 5. Ablation: State-Dependent Reward Weighting
    run_pipeline "ablation" "--gamma_0 0.99 --lambda_danger 5.0 --lambda_normal 1.0" "ablation_reward_weighting" 500000

    # Final summary plotting
    echo ""
    echo ">>> Step 4: Generating final Pareto and Scenario summary charts..."
    "$PYTHON_CMD" src/eval/plot_pareto_and_metrics.py
    "$PYTHON_CMD" src/eval/plot_oracle_interpretability.py
fi

echo ""
echo "========================================================================"
echo " Framework setup and diagnostics completed successfully."
echo " To launch the complete training campaign, run:"
echo "   bash scripts/run_all_experiments.sh --train-all"
echo "========================================================================"

