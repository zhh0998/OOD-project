#!/bin/bash
# RW2 Temporal Network Embedding - Full Experiment Pipeline
#
# This script executes the complete experimental pipeline:
# 1. Data validation
# 2. Train baseline model
# 3. Train Scheme 4 (DyGPrompt) - fastest
# 4. Train Scheme 3 (TPNet)
# 5. Train Scheme 0 (SSM-Memory-LLM) - most innovative
# 6. Evaluate all models
# 7. Generate report
#
# Usage:
#   ./run_experiments.sh                    # Run all experiments
#   ./run_experiments.sh --dataset tgbl-wiki  # Run on specific dataset
#   ./run_experiments.sh --model ssm_memory_llm  # Train specific model

set -e  # Exit on error

# Configuration
DATASETS=("tgbl-wiki" "tgbl-review" "tgbl-coin")
MODELS=("baseline" "dygprompt" "tpnet" "ssm_memory_llm")
GPU=0
NUM_RUNS=5
RESULTS_DIR="./results"
CHECKPOINTS_DIR="./checkpoints"
REPORTS_DIR="./reports"

# Parse arguments
SPECIFIC_DATASET=""
SPECIFIC_MODEL=""
SKIP_DATA_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --model|-m)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_DATA_VALIDATION=true
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update lists if specific dataset/model provided
if [ -n "$SPECIFIC_DATASET" ]; then
    DATASETS=("$SPECIFIC_DATASET")
fi

if [ -n "$SPECIFIC_MODEL" ]; then
    MODELS=("$SPECIFIC_MODEL")
fi

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$REPORTS_DIR"

echo "============================================================"
echo "RW2 Temporal Network Embedding - Experiment Pipeline"
echo "============================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "GPU: $GPU"
echo "============================================================"

# Step 1: Data Validation
if [ "$SKIP_DATA_VALIDATION" = false ]; then
    echo ""
    echo "[Step 1/7] Validating data..."
    echo "============================================================"
    python validate_data.py --all
fi

# Step 2-5: Train models
train_model() {
    local model=$1
    local dataset=$2

    echo ""
    echo "Training $model on $dataset..."

    python train.py \
        --model "$model" \
        --dataset "$dataset" \
        --gpu "$GPU" \
        --save_dir "$CHECKPOINTS_DIR" \
        --epochs 100 \
        --batch_size 200 \
        --patience 20

    # Copy results to results directory
    if [ -f "$CHECKPOINTS_DIR/${model}_${dataset}_results.json" ]; then
        cp "$CHECKPOINTS_DIR/${model}_${dataset}_results.json" "$RESULTS_DIR/"
    fi
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing dataset: $dataset"
    echo "============================================================"

    for model in "${MODELS[@]}"; do
        echo ""
        echo "[Training] $model on $dataset"
        echo "------------------------------------------------------------"

        # Check if already trained
        if [ -f "$CHECKPOINTS_DIR/${model}_${dataset}_best.pth" ]; then
            echo "Model already trained. Skipping..."
            continue
        fi

        train_model "$model" "$dataset"
    done
done

# Step 6: Evaluate all models
echo ""
echo "[Step 6/7] Evaluating models..."
echo "============================================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Evaluating on $dataset..."

    python evaluate.py \
        --compare "${MODELS[@]}" \
        --dataset "$dataset" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --output "$REPORTS_DIR" \
        --num_runs "$NUM_RUNS" \
        --gpu "$GPU"
done

# Step 7: Generate final report
echo ""
echo "[Step 7/7] Generating report..."
echo "============================================================"

python generate_report.py \
    --results_dir "$RESULTS_DIR" \
    --output "$REPORTS_DIR/RW2_Experiment_Report.md" \
    --datasets "${DATASETS[@]}" \
    --models "${MODELS[@]}" \
    --baseline "baseline"

echo ""
echo "============================================================"
echo "EXPERIMENT PIPELINE COMPLETE"
echo "============================================================"
echo "Results: $RESULTS_DIR"
echo "Checkpoints: $CHECKPOINTS_DIR"
echo "Reports: $REPORTS_DIR"
echo ""
echo "Main report: $REPORTS_DIR/RW2_Experiment_Report.md"
echo "============================================================"
