#!/bin/bash
#
# Run all hypothesis verification experiments
# RW1 Preliminary Experiments
#

set -e

echo "=============================================="
echo "RW1 Preliminary Experiments - Full Pipeline"
echo "=============================================="
echo ""
echo "Start time: $(date)"
echo ""

# Configuration
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_ROOT

DATA_DIR="./data"
RESULTS_DIR="./results"
FIGURES_DIR="./figures"

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR

# Function to run an experiment with timing
run_experiment() {
    local name=$1
    local script=$2
    local args=$3

    echo ""
    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    start_time=$(date +%s)

    if python $script $args; then
        echo "[PASSED] $name"
        result="PASSED"
    else
        echo "[FAILED] $name"
        result="FAILED"
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Duration: ${duration}s"

    return 0
}

# Run all hypothesis tests
echo ""
echo "Phase 1: Running Hypothesis Tests"
echo "=================================="

run_experiment "H1: Distribution Shift vs F1 Drop" \
    "prelim_experiments/h1_distribution_shift/verify_h1.py" \
    "--data_dir ./nyt10 --output_dir $RESULTS_DIR/h1"

run_experiment "H2: ARS vs Forgetting Rate" \
    "prelim_experiments/h2_analogous_forgetting/verify_h2.py" \
    "--data_dir ./fewrel --output_dir $RESULTS_DIR/h2"

run_experiment "H3: PDI vs Noise Rate" \
    "prelim_experiments/h3_prototype_dispersion/verify_h3.py" \
    "--data_dir ./nyt10 --output_dir $RESULTS_DIR/h3"

run_experiment "H4: Path Length vs False Negative" \
    "prelim_experiments/h4_path_length/verify_h4.py" \
    "--data_dir ./docred --output_dir $RESULTS_DIR/h4"

run_experiment "H5: Bag Size vs Reliability" \
    "prelim_experiments/h5_bag_reliability/verify_h5.py" \
    "--data_dir ./nyth --output_dir $RESULTS_DIR/h5"

# Generate report
echo ""
echo "Phase 2: Generating Report"
echo "=========================="

python scripts/generate_report.py \
    --results_dir $RESULTS_DIR \
    --output $RESULTS_DIR/preliminary_experiment_report.md

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Report: $RESULTS_DIR/preliminary_experiment_report.md"
echo "=============================================="
