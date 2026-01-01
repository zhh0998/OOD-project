# RW1 Preliminary Experiments

Hypothesis verification framework for Remote Supervision Relation Extraction research.

## Overview

This project implements **5 hypothesis verification experiments** for the RW1 research work. The goal is to validate research hypotheses **before** implementing the main methods, following the principle:

> **Preliminary experiments = Problem verification, NOT method verification**

## Research Hypotheses

| ID | Hypothesis | Method | Metric | Threshold |
|----|------------|--------|--------|-----------|
| H1 | Distribution shift → F1 drop | LDA-LLM | Pearson r | > 0.8 |
| H2 | ARS → Forgetting rate | LLM-RFCRE | Spearman ρ | > 0.5 |
| H3 | PDI → Noise rate | LLM-PUQ | Pearson r | > 0.5 |
| H4 | Path length → False negatives | HGT-LC | Cohen's d | > 0.5 |
| H5 | Bag size → Label reliability | PGCDN | Cohen's d | 0.5-0.8 |

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup_env.sh
source venv/bin/activate

# 2. Run all experiments
bash scripts/run_all_hypothesis_tests.sh

# 3. View report
cat results/preliminary_experiment_report.md
```

## Project Structure

```
OOD-project/
├── src/
│   ├── data/           # Dataset loaders
│   │   ├── nyt10_loader.py
│   │   ├── fewrel_loader.py
│   │   ├── docred_loader.py
│   │   └── nyth_loader.py
│   ├── models/         # Baseline models
│   │   ├── baseline_re.py
│   │   ├── gaussian_prototype.py
│   │   └── prototype_network.py
│   └── utils/          # Utilities
│       ├── statistics.py
│       └── visualization.py
├── prelim_experiments/
│   ├── h1_distribution_shift/
│   ├── h2_analogous_forgetting/
│   ├── h3_prototype_dispersion/
│   ├── h4_path_length/
│   └── h5_bag_reliability/
├── scripts/
│   ├── setup_env.sh
│   ├── run_all_hypothesis_tests.sh
│   └── generate_report.py
├── configs/
│   └── experiment_config.yaml
├── nyt10/              # NYT10 dataset
├── results/            # Experiment outputs
└── figures/            # Generated plots
```

## Running Individual Experiments

### H1: Distribution Shift vs F1 Drop

```bash
python prelim_experiments/h1_distribution_shift/verify_h1.py \
    --data_dir ./nyt10 \
    --n_scenarios 10 \
    --output_dir ./results/h1
```

### H2: ARS vs Forgetting Rate

```bash
python prelim_experiments/h2_analogous_forgetting/verify_h2.py \
    --data_dir ./fewrel \
    --n_tasks 10 \
    --n_runs 5 \
    --output_dir ./results/h2
```

### H3: PDI vs Noise Rate

```bash
python prelim_experiments/h3_prototype_dispersion/verify_h3.py \
    --data_dir ./nyt10 \
    --noise_rates 0.0,0.1,0.2,0.3,0.4,0.5 \
    --output_dir ./results/h3
```

### H4: Path Length vs False Negative

```bash
python prelim_experiments/h4_path_length/verify_h4.py \
    --data_dir ./docred \
    --output_dir ./results/h4
```

### H5: Bag Size vs Reliability

```bash
python prelim_experiments/h5_bag_reliability/verify_h5.py \
    --data_dir ./nyth \
    --output_dir ./results/h5
```

## Output Format

Each experiment generates:

1. **JSON results**: `results/hX/hX_results.json`
2. **Figures**: `results/hX/figures/`
3. **Console output**: Detailed statistics

## Statistical Standards

- **Significance level**: p < 0.05
- **Effect size interpretation**:
  - Cohen's d: 0.2 (small), 0.5 (medium), 0.8 (large)
  - Correlation: 0.3 (weak), 0.5 (moderate), 0.7 (strong)

## Success Criteria

| Result | Condition | Action |
|--------|-----------|--------|
| ✅ Success | ≥4/5 hypotheses pass | Proceed to full experiments |
| ⚠️ Partial | 2-3 hypotheses pass | Adjust research scope |
| ❌ Failure | ≤1 hypothesis passes | Re-evaluate direction |

## Key Design Principles

1. **No circular reasoning**: Features computed independently of target labels
2. **Controlled experiments**: Synthetic noise injection, distribution resampling
3. **Human ground truth**: NYT-H annotations for H5
4. **Multiple validation**: Quantitative + case analysis

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- scipy
- matplotlib
- seaborn
- networkx

See `requirements.txt` for complete list.

## Citation

If you use this code, please cite:

```bibtex
@misc{rw1_prelim_2026,
  title={RW1 Preliminary Experiments for Remote Supervision Relation Extraction},
  author={Author Name},
  year={2026}
}
```

## License

MIT License
