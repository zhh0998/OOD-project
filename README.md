# Heterophily-Aware Out-of-Distribution Detection for Text Classification

Official implementation of **"Leveraging Graph Heterophily for Robust Text OOD Detection"**.

## Overview

This repository contains the code for **Heterophily-Enhanced OOD Detection**, a novel framework that leverages neighborhood heterophily ratio (NHR) as an indicator of distribution shifts for text out-of-distribution detection.

### Key Features

- **Heterophily-Aware Detection**: Uses NHR to identify near-OOD samples that traditional methods miss
- **Graph-Based Learning**: Automatic k-NN graph construction from text embeddings
- **Adaptive Boundary Refinement**: Dynamic threshold adjustment based on local density
- **State-of-the-Art Performance**: Outperforms existing methods on multiple benchmarks

### Main Results

| Dataset | AUROC | FPR@95 | Type |
|---------|-------|--------|------|
| CLINC150 | **96.23%** | 16.84% | Far-OOD |
| Banking77 | **88.99%** | 32.15% | Near-OOD |
| ROSTD | **99.23%** | 2.10% | Cross-domain |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- sentence-transformers >= 2.2.0

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/heterophily-ood-detection.git
cd heterophily-ood-detection

# Install dependencies
pip install -r requirements.txt

# Run main experiments
python scripts/run_main_experiments.py
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── heterophily_detector.py  # Main model
│   │   ├── knn_detector.py          # KNN-based detector
│   │   └── sota_detectors.py        # SOTA methods (DA-ADB, FLatS, RMD)
│   ├── baselines/         # Baseline methods
│   │   ├── probability_based.py     # MSP, Energy, MaxLogits
│   │   └── knn_contrastive.py       # KNN-Contrastive, VI-OOD
│   ├── datasets/          # Dataset loaders
│   │   └── ood_datasets.py
│   └── utils/             # Utilities
│       ├── visualization.py
│       └── evaluation.py
├── scripts/               # Experiment scripts
│   ├── run_main_experiments.py
│   ├── run_ablation.py
│   └── train.py
├── experiments/           # Results and figures
│   ├── results/          # JSON results
│   └── figures/          # Paper figures (PDF)
├── data/                  # Datasets
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## Usage

### Training & Evaluation

```bash
# Run on CLINC150
python scripts/train.py --dataset clinc150 --k 5

# Run on Banking77
python scripts/train.py --dataset banking77 --k 10

# Run ablation study
python scripts/run_ablation.py
```

### Reproducing Results

```bash
# Main experiments (Table 1)
python scripts/run_main_experiments.py

# Ablation study (Table 2)
python scripts/run_ablation.py

# Generate figures
python src/utils/visualization.py
```

## Datasets

| Dataset | Train | Test (ID) | Test (OOD) | Classes |
|---------|-------|-----------|------------|---------|
| CLINC150 | 18,000 | 4,500 | 1,000 | 150 |
| Banking77 | 10,003 | 3,080 | ~500 | 77 |
| ROSTD | 30,521 | 4,481 | 4,000 | 12 |

See [data/README.md](data/README.md) for download instructions.

## Results

### Comparison with Baselines

| Method | CLINC150 | Banking77 | ROSTD |
|--------|----------|-----------|-------|
| MSP | 86.50% | 75.00% | 92.30% |
| Mahalanobis | 89.44% | 82.00% | 95.00% |
| KNN-Contrastive | 89.98% | 85.50% | 97.50% |
| DA-ADB | 94.54% | 88.00% | 97.60% |
| **Ours** | **96.23%** | **88.99%** | **99.23%** |

### Ablation Study

See [experiments/results/PRIORITY1_FINAL_REPORT.md](experiments/results/PRIORITY1_FINAL_REPORT.md) for detailed ablation results.

## Citation

```bibtex
@article{author2025heterophily,
  title={Leveraging Graph Heterophily for Robust Text OOD Detection},
  author={Author Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- CLINC150, Banking77, ROSTD dataset providers
- Sentence-Transformers team
- PyTorch community
