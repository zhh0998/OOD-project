# Dataset Download Instructions

This directory is intentionally empty to keep the repository lightweight.
Datasets are downloaded automatically when running experiments.

## Available Datasets

| Dataset | Type | Size | Source |
|---------|------|------|--------|
| CLINC150 | Far-OOD | ~2.4MB | [clinc/oos-eval](https://github.com/clinc/oos-eval) |
| Banking77 | Near-OOD | ~1MB | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| ROSTD | Cross-Domain | ~1MB | [LR_GC_OOD](https://github.com/vgtomahawk/LR_GC_OOD) |
| HWU64 | Multi-Domain | ~0.5MB | [hwu_64](https://huggingface.co/datasets/hwu_64) |

## Manual Download

```python
from datasets import load_dataset

# CLINC150
clinc = load_dataset("contemmcm/clinc150")

# Banking77
banking = load_dataset("PolyAI/banking77")

# HWU64
hwu = load_dataset("hwu_64")
```

## Using with Project

```python
from src.datasets.ood_datasets import load_clinc150, load_banking77

# Datasets are automatically downloaded and cached
train_texts, test_texts, test_labels, test_intents = load_clinc150()
```

## Data Format

All datasets are processed to return:
- `train_texts`: List of training sentences
- `test_texts`: List of test sentences
- `test_labels`: Binary labels (0=ID, 1=OOD)
- `test_intents`: Original intent labels (for analysis)
