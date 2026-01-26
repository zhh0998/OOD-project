# Dataset Documentation

## Overview

This project evaluates on three OOD detection benchmarks covering different difficulty levels.

## CLINC150 (Far-OOD)

**Description**: Multi-domain intent classification dataset with 150 intent classes.

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 18,000 | 150 (ID) |
| Test ID | 4,500 | 150 |
| Test OOD | 1,000 | out-of-scope |

**Download**:
```python
from datasets import load_dataset
dataset = load_dataset("contemmcm/clinc150")
```

**Characteristics**:
- Far-OOD: OOD samples are clearly out-of-domain
- Well-balanced classes
- Standard benchmark for intent OOD detection

## Banking77 (Near-OOD)

**Description**: Single-domain fine-grained banking intent dataset.

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 10,003 | 77 |
| Test | 3,080 | 77 |
| OOD | ~500 | held-out classes |

**Download**:
```python
from datasets import load_dataset
dataset = load_dataset("PolyAI/banking77")
```

**Characteristics**:
- Near-OOD: High semantic similarity between ID and OOD
- Challenging for traditional methods
- Tests fine-grained discrimination

## ROSTD (Cross-Domain)

**Description**: Real-world OOD sentences from multiple domains.

| Split | Samples | Domains |
|-------|---------|---------|
| Train | 30,521 | 12 |
| Test ID | 4,481 | 12 |
| Test OOD | 4,000 | out-of-domain |

**Source**: [LR_GC_OOD Repository](https://github.com/vgtomahawk/LR_GC_OOD)

**Characteristics**:
- Cross-domain OOD detection
- Large-scale evaluation
- Real-world distribution shifts

## Data Preparation

All datasets are automatically downloaded and cached. To manually prepare:

```bash
# Using the data loader
python -c "from src.datasets.ood_datasets import load_clinc150; load_clinc150()"
```

## Data Format

Each dataset is processed to return:
- `train_texts`: List of training sentences
- `test_texts`: List of test sentences
- `test_labels`: Binary labels (0=ID, 1=OOD)
- `test_intents`: Original intent labels (for analysis)
