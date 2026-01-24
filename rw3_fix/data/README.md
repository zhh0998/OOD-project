# RW3项目数据集说明

本目录包含OOD检测实验所需的数据集文件。

## 数据集列表

### 1. CLINC150 (Far-OOD)
- **位置**: `clinc150/data_full.json`
- **规模**: ~2.4MB
- **内容**: 150个意图类别 + OOD样本
- **来源**: [clinc/oos-eval](https://github.com/clinc/oos-eval)

### 2. Banking77 (Near-OOD)
- **位置**: `banking77_oos/`
- **规模**: ~1MB
- **内容**: 77个银行业务意图
- **来源**: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)

### 3. ROSTD
- **位置**: `rostd/`
- **内容**: 跨域OOD检测数据
- **来源**: [LR_GC_OOD](https://github.com/vgtomahawk/LR_GC_OOD)

### 4. HWU64
- **位置**: `hwu64/`
- **内容**: 64个意图类别
- **来源**: [hwu_64](https://huggingface.co/datasets/hwu_64)

## 数据集下载 (如需完整版)

```python
from datasets import load_dataset

# CLINC150
clinc = load_dataset("contemmcm/clinc150")

# Banking77
banking = load_dataset("PolyAI/banking77")

# HWU64
hwu = load_dataset("hwu_64")
```

## 数据格式

所有数据集已预处理为统一格式:
- `text`: 输入文本
- `label`: 意图标签 (ID样本)
- OOD样本标记为特定标签或单独文件

## 使用方法

```python
from data_loader import load_clinc150, load_banking77_oos, load_rostd

# 加载CLINC150
train_texts, test_texts, test_labels, test_intents = load_clinc150()

# 加载Banking77
train_texts, test_texts, test_labels, test_intents = load_banking77_oos()
```
