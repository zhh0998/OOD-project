# 股票多层次嵌入降维系统

本系统对股票生成三种不同层次的嵌入表示：基于概念关联的**静态嵌入**、基于因子时序数据的**动态嵌入**（LSTM或PCA），以及将两者融合的**混合嵌入**。系统设计通用、健壮，所有参数可配置，支持客户使用自有数据直接替换运行。

## 目录结构

```
stock_embedding/
├── README.md                           # 本文档
├── requirements.txt                    # Python 依赖
├── config.py                           # 共享配置和工具函数
├── generate_mock_data.py               # 模拟数据生成器
├── convert_data.py                     # 客户数据格式转换工具
├── run_all.py                          # 一键运行脚本
├── concept_embedding.py                # 模块1: 静态概念嵌入 (SVD/Node2Vec)
├── dynamic_temporal_embedding.py       # 模块2: 动态时序嵌入 (LSTM+注意力)
├── hybrid_embedding.py                 # 模块3: 动静融合嵌入 (自编码器)
├── quick_4emb_fastest.py               # 快速替代: PCA版动态嵌入
└── outputs/                            # 输出目录（运行时自动创建）
```

## 环境安装

```bash
# 基础依赖
pip install numpy pandas scikit-learn torch tqdm

# 或通过 requirements.txt
pip install -r requirements.txt

# 可选: Node2Vec 方法（模块1的可选方法）
pip install node2vec networkx
```

**GPU 支持**: 如果有 NVIDIA GPU，安装对应版本的 PyTorch 可自动启用 GPU 加速。没有 GPU 也可正常运行（自动回退 CPU）。

## 快速开始

三条命令跑通整个系统：

```bash
# 1. 生成模拟数据
python generate_mock_data.py

# 2. 一键运行所有模块
python run_all.py --mock

# 3. 查看结果
ls -la outputs/
```

或使用快速模式（跳过 LSTM，只用 PCA）：

```bash
python run_all.py --mock --skip-lstm
```

## 各模块详细说明

### 模块1: 静态概念嵌入 (`concept_embedding.py`)

**功能**: 从股票×概念关联矩阵生成静态嵌入向量

**输入**:
- `.npy` 文件: shape `(n_stocks, n_concepts)`, float64, 高度稀疏（~99%为0）
- 可选: 股票代码文件（每行一个代码）

**输出**:
- `embeddings_{dim}d.npy`: 嵌入矩阵
- `embeddings_{dim}d_with_codes.csv`: 带股票代码的嵌入CSV

**方法**:
- `svd`（默认）: TruncatedSVD，适合稀疏数据
- `node2vec`: GNN Node2Vec（需额外安装 `node2vec` 包）

**示例命令**:
```bash
# SVD 方法，16维嵌入
python concept_embedding.py --input data/concept_matrix.npy --codes data/stock_codes.txt --method svd --dim 16 --output outputs

# Node2Vec 方法，32维嵌入
python concept_embedding.py --input data/concept_matrix.npy --method node2vec --dim 32 --output outputs
```

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | (必填) | 概念矩阵 .npy 文件路径 |
| `--codes` | None | 股票代码文件路径 |
| `--method` | svd | 嵌入方法: svd 或 node2vec |
| `--dim` | 16 | 嵌入维度 |
| `--output` | outputs | 输出目录 |

---

### 模块2: 动态时序嵌入 (`dynamic_temporal_embedding.py`)

**功能**: 使用 LSTM + 时间注意力机制生成每只股票每天的动态嵌入

**输入**: 因子CSV文件夹（每个CSV为一个因子，含 TradingDate, SecurityID, Value 三列）

**输出**: `outputs/embeddings_lstm.csv`

**训练方法**: 自监督学习（最大化 embedding 方差 + 去相关正则）

**示例命令**:
```bash
python dynamic_temporal_embedding.py --input data/factors --output outputs --window 20 --embed_dim 16 --epochs 30
```

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | (必填) | 因子CSV文件夹 |
| `--window` | 20 | 时间窗口天数 |
| `--embed_dim` | 16 | 嵌入维度 |
| `--hidden_size` | 64 | LSTM隐藏层大小 |
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 0.001 | 学习率 |

---

### 模块3: 动静融合嵌入 (`hybrid_embedding.py`)

**功能**: 将静态概念嵌入与动态因子序列融合，通过自编码器生成统一嵌入

**输入**:
- 静态嵌入CSV（模块1的输出）
- 因子CSV文件夹

**输出**: `outputs/hybrid_embeddings.csv`

**自动处理**:
- 股票代码格式不一致：自动检测并统一（如 `SH600000` → `600000.SH`）
- 因子数量和静态嵌入维度：从数据自动推断

**示例命令**:
```bash
python hybrid_embedding.py --static outputs/embeddings_16d_with_codes.csv --factors data/factors --output outputs --embed_dim 16
```

---

### 快速替代版 (`quick_4emb_fastest.py`)

**功能**: 用PCA替代LSTM生成动态嵌入，无需GPU

**特性**:
- 多进程并行读取因子CSV
- 缓存机制，避免重复加载
- `PCA(n_components=min(target_dim, n_factors))` 自适应维度

**示例命令**:
```bash
python quick_4emb_fastest.py --input data/factors --output outputs --embed_dim 16
```

## 数据格式说明

### 概念关联矩阵 (.npy)

| 属性 | 说明 |
|------|------|
| 格式 | NumPy .npy 文件 |
| 形状 | `(n_stocks, n_concepts)` |
| 类型 | float64 |
| 特征 | 高度稀疏，~99% 为0 |

### 因子CSV文件

每个因子一个CSV文件，3列：

| 列名 | 类型 | 示例 | 说明 |
|------|------|------|------|
| TradingDate | string | `2024.06.03` | 交易日期，点号分隔 |
| SecurityID | string | `000001.SZ` | 股票代码 |
| Value | float | `0.523456` | 因子值 |

示例:
```csv
TradingDate,SecurityID,Value
2024.01.02,000001.SZ,0.123456
2024.01.02,600000.SH,-0.234567
2024.01.03,000001.SZ,0.345678
```

### 输出CSV格式

所有输出CSV格式统一：

| 列名 | 类型 | 说明 |
|------|------|------|
| TradingDate | string | 交易日期 (YYYY.MM.DD)，静态嵌入无此列 |
| SecurityID | string | 股票代码 (XXXXXX.SZ/SH) |
| emb_0 ~ emb_{dim-1} | float | 嵌入向量各维度 |

## 客户数据接入指南

### 方法一: 使用格式转换工具

如果客户数据格式与本系统不完全一致，使用 `convert_data.py` 自动转换：

```bash
# 转换整个文件夹
python convert_data.py --input ./客户原始数据/ --output ./stock_transaction_features1/

# 转换单个文件
python convert_data.py --input raw.csv --output converted.csv

# 跳过确认提示
python convert_data.py --input ./raw/ --output ./converted/ -y
```

**支持的输入格式**:

| 数据项 | 支持的格式 |
|--------|-----------|
| 日期 | `YYYY-MM-DD`, `YYYYMMDD`, `YYYY/MM/DD`, `YYYY.MM.DD` |
| 股票代码 | `000001`, `000001.SZ`, `SZ000001`, `000001.XSHE` |
| 列名 | 支持中英文（日期/date/TradingDate, 股票代码/stock/SecurityID 等） |

### 方法二: 手动准备数据

1. **概念矩阵**: 保存为 `.npy` 文件，shape `(股票数, 概念数)`, float64
2. **因子CSV**: 每个因子一个文件，放在同一个文件夹下，每个文件3列: `TradingDate,SecurityID,Value`
3. **股票代码**: 可选，一个文本文件，每行一个代码，顺序与概念矩阵行对应

### 运行自有数据

```bash
# 方式1: 指定数据目录
python concept_embedding.py --input /path/to/concept_matrix.npy --codes /path/to/codes.txt --output outputs
python dynamic_temporal_embedding.py --input /path/to/factors/ --output outputs

# 方式2: 使用 run_all.py
python run_all.py --data_dir /path/to/data/
```

## 输出格式说明

| 文件 | 说明 | 格式 |
|------|------|------|
| `embeddings_{dim}d.npy` | 静态嵌入矩阵 | NumPy数组 `(n_stocks, dim)` |
| `embeddings_{dim}d_with_codes.csv` | 静态嵌入+代码 | CSV: SecurityID, emb_0, ..., emb_{dim-1} |
| `embeddings_lstm.csv` | LSTM动态嵌入 | CSV: TradingDate, SecurityID, emb_0, ..., emb_{dim-1} |
| `embeddings_pca.csv` | PCA动态嵌入 | CSV: TradingDate, SecurityID, emb_0, ..., emb_{dim-1} |
| `hybrid_embeddings.csv` | 混合嵌入 | CSV: TradingDate, SecurityID, emb_0, ..., emb_{dim-1} |

## 常见问题 FAQ

**Q: 运行时提示 "无法检测日期列/股票代码列"**
A: 您的CSV列名不在自动检测范围内。请将日期列重命名为 `TradingDate`，股票代码列重命名为 `SecurityID`，或使用 `convert_data.py` 转换。

**Q: LSTM 训练太慢**
A: 使用 `--skip-lstm` 跳过 LSTM 模块，改用 PCA 快速版（`quick_4emb_fastest.py`）。或者减小 `--epochs` 和 `--window` 参数。

**Q: 报错 "CUDA out of memory"**
A: 减小 `--batch_size`（如改为128或64），或设置环境变量 `CUDA_VISIBLE_DEVICES=` 强制使用 CPU。

**Q: 静态嵌入和因子数据的股票代码不一致**
A: 系统会自动尝试统一股票代码格式。如果仍然不匹配，请用 `convert_data.py` 先统一格式。

**Q: 因子文件数量可以是多少？**
A: 没有限制，1个或200个都可以。系统自动检测文件夹中的所有CSV文件。

**Q: 没有 GPU 可以运行吗？**
A: 可以。系统自动检测，没有 GPU 会自动使用 CPU。PCA 快速版完全不依赖 GPU。

**Q: 如何调整嵌入维度？**
A: 所有模块都支持 `--dim` 或 `--embed_dim` 参数。推荐值: 8-32 维。

**Q: 数据量太大导致内存不足？**
A: 减小 `--n_stocks` 和 `--n_dates`（模拟数据），或分批处理真实数据。系统内部已做内存优化（分块处理、及时GC）。
