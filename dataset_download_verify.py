"""
完整的Banking77、ToxiGen、ROSTD数据集下载验证脚本

基于GPT 2024-2025年最新指南
每个数据集都会：
1. 尝试下载
2. 验证字段
3. 保存metadata
4. 输出样本示例

运行前准备：
pip install datasets requests huggingface_hub
huggingface-cli login  # ToxiGen需要
"""

import json
import pathlib
import zipfile
import csv
import hashlib
from datetime import datetime
import requests
from datasets import load_dataset, Dataset, DatasetDict

print("="*80)
print("三数据集下载验证脚本（Banking77、ToxiGen、ROSTD）")
print("="*80)

# 创建输出目录
output_dir = pathlib.Path("dataset_downloads")
output_dir.mkdir(exist_ok=True)

metadata = {
    "download_date": datetime.now().isoformat(),
    "datasets": {}
}

# ============================================================================
# 数据集1：Banking77
# ============================================================================
print("\n" + "="*80)
print("【数据集1/3】Banking77")
print("="*80)

banking77_success = False
banking77_info = {}

try:
    print("\n方法1：尝试加载 mteb/banking77（推荐）...")
    ds_banking = load_dataset("mteb/banking77")

    print(f"✅ 成功加载 mteb/banking77")
    print(f"   训练集: {ds_banking['train'].num_rows} 样本")
    print(f"   测试集: {ds_banking['test'].num_rows} 样本")
    print(f"   字段: {ds_banking['train'].column_names}")

    # 验证必需字段
    assert "text" in ds_banking['train'].column_names
    assert "label" in ds_banking['train'].column_names

    print(f"\n示例数据:")
    print(f"   {ds_banking['train'][0]}")

    # 统计类别数
    unique_labels = set(ds_banking['train']['label'])
    print(f"\n类别统计:")
    print(f"   类别数: {len(unique_labels)}")

    banking77_info = {
        "source": "mteb/banking77",
        "train_size": ds_banking['train'].num_rows,
        "test_size": ds_banking['test'].num_rows,
        "num_classes": len(unique_labels),
        "columns": ds_banking['train'].column_names,
        "sample": str(ds_banking['train'][0])
    }

    banking77_success = True

except Exception as e1:
    print(f"⚠️ 方法1失败: {e1}")

    # 备用：从GitHub下载
    try:
        print("\n方法2：尝试从GitHub下载CSV文件...")
        train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
        test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

        ds_banking = load_dataset(
            "csv",
            data_files={"train": train_url, "test": test_url},
        )

        print(f"✅ 成功从GitHub加载")
        print(f"   训练集: {ds_banking['train'].num_rows} 样本")
        print(f"   测试集: {ds_banking['test'].num_rows} 样本")
        print(f"   字段: {ds_banking['train'].column_names}")

        # 处理字段名
        if "category" in ds_banking['train'].column_names:
            ds_banking = ds_banking.rename_column("category", "label_text")

        # 构造数值label
        labels = sorted(set(ds_banking['train']['label_text']))
        label2id = {name: i for i, name in enumerate(labels)}

        def add_label(ex):
            ex["label"] = label2id[ex["label_text"]]
            return ex

        ds_banking = ds_banking.map(add_label)

        print(f"\n类别统计:")
        print(f"   类别数: {len(labels)}")

        banking77_info = {
            "source": "GitHub CSV",
            "train_size": ds_banking['train'].num_rows,
            "test_size": ds_banking['test'].num_rows,
            "num_classes": len(labels),
            "columns": ds_banking['train'].column_names
        }

        banking77_success = True

    except Exception as e2:
        print(f"❌ 方法2也失败: {e2}")
        banking77_info = {"error": str(e2)}

metadata["datasets"]["Banking77"] = {
    "success": banking77_success,
    "info": banking77_info
}

# ============================================================================
# 数据集2：ToxiGen
# ============================================================================
print("\n" + "="*80)
print("【数据集2/3】ToxiGen")
print("="*80)

toxigen_success = False
toxigen_info = {}

print("\n⚠️ 注意：ToxiGen需要HuggingFace登录")
print("如果未登录，请运行：huggingface-cli login")
print("或在代码中调用：login(token='hf_xxx')")

try:
    print("\n尝试加载 toxigen/toxigen-data (annotated配置)...")

    # 尝试使用token
    tox = load_dataset("toxigen/toxigen-data", "annotated", token=True)

    print(f"✅ 成功加载 toxigen/toxigen-data (annotated)")

    # 检查split
    splits = list(tox.keys())
    print(f"   可用splits: {splits}")

    for split in splits:
        print(f"   {split}: {tox[split].num_rows} 样本")

    # 验证关键字段
    train_cols = tox['train'].column_names if 'train' in tox else tox[splits[0]].column_names
    print(f"\n字段列表: {train_cols}")

    assert "text" in train_cols
    assert "toxicity_ai" in train_cols, "❌ 缺少toxicity_ai字段！"

    print("✅ 验证通过：toxicity_ai字段存在")

    # 检查toxicity_ai的值范围
    sample_split = 'train' if 'train' in tox else splits[0]
    import numpy as np
    toxicity_values = np.array(tox[sample_split]['toxicity_ai'])
    valid_mask = ~np.isnan(toxicity_values)

    print(f"\ntoxicity_ai统计:")
    print(f"   有效值: {valid_mask.sum()}/{len(toxicity_values)}")
    print(f"   范围: [{toxicity_values[valid_mask].min():.2f}, {toxicity_values[valid_mask].max():.2f}]")
    print(f"   均值: {toxicity_values[valid_mask].mean():.2f}")

    print(f"\n示例数据:")
    print(f"   {tox[sample_split][0]}")

    toxigen_info = {
        "source": "toxigen/toxigen-data (annotated)",
        "splits": {split: tox[split].num_rows for split in splits},
        "columns": train_cols,
        "toxicity_ai_range": [float(toxicity_values[valid_mask].min()),
                             float(toxicity_values[valid_mask].max())],
        "toxicity_ai_mean": float(toxicity_values[valid_mask].mean()),
        "valid_samples": int(valid_mask.sum())
    }

    toxigen_success = True

except Exception as e:
    print(f"❌ ToxiGen加载失败: {e}")
    print("\n可能的原因:")
    print("  1. 未登录HuggingFace（运行: huggingface-cli login）")
    print("  2. 需要在HF网站上Accept数据集使用条款")
    print("  3. 网络问题")

    toxigen_info = {
        "error": str(e),
        "troubleshooting": [
            "Run: huggingface-cli login",
            "Visit: https://huggingface.co/datasets/toxigen/toxigen-data",
            "Accept terms if prompted"
        ]
    }

metadata["datasets"]["ToxiGen"] = {
    "success": toxigen_success,
    "info": toxigen_info
}

# ============================================================================
# 数据集3：ROSTD（OOD + ID）
# ============================================================================
print("\n" + "="*80)
print("【数据集3/3】ROSTD")
print("="*80)

rostd_success = False
rostd_info = {}

# Part 1: OOD数据（LR_GC_OOD GitHub）
print("\nPart 1: 下载ROSTD OOD数据（Gangal et al. AAAI 2020）...")

try:
    ZIP_URL = "https://github.com/vgtomahawk/LR_GC_OOD/archive/refs/heads/master.zip"
    out_dir = output_dir / "LR_GC_OOD_data"
    out_dir.mkdir(exist_ok=True)

    zip_path = out_dir / "LR_GC_OOD-master.zip"

    if not zip_path.exists():
        print(f"下载GitHub仓库zip...")
        r = requests.get(ZIP_URL, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print(f"✅ 下载完成: {len(r.content) / 1024 / 1024:.1f} MB")
    else:
        print(f"✅ zip文件已存在，跳过下载")

    # 解压
    print("解压中...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)

    repo_root = next(out_dir.glob("LR_GC_OOD-*"))

    # 查找OOD文件
    ood_path = repo_root / "data" / "fbrelease" / "OODrelease.tsv"

    if not ood_path.exists():
        # 尝试找其他TSV
        tsv_files = list(repo_root.rglob("*.tsv"))
        print(f"找到的TSV文件: {[f.name for f in tsv_files]}")
        ood_path = tsv_files[0] if tsv_files else None

    if ood_path and ood_path.exists():
        print(f"✅ 找到OOD文件: {ood_path.name}")

        # 解析第3列（按README说明）
        texts = []
        with ood_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 3:
                    texts.append(row[2])

        ood_ds = Dataset.from_dict({"text": texts, "is_ood": [1] * len(texts)})
        print(f"✅ OOD样本数: {len(ood_ds)}")
        print(f"   示例: {ood_ds[0]['text'][:100]}...")

        rostd_info["ood"] = {
            "source": "GitHub LR_GC_OOD",
            "file": ood_path.name,
            "num_samples": len(ood_ds),
            "sample": ood_ds[0]['text'][:200]
        }

        rostd_success = True
    else:
        print(f"❌ 未找到OOD文件")
        rostd_info["ood"] = {"error": "OODrelease.tsv not found"}

except Exception as e:
    print(f"❌ OOD数据下载失败: {e}")
    rostd_info["ood"] = {"error": str(e)}

# Part 2: ID数据说明
print("\nPart 2: ID数据说明")
print("Schuster et al. NAACL 2019 多语言任务对话数据集")
print("fb.me链接可能不稳定，建议使用替代数据集")
print("推荐替代方案：ATIS、SNIPS、Banking77子集")

rostd_info["id"] = {
    "note": "ID data can use ATIS/SNIPS/Banking77 as alternative",
    "recommendation": "Use load_dataset('tuetschek/atis') or load_dataset('benayas/snips')"
}

metadata["datasets"]["ROSTD"] = {
    "success": rostd_success,
    "info": rostd_info
}

# ============================================================================
# 保存metadata
# ============================================================================
print("\n" + "="*80)
print("保存metadata")
print("="*80)

metadata_path = output_dir / "download_metadata.json"
with metadata_path.open("w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata已保存: {metadata_path}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("下载验证总结")
print("="*80)

total_success = sum([
    metadata["datasets"]["Banking77"]["success"],
    metadata["datasets"]["ToxiGen"]["success"],
    metadata["datasets"]["ROSTD"]["success"]
])

print(f"\n成功: {total_success}/3 个数据集\n")

for dataset_name, info in metadata["datasets"].items():
    status = "✅" if info["success"] else "❌"
    print(f"{status} {dataset_name}")
    if info["success"]:
        if "info" in info and "source" in info["info"]:
            print(f"   来源: {info['info']['source']}")
    else:
        if "info" in info and "error" in info["info"]:
            print(f"   错误: {info['info']['error'][:100]}")

print("\n" + "="*80)
print("详细信息请查看: " + str(metadata_path))
print("="*80)

if total_success == 3:
    print("\n🎉 所有数据集下载成功！")
    print("下一步：运行多数据集实验")
elif total_success >= 2:
    print("\n⚠️ 部分数据集下载成功")
    print("可以先用成功的数据集进行实验")
else:
    print("\n❌ 大部分数据集下载失败")
    print("请检查网络连接和登录状态")
