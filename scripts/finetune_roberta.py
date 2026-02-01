#!/usr/bin/env python3
"""
RoBERTa-base微调脚本
用于CLINC150和Banking77的ID分类任务

微调后的模型将用于提取更好的特征表示，
预期可提升OOD检测性能约6%（基于SOTA论文报告）

使用示例:
    # CLINC150 Native OOS
    python scripts/finetune_roberta.py --dataset clinc150 --protocol native_oos

    # CLINC150 Class-holdout 25% KIR
    python scripts/finetune_roberta.py --dataset clinc150 --protocol class_holdout --kir 0.25

    # Banking77 (50/27 FLatS标准)
    python scripts/finetune_roberta.py --dataset banking77 --protocol class_holdout
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}


def main():
    parser = argparse.ArgumentParser(description='RoBERTa微调脚本')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['clinc150', 'banking77'],
                       help='数据集名称')
    parser.add_argument('--protocol', type=str, required=True,
                       choices=['native_oos', 'class_holdout'],
                       help='实验协议')
    parser.add_argument('--kir', type=float, default=None,
                       help='Known Intent Ratio (仅class_holdout需要)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度')
    parser.add_argument('--output_dir', type=str, default='./finetuned_models',
                       help='模型保存目录')
    parser.add_argument('--data_root', type=str, default='data',
                       help='数据根目录')
    parser.add_argument('--model_name', type=str, default='distilroberta-base',
                       help='预训练模型名称 (distilroberta-base更快)')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*80)
    print("RoBERTa微调训练")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    from src.datasets.unified_loader import load_ood_dataset

    load_kwargs = {
        'dataset': args.dataset,
        'protocol': args.protocol,
        'seed': args.seed,
        'data_root': args.data_root
    }

    if args.protocol == 'class_holdout' and args.dataset == 'clinc150':
        if args.kir is None:
            raise ValueError("CLINC150 class_holdout协议需要指定--kir参数")
        load_kwargs['known_intent_ratio'] = args.kir

    data = load_ood_dataset(**load_kwargs)

    num_labels = len(set(data['train_labels']))

    print(f"\n微调配置:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Protocol: {args.protocol}")
    if args.kir:
        print(f"  KIR: {args.kir}")
    print(f"  Num labels (ID classes): {num_labels}")
    print(f"  Training samples: {len(data['train_texts'])}")
    print(f"  Seed: {args.seed}")

    # 划分训练/验证集 (90/10)
    n_samples = len(data['train_texts'])
    n_val = int(n_samples * 0.1)
    indices = np.random.permutation(n_samples)

    train_texts = [data['train_texts'][i] for i in indices[:-n_val]]
    train_labels = [data['train_labels'][i] for i in indices[:-n_val]]
    val_texts = [data['train_texts'][i] for i in indices[-n_val:]]
    val_labels = [data['train_labels'][i] for i in indices[-n_val:]]

    print(f"  Train split: {len(train_texts)}")
    print(f"  Val split: {len(val_texts)}")

    # 创建Dataset
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })

    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    # 初始化tokenizer和模型
    print(f"\n初始化模型: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=args.max_length
        )

    print("Tokenizing数据...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # 设置格式
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # 输出目录名
    output_name = f"{args.dataset}_{args.protocol}"
    if args.kir:
        output_name += f"_kir{int(args.kir*100)}"
    output_name += f"_seed{args.seed}"

    training_output_dir = Path(args.output_dir) / output_name
    training_output_dir.mkdir(parents=True, exist_ok=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",  # 禁用wandb等
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,  # 节省内存
        dataloader_num_workers=0,  # CPU友好
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 训练
    print("\n开始训练...")
    print("-"*60)
    trainer.train()

    # 最终评估
    print("\n最终评估...")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    print(f"  F1 Macro: {eval_results['eval_f1_macro']*100:.2f}%")

    # 保存最终模型
    final_path = Path(args.output_dir) / f"{output_name}_final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # 保存训练配置
    config = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'protocol': args.protocol,
        'kir': args.kir,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'num_labels': num_labels,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'final_accuracy': eval_results['eval_accuracy'],
        'final_f1_macro': eval_results['eval_f1_macro'],
        'timestamp': datetime.now().isoformat()
    }

    with open(final_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ 模型已保存: {final_path}")
    print("="*80)


if __name__ == "__main__":
    main()
