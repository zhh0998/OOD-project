#!/usr/bin/env python3
"""
Main Training Script for RW2 Temporal Network Embedding.

Supports training all three schemes:
- Scheme 0: SSM-Memory-LLM (DyGMamba-based)
- Scheme 3: TPNet-Walk-Matrix-LLM
- Scheme 4: DyGPrompt-TempMem-LLM

Usage:
    python train.py --model ssm_memory_llm --dataset tgbl-wiki
    python train.py --model tpnet --dataset tgbl-wiki
    python train.py --model dygprompt --dataset tgbl-wiki

Author: RW2 Temporal Network Embedding Project
"""

import argparse
import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import RealDataLoader, TemporalDataset
from models.base_model import TempMemLLM
from models.ssm_memory_llm import SSMMemoryLLM, create_ssm_memory_llm
from models.tpnet_llm import TPNetLLM, create_tpnet_llm
from models.dygprompt import StandaloneDyGPrompt, DyGPromptTempMem, create_dygprompt
from utils.metrics import compute_mrr, compute_hits_at_k, compute_metrics, StatisticalAnalysis
from utils.negative_sampling import create_negative_sampler


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(
    model_name: str,
    num_nodes: int,
    config: Dict,
    pretrained_path: Optional[str] = None
) -> nn.Module:
    """
    Create model based on name.

    Args:
        model_name: One of 'ssm_memory_llm', 'tpnet', 'dygprompt', 'baseline'
        num_nodes: Number of nodes in graph
        config: Model configuration
        pretrained_path: Path to pretrained model (for dygprompt)

    Returns:
        Model instance
    """
    if model_name == 'ssm_memory_llm':
        return create_ssm_memory_llm(num_nodes, config)

    elif model_name == 'tpnet':
        return create_tpnet_llm(num_nodes, config)

    elif model_name == 'dygprompt':
        if pretrained_path and os.path.exists(pretrained_path):
            # Load pretrained backbone
            backbone = TempMemLLM(num_nodes=num_nodes, **config)
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            backbone.load_state_dict(checkpoint['model_state_dict'])
            return DyGPromptTempMem(backbone, d_model=config.get('d_model', 128))
        else:
            # Use standalone version
            return StandaloneDyGPrompt(num_nodes=num_nodes, **config)

    elif model_name == 'baseline':
        return TempMemLLM(num_nodes=num_nodes, **config)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    neg_sampler,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_link_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        src = batch['src'].to(device)
        dst = batch['dst'].to(device)
        timestamp = batch['timestamp'].to(device)
        src_neighbor_seq = batch['src_neighbor_seq'].to(device)
        src_time_seq = batch['src_time_seq'].to(device)
        dst_neighbor_seq = batch['dst_neighbor_seq'].to(device)
        dst_time_seq = batch['dst_time_seq'].to(device)

        # Sample negatives
        neg_dst = neg_sampler.sample_batch(
            src.cpu().numpy(),
            dst.cpu().numpy(),
            timestamp.cpu().numpy()
        )
        neg_dst = torch.tensor(neg_dst, dtype=torch.long, device=device)

        # Forward pass
        output = model(
            src, dst, timestamp,
            src_neighbor_seq, src_time_seq,
            dst_neighbor_seq, dst_time_seq,
            neg_dst=neg_dst
        )

        # Compute loss
        loss_dict = model.compute_loss(output['pos_score'], output['neg_score'])
        loss = loss_dict['total_loss'] / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict['total_loss'].item()
        total_link_loss += loss_dict['link_loss'].item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{total_loss/num_batches:.4f}"})

    return {
        'total_loss': total_loss / num_batches,
        'link_loss': total_link_loss / num_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_dataset: TemporalDataset,
    device: torch.device,
    neg_sampler,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()

    if not hasattr(eval_dataset, 'eval_edges'):
        print("Warning: No eval_edges found, using full dataset")
        eval_edges = eval_dataset.edges
    else:
        eval_edges = eval_dataset.eval_edges

    # Sample subset for faster evaluation
    if len(eval_edges) > num_samples:
        indices = np.random.choice(len(eval_edges), num_samples, replace=False)
        eval_edges = [eval_edges[i] for i in indices]

    all_ranks = []

    for edge in tqdm(eval_edges, desc="Evaluating", leave=False):
        # Prepare input
        src = torch.tensor([edge.src], device=device)
        dst = torch.tensor([edge.dst], device=device)
        timestamp = torch.tensor([edge.timestamp], device=device)

        # Get neighbor sequences
        src_neighbors, src_times = eval_dataset.get_temporal_neighbors(
            edge.src, edge.timestamp
        )
        dst_neighbors, dst_times = eval_dataset.get_temporal_neighbors(
            edge.dst, edge.timestamp
        )

        # Pad sequences
        max_len = 64
        def pad(seq, val=0):
            if len(seq) >= max_len:
                return seq[-max_len:]
            return [val] * (max_len - len(seq)) + seq

        src_neighbor_seq = torch.tensor([pad(src_neighbors)], device=device)
        src_time_seq = torch.tensor([pad(src_times, 0.0)], device=device)
        dst_neighbor_seq = torch.tensor([pad(dst_neighbors)], device=device)
        dst_time_seq = torch.tensor([pad(dst_times, 0.0)], device=device)

        # Sample negatives for ranking
        num_neg = 100
        neg_dst = neg_sampler.sample(edge.src, edge.dst, edge.timestamp)
        neg_dst = torch.tensor(neg_dst[:num_neg].reshape(1, -1), device=device)

        # Forward pass
        output = model(
            src, dst, timestamp,
            src_neighbor_seq, src_time_seq,
            dst_neighbor_seq, dst_time_seq,
            neg_dst=neg_dst
        )

        # Compute rank
        pos_score = output['pos_score'].item()
        neg_scores = output['neg_score'].squeeze(0).cpu().numpy()

        # Rank = number of negatives with higher score + 1
        rank = (neg_scores > pos_score).sum() + 1
        all_ranks.append(rank)

    ranks = np.array(all_ranks)
    metrics = compute_metrics(ranks, ks=[1, 3, 10, 50])

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    config: Dict,
    path: str
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: str
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def train(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    )
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    try:
        data_loader = RealDataLoader(
            dataset_name=args.dataset,
            root=args.data_root
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please run 'python validate_data.py' first to check data availability.")
        sys.exit(1)

    # Get data splits
    train_dataset, val_dataset, test_dataset = data_loader.get_temporal_split()

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create negative sampler
    data = data_loader.get_data()
    neg_sampler = create_negative_sampler(
        src_nodes=data.src,
        dst_nodes=data.dst,
        timestamps=data.timestamps,
        num_nodes=data.num_nodes,
        strategy='temporal',
        num_negatives=args.num_negatives,
        seed=args.seed
    )

    # Model configuration
    config = {
        'd_model': args.d_model,
        'dropout': args.dropout,
        'llm_dim': args.llm_dim,
    }

    if args.model == 'ssm_memory_llm':
        config.update({
            'd_state': args.d_state,
            'd_conv': args.d_conv,
            'expand': args.expand,
            'num_ssm_layers': args.num_layers,
        })
    elif args.model == 'tpnet':
        config.update({
            'num_random_features': args.num_random_features,
            'num_neighbors': args.num_neighbors,
            'num_walk_layers': args.num_layers,
        })
    elif args.model == 'dygprompt':
        config.update({
            'alpha': args.alpha,
            'prompt_scale': args.prompt_scale,
        })

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        args.model,
        data.num_nodes,
        config,
        pretrained_path=args.pretrained
    )
    model = model.to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    if args.model == 'dygprompt' and hasattr(model, 'enable_prompt_mode'):
        model.enable_prompt_mode()
        optimizer = optim.AdamW(
            model.get_trainable_parameters() if hasattr(model, 'get_trainable_parameters')
            else filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 10,  # Higher LR for prompt learning
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    best_mrr = 0.0
    patience_counter = 0
    train_history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, neg_sampler,
            accumulation_steps=args.accumulation_steps
        )

        # Evaluate
        val_metrics = evaluate(
            model, val_dataset, device, neg_sampler,
            num_samples=args.eval_samples
        )

        epoch_time = time.time() - epoch_start

        # Logging
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_metrics['total_loss']:.4f} | "
            f"Val MRR: {val_metrics['mrr']:.4f} | "
            f"Val H@10: {val_metrics['hits@10']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'val_mrr': val_metrics['mrr'],
            'val_hits10': val_metrics['hits@10'],
            'time': epoch_time
        })

        # Save best model
        if val_metrics['mrr'] > best_mrr:
            best_mrr = val_metrics['mrr']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                os.path.join(args.save_dir, f"{args.model}_{args.dataset}_best.pth")
            )
            print(f"  -> New best MRR: {best_mrr:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Update scheduler
        scheduler.step()

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                os.path.join(args.save_dir, f"{args.model}_{args.dataset}_epoch{epoch}.pth")
            )

    # Final evaluation on test set
    print("\nLoading best model for final evaluation...")
    load_checkpoint(
        model, None,
        os.path.join(args.save_dir, f"{args.model}_{args.dataset}_best.pth")
    )

    test_metrics = evaluate(
        model, test_dataset, device, neg_sampler,
        num_samples=args.eval_samples * 2
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"MRR: {test_metrics['mrr']:.4f}")
    print(f"Hits@1: {test_metrics['hits@1']:.4f}")
    print(f"Hits@3: {test_metrics['hits@3']:.4f}")
    print(f"Hits@10: {test_metrics['hits@10']:.4f}")
    print(f"Hits@50: {test_metrics['hits@50']:.4f}")

    # Save results
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'config': config,
        'test_metrics': test_metrics,
        'best_val_mrr': best_mrr,
        'train_history': train_history,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'seed': args.seed
    }

    results_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal network embedding models"
    )

    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['ssm_memory_llm', 'tpnet', 'dygprompt', 'baseline'],
        default='ssm_memory_llm',
        help="Model to train"
    )

    # Data
    parser.add_argument('--dataset', '-d', type=str, default='tgbl-wiki')
    parser.add_argument('--data_root', type=str, default='./datasets')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_layers', type=int, default=2)

    # SSM-specific
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)

    # TPNet-specific
    parser.add_argument('--num_random_features', type=int, default=64)
    parser.add_argument('--num_neighbors', type=int, default=20)

    # DyGPrompt-specific
    parser.add_argument('--alpha', type=int, default=2)
    parser.add_argument('--prompt_scale', type=float, default=0.1)
    parser.add_argument('--pretrained', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Negative sampling
    parser.add_argument('--num_negatives', type=int, default=100)

    # Evaluation
    parser.add_argument('--eval_samples', type=int, default=1000)

    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Run training
    train(args)


if __name__ == '__main__':
    main()
