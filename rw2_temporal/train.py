#!/usr/bin/env python3
"""Training script for RW2 Temporal Network Embedding models"""

import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from data.data_loader import TGBDataLoader
from models import create_model, MODEL_REGISTRY


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_mrr(pos_scores, neg_scores):
    pos_scores = pos_scores.unsqueeze(-1)
    all_scores = torch.cat([pos_scores, neg_scores], dim=-1)
    rankings = (all_scores >= pos_scores).sum(dim=-1).float()
    return (1.0 / rankings).mean().item()


def compute_hits_at_k(pos_scores, neg_scores, k=10):
    pos_scores = pos_scores.unsqueeze(-1)
    all_scores = torch.cat([pos_scores, neg_scores], dim=-1)
    rankings = (all_scores >= pos_scores).sum(dim=-1)
    return (rankings <= k).float().mean().item()


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, total_mrr, num_batches = 0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

    for batch in pbar:
        src = batch['src'].to(device)
        pos_dst = batch['pos_dst'].to(device)
        neg_dst = batch['neg_dst'].to(device)
        timestamps = batch['timestamp'].to(device)

        optimizer.zero_grad()
        pos_score, neg_score = model(src, pos_dst, neg_dst, timestamps)
        loss = model.compute_loss(pos_score, neg_score)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            mrr = compute_mrr(pos_score, neg_score)
            total_mrr += mrr
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mrr': f'{mrr:.4f}'})

    return total_loss / num_batches, total_mrr / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_mrr, total_hits10, num_batches = 0, 0, 0

    for batch in dataloader:
        src = batch['src'].to(device)
        pos_dst = batch['pos_dst'].to(device)
        neg_dst = batch['neg_dst'].to(device)
        timestamps = batch['timestamp'].to(device)

        pos_score, neg_score = model(src, pos_dst, neg_dst, timestamps)
        total_mrr += compute_mrr(pos_score, neg_score)
        total_hits10 += compute_hits_at_k(pos_score, neg_score, k=10)
        num_batches += 1

    return {'mrr': total_mrr / num_batches, 'hits@10': total_hits10 / num_batches}


def train_single_run(args, run_idx, device):
    seed = args.seed + run_idx * 100
    set_seed(seed)
    print(f"\n{'='*60}\nRun {run_idx + 1}/{args.num_runs} | Seed: {seed}\n{'='*60}")

    data_loader = TGBDataLoader(args.dataset, root='./data', subsample=args.subsample)
    if not data_loader.load():
        raise RuntimeError(f"Failed to load dataset: {args.dataset}")

    num_nodes = data_loader.num_nodes
    stats = data_loader.get_statistics()
    print(f"Dataset: {stats['dataset_name']}, Nodes: {stats['num_nodes']:,}, Edges: {stats['num_edges']:,}")

    train_loader = data_loader.get_dataloader('train', batch_size=args.batch_size, shuffle=True)
    val_loader = data_loader.get_dataloader('val', batch_size=args.batch_size, shuffle=False)
    test_loader = data_loader.get_dataloader('test', batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model, num_nodes, embedding_dim=args.embedding_dim, time_dim=args.time_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mrr, best_epoch = 0, 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mrr = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f}, train_mrr={train_mrr:.4f}, val_mrr={val_metrics['mrr']:.4f}")

        if val_metrics['mrr'] > best_val_mrr:
            best_val_mrr = val_metrics['mrr']
            best_epoch = epoch
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_mrr': best_val_mrr},
                           os.path.join(args.save_dir, f'best_model_run{run_idx}.pt'))

    test_metrics = evaluate(model, test_loader, device)
    train_time = time.time() - start_time

    print(f"\nRun {run_idx + 1} Results: Best Val MRR: {best_val_mrr:.4f} (Epoch {best_epoch})")
    print(f"  Test MRR: {test_metrics['mrr']:.4f}, Test Hits@10: {test_metrics['hits@10']:.4f}")
    print(f"  Training Time: {train_time:.1f}s ({train_time/args.epochs:.2f}s/epoch)")

    return {'run_idx': run_idx, 'seed': seed, 'best_val_mrr': best_val_mrr, 'best_epoch': best_epoch,
            'test_mrr': test_metrics['mrr'], 'test_hits10': test_metrics['hits@10'],
            'train_time': train_time, 'param_count': param_count}


def main():
    parser = argparse.ArgumentParser(description='RW2 Temporal Network Embedding Training')
    parser.add_argument('--dataset', type=str, default='tgbl-wiki')
    parser.add_argument('--subsample', type=int, default=None)
    parser.add_argument('--model', type=str, default='baseline', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--time_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using {'GPU: ' + str(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'CPU'}")

    print(f"\n{'='*60}\nRW2 Training: {args.model} on {args.dataset}\n{'='*60}")

    all_results = [train_single_run(args, run_idx, device) for run_idx in range(args.num_runs)]

    test_mrrs = [r['test_mrr'] for r in all_results]
    test_hits10 = [r['test_hits10'] for r in all_results]

    print(f"\n{'='*60}\nFinal Results Summary\n{'='*60}")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Parameters: {all_results[0]['param_count']:,}")
    print(f"Test MRR: {np.mean(test_mrrs):.4f} ± {np.std(test_mrrs):.4f}")
    print(f"Test Hits@10: {np.mean(test_hits10):.4f} ± {np.std(test_hits10):.4f}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
            json.dump({
                'model': args.model, 'dataset': args.dataset, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'lr': args.lr, 'num_runs': args.num_runs,
                'param_count': all_results[0]['param_count'],
                'test_mrr_mean': float(np.mean(test_mrrs)), 'test_mrr_std': float(np.std(test_mrrs)),
                'test_hits10_mean': float(np.mean(test_hits10)), 'test_hits10_std': float(np.std(test_hits10)),
                'individual_runs': [{'run_idx': r['run_idx'], 'seed': r['seed'], 'test_mrr': float(r['test_mrr']),
                                     'test_hits10': float(r['test_hits10']), 'best_val_mrr': float(r['best_val_mrr']),
                                     'train_time': float(r['train_time'])} for r in all_results]
            }, f, indent=2)
        print(f"Results saved to: {args.save_dir}/results.json")


if __name__ == '__main__':
    main()
