"""TPNet: Temporal Pattern Network - Scheme 3 (P1)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseTemporalModel, TimeEncoder


class TPNet(BaseTemporalModel):
    def __init__(self, num_nodes, embedding_dim=64, time_dim=64, num_heads=4, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(num_nodes, embedding_dim, time_dim)
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.temporal_fusion = nn.Sequential(
            nn.Linear(embedding_dim + time_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 4), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(embedding_dim * 4, embedding_dim), nn.Dropout(dropout))
            for _ in range(num_layers)
        ])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])

        self.link_pred = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )

    def encode_node_temporal(self, node_ids, timestamps):
        node_emb = self.get_node_embedding(node_ids)
        time_emb = self.encode_time(timestamps)
        combined = torch.cat([node_emb, time_emb], dim=-1)
        return self.temporal_fusion(combined)

    def forward(self, src, pos_dst, neg_dst, timestamps):
        batch_size = src.size(0)
        num_neg = neg_dst.size(1) if neg_dst.dim() > 1 else 1

        src_emb = self.encode_node_temporal(src, timestamps)
        pos_dst_emb = self.encode_node_temporal(pos_dst, timestamps)

        x = src_emb.unsqueeze(1)
        for attn, ln, ff, ff_ln in zip(self.attention_layers, self.layer_norms, self.ff_layers, self.ff_norms):
            attn_out, _ = attn(x, x, x)
            x = ln(x + attn_out)
            x = ff_ln(x + ff(x))
        src_final = x.squeeze(1)

        pos_features = torch.cat([src_final, pos_dst_emb, src_final * pos_dst_emb], dim=-1)
        pos_score = self.link_pred(pos_features).squeeze(-1)

        neg_dst_flat = neg_dst.view(-1)
        timestamps_expanded = timestamps.unsqueeze(1).expand(-1, num_neg).reshape(-1)
        neg_dst_emb = self.encode_node_temporal(neg_dst_flat, timestamps_expanded)
        neg_dst_emb = neg_dst_emb.view(batch_size, num_neg, -1)

        src_final_expanded = src_final.unsqueeze(1).expand(-1, num_neg, -1)
        neg_features = torch.cat([src_final_expanded, neg_dst_emb, src_final_expanded * neg_dst_emb], dim=-1)
        neg_score = self.link_pred(neg_features).squeeze(-1)

        return pos_score, neg_score

    def compute_loss(self, pos_score, neg_score):
        batch_size = pos_score.size(0)
        pos_score = pos_score.unsqueeze(-1)
        logits = torch.cat([pos_score, neg_score], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def create_tpnet(num_nodes, **kwargs):
    return TPNet(num_nodes, **kwargs)
