"""SSM-Memory-LLM: State Space Model with Memory - Scheme 0 (P0 Core Innovation)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseTemporalModel, TimeEncoder


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.dt_min, self.dt_max = dt_min, dt_max
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, timestamps=None):
        batch, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log)
        B = self.B_proj(x)
        C = self.C_proj(x)
        dt = torch.clamp(F.softplus(self.dt_proj(x)), self.dt_min, self.dt_max)

        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            dt_A = dt[:, t].unsqueeze(-1) * A
            A_bar = torch.exp(dt_A)
            dt_B = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            h = A_bar * h + dt_B * x[:, t:t+1, :].transpose(1, 2)
            y_t = (C[:, t].unsqueeze(1) * h).sum(-1) + self.D * x[:, t]
            outputs.append(y_t)
        return self.out_proj(torch.stack(outputs, dim=1))


class MemoryModule(nn.Module):
    def __init__(self, d_model, memory_size=64, num_heads=4):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)
        self.read_head = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.write_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x):
        batch = x.size(0)
        memory = self.memory.unsqueeze(0).expand(batch, -1, -1)
        read_output, _ = self.read_head(x, memory, memory)
        gate = self.write_gate(torch.cat([x, read_output], dim=-1))
        return gate * read_output + (1 - gate) * x


class SSMMemoryLLM(BaseTemporalModel):
    def __init__(self, num_nodes, embedding_dim=64, time_dim=64, d_state=16,
                 memory_size=64, num_heads=4, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(num_nodes, embedding_dim, time_dim)
        self.d_state = d_state
        self.memory_size = memory_size

        self.temporal_fusion = nn.Sequential(
            nn.Linear(embedding_dim + time_dim, embedding_dim),
            nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout)
        )

        self.ssm_layers = nn.ModuleList([SelectiveSSM(embedding_dim, d_state) for _ in range(num_layers)])
        self.ssm_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.memory = MemoryModule(embedding_dim, memory_size, num_heads)
        self.memory_norm = nn.LayerNorm(embedding_dim)

        self.llm_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.llm_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embedding_dim * 4, embedding_dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

        self.link_pred = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2), nn.LayerNorm(embedding_dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(), nn.Linear(embedding_dim, 1)
        )

    def encode_with_ssm(self, node_ids, timestamps):
        node_emb = self.get_node_embedding(node_ids)
        time_emb = self.encode_time(timestamps)
        x = self.temporal_fusion(torch.cat([node_emb, time_emb], dim=-1)).unsqueeze(1)

        for ssm, norm in zip(self.ssm_layers, self.ssm_norms):
            x = norm(x + ssm(x))
        x = self.memory_norm(x + self.memory(x))
        attn_out, _ = self.llm_attention(x, x, x)
        x = self.llm_norm(x + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return x.squeeze(1)

    def forward(self, src, pos_dst, neg_dst, timestamps):
        batch_size = src.size(0)
        num_neg = neg_dst.size(1) if neg_dst.dim() > 1 else 1

        src_enc = self.encode_with_ssm(src, timestamps)
        pos_dst_enc = self.encode_with_ssm(pos_dst, timestamps)

        pos_features = torch.cat([src_enc, pos_dst_enc, src_enc * pos_dst_enc, src_enc - pos_dst_enc], dim=-1)
        pos_score = self.link_pred(pos_features).squeeze(-1)

        neg_dst_flat = neg_dst.view(-1)
        timestamps_expanded = timestamps.unsqueeze(1).expand(-1, num_neg).reshape(-1)
        neg_dst_enc = self.encode_with_ssm(neg_dst_flat, timestamps_expanded).view(batch_size, num_neg, -1)

        src_enc_expanded = src_enc.unsqueeze(1).expand(-1, num_neg, -1)
        neg_features = torch.cat([src_enc_expanded, neg_dst_enc, src_enc_expanded * neg_dst_enc,
                                  src_enc_expanded - neg_dst_enc], dim=-1)
        neg_score = self.link_pred(neg_features).squeeze(-1)

        return pos_score, neg_score

    def compute_loss(self, pos_score, neg_score):
        batch_size = pos_score.size(0)
        pos_score_expanded = pos_score.unsqueeze(-1)
        logits = torch.cat([pos_score_expanded, neg_score], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def create_ssm_memory_llm(num_nodes, **kwargs):
    return SSMMemoryLLM(num_nodes, **kwargs)
