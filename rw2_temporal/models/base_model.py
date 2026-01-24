"""Base Model for Temporal Network Embedding"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.w.weight = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim)).float().reshape(dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.w(t))


class BaseTemporalModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, time_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.src_proj = nn.Linear(embedding_dim + time_dim, embedding_dim)
        self.dst_proj = nn.Linear(embedding_dim, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.src_proj.weight)
        nn.init.xavier_uniform_(self.dst_proj.weight)

    def encode_time(self, timestamps):
        return self.time_encoder(timestamps)

    def get_node_embedding(self, node_ids):
        return self.node_embedding(node_ids)

    def forward(self, src, pos_dst, neg_dst, timestamps):
        batch_size = src.size(0)
        src_emb = self.get_node_embedding(src)
        pos_dst_emb = self.get_node_embedding(pos_dst)
        time_emb = self.encode_time(timestamps)

        src_combined = torch.cat([src_emb, time_emb], dim=-1)
        src_proj = self.src_proj(src_combined)
        pos_dst_proj = self.dst_proj(pos_dst_emb)
        pos_score = (src_proj * pos_dst_proj).sum(dim=-1)

        neg_dst_flat = neg_dst.view(-1)
        neg_dst_emb = self.get_node_embedding(neg_dst_flat)
        neg_dst_proj = self.dst_proj(neg_dst_emb)
        neg_dst_proj = neg_dst_proj.view(batch_size, -1, self.embedding_dim)
        src_proj_expanded = src_proj.unsqueeze(1)
        neg_score = (src_proj_expanded * neg_dst_proj).sum(dim=-1)

        return pos_score, neg_score

    def compute_loss(self, pos_score, neg_score):
        pos_score = pos_score.unsqueeze(-1)
        return F.relu(1.0 - pos_score + neg_score).mean()

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
