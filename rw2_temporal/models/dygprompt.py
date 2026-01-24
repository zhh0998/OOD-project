"""DyGPrompt: Dynamic Graph Prompting - Scheme 4 (P1)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseTemporalModel


class TemporalPrompt(nn.Module):
    def __init__(self, prompt_dim, num_prompts=4):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.time_proj = nn.Linear(prompt_dim, num_prompts)

    def forward(self, node_emb, time_emb):
        attn_weights = F.softmax(self.time_proj(time_emb), dim=-1)
        prompt = torch.matmul(attn_weights, self.prompt_tokens)
        return node_emb + prompt


class DyGPrompt(BaseTemporalModel):
    def __init__(self, num_nodes, embedding_dim=64, time_dim=64, num_prompts=4, **kwargs):
        super().__init__(num_nodes, embedding_dim, time_dim)
        self.num_prompts = num_prompts
        self.temporal_prompt = TemporalPrompt(embedding_dim, num_prompts)
        self.pred_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, src, pos_dst, neg_dst, timestamps):
        batch_size = src.size(0)
        num_neg = neg_dst.size(1) if neg_dst.dim() > 1 else 1

        src_emb = self.get_node_embedding(src)
        pos_dst_emb = self.get_node_embedding(pos_dst)
        time_emb = self.encode_time(timestamps)

        src_prompted = self.temporal_prompt(src_emb, time_emb)
        pos_dst_prompted = self.temporal_prompt(pos_dst_emb, time_emb)

        pos_pair = torch.cat([src_prompted, pos_dst_prompted], dim=-1)
        pos_score = self.pred_head(pos_pair).squeeze(-1)

        neg_dst_flat = neg_dst.view(-1)
        neg_dst_emb = self.get_node_embedding(neg_dst_flat)
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, self.time_dim)
        neg_dst_prompted = self.temporal_prompt(neg_dst_emb, time_emb_expanded)
        neg_dst_prompted = neg_dst_prompted.view(batch_size, num_neg, -1)

        src_prompted_expanded = src_prompted.unsqueeze(1).expand(-1, num_neg, -1)
        neg_pairs = torch.cat([src_prompted_expanded, neg_dst_prompted], dim=-1)
        neg_score = self.pred_head(neg_pairs).squeeze(-1)

        return pos_score, neg_score

    def compute_loss(self, pos_score, neg_score):
        pos_score = pos_score.unsqueeze(-1)
        diff = pos_score - neg_score
        return -F.logsigmoid(diff).mean()


def create_dygprompt(num_nodes, **kwargs):
    return DyGPrompt(num_nodes, **kwargs)
