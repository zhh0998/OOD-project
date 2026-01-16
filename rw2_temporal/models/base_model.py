"""
Base model for temporal network embedding.
Implements TempMem-LLM as the baseline backbone.

Author: RW2 Temporal Network Embedding Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class TimeEncoding(nn.Module):
    """Fourier-based time encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps."""
        original_shape = t.shape
        t_flat = t.reshape(-1, 1)

        pe = torch.zeros(t_flat.size(0), self.d_model, device=t.device)
        pe[:, 0::2] = torch.sin(t_flat * self.div_term)
        pe[:, 1::2] = torch.cos(t_flat * self.div_term)

        return self.dropout(pe.reshape(*original_shape, self.d_model))


class LogNormalMixture(nn.Module):
    """
    Log-normal mixture model for time prediction.
    Predicts parameters of a mixture of log-normal distributions.
    """

    def __init__(self, d_model: int, n_components: int = 3):
        super().__init__()
        self.n_components = n_components

        # Predict mixture weights, means, and stds
        self.weight_head = nn.Linear(d_model, n_components)
        self.mean_head = nn.Linear(d_model, n_components)
        self.std_head = nn.Linear(d_model, n_components)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict log-normal mixture parameters.

        Args:
            h: Hidden representation [batch_size, d_model]

        Returns:
            Dict with 'weights', 'means', 'stds'
        """
        weights = F.softmax(self.weight_head(h), dim=-1)
        means = self.mean_head(h)
        stds = F.softplus(self.std_head(h)) + 1e-6

        return {'weights': weights, 'means': means, 'stds': stds}

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """Sample from the predicted distribution."""
        params = self.forward(h)

        # Sample component
        comp_idx = torch.multinomial(params['weights'], 1).squeeze(-1)

        # Sample from that component
        batch_idx = torch.arange(h.size(0), device=h.device)
        mean = params['means'][batch_idx, comp_idx]
        std = params['stds'][batch_idx, comp_idx]

        # Sample from log-normal
        z = torch.randn_like(mean)
        sample = torch.exp(mean + std * z)

        return sample


class MemoryModule(nn.Module):
    """
    Memory module for storing and aggregating historical node information.
    Uses GRU for temporal aggregation.
    """

    def __init__(
        self,
        d_model: int,
        num_nodes: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim or d_model

        # Memory aggregator (GRU)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=self.memory_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Initialize node memory
        self.register_buffer(
            'memory',
            torch.zeros(num_nodes, self.memory_dim)
        )
        self.register_buffer(
            'last_update',
            torch.zeros(num_nodes)
        )

    def compute_message(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute message for memory update."""
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.message_encoder(combined)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get memory for specified nodes."""
        return self.memory[node_ids]

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Update memory for nodes."""
        # Get current memory
        current_memory = self.memory[node_ids].unsqueeze(1)  # [N, 1, d]

        # Update with GRU
        messages = messages.unsqueeze(1)  # [N, 1, d]
        _, new_memory = self.gru(messages, current_memory.transpose(0, 1))
        new_memory = new_memory.squeeze(0)  # [N, d]

        # Store updated memory
        self.memory[node_ids] = new_memory.detach()
        self.last_update[node_ids] = timestamps.detach()

    def reset_memory(self):
        """Reset all node memories."""
        self.memory.zero_()
        self.last_update.zero_()

    def forward(
        self,
        neighbor_seq: torch.Tensor,
        neighbor_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate neighbor sequence with GRU.

        Args:
            neighbor_seq: Neighbor node IDs [batch_size, seq_len]
            neighbor_emb: Neighbor embeddings [batch_size, seq_len, d_model]
            mask: Padding mask [batch_size, seq_len]

        Returns:
            Aggregated representation [batch_size, d_model]
        """
        # Apply GRU
        output, hidden = self.gru(neighbor_emb)

        # Use last hidden state
        return hidden.squeeze(0)


class LLMProjection(nn.Module):
    """
    LLM projection layer for mapping temporal representations
    to LLM semantic space.
    """

    def __init__(
        self,
        d_model: int,
        llm_dim: int = 768,  # GPT-2 hidden dimension
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.llm_dim = llm_dim

        self.proj = nn.Sequential(
            nn.Linear(d_model, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through LLM-like transformation."""
        return x + self.proj(x)  # Residual connection


class BaseTemporalModel(nn.Module):
    """
    Abstract base class for temporal network embedding models.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # Node embedding
        self.node_emb = nn.Embedding(num_nodes, d_model)

        # Time encoding
        self.time_enc = TimeEncoding(d_model, dropout)

    def encode(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source and destination nodes.

        Returns:
            src_repr, dst_repr: Node representations
        """
        raise NotImplementedError

    def compute_score(
        self,
        src_repr: torch.Tensor,
        dst_repr: torch.Tensor
    ) -> torch.Tensor:
        """Compute link prediction score."""
        raise NotImplementedError

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        neg_dst: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for link prediction.

        Args:
            src: Source nodes [batch_size]
            dst: Destination nodes [batch_size]
            timestamp: Timestamps [batch_size]
            neg_dst: Negative destinations [batch_size, num_negatives]

        Returns:
            Dict with 'pos_score', 'neg_score', optionally 'time_pred'
        """
        raise NotImplementedError


class TempMemLLM(BaseTemporalModel):
    """
    TempMem-LLM: Temporal Memory Network with LLM Projection.
    This is the baseline model from the pre-experiment.

    Architecture:
    1. Node embedding + Time encoding
    2. Neighbor aggregation with GRU-based Memory
    3. LLM projection layer
    4. Link decoder + Time decoder
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 128,
        memory_dim: int = 128,
        llm_dim: int = 768,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        use_time_prediction: bool = True
    ):
        super().__init__(num_nodes, d_model, dropout)

        self.memory_dim = memory_dim
        self.max_seq_len = max_seq_len
        self.use_time_prediction = use_time_prediction

        # Memory module
        self.memory = MemoryModule(
            d_model=d_model,
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            dropout=dropout
        )

        # LLM projection
        self.llm_proj = LLMProjection(
            d_model=memory_dim,
            llm_dim=llm_dim,
            dropout=dropout
        )

        # Link decoder (bilinear scoring)
        self.link_decoder = nn.Bilinear(memory_dim, memory_dim, 1)

        # Time decoder
        if use_time_prediction:
            self.time_decoder = LogNormalMixture(memory_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(memory_dim)

    def encode_neighbor_sequence(
        self,
        neighbor_seq: torch.Tensor,
        time_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a sequence of neighbors with their timestamps.

        Args:
            neighbor_seq: Neighbor IDs [batch_size, seq_len]
            time_seq: Timestamps [batch_size, seq_len]

        Returns:
            Encoded representation [batch_size, d_model]
        """
        # Get neighbor embeddings
        neighbor_emb = self.node_emb(neighbor_seq)  # [B, L, d]

        # Add time encoding
        time_emb = self.time_enc(time_seq)  # [B, L, d]
        combined = neighbor_emb + time_emb  # [B, L, d]

        # Aggregate with memory module
        h = self.memory(neighbor_seq, combined)  # [B, d]

        return h

    def encode(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        src_neighbor_seq: torch.Tensor,
        src_time_seq: torch.Tensor,
        dst_neighbor_seq: torch.Tensor,
        dst_time_seq: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode source and destination with their neighbor histories."""

        # Encode source
        src_base = self.node_emb(src)  # [B, d]
        src_time = self.time_enc(timestamp)  # [B, d]
        src_neighbor_repr = self.encode_neighbor_sequence(
            src_neighbor_seq, src_time_seq
        )  # [B, d]
        src_repr = src_base + src_time.squeeze(1) + src_neighbor_repr

        # Encode destination
        dst_base = self.node_emb(dst)  # [B, d]
        dst_neighbor_repr = self.encode_neighbor_sequence(
            dst_neighbor_seq, dst_time_seq
        )  # [B, d]
        dst_repr = dst_base + dst_neighbor_repr

        # Apply LLM projection
        src_repr = self.llm_proj(self.layer_norm(src_repr))
        dst_repr = self.llm_proj(self.layer_norm(dst_repr))

        return src_repr, dst_repr

    def compute_score(
        self,
        src_repr: torch.Tensor,
        dst_repr: torch.Tensor
    ) -> torch.Tensor:
        """Compute link prediction score using bilinear decoder."""
        return self.link_decoder(src_repr, dst_repr).squeeze(-1)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        src_neighbor_seq: torch.Tensor,
        src_time_seq: torch.Tensor,
        dst_neighbor_seq: torch.Tensor,
        dst_time_seq: torch.Tensor,
        neg_dst: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for link prediction and time prediction."""

        # Encode source and destination
        src_repr, dst_repr = self.encode(
            src, dst, timestamp,
            src_neighbor_seq, src_time_seq,
            dst_neighbor_seq, dst_time_seq
        )

        # Positive score
        pos_score = self.compute_score(src_repr, dst_repr)

        result = {'pos_score': pos_score, 'src_repr': src_repr, 'dst_repr': dst_repr}

        # Negative scores
        if neg_dst is not None:
            batch_size, num_negatives = neg_dst.shape

            # Encode negative destinations (simplified - no neighbor history)
            neg_dst_emb = self.node_emb(neg_dst)  # [B, N, d]

            # Expand source representation
            src_repr_expanded = src_repr.unsqueeze(1).expand(-1, num_negatives, -1)

            # Compute scores
            neg_scores = []
            for i in range(num_negatives):
                score = self.compute_score(
                    src_repr_expanded[:, i],
                    self.llm_proj(self.layer_norm(neg_dst_emb[:, i]))
                )
                neg_scores.append(score)

            result['neg_score'] = torch.stack(neg_scores, dim=1)

        # Time prediction
        if self.use_time_prediction:
            result['time_pred'] = self.time_decoder(src_repr)

        return result

    def compute_loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        time_pred: Optional[Dict] = None,
        time_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            pos_score: Positive edge scores [batch_size]
            neg_score: Negative edge scores [batch_size, num_negatives]
            time_pred: Time prediction parameters (optional)
            time_target: Target time differences (optional)

        Returns:
            Dict with 'total_loss', 'link_loss', 'time_loss'
        """
        # Link prediction loss (BPR-style)
        # pos_score should be higher than neg_score
        diff = pos_score.unsqueeze(1) - neg_score  # [B, N]
        link_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        result = {'link_loss': link_loss}

        # Time prediction loss (optional)
        if time_pred is not None and time_target is not None:
            # Negative log-likelihood of mixture model
            weights = time_pred['weights']
            means = time_pred['means']
            stds = time_pred['stds']

            # Log-likelihood for each component
            log_target = torch.log(time_target + 1e-10).unsqueeze(-1)
            log_prob = (
                -0.5 * ((log_target - means) / stds) ** 2
                - torch.log(stds)
                - 0.5 * math.log(2 * math.pi)
                - log_target  # Jacobian for log-normal
            )

            # Mixture log-likelihood
            log_prob = torch.logsumexp(
                torch.log(weights + 1e-10) + log_prob,
                dim=-1
            )
            time_loss = -log_prob.mean()

            result['time_loss'] = time_loss
            result['total_loss'] = link_loss + 0.1 * time_loss
        else:
            result['total_loss'] = link_loss

        return result


if __name__ == '__main__':
    # Test the model
    print("Testing TempMemLLM...")

    num_nodes = 1000
    batch_size = 32
    seq_len = 20

    model = TempMemLLM(
        num_nodes=num_nodes,
        d_model=128,
        memory_dim=128,
        llm_dim=768
    )

    # Create dummy inputs
    src = torch.randint(0, num_nodes, (batch_size,))
    dst = torch.randint(0, num_nodes, (batch_size,))
    timestamp = torch.rand(batch_size) * 1000
    src_neighbor_seq = torch.randint(0, num_nodes, (batch_size, seq_len))
    src_time_seq = torch.rand(batch_size, seq_len) * 1000
    dst_neighbor_seq = torch.randint(0, num_nodes, (batch_size, seq_len))
    dst_time_seq = torch.rand(batch_size, seq_len) * 1000
    neg_dst = torch.randint(0, num_nodes, (batch_size, 10))

    # Forward pass
    output = model(
        src, dst, timestamp,
        src_neighbor_seq, src_time_seq,
        dst_neighbor_seq, dst_time_seq,
        neg_dst=neg_dst
    )

    print(f"Positive scores shape: {output['pos_score'].shape}")
    print(f"Negative scores shape: {output['neg_score'].shape}")

    # Compute loss
    loss = model.compute_loss(
        output['pos_score'],
        output['neg_score']
    )
    print(f"Total loss: {loss['total_loss'].item():.4f}")

    print("Test passed!")
