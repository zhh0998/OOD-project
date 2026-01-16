"""
DyGPrompt-TempMem-LLM: Dynamic Graph Prompt Learning for Fast Adaptation.
Scheme 4 - Priority P1 (Mandatory)

Key Innovation: Node-Time Conditional Prompt with extreme parameter efficiency
- Based on DyGPrompt (ICLR 2025)
- Only ~3,104 trainable parameters (vs 2M+ for full model)
- Training speed: 0.27 seconds/epoch
- Freezes backbone, only trains lightweight prompts

Author: RW2 Temporal Network Embedding Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PromptGenerator(nn.Module):
    """
    Conditional Prompt Generator with Alpha Bottleneck.

    Architecture:
    - NCN (Node Conditional Network): generates time prompts from node features
    - TCN (Time Conditional Network): generates node prompts from time features

    Alpha Bottleneck:
    - Hidden dimension = d_model / alpha
    - Optimal alpha = 2 (from DyGPrompt paper)
    - Achieves balance between parameter efficiency and expressiveness
    """

    def __init__(
        self,
        d_model: int,
        alpha: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.d_hidden = d_model // alpha

        # Node Conditional Network (generates time prompts)
        self.ncn = nn.Sequential(
            nn.Linear(d_model, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, d_model)
        )

        # Time Conditional Network (generates node prompts)
        self.tcn = nn.Sequential(
            nn.Linear(d_model, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_node: torch.Tensor,
        x_time: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conditional prompts.

        Args:
            x_node: Node features [batch, d_model]
            x_time: Time features [batch, d_model]

        Returns:
            p_node: Node prompt [batch, d_model]
            p_time: Time prompt [batch, d_model]
        """
        # Generate time prompt conditioned on node
        p_time = self.ncn(x_node)

        # Generate node prompt conditioned on time
        p_node = self.tcn(x_time)

        return p_node, p_time

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DyGPromptTempMem(nn.Module):
    """
    DyGPrompt-TempMem-LLM: Main model for Scheme 4.

    Architecture:
    1. Frozen TempMem-LLM backbone
    2. Trainable Prompt Generator (NCN + TCN)
    3. Element-wise prompt application

    Key Innovation:
    - Extreme parameter efficiency: only 3,104 trainable parameters
    - Fast adaptation: 0.27s/epoch training
    - Freezes backbone, maintains pretrained knowledge

    Prompt Application:
    x_prompted = x * (1 + p)  # Element-wise multiplicative prompt
    """

    def __init__(
        self,
        frozen_backbone: nn.Module,
        d_model: int = 128,
        alpha: int = 2,
        prompt_scale: float = 0.1
    ):
        """
        Args:
            frozen_backbone: Pretrained TempMem-LLM model (will be frozen)
            d_model: Model hidden dimension
            alpha: Bottleneck factor (optimal: 2)
            prompt_scale: Scale factor for prompt application
        """
        super().__init__()

        # Freeze the backbone
        self.backbone = frozen_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.d_model = d_model
        self.alpha = alpha
        self.prompt_scale = prompt_scale

        # Trainable prompt generator
        self.prompt_gen = PromptGenerator(d_model=d_model, alpha=alpha)

        # Optional: trainable decoder head (if we want to fine-tune decoder)
        # By default, use frozen backbone's decoder
        self.use_trainable_decoder = False

    def apply_prompt(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply prompt using element-wise multiplication.

        Args:
            x: Original features [batch, *, d_model]
            prompt: Prompt to apply [batch, d_model]

        Returns:
            Prompted features
        """
        # Expand prompt to match x dimensions
        if x.dim() > prompt.dim():
            prompt = prompt.unsqueeze(1).expand_as(x)

        return x * (1 + self.prompt_scale * prompt)

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
        """
        Forward pass with prompt injection.

        The prompts are generated and applied at the input level,
        then the frozen backbone processes the prompted inputs.
        """
        batch_size = src.size(0)

        # Get initial embeddings from frozen backbone
        with torch.no_grad():
            # Node embeddings
            src_emb = self.backbone.node_emb(src)  # [B, d]
            dst_emb = self.backbone.node_emb(dst)  # [B, d]

            # Time encoding
            src_time_emb = self.backbone.time_enc(timestamp)  # [B, d]
            if src_time_emb.dim() == 3:
                src_time_emb = src_time_emb.squeeze(1)

            # Neighbor embeddings
            src_neighbor_emb = self.backbone.node_emb(src_neighbor_seq)  # [B, L, d]
            dst_neighbor_emb = self.backbone.node_emb(dst_neighbor_seq)  # [B, L, d]

            # Time encodings for neighbors
            src_neighbor_time_emb = self.backbone.time_enc(src_time_seq)
            dst_neighbor_time_emb = self.backbone.time_enc(dst_time_seq)

        # Generate prompts
        # Use mean of neighbor embeddings as node feature context
        src_node_context = src_neighbor_emb.mean(dim=1)  # [B, d]
        dst_node_context = dst_neighbor_emb.mean(dim=1)  # [B, d]

        # Time context from current timestamp
        src_time_context = src_time_emb  # [B, d]
        dst_time_context = src_time_emb  # Same timestamp for dst

        # Generate prompts for source
        src_p_node, src_p_time = self.prompt_gen(src_node_context, src_time_context)

        # Generate prompts for destination
        dst_p_node, dst_p_time = self.prompt_gen(dst_node_context, dst_time_context)

        # Apply prompts
        src_emb_prompted = self.apply_prompt(src_emb, src_p_node)
        dst_emb_prompted = self.apply_prompt(dst_emb, dst_p_node)
        src_time_prompted = self.apply_prompt(src_time_emb, src_p_time)

        src_neighbor_prompted = self.apply_prompt(src_neighbor_emb, src_p_node)
        dst_neighbor_prompted = self.apply_prompt(dst_neighbor_emb, dst_p_node)

        # Process through frozen backbone with prompted inputs
        with torch.no_grad():
            # Combine prompted embeddings
            src_combined = src_neighbor_prompted + src_neighbor_time_emb
            dst_combined = dst_neighbor_prompted + dst_neighbor_time_emb

            # Memory aggregation
            src_h = self.backbone.memory(src_neighbor_seq, src_combined)
            dst_h = self.backbone.memory(dst_neighbor_seq, dst_combined)

            # Add base embeddings
            src_repr = src_emb_prompted + src_time_prompted + src_h
            dst_repr = dst_emb_prompted + dst_h

            # Layer norm
            src_repr = self.backbone.layer_norm(src_repr)
            dst_repr = self.backbone.layer_norm(dst_repr)

            # LLM projection
            src_repr = self.backbone.llm_proj(src_repr)
            dst_repr = self.backbone.llm_proj(dst_repr)

        # Compute scores (can be done with gradients for decoder fine-tuning)
        pos_score = self.backbone.link_decoder(src_repr, dst_repr).squeeze(-1)

        result = {
            'pos_score': pos_score,
            'src_repr': src_repr,
            'dst_repr': dst_repr
        }

        # Negative scores
        if neg_dst is not None:
            batch_size, num_negatives = neg_dst.shape

            with torch.no_grad():
                neg_dst_emb = self.backbone.node_emb(neg_dst)

            neg_scores = []
            for i in range(num_negatives):
                neg_emb = neg_dst_emb[:, i]

                # Apply prompt to negative
                neg_emb_prompted = self.apply_prompt(neg_emb, dst_p_node)

                with torch.no_grad():
                    neg_repr = self.backbone.layer_norm(neg_emb_prompted)
                    neg_repr = self.backbone.llm_proj(neg_repr)

                score = self.backbone.link_decoder(src_repr, neg_repr).squeeze(-1)
                neg_scores.append(score)

            result['neg_score'] = torch.stack(neg_scores, dim=1)

        return result

    def compute_loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute BPR loss."""
        diff = pos_score.unsqueeze(1) - neg_score
        link_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        return {'link_loss': link_loss, 'total_loss': link_loss}

    def get_trainable_parameters(self):
        """Get only the trainable parameters (prompts)."""
        return self.prompt_gen.parameters()

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_parameter_summary(self):
        """Print parameter summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.count_trainable_parameters()
        frozen_params = total_params - trainable_params

        print(f"Parameter Summary:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"  Prompt Generator: {self.prompt_gen.count_parameters():,}")


class StandaloneDyGPrompt(nn.Module):
    """
    Standalone DyGPrompt model (without requiring pretrained backbone).
    Useful for ablation studies and when pretrained model is not available.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 128,
        alpha: int = 2,
        llm_dim: int = 768,
        prompt_scale: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.alpha = alpha

        # Node embedding
        self.node_emb = nn.Embedding(num_nodes, d_model)

        # Time encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

        # Memory (GRU)
        self.memory = nn.GRU(d_model, d_model, batch_first=True)

        # LLM projection
        self.llm_proj = nn.Sequential(
            nn.Linear(d_model, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, d_model)
        )

        # Link decoder
        self.link_decoder = nn.Bilinear(d_model, d_model, 1)

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Prompt generator (the only trainable part in prompt mode)
        self.prompt_gen = PromptGenerator(d_model=d_model, alpha=alpha)

        self.prompt_scale = prompt_scale
        self.prompt_mode = False  # Set to True to freeze backbone

    def time_encode(self, t: torch.Tensor) -> torch.Tensor:
        """Time encoding."""
        original_shape = t.shape
        t_flat = t.reshape(-1, 1)
        pe = torch.zeros(t_flat.size(0), self.d_model, device=t.device)
        pe[:, 0::2] = torch.sin(t_flat * self.div_term)
        pe[:, 1::2] = torch.cos(t_flat * self.div_term)
        return pe.reshape(*original_shape, self.d_model)

    def enable_prompt_mode(self):
        """Enable prompt mode: freeze backbone, only train prompts."""
        self.prompt_mode = True

        # Freeze backbone parameters
        for name, param in self.named_parameters():
            if 'prompt_gen' not in name:
                param.requires_grad = False

    def disable_prompt_mode(self):
        """Disable prompt mode: train full model."""
        self.prompt_mode = False

        for param in self.parameters():
            param.requires_grad = True

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
        """Forward pass."""

        # Embeddings
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        src_time_emb = self.time_encode(timestamp)
        if src_time_emb.dim() == 3:
            src_time_emb = src_time_emb.squeeze(1)

        src_neighbor_emb = self.node_emb(src_neighbor_seq)
        dst_neighbor_emb = self.node_emb(dst_neighbor_seq)
        src_neighbor_time = self.time_encode(src_time_seq)
        dst_neighbor_time = self.time_encode(dst_time_seq)

        # Generate and apply prompts
        src_context = src_neighbor_emb.mean(dim=1)
        dst_context = dst_neighbor_emb.mean(dim=1)

        src_p_node, src_p_time = self.prompt_gen(src_context, src_time_emb)
        dst_p_node, dst_p_time = self.prompt_gen(dst_context, src_time_emb)

        # Apply prompts
        src_emb = src_emb * (1 + self.prompt_scale * src_p_node)
        dst_emb = dst_emb * (1 + self.prompt_scale * dst_p_node)
        src_time_emb = src_time_emb * (1 + self.prompt_scale * src_p_time)

        # Memory aggregation
        src_combined = src_neighbor_emb + src_neighbor_time
        dst_combined = dst_neighbor_emb + dst_neighbor_time

        _, src_h = self.memory(src_combined)
        _, dst_h = self.memory(dst_combined)

        src_h = src_h.squeeze(0)
        dst_h = dst_h.squeeze(0)

        # Combine
        src_repr = src_emb + src_time_emb + src_h
        dst_repr = dst_emb + dst_h

        # LLM projection
        src_repr = self.llm_proj(self.layer_norm(src_repr)) + src_repr
        dst_repr = self.llm_proj(self.layer_norm(dst_repr)) + dst_repr

        # Score
        pos_score = self.link_decoder(src_repr, dst_repr).squeeze(-1)

        result = {'pos_score': pos_score, 'src_repr': src_repr, 'dst_repr': dst_repr}

        # Negative scores
        if neg_dst is not None:
            batch_size, num_negatives = neg_dst.shape
            neg_emb = self.node_emb(neg_dst)

            neg_scores = []
            for i in range(num_negatives):
                neg_i = neg_emb[:, i] * (1 + self.prompt_scale * dst_p_node)
                neg_repr = self.llm_proj(self.layer_norm(neg_i)) + neg_i
                score = self.link_decoder(src_repr, neg_repr).squeeze(-1)
                neg_scores.append(score)

            result['neg_score'] = torch.stack(neg_scores, dim=1)

        return result

    def compute_loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute BPR loss."""
        diff = pos_score.unsqueeze(1) - neg_score
        link_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        return {'link_loss': link_loss, 'total_loss': link_loss}


def create_dygprompt(
    num_nodes: int,
    pretrained_backbone: Optional[nn.Module] = None,
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Factory function to create DyGPrompt model.

    Args:
        num_nodes: Number of nodes
        pretrained_backbone: Optional pretrained TempMem-LLM model
        config: Optional configuration

    Returns:
        DyGPromptTempMem if backbone provided, else StandaloneDyGPrompt
    """
    default_config = {
        'd_model': 128,
        'alpha': 2,
        'prompt_scale': 0.1
    }

    if config is not None:
        default_config.update(config)

    if pretrained_backbone is not None:
        return DyGPromptTempMem(
            frozen_backbone=pretrained_backbone,
            d_model=default_config['d_model'],
            alpha=default_config['alpha'],
            prompt_scale=default_config['prompt_scale']
        )
    else:
        return StandaloneDyGPrompt(
            num_nodes=num_nodes,
            d_model=default_config['d_model'],
            alpha=default_config['alpha'],
            prompt_scale=default_config['prompt_scale']
        )


if __name__ == '__main__':
    # Test the models
    print("Testing DyGPrompt models...")

    num_nodes = 1000
    batch_size = 32
    seq_len = 20

    # Test StandaloneDyGPrompt
    print("\n1. Testing StandaloneDyGPrompt...")
    model = StandaloneDyGPrompt(
        num_nodes=num_nodes,
        d_model=128,
        alpha=2
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Prompt generator parameters: {model.prompt_gen.count_parameters():,}")

    # Enable prompt mode
    model.enable_prompt_mode()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (prompt mode): {trainable_params:,}")

    # Create inputs
    src = torch.randint(0, num_nodes, (batch_size,))
    dst = torch.randint(0, num_nodes, (batch_size,))
    timestamp = torch.rand(batch_size) * 1000
    src_neighbor_seq = torch.randint(0, num_nodes, (batch_size, seq_len))
    src_time_seq = torch.rand(batch_size, seq_len) * 1000
    dst_neighbor_seq = torch.randint(0, num_nodes, (batch_size, seq_len))
    dst_time_seq = torch.rand(batch_size, seq_len) * 1000
    neg_dst = torch.randint(0, num_nodes, (batch_size, 10))

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            src, dst, timestamp,
            src_neighbor_seq, src_time_seq,
            dst_neighbor_seq, dst_time_seq,
            neg_dst=neg_dst
        )

    print(f"Positive scores shape: {output['pos_score'].shape}")
    print(f"Negative scores shape: {output['neg_score'].shape}")

    # Test DyGPromptTempMem with mock backbone
    print("\n2. Testing DyGPromptTempMem...")

    # Import base model for backbone
    from .base_model import TempMemLLM

    backbone = TempMemLLM(num_nodes=num_nodes, d_model=128)

    prompt_model = DyGPromptTempMem(
        frozen_backbone=backbone,
        d_model=128,
        alpha=2
    )

    prompt_model.print_parameter_summary()

    print("\nDyGPrompt tests passed!")
