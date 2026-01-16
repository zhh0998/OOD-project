"""
Time Encoding utilities for temporal network embedding.
Implements various time encoding strategies from the literature.

Author: RW2 Temporal Network Embedding Project
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TimeEncoding(nn.Module):
    """
    Fourier-based time encoding similar to positional encoding.
    Uses sinusoidal functions to encode continuous timestamps.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create the frequency divisors
        # div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps into d_model dimensional vectors.

        Args:
            t: Tensor of timestamps [batch_size] or [batch_size, seq_len]

        Returns:
            Time encodings of shape [..., d_model]
        """
        # Ensure t has at least 1 dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)

        original_shape = t.shape

        # Flatten for encoding
        t_flat = t.reshape(-1, 1)  # [N, 1]

        # Compute sinusoidal encoding
        pe = torch.zeros(t_flat.size(0), self.d_model, device=t.device)
        pe[:, 0::2] = torch.sin(t_flat * self.div_term)
        pe[:, 1::2] = torch.cos(t_flat * self.div_term)

        # Reshape to original batch shape
        pe = pe.reshape(*original_shape, self.d_model)

        return self.dropout(pe)


class LearnableTimeEncoding(nn.Module):
    """
    Learnable time encoding with basis functions.
    Combines fixed Fourier basis with learned linear transformations.
    """

    def __init__(
        self,
        d_model: int,
        n_basis: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_basis = n_basis

        # Fixed Fourier basis
        self.register_buffer(
            'basis_freq',
            torch.linspace(0, np.pi, n_basis).unsqueeze(0)  # [1, n_basis]
        )

        # Learnable projection
        self.proj = nn.Sequential(
            nn.Linear(n_basis * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps with learned basis functions.

        Args:
            t: Tensor of timestamps

        Returns:
            Time encodings of shape [..., d_model]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        original_shape = t.shape
        t_flat = t.reshape(-1, 1)  # [N, 1]

        # Compute basis functions
        angles = t_flat * self.basis_freq  # [N, n_basis]
        basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, 2*n_basis]

        # Project to d_model
        encoding = self.proj(basis)  # [N, d_model]

        return encoding.reshape(*original_shape, self.d_model)


class ContinuousTimeEncoding(nn.Module):
    """
    Time encoding specifically designed for continuous-time dynamic graphs.
    Uses time differences and exponential decay for temporal attention.
    """

    def __init__(
        self,
        d_model: int,
        decay_init: float = 1.0,
        learnable_decay: bool = True
    ):
        super().__init__()
        self.d_model = d_model

        # Time decay parameter
        if learnable_decay:
            self.decay = nn.Parameter(torch.tensor(decay_init))
        else:
            self.register_buffer('decay', torch.tensor(decay_init))

        # Time projection
        self.time_proj = nn.Linear(1, d_model)

        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(
        self,
        t: torch.Tensor,
        t_ref: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode timestamps with optional reference time.

        Args:
            t: Current timestamps
            t_ref: Reference timestamps (for computing time differences)

        Returns:
            Time encodings
        """
        if t_ref is None:
            # Use positional encoding style
            t_enc = self.time_proj(t.unsqueeze(-1))
            return t_enc

        # Compute time difference
        dt = t_ref - t
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        # Exponential decay
        decay_weight = torch.exp(-self.decay * torch.abs(dt))

        # Project time difference
        dt_enc = self.time_proj(dt)

        # Combine with decay
        combined = torch.cat([dt_enc, decay_weight.expand_as(dt_enc)], dim=-1)
        return self.combine(combined)


class RelativeTimeEncoding(nn.Module):
    """
    Relative time encoding for temporal sequences.
    Encodes the time difference between consecutive events.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Log-linear time encoding (handles varying time scales)
        self.log_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Dropout(dropout)
        )

        # Sinusoidal component
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        Encode time differences.

        Args:
            time_diffs: Time differences [...] (can be any shape)

        Returns:
            Time encodings [..., d_model]
        """
        original_shape = time_diffs.shape
        dt = time_diffs.reshape(-1, 1)

        # Handle zero and negative time differences
        dt_safe = torch.clamp(torch.abs(dt), min=1e-6)

        # Log-linear component (good for varying scales)
        log_dt = torch.log(dt_safe + 1.0)
        log_enc = self.log_proj(log_dt)

        # Sinusoidal component
        sin_enc = torch.zeros(dt.size(0), self.d_model, device=dt.device)
        sin_enc[:, 0::2] = torch.sin(dt * self.div_term)
        sin_enc[:, 1::2] = torch.cos(dt * self.div_term)

        # Combine
        encoding = log_enc + sin_enc
        return encoding.reshape(*original_shape, self.d_model)


if __name__ == '__main__':
    # Test time encodings
    print("Testing time encodings...")

    d_model = 128
    batch_size = 32
    seq_len = 10

    # Test TimeEncoding
    te = TimeEncoding(d_model)
    t = torch.rand(batch_size, seq_len) * 1000
    enc = te(t)
    print(f"TimeEncoding: {t.shape} -> {enc.shape}")

    # Test LearnableTimeEncoding
    lte = LearnableTimeEncoding(d_model)
    enc = lte(t)
    print(f"LearnableTimeEncoding: {t.shape} -> {enc.shape}")

    # Test RelativeTimeEncoding
    rte = RelativeTimeEncoding(d_model)
    dt = t[:, 1:] - t[:, :-1]
    enc = rte(dt)
    print(f"RelativeTimeEncoding: {dt.shape} -> {enc.shape}")

    print("All tests passed!")
