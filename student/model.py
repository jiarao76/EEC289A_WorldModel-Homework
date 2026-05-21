"""Student world model — improved GRU + residual MLP with LayerNorm.

Architecture:
  Input projection  : Linear(obs_dim + act_dim → hidden_dim)
  Residual blocks   : N × (LayerNorm → Linear → SiLU → Linear) + skip
  GRU cell          : optional, wraps the encoded feature for temporal memory
  Output head       : Linear(hidden_dim → obs_dim) + tanh clamp

The public interface (initial_hidden / forward) is unchanged so all locked
tests, checkpoint utilities, and eval scripts remain compatible.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → SiLU → Linear → add."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 2, dim)
        # zero-init last layer so residual starts as identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class StudentWorldModel(nn.Module):
    """GRU-enhanced residual world model for InvertedPendulum-v5.

    Args:
        obs_dim:     Observation dimension (locked at 4).
        act_dim:     Action dimension (locked at 1).
        hidden_dim:  Width of all hidden layers.
        num_layers:  Number of residual blocks before the GRU.
        use_gru:     If True, adds a GRUCell for temporal memory.
        delta_limit: tanh output is scaled to [-delta_limit, delta_limit].
    """

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self._hidden_dim = int(hidden_dim)

        # Project concatenated (obs, act) into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
        )

        # Stack of pre-norm residual blocks
        self.res_blocks = nn.ModuleList(
            [_ResBlock(hidden_dim) for _ in range(int(num_layers))]
        )

        # Optional GRU for temporal context
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Output LayerNorm before head for training stability
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, obs_dim)

    # ------------------------------------------------------------------
    # Locked public interface
    # ------------------------------------------------------------------

    def initial_hidden(self, batch_size: int, device: torch.device):
        """Return zero GRU state, or None when use_gru=False."""
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self._hidden_dim, device=device)

    def forward(
        self,
        obs_norm: torch.Tensor,
        act_norm: torch.Tensor,
        hidden=None,
    ):
        """Predict normalised delta and update hidden state.

        Args:
            obs_norm: (B, obs_dim) normalised observation.
            act_norm: (B, act_dim) normalised action.
            hidden:   GRU hidden state tensor or None.

        Returns:
            delta_norm: (B, obs_dim) predicted normalised delta.
            hidden:     Updated hidden state (None if use_gru=False).
        """
        x = self.input_proj(torch.cat([obs_norm, act_norm], dim=-1))

        for block in self.res_blocks:
            x = block(x)

        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(x, hidden)
            x = hidden

        raw_delta = self.head(self.out_norm(x))
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
