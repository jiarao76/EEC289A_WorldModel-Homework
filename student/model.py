"""Student world model - back to the best working version (Experiment 1 style).

Simple residual MLP with LayerNorm. No physics assumptions.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * 2)
        self.act  = nn.SiLU()
        self.fc2  = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gru: bool = False,
        delta_limit: float = 0.5,
    ):
        super().__init__()
        self.use_gru    = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.hidden_dim  = int(hidden_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(int(num_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.trunk(self.input_proj(torch.cat([obs_norm, act_norm], dim=-1)))
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw = self.head(self.out_norm(feat))
        return self.delta_limit * torch.tanh(raw / self.delta_limit), hidden
