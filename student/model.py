"""Student world model - proper ResNet + GRU."""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class StudentWorldModel(nn.Module):
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
        self.use_gru = True
        self.delta_limit = float(delta_limit)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(int(num_layers))]
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.decode_res = ResBlock(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        if hidden is None:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
        x = self.input_norm(self.input_proj(torch.cat([obs_norm, act_norm], dim=-1)))
        for block in self.res_blocks:
            x = block(x)
        h = self.gru(x, hidden)
        out = self.decode_res(h)
        raw_delta = self.head(self.out_norm(out))
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, h
