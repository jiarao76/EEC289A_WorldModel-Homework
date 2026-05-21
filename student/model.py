"""Student world model — GRU + deep residual MLP, with EnsembleModel wrapper.

EnsembleModel wraps N independently-trained StudentWorldModel instances.
At inference, it averages the predicted delta across all members, which
reduces variance and prevents individual model failures from dominating.

The public interface (initial_hidden / forward) is identical for both
StudentWorldModel and EnsembleModel, so official_rollout.py works unchanged.
"""

from __future__ import annotations
import torch
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
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
    """GRU + deep residual MLP world model."""

    def __init__(
        self,
        obs_dim:     int   = 4,
        act_dim:     int   = 1,
        hidden_dim:  int   = 256,
        num_layers:  int   = 3,
        use_gru:     bool  = True,
        delta_limit: float = 3.0,
    ) -> None:
        super().__init__()
        self.use_gru     = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self._hidden_dim = int(hidden_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList(
            [_ResBlock(hidden_dim) for _ in range(int(num_layers))]
        )
        self.gru      = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head     = nn.Linear(hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self._hidden_dim, device=device)

    def forward(self, obs_norm, act_norm, hidden=None):
        x = self.input_proj(torch.cat([obs_norm, act_norm], dim=-1))
        for block in self.res_blocks:
            x = block(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(x, hidden)
            x = hidden
        raw   = self.head(self.out_norm(x))
        delta = self.delta_limit * torch.tanh(raw / self.delta_limit)
        return delta, hidden


class EnsembleModel:
    """Wraps N StudentWorldModel instances; averages their delta predictions.

    Satisfies the same interface as StudentWorldModel so it can be passed
    directly to official_open_loop_rollout and predict_next without any
    changes to locked files.

    hidden = list of per-member hidden states, one per member model.
    """

    def __init__(self, members: list[StudentWorldModel]) -> None:
        assert len(members) > 0
        self.members = members

    def initial_hidden(self, batch_size: int, device: torch.device):
        return [m.initial_hidden(batch_size, device) for m in self.members]

    def eval(self):
        for m in self.members:
            m.eval()
        return self

    def train(self, mode: bool = True):
        for m in self.members:
            m.train(mode)
        return self

    def __call__(self, obs_norm, act_norm, hidden=None):
        if hidden is None:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)

        deltas     = []
        new_hidden = []
        for i, member in enumerate(self.members):
            d, h = member(obs_norm, act_norm, hidden[i])
            deltas.append(d)
            new_hidden.append(h)

        # average delta across ensemble members
        delta_mean = torch.stack(deltas, dim=0).mean(dim=0)
        return delta_mean, new_hidden
