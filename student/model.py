"""Student world model with physics-informed architecture.

InvertedPendulum state: [x, theta, x_dot, theta_dot]

Key insight: position deltas are approximately linear in velocities:
    dx      ≈ x_dot * dt
    dtheta  ≈ theta_dot * dt

So the model only needs to accurately predict velocity changes (dv),
and position changes follow from the current velocities. This halves
the effective learning problem and eliminates a major source of drift.

Architecture:
    1. Velocity head: MLP predicts [dx_dot, dtheta_dot] (the hard part)
    2. Position delta: computed from current velocities * learned dt scale
    3. Residual correction: small MLP corrects both (nonlinear effects)
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
        self.obs_dim     = int(obs_dim)
        self.delta_limit = float(delta_limit)
        self.use_gru     = bool(use_gru)
        self.hidden_dim  = int(hidden_dim)

        # --- shared encoder ---
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(int(num_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

        # --- velocity head: predicts delta for [x_dot, theta_dot] ---
        # This is the physically meaningful part the network must learn
        self.vel_head = nn.Linear(hidden_dim, 2)   # [dx_dot, dtheta_dot]
        nn.init.zeros_(self.vel_head.weight)
        nn.init.zeros_(self.vel_head.bias)

        # --- learned dt scale for position integration ---
        # pos_delta ≈ vel * dt; initialise dt_scale near 1.0
        self.dt_scale = nn.Parameter(torch.ones(2))  # [scale_x, scale_theta]

        # --- residual correction for all 4 dims (nonlinear effects) ---
        self.res_head = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

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

        feat_out = self.out_norm(feat)

        # Velocity delta (the hard part: nonlinear dynamics)
        raw_vel_delta = self.vel_head(feat_out)
        vel_delta = self.delta_limit * torch.tanh(raw_vel_delta / self.delta_limit)

        # Position delta from current velocities (physics prior)
        # obs_norm[:, 2:4] are normalised [x_dot, theta_dot]
        # We use the normalised velocities directly as a proxy for integration
        pos_delta_phys = obs_norm[:, 2:4] * self.dt_scale.unsqueeze(0)
        pos_delta_phys = self.delta_limit * torch.tanh(pos_delta_phys / self.delta_limit)

        # Combine: [pos_delta, vel_delta] + small residual correction
        physics_delta = torch.cat([pos_delta_phys, vel_delta], dim=-1)
        residual      = self.delta_limit * torch.tanh(self.res_head(feat_out) / self.delta_limit)

        delta = physics_delta + 0.1 * residual   # residual is small correction
        return delta, hidden
