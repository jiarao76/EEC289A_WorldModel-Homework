"""Student world model — improved GRU-based dynamics predictor.

Architecture highlights
-----------------------
* Deep encoder with residual blocks and LayerNorm for training stability.
* GRU cell to capture temporal correlations across rollout steps.
* Separate decoder head with an additional hidden layer.
* Soft delta clamping via tanh keeps predictions in a plausible range.
* All extra hyper-parameters have safe defaults so the locked build_model()
  call (which only forwards hidden_dim / num_layers / use_gru) still works.
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Two-layer pre-activation residual block with LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.act(self.norm1(x)))
        h = self.drop(h)
        h = self.fc2(self.act(self.norm2(h)))
        return x + h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StudentWorldModel(nn.Module):
    """Improved GRU world model for InvertedPendulum-v5.

    Parameters
    ----------
    obs_dim:        observation dimension (4 for InvertedPendulum-v5).
    act_dim:        action dimension (1).
    hidden_dim:     width of every hidden layer.
    num_layers:     controls the number of residual blocks (num_layers - 1,
                    minimum 1).  Matches the build_model() kwarg name.
    use_gru:        whether to use a GRU cell for temporal memory.
    delta_limit:    soft clamp magnitude via tanh (normalised delta space).
    dropout:        dropout probability inside residual blocks (0 = off).
    """

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 5.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        in_dim = obs_dim + act_dim
        # number of residual blocks: at least 1, scales with num_layers
        n_res = max(1, int(num_layers) - 1)

        # ---- input projection ----
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        # ---- residual encoder ----
        self.res_blocks = nn.Sequential(
            *[_ResBlock(hidden_dim, dropout=float(dropout)) for _ in range(n_res)]
        )

        # ---- temporal memory ----
        if self.use_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None  # type: ignore[assignment]

        # ---- decoder head ----
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, obs_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_norm: torch.Tensor,
        act_norm: torch.Tensor,
        hidden=None,
    ):
        """Predict normalised delta given normalised obs + action.

        Returns
        -------
        delta_norm : (B, obs_dim) predicted state delta in normalised space.
        hidden     : updated GRU hidden state (or None if use_gru=False).
        """
        x = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.input_proj(x)
        feat = self.res_blocks(feat)

        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden

        raw_delta = self.head(feat)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
