"""Student world model — RSSM (Recurrent State Space Model).

Architecture
------------
* Deterministic path: GRUCell captures temporal history.
* Stochastic path: learned prior p(z|h) and posterior q(z|h,o).
* At training time, posterior samples z from (h, obs) via reparameterization.
* At prediction time, prior samples z from h only — no ground-truth obs needed.
* KL divergence between posterior and prior is minimized during training.
* Decoder predicts obs delta from (h, z).
"""

from __future__ import annotations

import torch
from torch import nn


class _ResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.act   = nn.SiLU()
        self.norm2 = nn.LayerNorm(dim)
        self.fc2   = nn.Linear(dim, dim)
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.act(self.norm1(x)))
        h = self.drop(h)
        h = self.fc2(self.act(self.norm2(h)))
        return x + h


class StudentWorldModel(nn.Module):
    """RSSM world model for InvertedPendulum-v5.

    Parameters
    ----------
    obs_dim     : observation dimension (4).
    act_dim     : action dimension (1).
    hidden_dim  : width of GRU and MLP layers.
    num_layers  : number of residual blocks (num_layers - 1, min 1).
    use_gru     : must be True for RSSM; kept for interface compatibility.
    delta_limit : soft clamp via tanh in normalised delta space.
    dropout     : dropout inside residual blocks.
    latent_dim  : dimension of stochastic latent z (default hidden_dim // 4).
    """

    def __init__(
        self,
        obs_dim:     int   = 4,
        act_dim:     int   = 1,
        hidden_dim:  int   = 256,
        num_layers:  int   = 3,
        use_gru:     bool  = True,
        delta_limit: float = 5.0,
        dropout:     float = 0.02,
        latent_dim:  int   = None,
    ) -> None:
        super().__init__()
        self.obs_dim     = int(obs_dim)
        self.act_dim     = int(act_dim)
        self.hidden_dim  = int(hidden_dim)
        self.delta_limit = float(delta_limit)
        self.latent_dim  = int(latent_dim) if latent_dim else int(hidden_dim) // 4

        n_res = max(1, int(num_layers) - 1)
        D = self.hidden_dim
        L = self.latent_dim

        # ── 1. Input projection (obs + action → hidden) ──────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, D),
            nn.SiLU(),
        )

        # ── 2. Residual encoder ───────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[_ResBlock(D, dropout=float(dropout)) for _ in range(n_res)]
        )

        # ── 3. Deterministic GRU ──────────────────────────────────────────
        # Input: encoded features (D) + previous latent z (L)
        self.gru = nn.GRUCell(D + L, D)

        # ── 4. Prior p(z | h): only uses GRU hidden state ────────────────
        self.prior_net = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, L * 2),   # → mean, log_std
        )

        # ── 5. Posterior q(z | h, o): uses h + encoded obs ───────────────
        # Encodes obs separately to compute posterior
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, D // 2),
        )
        self.posterior_net = nn.Sequential(
            nn.LayerNorm(D + D // 2),
            nn.Linear(D + D // 2, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, L * 2),   # → mean, log_std
        )

        # ── 6. Decoder: (h, z) → delta ───────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(D + L),
            nn.Linear(D + L, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, obs_dim),
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def initial_hidden(self, batch_size: int, device: torch.device):
        """Return (h, z) tuple as the initial hidden state."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return (h, z)

    def _dist_params(self, raw: torch.Tensor):
        """Split raw output into mean and clamped log_std."""
        mean, log_std = raw.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -4.0, 4.0)
        return mean, log_std

    def _sample(self, mean, log_std):
        """Reparameterisation trick."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + std * eps

    def _kl(self, post_mean, post_log_std, prior_mean, prior_log_std):
        """KL(posterior || prior) in closed form."""
        post_var  = torch.exp(2 * post_log_std)
        prior_var = torch.exp(2 * prior_log_std)
        kl = (
            prior_log_std - post_log_std
            + (post_var + (post_mean - prior_mean) ** 2) / (2 * prior_var)
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    # ── Forward (training mode: uses posterior) ───────────────────────────

    def forward(
        self,
        obs_norm: torch.Tensor,
        act_norm: torch.Tensor,
        hidden=None,
    ):
        """One-step prediction using posterior (training).

        Returns
        -------
        delta_norm : (B, obs_dim) predicted delta in normalised space.
        hidden     : updated (h, z) tuple.
        """
        if hidden is None:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
        h, z_prev = hidden

        # Encode (obs, action)
        x    = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.res_blocks(self.input_proj(x))

        # GRU update: input = encoded features + previous z
        h_new = self.gru(torch.cat([feat, z_prev], dim=-1), h)

        # Prior p(z | h_new)
        prior_raw               = self.prior_net(h_new)
        prior_mean, prior_log_std = self._dist_params(prior_raw)

        # Posterior q(z | h_new, obs)
        obs_feat                    = self.obs_encoder(obs_norm)
        post_raw                    = self.posterior_net(
            torch.cat([h_new, obs_feat], dim=-1)
        )
        post_mean, post_log_std     = self._dist_params(post_raw)

        # Sample z from posterior (reparameterisation)
        z_new = self._sample(post_mean, post_log_std)

        # Decode
        raw_delta  = self.head(torch.cat([h_new, z_new], dim=-1))
        delta_norm = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        # Pack extra info needed by loss into hidden tuple
        hidden_new = (h_new, z_new, prior_mean, prior_log_std,
                      post_mean, post_log_std)
        return delta_norm, hidden_new

    # ── Predict (inference mode: uses prior only) ─────────────────────────

    def predict(
        self,
        obs_norm: torch.Tensor,
        act_norm: torch.Tensor,
        hidden=None,
    ):
        """One-step prediction using prior only (open-loop inference).

        Returns
        -------
        delta_norm : (B, obs_dim) predicted delta.
        hidden     : updated (h, z) tuple (compact, no dist params).
        """
        if hidden is None:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)

        # Support both compact (h, z) and full training tuple
        h, z_prev = hidden[0], hidden[1]

        x    = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.res_blocks(self.input_proj(x))

        h_new = self.gru(torch.cat([feat, z_prev], dim=-1), h)

        # Prior only — no obs needed
        prior_raw             = self.prior_net(h_new)
        prior_mean, prior_log_std = self._dist_params(prior_raw)
        z_new                 = self._sample(prior_mean, prior_log_std)

        raw_delta  = self.head(torch.cat([h_new, z_new], dim=-1))
        delta_norm = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        return delta_norm, (h_new, z_new)
