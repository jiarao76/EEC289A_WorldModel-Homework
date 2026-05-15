"""Student open-loop rollout — uses RSSM prior for prediction."""

from __future__ import annotations

import torch
from wm_hw.model_utils import predict_next


def open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
):
    """Warmup with posterior, then predict open-loop with prior."""
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)

    # Warmup: use forward() which uses posterior (has access to true obs)
    for t in range(int(warmup_steps)):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        _, hidden = model(obs_norm, act_norm, hidden)

    cur = states[:, int(warmup_steps)]
    preds = []

    # Open-loop: use predict() which uses prior only
    for h in range(int(horizon)):
        obs_norm = normalizer.normalize_obs(cur)
        act_norm = normalizer.normalize_act(
            actions[:, int(warmup_steps) + h]
        )
        delta_norm, hidden = model.predict(obs_norm, act_norm, hidden)
        delta = normalizer.unnormalize_delta(delta_norm)
        cur = cur + delta
        preds.append(cur)

    return torch.stack(preds, dim=1)
