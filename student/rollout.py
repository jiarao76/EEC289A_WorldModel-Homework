"""Student open-loop rollout — uses RSSM prior for prediction."""

from __future__ import annotations

import torch


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

    # Warmup: use forward() which uses posterior
    for t in range(int(warmup_steps)):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        _, hidden = model(obs_norm, act_norm, hidden)
        # Keep only (h, z) if hidden is a tuple with more than 2 elements
        if isinstance(hidden, tuple) and len(hidden) > 2:
            hidden = (hidden[0], hidden[1])

    cur = states[:, int(warmup_steps)]
    preds = []

    use_prior = hasattr(model, 'predict')

    for h in range(int(horizon)):
        obs_norm = normalizer.normalize_obs(cur)
        act_norm = normalizer.normalize_act(
            actions[:, int(warmup_steps) + h]
        )
        if use_prior:
            delta_norm, hidden = model.predict(obs_norm, act_norm, hidden)
        else:
            delta_
