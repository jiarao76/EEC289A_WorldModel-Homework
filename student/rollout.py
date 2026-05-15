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
        # Compact hidden to (h, z) if RSSM returns full 6-tuple
        if isinstance(hidden, tuple) and len(hidden) > 2:
            hidden = (hidden[0], hidden[1])

    cur = states[:, int(warmup_steps)]
    preds = []

    # Use predict() for RSSM prior path, fallback to forward() otherwise
    predict_fn = getattr(model, 'predict', None)

    for step in range(int(horizon)):
        obs_norm = normalizer.normalize_obs(cur)
        act_norm = normalizer.normalize_act(
            actions[:, int(warmup_steps) + step]
        )
        if predict_fn is not None:
            delta_norm, hidden = predict_fn(obs_norm, act_norm, hidden)
        else:
            delta_norm, hidden = model(obs_norm, act_norm, hidden)
            if isinstance(hidden, tuple) and len(hidden) > 2:
                hidden = (hidden[0], hidden[1])

        delta = normalizer.denormalize_delta(delta_norm)
        cur = cur + delta
        preds.append(cur)

    return torch.stack(preds, dim=1)
