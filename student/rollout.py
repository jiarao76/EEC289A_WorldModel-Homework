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
        # Keep only (h, z) for next step
        hidden = (hidden[0], hidden[1])

    cur = states[:, int(warmup_steps)]
    preds = []

    # Open-loop: use prior if available, else fallback to forward()
    use_prior = hasattr(model, 'predict')

    for h in range(int(horizon)):
        obs_norm = normalizer.normalize_obs(cur)
        act_norm = normalizer.normalize_act(
            actions[:, int(warmup_steps) + h]
        )
        if use_prior:
            delta_norm, hidden = model.predict(obs_norm, act_norm, hidden)
        else:
            delta_norm, hidden = model(obs_norm, act_norm, hidden)
            hidden = (hidden[0], hidden[1]) if isinstance(hidden, tuple) else hidden

        delta = normalizer.unnormalize_delta(delta_norm)
        cur = cur + delta
        preds.append(cur)

    return torch.stack(preds, dim=1)
