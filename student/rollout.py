"""Student open-loop rollout — uses RSSM prior mean for prediction.

Key fix vs original:
  The warmup phase uses model.forward() (posterior) to build up an accurate
  hidden state h from ground-truth observations.  At the transition point we
  keep only (h, z_posterior) and hand it to model.predict(), which then uses
  the prior mean deterministically.  This preserves the information gathered
  during warmup instead of discarding it.
"""

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
    """Warmup with posterior, then predict open-loop with prior mean.

    Parameters
    ----------
    model        : StudentWorldModel instance.
    states       : (B, warmup_steps + horizon + 1, obs_dim) ground-truth obs.
    actions      : (B, warmup_steps + horizon, act_dim) actions.
    normalizer   : obs/action/delta normalizer.
    warmup_steps : number of steps to condition on ground-truth obs.
    horizon      : number of open-loop prediction steps.

    Returns
    -------
    preds : (B, horizon, obs_dim) predicted states in raw (un-normalised) space.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)

    # ── Warmup: use posterior (forward) to build accurate h ──────────────
    for t in range(int(warmup_steps)):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        _, hidden = model(obs_norm, act_norm, hidden)
        # Keep only (h, z) — discard dist params if returned as 6-tuple
        if isinstance(hidden, tuple) and len(hidden) > 2:
            hidden = (hidden[0], hidden[1])

    # ── Open-loop: use prior mean (predict) — no ground-truth obs ────────
    cur = states[:, int(warmup_steps)]
    preds = []

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
        cur   = cur + delta
        preds.append(cur)

    return torch.stack(preds, dim=1)   # (B, horizon, obs_dim)
