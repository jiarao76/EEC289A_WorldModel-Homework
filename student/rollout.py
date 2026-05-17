"""Student open-loop rollout."""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(model, states, actions, normalizer, warmup_steps, horizon, teacher_forcing_ratio=0.0):
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        cur, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)
