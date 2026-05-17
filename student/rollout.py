"""Student open-loop rollout implementation."""

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
    teacher_forcing_ratio: float = 0.0,
):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.

    Args:
        teacher_forcing_ratio: probability of using ground-truth state instead
            of model prediction at each step. 1.0 = always use ground truth
            (pure teacher forcing), 0.0 = pure open-loop. Scheduled Sampling
            anneals this from high to low during training.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        pred, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(pred)
        # Scheduled sampling: mix ground truth and model prediction
        gt_idx = int(warmup_steps) + 1 + h
        if teacher_forcing_ratio > 0.0 and gt_idx < states.shape[1]:
            use_gt = torch.rand((), device=states.device) < teacher_forcing_ratio
            cur = states[:, gt_idx] if use_gt else pred
        else:
            cur = pred
    return torch.stack(preds, dim=1)
