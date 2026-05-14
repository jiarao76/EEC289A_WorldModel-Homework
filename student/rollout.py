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

    Future ground-truth states after `warmup_steps` must not be read
    during evaluation (teacher_forcing_ratio=0.0).

    During training, teacher_forcing_ratio > 0 allows occasional
    replacement of predicted states with ground-truth states, which
    stabilizes learning of long-horizon dependencies (scheduled sampling).
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)

    # Warmup: feed ground-truth states to build up GRU hidden state
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(
            model, states[:, t], actions[:, t], hidden, normalizer
        )

    cur = states[:, int(warmup_steps)]
    preds = []

    for h in range(int(horizon)):
        cur, hidden = predict_next(
            model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer
        )
        preds.append(cur)

        # Scheduled sampling: occasionally replace prediction with ground truth
        if teacher_forcing_ratio > 0.0:
            next_idx = int(warmup_steps) + h + 1
            if next_idx < states.shape[1]:
                use_truth = torch.rand(1, device=states.device).item() < teacher_forcing_ratio
                if use_truth:
                    cur = states[:, next_idx]

    return torch.stack(preds, dim=1)
