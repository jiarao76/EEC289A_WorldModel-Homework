"""Student one-step plus rollout loss with curriculum rollout horizon.

train.py calls:  compute_loss(model, batch, normalizer, cfg)
The `step` parameter defaults to 0, so curriculum is driven by an internal
counter that increments each call — no changes to train.py required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

# Internal step counter — persists across calls within one training run.
_call_count: int = 0


def _reset_curriculum() -> None:
    """Call at the start of training to reset the curriculum counter."""
    global _call_count
    _call_count = 0


def one_step_delta_loss(
    model, states: torch.Tensor, actions: torch.Tensor, normalizer
) -> torch.Tensor:
    """Single-step delta prediction loss in normalised space."""
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Open-loop rollout loss with later-step emphasis weighting."""
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    start = (
        int(torch.randint(0, max_start + 1, (), device=states.device).item())
        if max_start > 0
        else 0
    )
    sub_states  = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # Later steps weighted more heavily — encourages long-horizon stability.
    # weight[h] = 1 + h / H  →  range [1, 2], mean ≈ 1.5
    H = preds.shape[1]
    step_idx = torch.arange(H, dtype=preds.dtype, device=preds.device)
    weights  = 1.0 + step_idx / max(H, 1)
    weights  = weights / weights.mean()                        # keep overall scale
    sq_err   = (pred_norm - target_norm) ** 2                 # (B, H, obs_dim)
    return (sq_err * weights.unsqueeze(0).unsqueeze(-1)).mean()


def compute_loss(
    model,
    batch: dict[str, torch.Tensor],
    normalizer,
    cfg: dict,
    step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine one-step and rollout losses.

    Curriculum: rollout horizon grows linearly from `rollout_horizon_start`
    to `rollout_train_horizon` over `rollout_curriculum_steps` calls.
    When called without `step` (as train.py does), an internal counter drives
    the curriculum automatically.
    """
    global _call_count

    loss_cfg = cfg["loss"]
    states   = batch["states"]
    actions  = batch["actions"]

    # --- curriculum step resolution ---
    if step is None:
        cur_step = _call_count
        _call_count += 1
    else:
        cur_step = int(step)

    horizon_max      = int(loss_cfg.get("rollout_train_horizon", 30))
    horizon_start    = int(loss_cfg.get("rollout_horizon_start", 5))
    curriculum_steps = int(loss_cfg.get("rollout_curriculum_steps", 1500))

    if curriculum_steps > 0 and cur_step < curriculum_steps:
        frac    = cur_step / curriculum_steps
        horizon = int(horizon_start + frac * (horizon_max - horizon_start))
        horizon = max(horizon_start, min(horizon, horizon_max))
    else:
        horizon = horizon_max

    warmup = int(cfg["eval"].get("warmup_steps", 10))

    # --- losses ---
    one  = one_step_delta_loss(model, states, actions, normalizer)
    roll = rollout_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon,
    )

    one_w  = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.5))
    total  = one_w * one + roll_w * roll

    return total, {
        "loss/total":           float(total.detach().cpu()),
        "loss/one_step":        float(one.detach().cpu()),
        "loss/rollout":         float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
