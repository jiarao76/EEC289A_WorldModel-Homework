"""Student one-step + multi-scale rollout loss with curriculum learning.

Key improvements over the starter:
1. Multi-scale rollout: simultaneously train at short (H/4), medium (H/2),
   and full (H) horizons so the model learns both local accuracy and long-range
   stability.
2. Late-step emphasis: within each rollout loss the per-step MSE is weighted
   by its position (later steps get higher weight), directly penalising drift.
3. Curriculum warmup: rollout_train_horizon ramps from 5 to the configured max
   over the first half of training, preventing gradient explosion early on.
   When the training loop does not pass a global step, the full horizon is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def _single_rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    late_step_weight: float = 2.0,
) -> torch.Tensor:
    """MSE in normalised obs space over `horizon` open-loop steps.

    Steps closer to the end receive up to `late_step_weight` × more loss
    weight, focusing optimisation on long-horizon stability.
    """
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        raise ValueError(
            f"train_sequence_length too short: need {needed - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

    sub_states = states[:, start : start + needed]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )  # [B, horizon, obs_dim]
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # Per-step squared error: [B, horizon, obs_dim] -> [B, horizon]
    step_se = ((pred_norm - target_norm) ** 2).mean(dim=-1)

    # Linear ramp: weight_t = 1 + (late_step_weight - 1) * t / (horizon - 1)
    h = step_se.shape[1]
    if h > 1:
        ramp = torch.linspace(1.0, float(late_step_weight), h, device=step_se.device)
        step_se = step_se * ramp.unsqueeze(0)

    return step_se.mean()


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Multi-scale rollout loss at up to 4 anchors: 10, H//4, H//2, H.

    With horizon=400 this gives 10, 100, 200, 400 — directly covering
    the critical long-range stability range.
    All anchors are clamped to [1, horizon] so the function is safe even
    when horizon is small (e.g. during unit tests with short sequences).
    """
    def clamp(h: int) -> int:
        return max(1, min(int(h), horizon))

    scales = sorted({clamp(10), clamp(horizon // 4), clamp(horizon // 2), horizon})
    losses = []
    for h in scales:
        losses.append(_single_rollout_loss(model, states, actions, normalizer, warmup_steps, h))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Public API expected by train.py
# ---------------------------------------------------------------------------

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict, *, step: int = 0, total_steps: int = 0):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)

    max_horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup = int(cfg["eval"].get("warmup_steps", 10))

    # Curriculum: ramp horizon from 5 → max_horizon over first 40 % of training
    if total_steps > 0:
        frac = min(1.0, step / max(1, int(total_steps * 0.4)))
        horizon = max(5, int(5 + (max_horizon - 5) * frac))
    else:
        horizon = max_horizon

    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))
    total = one_w * one + roll_w * roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
