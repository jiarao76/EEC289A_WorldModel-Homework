"""Student losses with internal step counter for Scheduled Sampling.

Since train.py is locked and does not pass `step` to compute_loss, we use
a module-level counter that increments on every compute_loss call.
This gives SS a reliable clock without modifying the locked training script.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

# ---------------------------------------------------------------------------
# Internal step counter (replaces the missing `step` argument from train.py)
# ---------------------------------------------------------------------------
_CALL_COUNT = 0


def _get_and_increment() -> int:
    global _CALL_COUNT
    c = _CALL_COUNT
    _CALL_COUNT += 1
    return c


def reset_counter():
    """Call this at the start of training if you want reproducible SS."""
    global _CALL_COUNT
    _CALL_COUNT = 0


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
    teacher_forcing_ratio: float = 0.0,
    late_step_weight: float = 2.0,
) -> torch.Tensor:
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        raise ValueError(
            f"train_sequence_length too short: need {needed - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

    sub_states  = states[:, start : start + needed]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    step_se = ((pred_norm - target_norm) ** 2).mean(dim=-1)

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
    teacher_forcing_ratio: float = 0.0,
) -> torch.Tensor:
    def clamp(h: int) -> int:
        return max(1, min(int(h), horizon))

    scales = sorted({clamp(10), clamp(horizon // 4), clamp(horizon // 2), horizon})
    losses = []
    for h in scales:
        losses.append(_single_rollout_loss(
            model, states, actions, normalizer, warmup_steps, h,
            teacher_forcing_ratio=teacher_forcing_ratio,
        ))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict, *, step: int = 0, total_steps: int = 0):
    loss_cfg = cfg["loss"]
    states  = batch["states"]
    actions = batch["actions"]

    # Use internal counter as clock since train.py doesn't pass step
    call_step = _get_and_increment()

    one = one_step_delta_loss(model, states, actions, normalizer)

    horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup  = int(cfg["eval"].get("warmup_steps", 10))

    # Scheduled Sampling anneal
    ss_start        = float(loss_cfg.get("ss_start", 0.5))
    ss_anneal_steps = int(loss_cfg.get("ss_anneal_steps", 3000))
    if ss_anneal_steps > 0:
        frac = min(1.0, call_step / ss_anneal_steps)
        teacher_forcing_ratio = ss_start * (1.0 - frac)
    else:
        teacher_forcing_ratio = 0.0

    roll = rollout_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    one_w  = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))
    total  = one_w * one + roll_w * roll

    return total, {
        "loss/total":           float(total.detach().cpu()),
        "loss/one_step":        float(one.detach().cpu()),
        "loss/rollout":         float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
        "loss/teacher_forcing": float(teacher_forcing_ratio),
    }
