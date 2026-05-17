"""Student losses - pure open-loop, focus on single-step accuracy first."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


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
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    step_se = ((pred_norm - target_norm) ** 2).mean(dim=-1)

    if step_se.shape[1] > 1:
        ramp = torch.linspace(1.0, float(late_step_weight), step_se.shape[1], device=step_se.device)
        step_se = step_se * ramp.unsqueeze(0)

    return step_se.mean()


def rollout_loss(model, states, actions, normalizer, warmup_steps, horizon) -> torch.Tensor:
    def clamp(h: int) -> int:
        return max(1, min(int(h), horizon))
    scales = sorted({clamp(10), clamp(horizon // 4), clamp(horizon // 2), horizon})
    losses = [_single_rollout_loss(model, states, actions, normalizer, warmup_steps, h) for h in scales]
    return torch.stack(losses).mean()


def compute_loss(model, batch, normalizer, cfg, *, step: int = 0, total_steps: int = 0):
    loss_cfg = cfg["loss"]
    states  = batch["states"]
    actions = batch["actions"]

    one    = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup  = int(cfg["eval"].get("warmup_steps", 10))
    roll   = rollout_loss(model, states, actions, normalizer, warmup, horizon)

    one_w  = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))
    total  = one_w * one + roll_w * roll

    return total, {
        "loss/total":           float(total.detach().cpu()),
        "loss/one_step":        float(one.detach().cpu()),
        "loss/rollout":         float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
        "loss/teacher_forcing": 0.0,
    }
