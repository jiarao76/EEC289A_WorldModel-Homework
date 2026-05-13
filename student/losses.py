"""Student losses — one-step + exponentially-weighted multi-step rollout."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wm_hw.model_utils import predict_next
from .rollout import open_loop_rollout


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
) -> torch.Tensor:
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
    gamma: float = 0.97,
) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    h = pred_norm.shape[1]
    weights = torch.tensor(
        [gamma ** i for i in range(h)],
        dtype=pred_norm.dtype,
        device=pred_norm.device,
    ).unsqueeze(0).unsqueeze(-1)

    sq_err = (pred_norm - target_norm) ** 2
    weighted = (sq_err * weights).sum() / (weights.sum() * sq_err.shape[0] * sq_err.shape[2])
    return weighted


def velocity_consistency_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    if preds.shape[1] < 2:
        return torch.tensor(0.0, device=states.device)

    vel = preds[:, :, 2:]
    vel_diff = vel[:, 1:] - vel[:, :-1]
    return (vel_diff ** 2).mean()


def early_failure_penalty(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    check_horizon: int = 20,
    threshold: float = 0.25,
) -> torch.Tensor:
    """专门惩罚前check_horizon步内误差超过阈值的预测，直接针对VPT80@0.25。"""
    needed_states = int(warmup_steps) + int(check_horizon) + 1
    if states.shape[1] < needed_states:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(check_horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=check_horizon,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + check_horizon]

    obs_std = torch.as_tensor(normalizer.obs_std, dtype=preds.dtype, device=preds.device)
    per_step_nmse = torch.mean(((preds - targets) / obs_std) ** 2, dim=-1)  # (B, H)

    penalty = torch.relu(per_step_nmse - threshold)
    penalty = torch.clamp(penalty, max=2.0)
    return penalty.mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)

    horizon = int(loss_cfg.get("rollout_train_horizon", 20))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    gamma = float(loss_cfg.get("rollout_gamma", 0.97))

    roll = rollout_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon, gamma=gamma,
    )

    early_w = float(loss_cfg.get("early_failure_weight", 0.0))
    if early_w > 0.0:
        early = early_failure_penalty(
            model, states, actions, normalizer,
            warmup_steps=warmup, check_horizon=20, threshold=0.25,
        )
    else:
        early = torch.tensor(0.0, device=states.device)

    vel_w = float(loss_cfg.get("velocity_weight", 0.0))
    if vel_w > 0.0:
        vel = velocity_consistency_loss(
            model, states, actions, normalizer,
            warmup_steps=warmup, horizon=horizon,
        )
    else:
        vel = torch.tensor(0.0, device=states.device)

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 1.0))

    total = one_w * one + roll_w * roll + vel_w * vel + early_w * early
    return total, {
        "loss/total":         float(total.detach().cpu()),
        "loss/one_step":      float(one.detach().cpu()),
        "loss/rollout":       float(roll.detach().cpu()),
        "loss/velocity":      float(vel.detach().cpu()),
        "loss/early_penalty": float(early.detach().cpu()),
    }
