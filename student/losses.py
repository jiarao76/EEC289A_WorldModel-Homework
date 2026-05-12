"""Student losses — one-step + exponentially-weighted multi-step rollout.

Improvements over the starter
------------------------------
* **Exponential step weights** in the rollout loss: earlier prediction steps
  receive higher weight so the model learns to stay accurate before worrying
  about the far future.  Controlled by `rollout_gamma` (0 < γ ≤ 1).
  γ = 1.0 recovers the original uniform weighting.
* **Velocity-consistency loss**: penalises discontinuities between the
  predicted velocity components across consecutive rollout steps, which
  helps the GRU hidden state stay well-behaved during long rollouts.
* The warmup hidden state is always properly threaded through the rollout
  so the GRU benefits from the context window.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wm_hw.model_utils import predict_next
from .rollout import open_loop_rollout


# ---------------------------------------------------------------------------
# One-step delta loss  (unchanged interface)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rollout loss with exponential step weights
# ---------------------------------------------------------------------------

def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    gamma: float = 0.97,
) -> torch.Tensor:
    """Open-loop rollout loss with exponential step weights.

    Steps closer to the warmup boundary receive weight γ^0 = 1.0;
    step h receives weight γ^h.  This encourages early-step accuracy
    (which drives VPT) without completely ignoring the far future.
    """
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

    pred_norm = normalizer.normalize_obs(preds)       # (B, H, D)
    target_norm = normalizer.normalize_obs(targets)   # (B, H, D)

    # exponential weights: shape (1, H, 1)
    h = pred_norm.shape[1]
    weights = torch.tensor(
        [gamma ** i for i in range(h)],
        dtype=pred_norm.dtype,
        device=pred_norm.device,
    ).unsqueeze(0).unsqueeze(-1)  # (1, H, 1)

    sq_err = (pred_norm - target_norm) ** 2          # (B, H, D)
    weighted = (sq_err * weights).sum() / (weights.sum() * sq_err.shape[0] * sq_err.shape[2])
    return weighted


# ---------------------------------------------------------------------------
# Velocity-consistency regularisation
# ---------------------------------------------------------------------------

def velocity_consistency_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Penalise large prediction jumps in velocity dimensions (indices 2, 3).

    InvertedPendulum-v5 obs = [x, theta, x_dot, theta_dot].
    Smooth velocity changes keep the GRU hidden state stable.
    """
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

    # velocity dims: x_dot=2, theta_dot=3
    vel = preds[:, :, 2:]          # (B, H, 2)
    vel_diff = vel[:, 1:] - vel[:, :-1]  # (B, H-1, 2)
    return (vel_diff ** 2).mean()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    total = one_w * one + roll_w * roll + vel_w * vel
    return total, {
        "loss/total":    float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout":  float(roll.detach().cpu()),
        "loss/velocity": float(vel.detach().cpu()),
    }
