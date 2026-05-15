"""Student losses — RSSM version with KL divergence."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wm_hw.model_utils import predict_next
from .rollout import open_loop_rollout


def one_step_delta_loss(model, states, actions, normalizer):
    """One-step loss using posterior. Also extracts KL from hidden."""
    obs    = states[:, :-1].reshape(-1, states.shape[-1])
    act    = actions.reshape(-1, actions.shape[-1])
    target = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])

    obs_norm    = normalizer.normalize_obs(obs)
    act_norm    = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target)

    # Process sequence step by step to accumulate KL
    B, T, _ = states[:, :-1].shape
    hidden = model.initial_hidden(B, states.device)
    total_recon = torch.tensor(0.0, device=states.device)
    total_kl    = torch.tensor(0.0, device=states.device)

    for t in range(T):
        o = normalizer.normalize_obs(states[:, t])
        a = normalizer.normalize_act(actions[:, t])
        tgt = normalizer.normalize_delta(states[:, t+1] - states[:, t])

        pred, hidden = model(o, a, hidden)
        total_recon += F.mse_loss(pred, tgt)

        # Extract KL from hidden tuple
        if len(hidden) == 6:
            _, _, prior_mean, prior_log_std, post_mean, post_log_std = hidden
            kl = model._kl(post_mean, post_log_std,
                           prior_mean, prior_log_std)
            total_kl += kl

        # Keep only (h, z) for next step
        hidden = (hidden[0], hidden[1])

    return total_recon / T, total_kl / T


def rollout_loss(model, states, actions, normalizer,
                 warmup_steps, horizon, gamma=0.97):
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        raise ValueError("train_sequence_length too short for rollout loss.")

    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (),
                device=states.device).item()) if max_start > 0 else 0
    sub_states  = states[:, start : start + needed]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds   = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                                warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    h = pred_norm.shape[1]
    weights = torch.tensor(
        [gamma ** i for i in range(h)],
        dtype=pred_norm.dtype, device=pred_norm.device
    ).unsqueeze(0).unsqueeze(-1)

    sq_err  = (pred_norm - target_norm) ** 2
    return (sq_err * weights).sum() / (
        weights.sum() * sq_err.shape[0] * sq_err.shape[2]
    )


def early_failure_penalty(model, states, actions, normalizer,
                           warmup_steps, check_horizon=20, threshold=0.25):
    needed = int(warmup_steps) + int(check_horizon) + 1
    if states.shape[1] < needed:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (),
                device=states.device).item()) if max_start > 0 else 0
    sub_states  = states[:, start : start + needed]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(check_horizon)]

    preds   = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                                warmup_steps=warmup_steps, horizon=check_horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + check_horizon]

    obs_std      = torch.as_tensor(normalizer.obs_std,
                                   dtype=preds.dtype, device=preds.device)
    per_step_nmse = torch.mean(((preds - targets) / obs_std) ** 2, dim=-1)
    penalty       = torch.relu(per_step_nmse - threshold)
    return torch.clamp(penalty, max=2.0).mean()


def velocity_consistency_loss(model, states, actions, normalizer,
                               warmup_steps, horizon):
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (),
                device=states.device).item()) if max_start > 0 else 0
    sub_states  = states[:, start : start + needed]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                              warmup_steps=warmup_steps, horizon=horizon)
    if preds.shape[1] < 2:
        return torch.tensor(0.0, device=states.device)

    vel_diff = preds[:, 1:, 2:] - preds[:, :-1, 2:]
    return (vel_diff ** 2).mean()


def compute_loss(model, batch, normalizer, cfg):
    loss_cfg = cfg["loss"]
    states   = batch["states"]
    actions  = batch["actions"]

    warmup  = int(cfg["eval"].get("warmup_steps", 10))
    horizon = int(loss_cfg.get("rollout_train_horizon", 20))
    gamma   = float(loss_cfg.get("rollout_gamma", 0.97))

    # One-step recon + KL
    one, kl = one_step_delta_loss(model, states, actions, normalizer)
    kl_w    = float(loss_cfg.get("kl_weight", 0.1))

    # Rollout loss
    roll   = rollout_loss(model, states, actions, normalizer,
                          warmup_steps=warmup, horizon=horizon, gamma=gamma)
    roll_w = float(loss_cfg.get("rollout_weight", 3.0))

    # Early failure penalty
    early_w = float(loss_cfg.get("early_failure_weight", 0.0))
    early   = early_failure_penalty(
        model, states, actions, normalizer,
        warmup_steps=warmup
    ) if early_w > 0.0 else torch.tensor(0.0, device=states.device)

    # Velocity consistency
    vel_w = float(loss_cfg.get("velocity_weight", 0.0))
    vel   = velocity_consistency_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon
    ) if vel_w > 0.0 else torch.tensor(0.0, device=states.device)

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    total = one_w * one + kl_w * kl + roll_w * roll + \
            early_w * early + vel_w * vel

    return total, {
        "loss/total":        float(total.detach().cpu()),
        "loss/one_step":     float(one.detach().cpu()),
        "loss/kl":           float(kl.detach().cpu()),
        "loss/rollout":      float(roll.detach().cpu()),
        "loss/velocity":     float(vel.detach().cpu()),
        "loss/early_penalty":float(early.detach().cpu()),
    }
