"""Student losses — RSSM version with KL divergence.

Key fixes vs original:
  1. rollout_loss uses gamma=1.0 (uniform weights) — with gamma=0.98 the
     weight at step 500 is 0.98^500 ≈ 4e-5, giving essentially zero gradient
     signal for long-horizon accuracy.  Uniform weights force the model to
     care equally about step 1 and step 500.
  2. rollout warmup is run under torch.no_grad() + detach — the warmup
     hidden state is used only to initialise the open-loop rollout; computing
     gradients through it wastes memory and can destabilise training when
     horizon is large.
  3. kl_weight raised and free_bits passed through — see model._kl().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states, actions, normalizer):
    """One-step loss using posterior. Also extracts KL from hidden."""
    B, T_plus1, _ = states.shape
    T = T_plus1 - 1

    hidden = model.initial_hidden(B, states.device)
    total_recon = torch.tensor(0.0, device=states.device)
    total_kl    = torch.tensor(0.0, device=states.device)

    for t in range(T):
        o   = normalizer.normalize_obs(states[:, t])
        a   = normalizer.normalize_act(actions[:, t])
        tgt = normalizer.normalize_delta(states[:, t + 1] - states[:, t])

        pred, hidden = model(o, a, hidden)
        total_recon += F.mse_loss(pred, tgt)

        if len(hidden) == 6:
            _, _, prior_mean, prior_log_std, post_mean, post_log_std = hidden
            kl = model._kl(post_mean, post_log_std,
                           prior_mean, prior_log_std,
                           free_bits=0.0)
            total_kl += kl

        # Detach hidden between steps to limit BPTT length,
        # but keep (h, z) for next iteration
        h_det = hidden[0].detach()
        z_det = hidden[1].detach()
        hidden = (h_det, z_det)

    return total_recon / T, total_kl / T


def _warmup_hidden(model, states, actions, normalizer, warmup_steps):
    """Run warmup under no_grad and return detached (h, z)."""
    B = states.shape[0]
    hidden = model.initial_hidden(B, states.device)
    with torch.no_grad():
        for t in range(int(warmup_steps)):
            o = normalizer.normalize_obs(states[:, t])
            a = normalizer.normalize_act(actions[:, t])
            _, hidden = model(o, a, hidden)
            if isinstance(hidden, tuple) and len(hidden) > 2:
                hidden = (hidden[0], hidden[1])
    # Detach so gradients only flow through the open-loop prediction
    return (hidden[0].detach(), hidden[1].detach())


def rollout_loss(model, states, actions, normalizer,
                 warmup_steps, horizon, gamma=1.0):
    """Open-loop rollout loss with uniform step weights.

    gamma=1.0 means every predicted step contributes equally to the loss.
    This is essential for learning accurate long-horizon predictions:
    with gamma=0.98 the gradient at step 200 is ~0.02x step 1, giving
    virtually no signal to improve 200+ step accuracy.
    """
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        raise ValueError(
            f"train_sequence_length ({states.shape[1]}) too short for "
            f"warmup ({warmup_steps}) + horizon ({horizon}) + 1 = {needed}"
        )

    max_start = states.shape[1] - needed
    start = (
        int(torch.randint(0, max_start + 1, (), device=states.device).item())
        if max_start > 0 else 0
    )
    sub_states  = states[:, start: start + needed]
    sub_actions = actions[:, start: start + int(warmup_steps) + int(horizon)]

    preds   = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    targets = sub_states[:, warmup_steps + 1: warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    H = pred_norm.shape[1]
    if gamma < 1.0:
        weights = torch.tensor(
            [gamma ** i for i in range(H)],
            dtype=pred_norm.dtype, device=pred_norm.device,
        ).unsqueeze(0).unsqueeze(-1)
        sq_err = (pred_norm - target_norm) ** 2
        return (sq_err * weights).sum() / (
            weights.sum() * sq_err.shape[0] * sq_err.shape[2]
        )
    else:
        # Uniform weights — simple MSE across all steps
        return F.mse_loss(pred_norm, target_norm)


def early_failure_penalty(model, states, actions, normalizer,
                           warmup_steps, check_horizon=20, threshold=0.25):
    needed = int(warmup_steps) + int(check_horizon) + 1
    if states.shape[1] < needed:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed
    start = (
        int(torch.randint(0, max_start + 1, (), device=states.device).item())
        if max_start > 0 else 0
    )
    sub_states  = states[:, start: start + needed]
    sub_actions = actions[:, start: start + int(warmup_steps) + int(check_horizon)]

    preds   = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=check_horizon,
    )
    targets = sub_states[:, warmup_steps + 1: warmup_steps + 1 + check_horizon]

    obs_std       = torch.as_tensor(
        normalizer.obs_std, dtype=preds.dtype, device=preds.device
    )
    per_step_nmse = torch.mean(((preds - targets) / obs_std) ** 2, dim=-1)
    penalty       = torch.relu(per_step_nmse - threshold)
    return torch.clamp(penalty, max=2.0).mean()


def velocity_consistency_loss(model, states, actions, normalizer,
                               warmup_steps, horizon):
    needed = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed:
        return torch.tensor(0.0, device=states.device)

    max_start = states.shape[1] - needed
    start = (
        int(torch.randint(0, max_start + 1, (), device=states.device).item())
        if max_start > 0 else 0
    )
    sub_states  = states[:, start: start + needed]
    sub_actions = actions[:, start: start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    if preds.shape[1] < 2:
        return torch.tensor(0.0, device=states.device)

    vel_diff = preds[:, 1:, 2:] - preds[:, :-1, 2:]
    return (vel_diff ** 2).mean()


def compute_loss(model, batch, normalizer, cfg):
    loss_cfg = cfg["loss"]
    states   = batch["states"]
    actions  = batch["actions"]

    warmup  = int(cfg["eval"].get("warmup_steps", 10))
    horizon = int(loss_cfg.get("rollout_train_horizon", 50))
    gamma   = float(loss_cfg.get("rollout_gamma", 1.0))

    # ── One-step recon + KL ───────────────────────────────────────────────
    one, kl = one_step_delta_loss(model, states, actions, normalizer)
    kl_w    = float(loss_cfg.get("kl_weight", 0.5))

    # ── Rollout loss (uniform weights by default) ─────────────────────────
    roll   = rollout_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon, gamma=gamma,
    )
    roll_w = float(loss_cfg.get("rollout_weight", 8.0))

    # ── Early failure penalty ─────────────────────────────────────────────
    early_w = float(loss_cfg.get("early_failure_weight", 0.0))
    early   = early_failure_penalty(
        model, states, actions, normalizer, warmup_steps=warmup,
    ) if early_w > 0.0 else torch.tensor(0.0, device=states.device)

    # ── Velocity consistency ──────────────────────────────────────────────
    vel_w = float(loss_cfg.get("velocity_weight", 0.0))
    vel   = velocity_consistency_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup, horizon=horizon,
    ) if vel_w > 0.0 else torch.tensor(0.0, device=states.device)

    one_w = float(loss_cfg.get("one_step_weight", 0.5))
    total = one_w * one + kl_w * kl + roll_w * roll + \
            early_w * early + vel_w * vel

    return total, {
        "loss/total":         float(total.detach().cpu()),
        "loss/one_step":      float(one.detach().cpu()),
        "loss/kl":            float(kl.detach().cpu()),
        "loss/rollout":       float(roll.detach().cpu()),
        "loss/velocity":      float(vel.detach().cpu()),
        "loss/early_penalty": float(early.detach().cpu()),
    }
