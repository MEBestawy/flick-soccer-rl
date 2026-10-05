"""Clipped PPO update with free-head BC (teacher transfer) + target KL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import RLConfig
from .model import ActorCritic


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    config: RLConfig,
    *,
    bc_coef: Optional[float] = None,
    free_bc_coef: Optional[float] = None,
) -> Dict[str, float]:
    obs = batch["obs"]
    players = batch["players"]
    angles = batch["angles"]
    powers = batch["powers"]
    old_logp = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    old_values = batch["values"]

    n = obs.shape[0]
    idxs = torch.randperm(n, device=obs.device)
    mb = config.minibatch_size
    bc_w = config.bc_coef if bc_coef is None else bc_coef
    free_w = (
        config.free_bc_coef if free_bc_coef is None else free_bc_coef
    )

    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "bc_loss": 0.0,
        "free_bc_loss": 0.0,
        "n_updates": 0,
    }

    early_stop = False
    saved_mix = float(getattr(model, "teacher_mix", 1.0))

    for _ in range(config.update_epochs):
        if early_stop:
            break
        for start in range(0, n, mb):
            end = start + mb
            mb_idx = idxs[start:end]
            logp, entropy, values = model.evaluate_actions(
                obs[mb_idx],
                players[mb_idx],
                angles[mb_idx],
                powers[mb_idx],
            )
            ratio = torch.exp(logp - old_logp[mb_idx])
            adv = advantages[mb_idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
            ) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clipped = old_values[mb_idx] + torch.clamp(
                values - old_values[mb_idx],
                -config.clip_coef,
                config.clip_coef,
            )
            v_loss_unclipped = (values - returns[mb_idx]) ** 2
            v_loss_clipped = (v_clipped - returns[mb_idx]) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            bc_loss = torch.zeros((), device=obs.device)
            free_bc_loss = torch.zeros((), device=obs.device)
            if (bc_w > 0.0 or free_w > 0.0) and hasattr(
                model, "teacher_actions_from_obs"
            ):
                t_player, t_angle, t_power = model.teacher_actions_from_obs(
                    obs[mb_idx]
                )
                if bc_w > 0.0:
                    t_logp, _, _ = model.evaluate_actions(
                        obs[mb_idx], t_player, t_angle, t_power
                    )
                    bc_loss = -t_logp.mean()
                if free_w > 0.0:
                    # Train residual heads as if teacher_mix=0 so they can
                    # replace the prior when annealing.
                    model.set_teacher_mix(0.0)
                    f_logp, _, _ = model.evaluate_actions(
                        obs[mb_idx], t_player, t_angle, t_power
                    )
                    free_bc_loss = -f_logp.mean()
                    model.set_teacher_mix(saved_mix)

            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_loss
                + bc_w * bc_loss
                + free_w * free_bc_loss
            )

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                model.set_teacher_mix(saved_mix)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            model.set_teacher_mix(saved_mix)

            with torch.no_grad():
                approx_kl = (old_logp[mb_idx] - logp).mean().item()
                clip_frac = (
                    (ratio - 1.0).abs() > config.clip_coef
                ).float().mean().item()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy_loss.item())
            metrics["approx_kl"] += float(approx_kl)
            metrics["clip_fraction"] += float(clip_frac)
            metrics["bc_loss"] += float(bc_loss.item()) if bc_w > 0 else 0.0
            metrics["free_bc_loss"] += (
                float(free_bc_loss.item()) if free_w > 0 else 0.0
            )
            metrics["n_updates"] += 1

            if config.target_kl > 0 and approx_kl > config.target_kl:
                early_stop = True
                break

    model.set_teacher_mix(saved_mix)
    n_up = max(1, metrics["n_updates"])
    for k in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "bc_loss",
        "free_bc_loss",
    ):
        metrics[k] /= n_up
    return metrics
