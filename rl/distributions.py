"""Joint hybrid action distribution for PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Categorical, Normal


@dataclass
class ActionDistOutput:
    player_logits: torch.Tensor  # (B, 5)
    angle_mean: torch.Tensor  # (B,)
    angle_log_std: torch.Tensor  # (B,) or scalar broadcast
    power_alpha: torch.Tensor  # (B,)
    power_beta: torch.Tensor  # (B,)


@dataclass
class SampledAction:
    player: torch.Tensor  # (B,) long
    angle: torch.Tensor  # (B,)
    direction_raw: torch.Tensor  # (B, 2) unit vector used as "raw" for env
    power: torch.Tensor  # (B,)
    log_prob: torch.Tensor  # (B,)
    entropy: torch.Tensor  # (B,)
    value: torch.Tensor  # (B,)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class HybridActionDistribution:
    """
    π(a|s) = Cat(player) × Normal(angle) × Beta(power)

    Direction for the simulator is (cos θ, sin θ) in canonical L→R frame.
    Angle uses a diagonal Normal with wrapped mean; log_prob uses the sampled
    angle under that Normal (standard practical approach for 2D aiming).
    """

    def __init__(self, out: ActionDistOutput) -> None:
        self.out = out
        self.player_dist = Categorical(logits=out.player_logits)
        log_std = out.angle_log_std.clamp(-5.0, 2.0)
        self.angle_dist = Normal(out.angle_mean, log_std.exp())
        # Beta needs positive concentrations
        self.power_dist = Beta(out.power_alpha, out.power_beta)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player = self.player_dist.sample()
        angle = _wrap_angle(self.angle_dist.sample())
        # Beta sample can hit exact 0/1; clamp for numerical safety in env
        power = self.power_dist.sample().clamp(1e-4, 1.0 - 1e-4)
        return player, angle, power

    def deterministic(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player = torch.argmax(self.out.player_logits, dim=-1)
        angle = _wrap_angle(self.out.angle_mean)
        power = (
            self.out.power_alpha / (self.out.power_alpha + self.out.power_beta)
        ).clamp(1e-4, 1.0 - 1e-4)
        return player, angle, power

    def log_prob(
        self,
        player: torch.Tensor,
        angle: torch.Tensor,
        power: torch.Tensor,
    ) -> torch.Tensor:
        # Angle: evaluate unwrapped relative to mean for stability
        # Use wrapped difference inside Normal by shifting sample
        mean = self.out.angle_mean
        delta = _wrap_angle(angle - mean)
        # log prob of (mean + delta) under Normal(mean, std) == log prob of delta under Normal(0,std)
        std = self.out.angle_log_std.clamp(-5.0, 2.0).exp()
        angle_lp = Normal(torch.zeros_like(mean), std).log_prob(delta)
        power_c = power.clamp(1e-4, 1.0 - 1e-4)
        return (
            self.player_dist.log_prob(player)
            + angle_lp
            + self.power_dist.log_prob(power_c)
        )

    def entropy(self) -> torch.Tensor:
        return (
            self.player_dist.entropy()
            + self.angle_dist.entropy()
            + self.power_dist.entropy()
        )

    def direction_from_angle(self, angle: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def build_dist_from_network(
    player_logits: torch.Tensor,
    direction_params: torch.Tensor,
    power_params: torch.Tensor,
) -> HybridActionDistribution:
    """
    direction_params: (B, 3) -> mean_x, mean_y, log_std
    power_params: (B, 2) -> raw_alpha, raw_beta
    """
    mean_x = direction_params[..., 0]
    mean_y = direction_params[..., 1]
    angle_mean = torch.atan2(mean_y, mean_x)
    angle_log_std = direction_params[..., 2]
    alpha = F.softplus(power_params[..., 0]) + 1.0
    beta = F.softplus(power_params[..., 1]) + 1.0
    return HybridActionDistribution(
        ActionDistOutput(
            player_logits=player_logits,
            angle_mean=angle_mean,
            angle_log_std=angle_log_std,
            power_alpha=alpha,
            power_beta=beta,
        )
    )
