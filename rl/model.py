"""Small shared MLP actor-critic for flick soccer."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .actions import RLAction
from .config import RLConfig
from .distributions import HybridActionDistribution, SampledAction, build_dist_from_network


class ActorCritic(nn.Module):
    def __init__(self, config: Optional[RLConfig] = None) -> None:
        super().__init__()
        self.config = config or RLConfig()
        obs = self.config.obs_dim
        sizes = self.config.hidden_sizes

        layers: list[nn.Module] = []
        in_dim = obs
        for h in sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        feat = sizes[-1]
        self.player_head = nn.Linear(feat, self.config.num_players)
        # mean_x, mean_y, log_std
        self.direction_head = nn.Linear(feat, 3)
        # raw alpha/beta
        self.power_head = nn.Linear(feat, 2)
        self.value_head = nn.Linear(feat, 1)

        # Init
        nn.init.zeros_(self.direction_head.bias)
        with torch.no_grad():
            self.direction_head.bias[2] = -0.5  # moderate angle std
            self.power_head.bias[:] = 1.0  # mild Beta peaking mid

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        return (
            self.player_head(feat),
            self.direction_head(feat),
            self.power_head(feat),
            self.value_head(feat).squeeze(-1),
        )

    def dist(self, obs: torch.Tensor) -> Tuple[HybridActionDistribution, torch.Tensor]:
        player_logits, direction_params, power_params, value = self.forward(obs)
        dist = build_dist_from_network(player_logits, direction_params, power_params)
        return dist, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> SampledAction:
        dist, value = self.dist(obs)
        if deterministic:
            player, angle, power = dist.deterministic()
        else:
            player, angle, power = dist.sample()
        direction = dist.direction_from_angle(angle)
        log_prob = dist.log_prob(player, angle, power)
        entropy = dist.entropy()
        return SampledAction(
            player=player,
            angle=angle,
            direction_raw=direction,
            power=power,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        player: torch.Tensor,
        angle: torch.Tensor,
        power: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.dist(obs)
        log_prob = dist.log_prob(player, angle, power)
        entropy = dist.entropy()
        return log_prob, entropy, value

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = False) -> RLAction:
        self.eval()
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        device = next(self.parameters()).device
        t = t.to(device)
        sample = self.act(t, deterministic=deterministic)
        return RLAction(
            player_index=int(sample.player.item()),
            direction_raw=sample.direction_raw.squeeze(0).cpu().numpy(),
            power=float(sample.power.item()),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def describe_policy(model: ActorCritic, obs: np.ndarray) -> str:
    model.eval()
    device = next(model.parameters()).device
    t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, value = model.dist(t)
        probs = torch.softmax(dist.out.player_logits, dim=-1).squeeze(0).cpu().numpy()
        angle = float(dist.out.angle_mean.item())
        power_mean = float(
            (dist.out.power_alpha / (dist.out.power_alpha + dist.out.power_beta)).item()
        )
        dx, dy = float(np.cos(angle)), float(np.sin(angle))
    lines = ["Player:"]
    for i, p in enumerate(probs):
        lines.append(f"  [{i}] {p:.3f}")
    lines.append(f"Direction: ({dx:.2f}, {dy:.2f})")
    lines.append(f"Power: {power_mean:.2f}")
    lines.append(f"Value: {float(value.item()):+.2f}")
    return "\n".join(lines)
