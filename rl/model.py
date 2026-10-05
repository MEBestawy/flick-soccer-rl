"""Annealed teacher-residual actor-critic with optional dual value tower."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import RLAction
from .config import RLConfig
from .distributions import HybridActionDistribution, SampledAction, build_dist_from_network


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


def _mlp(in_dim: int, sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in sizes:
        layers.append(nn.Linear(d, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU())
        d = h
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Residual policy around a teacher/search hint embedded in the observation.

    ``teacher_mix`` (0..1) anneals the hard prior:
      1 → behavior ≈ hint, 0 → fully learned heads.
    """

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        super().__init__()
        self.config = config or RLConfig()
        obs = self.config.obs_dim
        sizes = self.config.hidden_sizes
        hint = self.config.teacher_hint_dim
        self._hint_dim = hint
        self._teacher_mix = float(self.config.teacher_mix_start)

        self.policy_backbone = _mlp(obs, sizes)
        self.policy_res1 = ResidualBlock(sizes[-1])
        self.policy_res2 = ResidualBlock(sizes[-1])

        if self.config.dual_value:
            self.value_backbone = _mlp(obs, sizes)
            self.value_res1 = ResidualBlock(sizes[-1])
            self.value_res2 = ResidualBlock(sizes[-1])
        else:
            self.value_backbone = None
            self.value_res1 = None
            self.value_res2 = None

        feat = sizes[-1]
        self.player_head = nn.Linear(feat, self.config.num_players)
        self.angle_delta_head = nn.Linear(feat, 1)
        self.angle_log_std_head = nn.Linear(feat, 1)
        self.power_delta_head = nn.Linear(feat, 1)
        self.power_conc_head = nn.Linear(feat, 1)
        self.value_head = nn.Linear(feat, 1)

        nn.init.zeros_(self.angle_delta_head.weight)
        nn.init.zeros_(self.angle_delta_head.bias)
        nn.init.zeros_(self.power_delta_head.weight)
        nn.init.zeros_(self.power_delta_head.bias)
        nn.init.zeros_(self.player_head.weight)
        nn.init.zeros_(self.player_head.bias)
        with torch.no_grad():
            self.angle_log_std_head.bias.fill_(-0.8)
            self.power_conc_head.bias.fill_(1.3)

    def set_teacher_mix(self, mix: float) -> None:
        self._teacher_mix = float(np.clip(mix, 0.0, 1.0))

    @property
    def teacher_mix(self) -> float:
        return self._teacher_mix

    def _split_hint(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hint = obs[..., -self._hint_dim :]
        n = self.config.num_players
        return (
            hint[..., :n],
            hint[..., n],
            hint[..., n + 1],
            hint[..., n + 2],
        )

    def _policy_feat(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.policy_backbone(obs)
        return self.policy_res2(self.policy_res1(feat))

    def _value_feat(self, obs: torch.Tensor) -> torch.Tensor:
        if self.value_backbone is None:
            return self._policy_feat(obs)
        feat = self.value_backbone(obs)
        assert self.value_res1 is not None and self.value_res2 is not None
        return self.value_res2(self.value_res1(feat))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_oh, teacher_cos, teacher_sin, teacher_power = self._split_hint(obs)
        mix = self._teacher_mix
        pfeat = self._policy_feat(obs)
        vfeat = self._value_feat(obs)

        # Learned player logits + annealed teacher bias
        player_logits = self.player_head(pfeat) + (
            mix * self.config.teacher_player_bias * teacher_oh
        )

        teacher_angle = torch.atan2(teacher_sin, teacher_cos)
        delta = torch.tanh(self.angle_delta_head(pfeat).squeeze(-1))
        # Residual capacity grows as mix falls; free BC keeps delta≈0 ≈ teacher.
        residual_scale = self.config.max_angle_residual * (
            0.3 + 0.7 * (1.0 - mix)
        )
        angle_mean = teacher_angle + delta * residual_scale
        log_std = self.angle_log_std_head(pfeat).squeeze(-1)
        log_std = log_std + (1.0 - mix) * 0.15
        direction_params = torch.stack(
            [torch.cos(angle_mean), torch.sin(angle_mean), log_std], dim=-1
        )

        base = teacher_power.clamp(0.05, 0.95)
        logit_t = torch.log(base) - torch.log1p(-base)
        free_logit = self.power_delta_head(pfeat).squeeze(-1).clamp(-4.0, 4.0)
        logit = mix * (logit_t + 0.35 * torch.tanh(free_logit)) + (
            1.0 - mix
        ) * (logit_t + free_logit)
        p = torch.sigmoid(logit).clamp(0.05, 0.95)
        conc = (F.softplus(self.power_conc_head(pfeat).squeeze(-1)) + 2.0).clamp(
            max=30.0
        )
        alpha = (p * conc).clamp(1.01, 28.0)
        beta = ((1.0 - p) * conc).clamp(1.01, 28.0)
        power_params = torch.stack(
            [
                torch.log(torch.expm1(alpha - 1.0)),
                torch.log(torch.expm1(beta - 1.0)),
            ],
            dim=-1,
        )

        # Protect player logits from exploding teacher bias compounding
        player_logits = player_logits.clamp(-20.0, 20.0)
        value = self.value_head(vfeat).squeeze(-1)
        return player_logits, direction_params, power_params, value

    def dist(self, obs: torch.Tensor) -> Tuple[HybridActionDistribution, torch.Tensor]:
        player_logits, direction_params, power_params, value = self.forward(obs)
        dist = build_dist_from_network(
            player_logits,
            direction_params,
            power_params,
            log_std_min=self.config.angle_log_std_min,
            log_std_max=self.config.angle_log_std_max,
        )
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
        return SampledAction(
            player=player,
            angle=angle,
            direction_raw=direction,
            power=power,
            log_prob=dist.log_prob(player, angle, power),
            entropy=dist.entropy(),
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
        return dist.log_prob(player, angle, power), dist.entropy(), value

    def teacher_actions_from_obs(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_oh, teacher_cos, teacher_sin, teacher_power = self._split_hint(obs)
        return (
            teacher_oh.argmax(dim=-1),
            torch.atan2(teacher_sin, teacher_cos),
            teacher_power,
        )

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = False) -> RLAction:
        self.eval()
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        device = next(self.parameters()).device
        sample = self.act(t.to(device), deterministic=deterministic)
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
    lines = [f"teacher_mix={model.teacher_mix:.2f}", "Player:"]
    for i, p in enumerate(probs):
        lines.append(f"  [{i}] {p:.3f}")
    lines.append(f"Direction: ({np.cos(angle):.2f}, {np.sin(angle):.2f})")
    lines.append(f"Power: {power_mean:.2f}")
    lines.append(f"Value: {float(value.item()):+.2f}")
    return "\n".join(lines)
