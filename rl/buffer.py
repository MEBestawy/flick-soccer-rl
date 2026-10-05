"""Rollout buffer with GAE advantages."""

from __future__ import annotations

from typing import Dict, Generator, Optional

import numpy as np
import torch

from .config import RLConfig


class RolloutBuffer:
    def __init__(self, config: RLConfig, num_envs: int) -> None:
        self.config = config
        self.num_envs = num_envs
        self.n_steps = config.rollout_steps
        self.obs_dim = config.obs_dim
        self.ptr = 0
        self.full = False

        T, N = self.n_steps, num_envs
        self.obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.players = np.zeros((T, N), dtype=np.int64)
        self.angles = np.zeros((T, N), dtype=np.float32)
        self.powers = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        player: np.ndarray,
        angle: np.ndarray,
        power: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ) -> None:
        assert self.ptr < self.n_steps
        self.obs[self.ptr] = obs
        self.players[self.ptr] = player
        self.angles[self.ptr] = angle
        self.powers[self.ptr] = power
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        cfg = self.config
        adv = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + cfg.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            adv = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def get(self, device: torch.device) -> Dict[str, torch.Tensor]:
        assert self.full
        b_obs = torch.as_tensor(self.obs.reshape(-1, self.obs_dim), device=device)
        b_players = torch.as_tensor(self.players.reshape(-1), device=device)
        b_angles = torch.as_tensor(self.angles.reshape(-1), device=device)
        b_powers = torch.as_tensor(self.powers.reshape(-1), device=device)
        b_logp = torch.as_tensor(self.log_probs.reshape(-1), device=device)
        b_adv = torch.as_tensor(self.advantages.reshape(-1), device=device)
        b_ret = torch.as_tensor(self.returns.reshape(-1), device=device)
        b_val = torch.as_tensor(self.values.reshape(-1), device=device)
        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        return {
            "obs": b_obs,
            "players": b_players,
            "angles": b_angles,
            "powers": b_powers,
            "log_probs": b_logp,
            "advantages": b_adv,
            "returns": b_ret,
            "values": b_val,
        }

    def reset(self) -> None:
        self.ptr = 0
        self.full = False
