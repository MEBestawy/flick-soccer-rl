"""Training opponents: random, heuristic, frozen policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from sim.agents.heuristic_agent import HeuristicAgent
from sim.config import SimConfig
from sim.geometry import Vec2
from sim.models import GameState, Team

from .actions import RLAction
from .config import RLConfig
from .model import ActorCritic
from .observations import friendly_player_ids, observation_from_state


class Opponent(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, state: GameState, team: Team) -> RLAction:
        ...


class RandomOpponent(Opponent):
    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state: GameState, team: Team) -> RLAction:
        ids = friendly_player_ids(state, team)
        idx = int(self.rng.integers(0, len(ids)))
        angle = float(self.rng.uniform(-np.pi, np.pi))
        return RLAction(
            player_index=idx,
            direction_raw=np.array([np.cos(angle), np.sin(angle)], dtype=np.float64),
            power=float(self.rng.uniform(0.3, 1.0)),
        )


class HeuristicRLOpponent(Opponent):
    """Wraps sim HeuristicAgent into RLAction space (canonical frame)."""

    name = "heuristic"

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        noise: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or SimConfig.default()
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._agents = {
            Team.A: HeuristicAgent(Team.A, self.config),
            Team.B: HeuristicAgent(Team.B, self.config),
        }

    def act(self, state: GameState, team: Team) -> RLAction:
        flick = self._agents[team].select_action(state)
        ids = friendly_player_ids(state, team)
        try:
            idx = ids.index(flick.player_id)
        except ValueError:
            idx = 0

        dx, dy = flick.direction.x, flick.direction.y
        # Convert world direction into canonical L→R frame for RLAction storage
        if team == Team.B:
            dx = -dx
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-8:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / n, dy / n

        # Small aiming noise in canonical frame
        angle = float(np.arctan2(dy, dx) + self.rng.normal(0.0, self.noise))
        power = float(np.clip(flick.power + self.rng.normal(0.0, 0.05), 0.05, 1.0))
        return RLAction(
            player_index=idx,
            direction_raw=np.array([np.cos(angle), np.sin(angle)], dtype=np.float64),
            power=power,
        )


class PolicyOpponent(Opponent):
    name = "policy"

    def __init__(
        self,
        model: ActorCritic,
        rl_config: RLConfig,
        sim_config: SimConfig,
        *,
        deterministic: bool = False,
        label: str = "policy",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.rl_config = rl_config
        self.sim_config = sim_config
        self.deterministic = deterministic
        self.name = label
        self.checkpoint_path = checkpoint_path

    def act(self, state: GameState, team: Team) -> RLAction:
        obs = observation_from_state(state, team, self.sim_config, self.rl_config)
        return self.model.act_numpy(obs, deterministic=self.deterministic)
