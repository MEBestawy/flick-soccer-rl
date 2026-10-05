"""Random legal-action agent."""

from __future__ import annotations

import math
import random
from typing import Optional

from ..actions import FlickAction
from ..config import SimConfig
from ..geometry import Vec2
from ..models import GameState, Team
from .base import Agent


class RandomAgent(Agent):
    """Selects a uniformly random controllable player, direction, and power."""

    name = "random"

    def __init__(
        self,
        team: Team,
        config: Optional[SimConfig] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(team, config)
        self._rng = random.Random(seed)

    def select_action(self, state: GameState) -> FlickAction:
        players = [
            p for p in state.get_controllable_players() if p.team == self.team
        ]
        if not players:
            raise RuntimeError(f"No controllable players for team {self.team.value}")

        player = self._rng.choice(players)
        angle = self._rng.uniform(0.0, 2.0 * math.pi)
        power = self._rng.uniform(0.25, 1.0)
        return FlickAction(
            player_id=player.id,
            direction=Vec2(math.cos(angle), math.sin(angle)),
            power=power,
        )
