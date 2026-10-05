"""Toy open-goal environments for PPO sanity checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from sim.config import SimConfig
from sim.geometry import Vec2
from sim.models import GamePhase, Team

from .actions import RLAction
from .config import RLConfig
from .env import FlickRLEnv, StepResult
from .observations import observation_from_state


class OpenGoalSanityEnv(FlickRLEnv):
    """
    Single decision scenario: one friendly player behind the ball, open goal.

    Opponents are parked far away so a forward flick should score.
    Episode ends on goal, miss timeout, or max turns.
    """

    def __init__(
        self,
        sim_config: Optional[SimConfig] = None,
        rl_config: Optional[RLConfig] = None,
        *,
        max_turns: int = 8,
    ) -> None:
        base = sim_config or SimConfig.default()
        cfg = replace(base, goals_to_win=1)
        rl = rl_config or RLConfig()
        rl = RLConfig.from_dict({**rl.to_dict(), "max_turns_per_game": max_turns})
        super().__init__(cfg, rl)
        self._max_turns = max_turns

    def _apply_open_goal_layout(self) -> None:
        assert self._state is not None
        physics = self.sim_config.physics
        w, h = physics.pitch_width, physics.pitch_height
        cy = h / 2.0

        # Ball near right goal, player just behind it
        ball_x = w - 18.0
        player_x = ball_x - 8.0
        self._state.ball.position = Vec2(ball_x, cy)
        self._state.ball.velocity = Vec2.zero()
        self._state.ball.is_sleeping = True

        for p in self._state.players:
            p.velocity = Vec2.zero()
            p.is_sleeping = True
            if p.team == Team.A:
                if p.id == "A1":
                    p.position = Vec2(player_x, cy)
                elif p.id == "A2":
                    p.position = Vec2(player_x - 10.0, cy - 12.0)
                elif p.id == "A3":
                    p.position = Vec2(player_x - 10.0, cy + 12.0)
                else:
                    p.position = Vec2(30.0, cy + (hash(p.id) % 7 - 3) * 4.0)
            else:
                # Park opponents near their own half, out of the shot lane
                idx = int(p.id[1]) if p.id[1:].isdigit() else 1
                p.position = Vec2(20.0 + idx * 3.0, 8.0 + idx * 4.0)

        self._state.phase = GamePhase.AIMING
        self._state.current_team = Team.A
        self._state.score_a = 0
        self._state.score_b = 0

        # Keep sim + wrapper in sync
        self.inner._state = self._state
        self.inner.simulator._state = self._state
        self.inner._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed)
        self._apply_open_goal_layout()
        assert self._state is not None
        obs = observation_from_state(
            self._state, Team.A, self.sim_config, self.rl_config
        )
        return obs, {**info, "scenario": "open_goal"}

    def step(self, action: RLAction) -> StepResult:
        result = super().step(action)
        # After a turn, if no goal, re-center for another attempt (single-agent toy)
        if not result.terminated and not result.truncated and self._state is not None:
            if self._state.score_a == 0 and self._state.current_team != Team.A:
                # Skip opponent turns: force back to Team A with fresh layout
                self._apply_open_goal_layout()
                result.obs = observation_from_state(
                    self._state, Team.A, self.sim_config, self.rl_config
                )
        return result
