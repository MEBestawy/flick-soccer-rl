"""Scripted teachers for BC / curriculum."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sim.config import SimConfig
from sim.models import GameState, Team

from .actions import RLAction
from .observations import friendly_player_ids
from .opponents import Opponent


class ShootAtGoalTeacher(Opponent):
    """
    Pick the player nearest the ball and flick toward the opponent goal mouth.

    Much clearer scoring prior than the generic heuristic for early BC.
    """

    name = "shoot_teacher"

    def __init__(self, config: Optional[SimConfig] = None, power: float = 0.85) -> None:
        self.config = config or SimConfig.default()
        self.power = power

    def act(self, state: GameState, team: Team) -> RLAction:
        ids = friendly_player_ids(state, team)
        best_i = 0
        best_d = 1e18
        for i, pid in enumerate(ids):
            p = state.get_player(pid)
            if p is None:
                continue
            d = (state.ball.position - p.position).length_squared()
            if d < best_d:
                best_d = d
                best_i = i

        player = state.get_player(ids[best_i])
        assert player is not None

        # Aim from player through ball toward goal center (world frame),
        # then convert to canonical for RLAction.
        goal_x = self.config.physics.pitch_width if team == Team.A else 0.0
        goal_y = self.config.physics.pitch_height * 0.5

        # Prefer aiming via the ball if we're close; else go straight to goal.
        target = state.ball.position
        if best_d > (8.0 ** 2):
            # Far from ball: move toward ball first.
            wx = target.x - player.position.x
            wy = target.y - player.position.y
        else:
            # Near ball: shoot toward goal, slightly through the ball.
            through = state.ball.position
            wx = (through.x - player.position.x) * 0.35 + (goal_x - through.x) * 0.65
            wy = (through.y - player.position.y) * 0.35 + (goal_y - through.y) * 0.65

        # Canonicalize for Team B
        if team == Team.B:
            wx = -wx
        n = (wx * wx + wy * wy) ** 0.5
        if n < 1e-8:
            wx, wy = 1.0, 0.0
        else:
            wx, wy = wx / n, wy / n

        return RLAction(
            player_index=best_i,
            direction_raw=np.array([wx, wy], dtype=np.float64),
            power=self.power,
        )
