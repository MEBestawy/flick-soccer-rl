"""Simple heuristic agent: flick the player nearest the ball toward the ball."""

from __future__ import annotations

from typing import Optional

from ..actions import FlickAction
from ..config import SimConfig
from ..geometry import Vec2
from ..models import GameState, Team
from .base import Agent


class HeuristicAgent(Agent):
    """
    Greedy baseline:
    - Pick controllable player closest to the ball
    - Aim through the ball toward the opponent goal
    - Scale power by distance to ball
    """

    name = "heuristic"

    def __init__(self, team: Team, config: Optional[SimConfig] = None) -> None:
        super().__init__(team, config)

    def select_action(self, state: GameState) -> FlickAction:
        players = [
            p for p in state.get_controllable_players() if p.team == self.team
        ]
        if not players:
            raise RuntimeError(f"No controllable players for team {self.team.value}")

        ball = state.ball.position
        player = min(players, key=lambda p: (p.position - ball).length_squared())

        # Aim: player → ball, with a small bias toward opponent goal center.
        physics = self.config.physics
        goal_x = physics.pitch_width if self.team == Team.A else 0.0
        goal_y = physics.pitch_height / 2.0
        goal = Vec2(goal_x, goal_y)

        to_ball = ball - player.position
        through_ball = (goal - ball).normalized()
        # Blend: mostly hit the ball, slightly aim for a useful follow-through.
        direction = (to_ball.normalized() * 0.75 + through_ball * 0.25)
        if direction.length() < 1e-6:
            direction = to_ball if to_ball.length() > 1e-6 else Vec2(1.0, 0.0)

        dist = to_ball.length()
        # Closer → softer; farther → stronger (clamped).
        power = min(1.0, max(0.35, dist / (physics.pitch_width * 0.35)))

        return FlickAction(
            player_id=player.id,
            direction=direction.normalized(),
            power=power,
        )
