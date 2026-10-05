"""
Match runner that uses swappable Agent strategies for each side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..actions import FlickAction
from ..config import SimConfig
from ..env import HeadlessEnv
from ..models import GamePhase, GameState, Team
from .base import Agent


@dataclass
class TurnRecord:
    """One completed turn in a match."""

    turn_number: int
    team: str
    agent_name: str
    action: FlickAction
    reward: float
    score_a: int
    score_b: int
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Outcome of a full match between two agents."""

    score_a: int
    score_b: int
    winner: Optional[str]
    turns: int
    history: List[TurnRecord]
    final_state: GameState


class MatchRunner:
    """
    Plays a match using strategy-pattern agents for each team.

    Example:
        runner = MatchRunner(
            team_a=RandomAgent(Team.A),
            team_b=JevAgent(Team.B),
        )
        result = runner.play()
    """

    def __init__(
        self,
        team_a: Agent,
        team_b: Agent,
        *,
        config: Optional[SimConfig] = None,
        max_turns: int = 200,
    ) -> None:
        if team_a.team != Team.A:
            raise ValueError("team_a agent must be assigned Team.A")
        if team_b.team != Team.B:
            raise ValueError("team_b agent must be assigned Team.B")

        self.team_a = team_a
        self.team_b = team_b
        self.config = config or SimConfig.default()
        self.max_turns = max_turns
        self.env = HeadlessEnv(self.config)

    def agent_for(self, team: Team) -> Agent:
        return self.team_a if team == Team.A else self.team_b

    def play(self, starting_team: Team = Team.A) -> MatchResult:
        state = self.env.reset(starting_team)
        self.team_a.on_match_start(state)
        self.team_b.on_match_start(state)

        history: List[TurnRecord] = []
        turns = 0

        while not self.env.is_done() and turns < self.max_turns:
            agent = self.agent_for(state.current_team)
            action = agent.select_action(state)
            state, reward, _done, info = self.env.step(action)
            turns += 1
            agent.on_turn_end(state)

            history.append(
                TurnRecord(
                    turn_number=turns,
                    team=agent.team.value,
                    agent_name=agent.name,
                    action=action,
                    reward=reward,
                    score_a=state.score_a,
                    score_b=state.score_b,
                    info=info,
                )
            )

        winner_team = state.winner(self.config.goals_to_win)
        winner = winner_team.value if winner_team else None
        if winner is None and state.phase == GamePhase.GAME_OVER:
            if state.score_a > state.score_b:
                winner = "A"
            elif state.score_b > state.score_a:
                winner = "B"

        return MatchResult(
            score_a=state.score_a,
            score_b=state.score_b,
            winner=winner,
            turns=turns,
            history=history,
            final_state=state,
        )
