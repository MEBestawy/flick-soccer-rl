"""
Agent strategy interface.

Agents decide FlickActions from GameState. Both sides of a match can use
any Agent implementation — human, random, heuristic, Jev, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..actions import FlickAction
from ..config import SimConfig
from ..models import GameState, Team


class Agent(ABC):
    """
    Strategy interface for selecting a flick action.

    Implementations must be replaceable without changing match / env code.
    """

    name: str = "base"

    def __init__(self, team: Team, config: Optional[SimConfig] = None) -> None:
        self.team = team
        self.config = config or SimConfig.default()

    @abstractmethod
    def select_action(self, state: GameState) -> FlickAction:
        """
        Choose a legal FlickAction for the agent's team.

        Args:
            state: Current authoritative game state (must be AIMING/KICKOFF
                   and current_team == self.team).

        Returns:
            A FlickAction to apply.
        """

    def on_match_start(self, state: GameState) -> None:
        """Optional hook when a match begins."""

    def on_turn_end(self, state: GameState) -> None:
        """Optional hook after a turn settles."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(team={self.team.value}, name={self.name!r})"
