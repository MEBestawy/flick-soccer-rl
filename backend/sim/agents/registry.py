"""
Factory / registry for replaceable agents.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ..config import SimConfig
from ..models import Team
from .base import Agent
from .heuristic_agent import HeuristicAgent
from .jev_agent import JevAgent
from .random_agent import RandomAgent
from .rl_agent import RLAgent

AgentFactory = Callable[[Team, Optional[SimConfig]], Agent]

_REGISTRY: Dict[str, AgentFactory] = {
    "random": lambda team, config: RandomAgent(team, config),
    "heuristic": lambda team, config: HeuristicAgent(team, config),
    "jev": lambda team, config: JevAgent(team, config),
    "rl": lambda team, config: RLAgent(team, config, name="rl"),
    "rl-first": lambda team, config: RLAgent(team, config, name="rl-first"),
}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Register or replace an agent strategy by name."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Agent name must be non-empty")
    _REGISTRY[key] = factory


def available_agents() -> list[str]:
    """Sorted list of registered agent strategy names."""
    return sorted(_REGISTRY.keys())


def create_agent(
    name: str,
    team: Team,
    config: Optional[SimConfig] = None,
) -> Agent:
    """
    Create an agent by strategy name.

    Raises:
        KeyError: unknown strategy name
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(available_agents())
        raise KeyError(f"Unknown agent '{name}'. Known: {known}")
    return _REGISTRY[key](team, config)
