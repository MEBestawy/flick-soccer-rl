"""
Replaceable agent strategies for flick football.

Usage:
    from sim.agents import create_agent, MatchRunner, JevAgent

    jev = create_agent("jev", Team.B)
    opp = create_agent("heuristic", Team.A)
    result = MatchRunner(team_a=opp, team_b=jev).play()
"""

from .base import Agent
from .heuristic_agent import HeuristicAgent
from .jev_agent import JevAgent
from .jev_client import JevClient, JevClientError
from .match import MatchResult, MatchRunner, TurnRecord
from .prompt import (
    build_normalized_observation,
    normalize_drag,
    normalize_position,
    normalize_size,
    observation_to_prompt_text,
)
from .random_agent import RandomAgent
from .registry import available_agents, create_agent, register_agent
from .rl_agent import RLAgent, resolve_rl_checkpoint, resolve_rl_first_checkpoint

__all__ = [
    "Agent",
    "RandomAgent",
    "HeuristicAgent",
    "JevAgent",
    "RLAgent",
    "JevClient",
    "JevClientError",
    "MatchRunner",
    "MatchResult",
    "TurnRecord",
    "create_agent",
    "register_agent",
    "available_agents",
    "resolve_rl_checkpoint",
    "resolve_rl_first_checkpoint",
    "build_normalized_observation",
    "observation_to_prompt_text",
    "normalize_position",
    "normalize_size",
    "normalize_drag",
]
