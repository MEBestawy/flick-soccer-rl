"""
Headless physics-based soccer simulation.

This package provides a deterministic 2D physics simulation for turn-based
5v5 soccer (flick football). All game state and physics are handled here,
completely independent of any UI or rendering.

Usage:
    from sim import GameSimulator, GameState, FlickAction, HeadlessEnv
"""

from .config import SimConfig, PhysicsConfig
from .models import (
    GameState,
    Player,
    Ball,
    Team,
    GamePhase,
    Vec2,
)
from .actions import FlickAction, ActionResult
from .simulator import GameSimulator
from .env import HeadlessEnv
from .events import GameEvent, EventType
from .formations import Formation, get_default_formation
from .serialization import serialize_state, deserialize_state
from .agents import (
    Agent,
    RandomAgent,
    HeuristicAgent,
    JevAgent,
    MatchRunner,
    create_agent,
    available_agents,
)

__all__ = [
    # Config
    "SimConfig",
    "PhysicsConfig",
    # Models
    "GameState",
    "Player",
    "Ball",
    "Team",
    "GamePhase",
    "Vec2",
    # Actions
    "FlickAction",
    "ActionResult",
    # Simulator
    "GameSimulator",
    # Environment
    "HeadlessEnv",
    # Events
    "GameEvent",
    "EventType",
    # Formations
    "Formation",
    "get_default_formation",
    # Serialization
    "serialize_state",
    "deserialize_state",
    # Agents
    "Agent",
    "RandomAgent",
    "HeuristicAgent",
    "JevAgent",
    "MatchRunner",
    "create_agent",
    "available_agents",
]

__version__ = "1.0.0"
