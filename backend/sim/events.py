"""
Game events system.

Events are emitted during simulation to track what happened.
Used for replays, logging, and UI notifications.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List

from .geometry import Vec2
from .models import Team


class EventType(Enum):
    """Types of game events."""
    
    # Match events
    MATCH_START = auto()
    MATCH_END = auto()
    
    # Turn events
    TURN_START = auto()
    TURN_END = auto()
    
    # Action events
    FLICK = auto()
    
    # Physics events
    COLLISION_BALL_PLAYER = auto()
    COLLISION_PLAYER_PLAYER = auto()
    COLLISION_WALL = auto()
    COLLISION_POST = auto()
    
    # Goal events
    GOAL_SCORED = auto()
    GOAL_SAVED = auto()  # Ball enters goal area but saved by player
    
    # State events
    ALL_AT_REST = auto()
    PHASE_CHANGE = auto()
    
    # Error events
    INVALID_ACTION = auto()
    TIMEOUT = auto()


@dataclass
class GameEvent:
    """
    A game event that occurred during simulation.
    
    Events are timestamped and contain relevant data for the event type.
    """
    
    type: EventType
    time: float  # Simulation time when event occurred
    data: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def match_start() -> GameEvent:
        """Create match start event."""
        return GameEvent(type=EventType.MATCH_START, time=0.0)
    
    @staticmethod
    def match_end(winner: Optional[Team], score_a: int, score_b: int) -> GameEvent:
        """Create match end event."""
        return GameEvent(
            type=EventType.MATCH_END,
            time=0.0,
            data={
                "winner": winner.value if winner else None,
                "score_a": score_a,
                "score_b": score_b,
            }
        )
    
    @staticmethod
    def turn_start(team: Team, turn_number: int) -> GameEvent:
        """Create turn start event."""
        return GameEvent(
            type=EventType.TURN_START,
            time=0.0,
            data={"team": team.value, "turn_number": turn_number}
        )
    
    @staticmethod
    def turn_end(team: Team, turn_number: int) -> GameEvent:
        """Create turn end event."""
        return GameEvent(
            type=EventType.TURN_END,
            time=0.0,
            data={"team": team.value, "turn_number": turn_number}
        )
    
    @staticmethod
    def flick(
        player_id: str,
        team: Team,
        direction: Vec2,
        power: float,
        time: float = 0.0
    ) -> GameEvent:
        """Create flick event."""
        return GameEvent(
            type=EventType.FLICK,
            time=time,
            data={
                "player_id": player_id,
                "team": team.value,
                "direction": direction.to_tuple(),
                "power": power,
            }
        )
    
    @staticmethod
    def collision_ball_player(
        player_id: str,
        position: Vec2,
        time: float
    ) -> GameEvent:
        """Create ball-player collision event."""
        return GameEvent(
            type=EventType.COLLISION_BALL_PLAYER,
            time=time,
            data={
                "player_id": player_id,
                "position": position.to_tuple(),
            }
        )
    
    @staticmethod
    def collision_player_player(
        player1_id: str,
        player2_id: str,
        position: Vec2,
        time: float
    ) -> GameEvent:
        """Create player-player collision event."""
        return GameEvent(
            type=EventType.COLLISION_PLAYER_PLAYER,
            time=time,
            data={
                "player1_id": player1_id,
                "player2_id": player2_id,
                "position": position.to_tuple(),
            }
        )
    
    @staticmethod
    def collision_wall(
        entity_id: str,
        wall: str,  # "top", "bottom", "left", "right"
        position: Vec2,
        time: float
    ) -> GameEvent:
        """Create wall collision event."""
        return GameEvent(
            type=EventType.COLLISION_WALL,
            time=time,
            data={
                "entity_id": entity_id,
                "wall": wall,
                "position": position.to_tuple(),
            }
        )
    
    @staticmethod
    def collision_post(
        entity_id: str,
        goal_side: str,  # "left" or "right"
        post: str,  # "top" or "bottom"
        position: Vec2,
        time: float
    ) -> GameEvent:
        """Create goal post collision event."""
        return GameEvent(
            type=EventType.COLLISION_POST,
            time=time,
            data={
                "entity_id": entity_id,
                "goal_side": goal_side,
                "post": post,
                "position": position.to_tuple(),
            }
        )
    
    @staticmethod
    def goal_scored(
        team: Team,
        scorer_id: Optional[str],
        time: float
    ) -> GameEvent:
        """Create goal scored event."""
        return GameEvent(
            type=EventType.GOAL_SCORED,
            time=time,
            data={
                "scoring_team": team.value,
                "scorer_id": scorer_id,
            }
        )
    
    @staticmethod
    def all_at_rest(time: float) -> GameEvent:
        """Create all-at-rest event."""
        return GameEvent(type=EventType.ALL_AT_REST, time=time)
    
    @staticmethod
    def phase_change(old_phase: str, new_phase: str, time: float) -> GameEvent:
        """Create phase change event."""
        return GameEvent(
            type=EventType.PHASE_CHANGE,
            time=time,
            data={"old_phase": old_phase, "new_phase": new_phase}
        )


class EventLog:
    """Collection of events for a simulation run."""
    
    def __init__(self) -> None:
        self._events: List[GameEvent] = []
    
    def add(self, event: GameEvent) -> None:
        """Add event to log."""
        self._events.append(event)
    
    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
    
    @property
    def events(self) -> List[GameEvent]:
        """Get all events (copy)."""
        return list(self._events)
    
    def get_by_type(self, event_type: EventType) -> List[GameEvent]:
        """Get events of specific type."""
        return [e for e in self._events if e.type == event_type]
    
    def __len__(self) -> int:
        return len(self._events)
    
    def __iter__(self):
        return iter(self._events)
