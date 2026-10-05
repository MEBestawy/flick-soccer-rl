"""
Core data models for game state.

All models are designed to be:
- Immutable or easily cloneable for state snapshots
- Serializable to JSON
- Free of any rendering or UI concerns
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import copy

from .geometry import Vec2


class GamePhase(Enum):
    """Current phase of the game."""
    
    KICKOFF = auto()      # Waiting for initial kick
    AIMING = auto()       # Player selecting shot
    SIMULATING = auto()   # Physics running
    TURN_END = auto()     # Turn complete, transitioning
    GOAL = auto()         # Goal scored, celebrating
    RESETTING = auto()    # Resetting positions after goal
    GAME_OVER = auto()    # Match complete


class Team(Enum):
    """Team identifier."""
    
    A = "A"  # Left side (yellow/cream)
    B = "B"  # Right side (pink/coral)
    
    @property
    def opponent(self) -> Team:
        """Get opposing team."""
        return Team.B if self == Team.A else Team.A


@dataclass
class Player:
    """
    A player (disc) on the field.
    
    Players are physics objects that can be flicked by the active team.
    """
    
    id: str
    team: Team
    position: Vec2
    velocity: Vec2 = field(default_factory=Vec2.zero)
    
    # Physics state
    radius: float = 2.3
    mass: float = 4.0
    is_sleeping: bool = False
    sleep_timer: float = 0.0
    
    def clone(self) -> Player:
        """Create a deep copy."""
        return Player(
            id=self.id,
            team=self.team,
            position=Vec2(self.position.x, self.position.y),
            velocity=Vec2(self.velocity.x, self.velocity.y),
            radius=self.radius,
            mass=self.mass,
            is_sleeping=self.is_sleeping,
            sleep_timer=self.sleep_timer,
        )
    
    @property
    def speed(self) -> float:
        """Current speed."""
        return self.velocity.length()
    
    def is_at_rest(self, threshold: float) -> bool:
        """Check if player is effectively stationary."""
        return self.velocity.length() < threshold


@dataclass
class Ball:
    """
    The ball.
    
    Lighter and less drag than players, travels farther when hit.
    """
    
    position: Vec2
    velocity: Vec2 = field(default_factory=Vec2.zero)
    
    # Physics state
    radius: float = 1.35
    mass: float = 1.0
    is_sleeping: bool = False
    sleep_timer: float = 0.0
    
    def clone(self) -> Ball:
        """Create a deep copy."""
        return Ball(
            position=Vec2(self.position.x, self.position.y),
            velocity=Vec2(self.velocity.x, self.velocity.y),
            radius=self.radius,
            mass=self.mass,
            is_sleeping=self.is_sleeping,
            sleep_timer=self.sleep_timer,
        )
    
    @property
    def speed(self) -> float:
        """Current speed."""
        return self.velocity.length()
    
    def is_at_rest(self, threshold: float) -> bool:
        """Check if ball is effectively stationary."""
        return self.velocity.length() < threshold


@dataclass
class GameState:
    """
    Complete game state at a point in time.
    
    This is the authoritative representation of the game.
    All state needed to continue simulation is contained here.
    """
    
    # Core state
    phase: GamePhase
    current_team: Team
    turn_number: int
    
    # Entities
    players: List[Player]
    ball: Ball
    
    # Scores
    score_a: int = 0
    score_b: int = 0
    
    # Timing
    match_time: float = 0.0  # Total elapsed match time
    turn_time: float = 0.0   # Time in current turn
    simulation_time: float = 0.0  # Time in current simulation
    
    # Metadata
    last_touch_team: Optional[Team] = None
    last_touch_player: Optional[str] = None
    
    def clone(self) -> GameState:
        """Create a deep copy of the state."""
        return GameState(
            phase=self.phase,
            current_team=self.current_team,
            turn_number=self.turn_number,
            players=[p.clone() for p in self.players],
            ball=self.ball.clone(),
            score_a=self.score_a,
            score_b=self.score_b,
            match_time=self.match_time,
            turn_time=self.turn_time,
            simulation_time=self.simulation_time,
            last_touch_team=self.last_touch_team,
            last_touch_player=self.last_touch_player,
        )
    
    def get_player(self, player_id: str) -> Optional[Player]:
        """Get player by ID."""
        for player in self.players:
            if player.id == player_id:
                return player
        return None
    
    def get_team_players(self, team: Team) -> List[Player]:
        """Get all players on a team."""
        return [p for p in self.players if p.team == team]
    
    def get_controllable_players(self) -> List[Player]:
        """Get players that current team can control (flick)."""
        if self.phase not in (GamePhase.AIMING, GamePhase.KICKOFF):
            return []
        return self.get_team_players(self.current_team)
    
    def all_at_rest(self, threshold: float) -> bool:
        """Check if all entities are at rest."""
        if not self.ball.is_at_rest(threshold):
            return False
        return all(p.is_at_rest(threshold) for p in self.players)
    
    def is_game_over(self, goals_to_win: int) -> bool:
        """Check if game has ended."""
        return self.score_a >= goals_to_win or self.score_b >= goals_to_win
    
    def winner(self, goals_to_win: int) -> Optional[Team]:
        """Get winning team, or None if game not over."""
        if self.score_a >= goals_to_win:
            return Team.A
        if self.score_b >= goals_to_win:
            return Team.B
        return None


@dataclass
class Frame:
    """
    A snapshot of entity positions for animation.
    
    Captured at regular intervals during simulation for smooth playback.
    """
    
    time: float  # Time within simulation
    ball_position: Vec2
    ball_velocity: Vec2
    player_positions: Dict[str, Vec2]
    player_velocities: Dict[str, Vec2]
    
    @staticmethod
    def from_state(state: GameState, time: float) -> Frame:
        """Create frame from game state."""
        return Frame(
            time=time,
            ball_position=Vec2(state.ball.position.x, state.ball.position.y),
            ball_velocity=Vec2(state.ball.velocity.x, state.ball.velocity.y),
            player_positions={
                p.id: Vec2(p.position.x, p.position.y) for p in state.players
            },
            player_velocities={
                p.id: Vec2(p.velocity.x, p.velocity.y) for p in state.players
            },
        )
