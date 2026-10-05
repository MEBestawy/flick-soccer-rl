"""
Player actions and validation.

Actions are the inputs to the simulation - what players do during their turn.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum, auto

from .geometry import Vec2
from .models import GameState, GamePhase, Team, Player


class ActionError(Enum):
    """Reasons an action might be invalid."""
    
    WRONG_PHASE = auto()
    WRONG_TEAM = auto()
    INVALID_PLAYER = auto()
    PLAYER_NOT_FOUND = auto()
    POWER_OUT_OF_RANGE = auto()
    ZERO_DIRECTION = auto()


@dataclass(frozen=True)
class FlickAction:
    """
    A flick action - applying impulse to a player.
    
    This is the primary action in the game. Players drag back and release
    to flick their disc toward the ball.
    
    Attributes:
        player_id: ID of the player to flick
        direction: Direction vector (will be normalized)
        power: Power of flick, 0.0 to 1.0
    """
    
    player_id: str
    direction: Vec2
    power: float  # 0.0 to 1.0
    
    def normalized_direction(self) -> Vec2:
        """Get normalized direction vector."""
        return self.direction.normalized()
    
    def validate(self, state: GameState) -> Optional[ActionError]:
        """
        Validate action against current game state.
        
        Returns:
            None if valid, ActionError if invalid
        """
        # Check phase
        if state.phase not in (GamePhase.AIMING, GamePhase.KICKOFF):
            return ActionError.WRONG_PHASE
        
        # Check power range
        if not (0.0 <= self.power <= 1.0):
            return ActionError.POWER_OUT_OF_RANGE
        
        # Check direction
        if self.direction.length() < 1e-6:
            return ActionError.ZERO_DIRECTION
        
        # Check player exists
        player = state.get_player(self.player_id)
        if player is None:
            return ActionError.PLAYER_NOT_FOUND
        
        # Check player belongs to current team
        if player.team != state.current_team:
            return ActionError.WRONG_TEAM
        
        return None
    
    def is_valid(self, state: GameState) -> bool:
        """Check if action is valid."""
        return self.validate(state) is None


@dataclass
class ActionResult:
    """
    Result of executing an action.
    
    Contains the final state, animation frames, and events.
    """
    
    success: bool
    error: Optional[ActionError] = None
    error_message: Optional[str] = None
    
    # State snapshots
    start_state: Optional[GameState] = None
    final_state: Optional[GameState] = None
    
    # Animation data
    frames: Optional[List] = None  # List[Frame]
    
    # Events that occurred
    events: Optional[List] = None  # List[GameEvent]
    
    # Timing
    simulation_duration: float = 0.0  # Real seconds to simulate
    simulated_time: float = 0.0  # In-game seconds simulated
    
    @staticmethod
    def failure(error: ActionError, message: str) -> ActionResult:
        """Create a failure result."""
        return ActionResult(
            success=False,
            error=error,
            error_message=message,
        )
    
    @staticmethod
    def invalid_action(error: ActionError) -> ActionResult:
        """Create result for invalid action."""
        messages = {
            ActionError.WRONG_PHASE: "Cannot perform action in current game phase",
            ActionError.WRONG_TEAM: "Player does not belong to current team",
            ActionError.INVALID_PLAYER: "Invalid player specified",
            ActionError.PLAYER_NOT_FOUND: "Player not found",
            ActionError.POWER_OUT_OF_RANGE: "Power must be between 0 and 1",
            ActionError.ZERO_DIRECTION: "Direction cannot be zero vector",
        }
        return ActionResult.failure(error, messages.get(error, "Unknown error"))


def get_legal_actions(
    state: GameState,
    direction_samples: int = 8,
    power_samples: int = 3,
) -> List[FlickAction]:
    """
    Generate a list of legal actions for current game state.
    
    Useful for AI/random play. Samples directions and powers.
    
    Args:
        state: Current game state
        direction_samples: Number of direction angles to sample (around circle)
        power_samples: Number of power levels to sample
    
    Returns:
        List of valid FlickAction objects
    """
    import math
    
    actions: List[FlickAction] = []
    controllable = state.get_controllable_players()
    
    if not controllable:
        return actions
    
    # Sample directions evenly around circle
    directions = [
        Vec2.from_angle(2 * math.pi * i / direction_samples)
        for i in range(direction_samples)
    ]
    
    # Sample power levels (excluding 0)
    powers = [
        (i + 1) / power_samples
        for i in range(power_samples)
    ]
    
    for player in controllable:
        for direction in directions:
            for power in powers:
                action = FlickAction(
                    player_id=player.id,
                    direction=direction,
                    power=power,
                )
                if action.is_valid(state):
                    actions.append(action)
    
    return actions
