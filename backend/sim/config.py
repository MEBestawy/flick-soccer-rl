"""
Configuration for physics simulation.

All tunable parameters are defined here as dataclasses.
Physics.py and other modules read from these configs, never hardcoding values.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics constants and tuning parameters."""

    # Simulation
    timestep: float = 1.0 / 120.0  # 120 Hz fixed timestep
    max_substeps: int = 8  # Adaptive substepping for high-speed objects
    
    # Pitch dimensions (world units, roughly meters scaled ~1.5x)
    pitch_width: float = 120.0
    pitch_height: float = 72.0
    
    # Goal dimensions
    goal_width: float = 3.0  # Depth into/behind the pitch
    goal_height: float = 14.0  # Width of goal mouth (Y range)
    goal_post_radius: float = 0.5
    
    # Player physics
    player_radius: float = 2.3
    player_mass: float = 4.0
    player_drag: float = 3.5  # Higher drag = stops faster
    
    # Ball physics
    ball_radius: float = 1.35
    ball_mass: float = 1.0
    ball_drag: float = 1.2  # Lower drag = travels farther
    
    # Collision
    restitution: float = 0.82  # Ball-player bounce
    player_player_restitution: float = 0.7
    wall_restitution: float = 0.8
    post_restitution: float = 0.9
    collision_friction: float = 0.35  # Tangential friction (reduces sliding)
    
    # Motion
    max_launch_speed: float = 120.0  # Maximum flick impulse speed (+50% from 80)
    min_launch_speed: float = 10.0  # Minimum useful flick
    sleep_threshold: float = 0.5  # Velocity below which object is "at rest"
    sleep_time_required: float = 0.1  # Seconds below threshold to sleep
    
    # CCD (Continuous Collision Detection)
    ccd_threshold: float = 20.0  # Speed above which CCD is used
    min_separation: float = 0.05  # Minimum separation after contact
    
    @property
    def goal_y_min(self) -> float:
        """Lower Y bound of goal mouth."""
        return (self.pitch_height - self.goal_height) / 2.0
    
    @property
    def goal_y_max(self) -> float:
        """Upper Y bound of goal mouth."""
        return (self.pitch_height + self.goal_height) / 2.0


@dataclass
class SimConfig:
    """Game simulation configuration."""
    
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    
    # Game rules
    team_size: int = 5
    goals_to_win: int = 3
    max_turns: int = 100  # Per team, prevents infinite games
    turn_time_limit: float = 30.0  # Seconds to make a move (for UI)
    match_time_limit: float = 600.0  # 10 minutes max match time
    
    # Simulation limits
    max_simulation_time: float = 30.0  # Max seconds to simulate one turn
    frame_capture_rate: float = 60.0  # Hz for captured animation frames
    
    # Team colors (for serialization, not used in physics)
    team_a_color: Tuple[int, int, int] = (255, 220, 100)  # Yellow/cream
    team_b_color: Tuple[int, int, int] = (255, 150, 150)  # Pink/coral
    
    @classmethod
    def default(cls) -> "SimConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def fast(cls) -> "SimConfig":
        """Create configuration for fast simulation (testing/benchmarks)."""
        return cls(
            physics=PhysicsConfig(
                sleep_threshold=1.0,  # Higher threshold = faster settling
                ball_drag=2.0,  # More drag = faster settling
            ),
            frame_capture_rate=30.0,
        )
