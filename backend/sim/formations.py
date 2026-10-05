"""
Team formations.

Defines starting positions for players in various formations.
Positions are normalized (0-1) and scaled to pitch dimensions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

from .geometry import Vec2
from .config import PhysicsConfig


@dataclass
class Formation:
    """
    A team formation defining relative player positions.
    
    Positions are normalized 0-1, where:
    - X: 0 = own goal line, 1 = opponent goal line
    - Y: 0 = bottom, 1 = top
    """
    
    name: str
    positions: List[Tuple[float, float]]  # List of (x, y) normalized positions
    
    def get_absolute_positions(
        self,
        config: PhysicsConfig,
        team_side: str,  # "left" or "right"
    ) -> List[Vec2]:
        """
        Get absolute positions for a team.
        
        Args:
            config: Physics config with pitch dimensions
            team_side: "left" (Team A) or "right" (Team B)
        
        Returns:
            List of Vec2 positions in world coordinates
        """
        w = config.pitch_width
        h = config.pitch_height
        
        result: List[Vec2] = []
        
        for nx, ny in self.positions:
            if team_side == "left":
                # Team A: x goes from 0 to half
                x = nx * (w / 2)
            else:
                # Team B: mirror, x goes from half to w
                x = w - nx * (w / 2)
            
            y = ny * h
            result.append(Vec2(x, y))
        
        return result


# Standard 5-player formations
FORMATION_2_1_2 = Formation(
    name="2-1-2",
    positions=[
        # Goalkeeper
        (0.08, 0.5),
        # Defenders
        (0.25, 0.25),
        (0.25, 0.75),
        # Midfielder
        (0.5, 0.5),
        # Forwards
        (0.75, 0.35),
    ]
)

FORMATION_1_2_2 = Formation(
    name="1-2-2",
    positions=[
        # Goalkeeper
        (0.08, 0.5),
        # Defender
        (0.3, 0.5),
        # Midfielders
        (0.5, 0.3),
        (0.5, 0.7),
        # Forward
        (0.75, 0.5),
    ]
)

FORMATION_2_2_1 = Formation(
    name="2-2-1",
    positions=[
        # Goalkeeper
        (0.08, 0.5),
        # Defenders
        (0.25, 0.3),
        (0.25, 0.7),
        # Midfielders
        (0.5, 0.35),
        (0.5, 0.65),
    ]
)

FORMATION_DIAMOND = Formation(
    name="Diamond",
    positions=[
        # Goalkeeper
        (0.08, 0.5),
        # Back
        (0.25, 0.5),
        # Wide
        (0.45, 0.25),
        (0.45, 0.75),
        # Forward
        (0.7, 0.5),
    ]
)

FORMATION_DEFAULT = FORMATION_2_1_2


def get_default_formation() -> Formation:
    """Get the default formation."""
    return FORMATION_DEFAULT


def get_all_formations() -> Dict[str, Formation]:
    """Get all available formations."""
    return {
        "2-1-2": FORMATION_2_1_2,
        "1-2-2": FORMATION_1_2_2,
        "2-2-1": FORMATION_2_2_1,
        "diamond": FORMATION_DIAMOND,
    }


def get_kickoff_positions(
    config: PhysicsConfig,
    kicking_team: str,  # "left" or "right"
) -> Tuple[List[Vec2], List[Vec2], Vec2]:
    """
    Get positions for kickoff.
    
    Returns:
        (team_a_positions, team_b_positions, ball_position)
    """
    formation = get_default_formation()
    
    # Standard positions
    team_a_pos = formation.get_absolute_positions(config, "left")
    team_b_pos = formation.get_absolute_positions(config, "right")
    
    # Ball at center
    ball_pos = Vec2(config.pitch_width / 2, config.pitch_height / 2)
    
    # Adjust kicking team's forward player closer to ball
    if kicking_team == "left":
        # Move Team A's most forward player
        max_x_idx = max(range(len(team_a_pos)), key=lambda i: team_a_pos[i].x)
        team_a_pos[max_x_idx] = Vec2(
            config.pitch_width / 2 - 5,
            config.pitch_height / 2
        )
    else:
        # Move Team B's most forward player
        min_x_idx = min(range(len(team_b_pos)), key=lambda i: team_b_pos[i].x)
        team_b_pos[min_x_idx] = Vec2(
            config.pitch_width / 2 + 5,
            config.pitch_height / 2
        )
    
    return team_a_pos, team_b_pos, ball_pos


def get_goal_reset_positions(
    config: PhysicsConfig,
    scored_on_team: str,  # "left" or "right" - team that was scored on
) -> Tuple[List[Vec2], List[Vec2], Vec2]:
    """
    Get positions after a goal is scored.
    
    The team that was scored on kicks off.
    """
    kicking_team = scored_on_team
    return get_kickoff_positions(config, kicking_team)
