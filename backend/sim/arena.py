"""
Arena definition - pitch, walls, goals.

Defines the playing field geometry including:
- Pitch boundaries
- Goal areas with posts
- Wall segments for collision
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from .geometry import Vec2, Segment, Circle
from .config import PhysicsConfig


@dataclass
class Goal:
    """Goal definition with posts and bounds."""
    
    side: str  # "left" or "right"
    mouth_y_min: float
    mouth_y_max: float
    x_position: float  # X coordinate of goal line
    depth: float  # How far goal extends beyond pitch
    
    # Posts as circles
    top_post: Circle
    bottom_post: Circle
    
    # Segments for collision
    back_wall: Segment  # Back of goal
    top_wall: Segment   # Top side of goal
    bottom_wall: Segment  # Bottom side of goal
    
    def ball_in_goal(self, ball_center: Vec2, ball_radius: float = 0.0) -> bool:
        """Check if ball center is in goal."""
        if self.side == "left":
            in_x = ball_center.x < self.x_position
        else:
            in_x = ball_center.x > self.x_position
        
        in_y = self.mouth_y_min < ball_center.y < self.mouth_y_max
        return in_x and in_y


@dataclass
class Arena:
    """
    Complete arena geometry.
    
    Includes pitch boundaries, goals, and collision segments.
    """
    
    width: float
    height: float
    
    # Goals
    left_goal: Goal
    right_goal: Goal
    
    # Wall segments (for collision detection)
    walls: List[Segment]
    
    # Goal post circles
    posts: List[Tuple[Circle, str, str]]  # (circle, goal_side, post_position)
    
    @staticmethod
    def create(config: PhysicsConfig) -> Arena:
        """Create arena from physics config."""
        w = config.pitch_width
        h = config.pitch_height
        goal_h = config.goal_height
        goal_d = config.goal_width
        post_r = config.goal_post_radius
        
        # Goal Y bounds
        goal_y_min = (h - goal_h) / 2.0
        goal_y_max = (h + goal_h) / 2.0
        
        # Left goal (Team A defends)
        left_goal = Goal(
            side="left",
            mouth_y_min=goal_y_min,
            mouth_y_max=goal_y_max,
            x_position=0.0,
            depth=goal_d,
            top_post=Circle(Vec2(0.0, goal_y_max), post_r),
            bottom_post=Circle(Vec2(0.0, goal_y_min), post_r),
            back_wall=Segment(Vec2(-goal_d, goal_y_min), Vec2(-goal_d, goal_y_max)),
            top_wall=Segment(Vec2(-goal_d, goal_y_max), Vec2(0.0, goal_y_max)),
            bottom_wall=Segment(Vec2(-goal_d, goal_y_min), Vec2(0.0, goal_y_min)),
        )
        
        # Right goal (Team B defends)
        right_goal = Goal(
            side="right",
            mouth_y_min=goal_y_min,
            mouth_y_max=goal_y_max,
            x_position=w,
            depth=goal_d,
            top_post=Circle(Vec2(w, goal_y_max), post_r),
            bottom_post=Circle(Vec2(w, goal_y_min), post_r),
            back_wall=Segment(Vec2(w + goal_d, goal_y_min), Vec2(w + goal_d, goal_y_max)),
            top_wall=Segment(Vec2(w, goal_y_max), Vec2(w + goal_d, goal_y_max)),
            bottom_wall=Segment(Vec2(w, goal_y_min), Vec2(w + goal_d, goal_y_min)),
        )
        
        # Pitch walls (excluding goal mouths)
        walls = [
            # Top wall
            Segment(Vec2(0.0, h), Vec2(w, h)),
            # Bottom wall
            Segment(Vec2(0.0, 0.0), Vec2(w, 0.0)),
            # Left wall above goal
            Segment(Vec2(0.0, goal_y_max), Vec2(0.0, h)),
            # Left wall below goal
            Segment(Vec2(0.0, 0.0), Vec2(0.0, goal_y_min)),
            # Right wall above goal
            Segment(Vec2(w, goal_y_max), Vec2(w, h)),
            # Right wall below goal
            Segment(Vec2(w, 0.0), Vec2(w, goal_y_min)),
            # Left goal walls
            left_goal.back_wall,
            left_goal.top_wall,
            left_goal.bottom_wall,
            # Right goal walls
            right_goal.back_wall,
            right_goal.top_wall,
            right_goal.bottom_wall,
        ]
        
        # Goal posts
        posts = [
            (left_goal.top_post, "left", "top"),
            (left_goal.bottom_post, "left", "bottom"),
            (right_goal.top_post, "right", "top"),
            (right_goal.bottom_post, "right", "bottom"),
        ]
        
        return Arena(
            width=w,
            height=h,
            left_goal=left_goal,
            right_goal=right_goal,
            walls=walls,
            posts=posts,
        )
    
    def point_in_bounds(self, point: Vec2, margin: float = 0.0) -> bool:
        """Check if point is within pitch bounds."""
        return (
            -margin <= point.x <= self.width + margin and
            -margin <= point.y <= self.height + margin
        )
    
    def clamp_to_bounds(self, point: Vec2, margin: float = 0.0) -> Vec2:
        """Clamp point to pitch bounds."""
        x = max(margin, min(self.width - margin, point.x))
        y = max(margin, min(self.height - margin, point.y))
        return Vec2(x, y)
    
    def get_wall_name(self, segment: Segment) -> str:
        """Get descriptive name for a wall segment."""
        # Check if it's a horizontal wall
        if abs(segment.start.y - segment.end.y) < 0.01:
            if segment.start.y > self.height / 2:
                return "top"
            else:
                return "bottom"
        # Vertical wall
        if segment.start.x < self.width / 2:
            return "left"
        else:
            return "right"
