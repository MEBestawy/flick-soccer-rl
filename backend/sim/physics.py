"""
Physics simulation engine.

Core physics step with:
- Exponential drag
- Adaptive substepping for high-speed objects
- Impulse-based collision response
- Sleep detection
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Protocol

from .geometry import Vec2, Circle
from .config import PhysicsConfig
from .models import Player, Ball


class PhysicsBody(Protocol):
    """Protocol for physics-enabled objects."""
    
    position: Vec2
    velocity: Vec2
    radius: float
    mass: float
    is_sleeping: bool
    sleep_timer: float


def apply_drag(velocity: Vec2, drag: float, dt: float) -> Vec2:
    """
    Apply exponential drag to velocity.
    
    v *= exp(-drag * dt)
    
    This provides smooth, stable deceleration that's independent
    of timestep (within numerical precision).
    """
    factor = math.exp(-drag * dt)
    return velocity * factor


def integrate_position(body: PhysicsBody, dt: float) -> Vec2:
    """Integrate position using velocity."""
    return body.position + body.velocity * dt


def compute_required_substeps(
    velocity: Vec2,
    radius: float,
    dt: float,
    config: PhysicsConfig
) -> int:
    """
    Compute number of substeps needed for safe collision detection.
    
    Uses CCD threshold and ensures object doesn't move more than
    its radius per substep.
    """
    speed = velocity.length()
    if speed < config.ccd_threshold:
        return 1
    
    # Maximum distance per substep should be fraction of radius
    max_step_dist = radius * 0.5
    total_dist = speed * dt
    
    substeps = int(math.ceil(total_dist / max_step_dist))
    return min(substeps, config.max_substeps)


def update_sleep_state(
    body: PhysicsBody,
    dt: float,
    threshold: float,
    time_required: float
) -> Tuple[bool, float]:
    """
    Update sleep state based on velocity.
    
    Returns:
        (is_sleeping, sleep_timer)
    """
    speed = body.velocity.length()
    
    if speed < threshold:
        new_timer = body.sleep_timer + dt
        if new_timer >= time_required:
            return (True, new_timer)
        return (False, new_timer)
    else:
        return (False, 0.0)


def impulse_response(
    pos1: Vec2, vel1: Vec2, mass1: float,
    pos2: Vec2, vel2: Vec2, mass2: float,
    normal: Vec2,
    restitution: float,
    friction: float = 0.0,
) -> Tuple[Vec2, Vec2]:
    """
    Compute impulse-based collision response.
    
    Args:
        pos1, vel1, mass1: First body state
        pos2, vel2, mass2: Second body state
        normal: Collision normal (from body 1 to body 2)
        restitution: Coefficient of restitution
        friction: Coulomb friction coefficient (0 = frictionless)
    
    Returns:
        (new_velocity1, new_velocity2)
    """
    # Relative velocity of body2 as seen from body1.
    # With normal pointing 1→2, gap rate is normal·(vel2-vel1).
    # Approaching when that rate is negative.
    rel_vel = vel2 - vel1
    vel_along_normal = rel_vel.dot(normal)

    # Already separating — do not push them apart again.
    if vel_along_normal > 0:
        return (vel1, vel2)

    inv_mass1 = 1.0 / mass1
    inv_mass2 = 1.0 / mass2
    inv_mass_sum = inv_mass1 + inv_mass2

    # Normal impulse (bounce)
    j_n = -(1.0 + restitution) * vel_along_normal
    j_n /= inv_mass_sum
    impulse = normal * j_n

    # Tangential friction impulse — kills sliding along the contact.
    if friction > 0.0:
        tangent = rel_vel - normal * vel_along_normal
        tangent_len = tangent.length()
        if tangent_len > 1e-8:
            tangent = tangent / tangent_len
            j_t = -rel_vel.dot(tangent) / inv_mass_sum
            # Coulomb clamp
            max_jt = abs(j_n) * friction
            if j_t > max_jt:
                j_t = max_jt
            elif j_t < -max_jt:
                j_t = -max_jt
            impulse = impulse + tangent * j_t

    # Apply equal-and-opposite impulse
    new_vel1 = vel1 - impulse * inv_mass1
    new_vel2 = vel2 + impulse * inv_mass2

    return (new_vel1, new_vel2)


def wall_impulse_response(
    velocity: Vec2,
    normal: Vec2,
    restitution: float
) -> Vec2:
    """
    Compute velocity after wall collision.
    
    Reflects velocity across normal with restitution.
    """
    # Velocity component along normal
    vel_normal = velocity.dot(normal)
    
    # Only reflect if moving into wall
    if vel_normal >= 0:
        return velocity
    
    # Reflect with restitution
    return velocity - normal * (1 + restitution) * vel_normal


def separate_circles(
    pos1: Vec2, radius1: float, mass1: float,
    pos2: Vec2, radius2: float, mass2: float,
    min_separation: float = 0.01
) -> Tuple[Vec2, Vec2]:
    """
    Separate overlapping circles.
    
    Moves circles apart proportional to their masses.
    """
    delta = pos2 - pos1
    dist = delta.length()
    
    if dist < 1e-10:
        # Coincident - push apart along arbitrary direction
        delta = Vec2(1.0, 0.0)
        dist = 0.0
    
    overlap = radius1 + radius2 - dist + min_separation
    if overlap <= 0:
        return (pos1, pos2)
    
    normal = delta / max(dist, 1e-10)
    
    # Move proportional to inverse mass
    total_mass = mass1 + mass2
    move1 = overlap * (mass2 / total_mass)
    move2 = overlap * (mass1 / total_mass)
    
    new_pos1 = pos1 - normal * move1
    new_pos2 = pos2 + normal * move2
    
    return (new_pos1, new_pos2)


def separate_from_wall(
    position: Vec2,
    radius: float,
    wall_point: Vec2,
    wall_normal: Vec2,
    min_separation: float = 0.01
) -> Vec2:
    """
    Separate circle from wall.
    
    Pushes circle along wall normal until no overlap.
    """
    # Distance from center to wall
    to_center = position - wall_point
    dist = to_center.dot(wall_normal)
    
    # Required distance is radius + separation
    required = radius + min_separation
    
    if dist >= required:
        return position
    
    # Push out along normal
    push = required - dist
    return position + wall_normal * push


class PhysicsEngine:
    """
    Physics simulation engine.
    
    Handles integration, collision detection, and response for
    all game entities.
    """
    
    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config
    
    def step_ball(self, ball: Ball, dt: float) -> None:
        """
        Step ball physics for one timestep.
        
        Updates position, velocity, and sleep state.
        """
        if ball.is_sleeping:
            return
        
        # Apply drag
        ball.velocity = apply_drag(ball.velocity, self.config.ball_drag, dt)
        
        # Integrate position
        ball.position = ball.position + ball.velocity * dt
        
        # Update sleep state
        ball.is_sleeping, ball.sleep_timer = update_sleep_state(
            ball, dt,
            self.config.sleep_threshold,
            self.config.sleep_time_required
        )
        
        if ball.is_sleeping:
            ball.velocity = Vec2.zero()
    
    def step_player(self, player: Player, dt: float) -> None:
        """
        Step player physics for one timestep.
        """
        if player.is_sleeping:
            return
        
        # Apply drag (higher than ball)
        player.velocity = apply_drag(player.velocity, self.config.player_drag, dt)
        
        # Integrate position
        player.position = player.position + player.velocity * dt
        
        # Update sleep state
        player.is_sleeping, player.sleep_timer = update_sleep_state(
            player, dt,
            self.config.sleep_threshold,
            self.config.sleep_time_required
        )
        
        if player.is_sleeping:
            player.velocity = Vec2.zero()
    
    def apply_impulse(self, body: PhysicsBody, impulse: Vec2) -> None:
        """Apply impulse to a body (wake it up)."""
        body.velocity = body.velocity + impulse / body.mass
        body.is_sleeping = False
        body.sleep_timer = 0.0
    
    def launch_player(self, player: Player, direction: Vec2, power: float) -> None:
        """
        Launch player with flick action.
        
        Args:
            player: Player to launch
            direction: Direction of flick (will be normalized)
            power: Power 0-1
        """
        # Compute launch speed
        speed = (
            self.config.min_launch_speed +
            (self.config.max_launch_speed - self.config.min_launch_speed) * power
        )
        
        # Set velocity
        player.velocity = direction.normalized() * speed
        player.is_sleeping = False
        player.sleep_timer = 0.0
