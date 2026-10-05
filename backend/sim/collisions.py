"""
Collision detection and resolution.

Handles all collision types:
- Ball-player
- Player-player
- Ball-wall
- Player-wall
- Ball/player-goalpost
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from .geometry import (
    Vec2, Circle, Segment,
    circle_circle_collision, circle_segment_collision
)
from .config import PhysicsConfig
from .models import Player, Ball
from .arena import Arena
from .physics import (
    impulse_response, wall_impulse_response,
    separate_circles, separate_from_wall
)
from .events import GameEvent, EventLog


@dataclass
class Collision:
    """Record of a collision that occurred."""
    type: str
    entity1: str
    entity2: Optional[str]
    position: Vec2
    normal: Vec2


def resolve_ball_player_collisions(
    ball: Ball,
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float
) -> List[Collision]:
    """
    Detect and resolve ball-player collisions.
    
    Returns list of collisions that occurred.
    """
    collisions: List[Collision] = []
    ball_circle = Circle(ball.position, ball.radius)
    
    for player in players:
        player_circle = Circle(player.position, player.radius)
        
        result = circle_circle_collision(ball_circle, player_circle)
        if result is None:
            continue
        
        normal, penetration = result
        
        # Collision point
        collision_pos = ball.position + normal * (ball.radius - penetration / 2)
        
        # Resolve velocities
        new_ball_vel, new_player_vel = impulse_response(
            ball.position, ball.velocity, ball.mass,
            player.position, player.velocity, player.mass,
            normal,
            config.restitution,
            config.collision_friction,
        )
        
        ball.velocity = new_ball_vel
        player.velocity = new_player_vel
        
        # Wake up both
        ball.is_sleeping = False
        ball.sleep_timer = 0.0
        player.is_sleeping = False
        player.sleep_timer = 0.0
        
        # Separate
        new_ball_pos, new_player_pos = separate_circles(
            ball.position, ball.radius, ball.mass,
            player.position, player.radius, player.mass,
            config.min_separation
        )
        ball.position = new_ball_pos
        player.position = new_player_pos
        
        # Record collision
        collision = Collision(
            type="ball_player",
            entity1="ball",
            entity2=player.id,
            position=collision_pos,
            normal=normal
        )
        collisions.append(collision)
        
        # Log event
        event_log.add(GameEvent.collision_ball_player(
            player.id, collision_pos, sim_time
        ))
    
    return collisions


def resolve_player_player_collisions(
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float
) -> List[Collision]:
    """
    Detect and resolve player-player collisions.
    """
    collisions: List[Collision] = []
    n = len(players)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = players[i], players[j]
            
            c1 = Circle(p1.position, p1.radius)
            c2 = Circle(p2.position, p2.radius)
            
            result = circle_circle_collision(c1, c2)
            if result is None:
                continue
            
            normal, penetration = result
            collision_pos = p1.position + normal * (p1.radius - penetration / 2)
            
            # Resolve velocities
            new_vel1, new_vel2 = impulse_response(
                p1.position, p1.velocity, p1.mass,
                p2.position, p2.velocity, p2.mass,
                normal,
                config.player_player_restitution,
                config.collision_friction,
            )
            
            p1.velocity = new_vel1
            p2.velocity = new_vel2
            
            # Wake up both
            p1.is_sleeping = False
            p1.sleep_timer = 0.0
            p2.is_sleeping = False
            p2.sleep_timer = 0.0
            
            # Separate
            new_pos1, new_pos2 = separate_circles(
                p1.position, p1.radius, p1.mass,
                p2.position, p2.radius, p2.mass,
                config.min_separation
            )
            p1.position = new_pos1
            p2.position = new_pos2
            
            collision = Collision(
                type="player_player",
                entity1=p1.id,
                entity2=p2.id,
                position=collision_pos,
                normal=normal
            )
            collisions.append(collision)
            
            event_log.add(GameEvent.collision_player_player(
                p1.id, p2.id, collision_pos, sim_time
            ))
    
    return collisions


def resolve_wall_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float
) -> List[Collision]:
    """
    Resolve collisions with arena walls.
    """
    collisions: List[Collision] = []
    
    # Ball vs walls
    ball_circle = Circle(ball.position, ball.radius)
    for wall in arena.walls:
        result = circle_segment_collision(ball_circle, wall)
        if result is None:
            continue
        
        normal, penetration = result
        
        # Check if ball was moving toward wall
        vel_toward = ball.velocity.dot(normal)
        if vel_toward < 0:
            ball.velocity = wall_impulse_response(
                ball.velocity, normal, config.wall_restitution
            )
            ball.is_sleeping = False
            ball.sleep_timer = 0.0
        
        # Separate from wall
        closest = wall.closest_point(ball.position)
        ball.position = separate_from_wall(
            ball.position, ball.radius, closest, normal, config.min_separation
        )
        
        wall_name = arena.get_wall_name(wall)
        collision = Collision(
            type="ball_wall",
            entity1="ball",
            entity2=wall_name,
            position=closest,
            normal=normal
        )
        collisions.append(collision)
        event_log.add(GameEvent.collision_wall("ball", wall_name, closest, sim_time))
    
    # Players vs walls
    for player in players:
        player_circle = Circle(player.position, player.radius)
        for wall in arena.walls:
            result = circle_segment_collision(player_circle, wall)
            if result is None:
                continue
            
            normal, penetration = result
            
            vel_toward = player.velocity.dot(normal)
            if vel_toward < 0:
                player.velocity = wall_impulse_response(
                    player.velocity, normal, config.wall_restitution
                )
                player.is_sleeping = False
                player.sleep_timer = 0.0
            
            closest = wall.closest_point(player.position)
            player.position = separate_from_wall(
                player.position, player.radius, closest, normal, config.min_separation
            )
            
            wall_name = arena.get_wall_name(wall)
            collision = Collision(
                type="player_wall",
                entity1=player.id,
                entity2=wall_name,
                position=closest,
                normal=normal
            )
            collisions.append(collision)
            event_log.add(GameEvent.collision_wall(player.id, wall_name, closest, sim_time))
    
    return collisions


def resolve_post_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float
) -> List[Collision]:
    """
    Resolve collisions with goal posts.
    """
    collisions: List[Collision] = []
    
    for post_circle, goal_side, post_pos in arena.posts:
        # Ball vs post
        ball_circle = Circle(ball.position, ball.radius)
        result = circle_circle_collision(ball_circle, post_circle)
        
        if result is not None:
            normal, penetration = result
            collision_pos = ball.position - normal * (ball.radius - penetration / 2)
            
            # Post is immovable, so ball bounces off
            vel_toward = ball.velocity.dot(-normal)
            if vel_toward > 0:
                ball.velocity = wall_impulse_response(
                    ball.velocity, -normal, config.post_restitution
                )
                ball.is_sleeping = False
                ball.sleep_timer = 0.0
            
            # Separate
            new_ball_pos, _ = separate_circles(
                ball.position, ball.radius, ball.mass,
                post_circle.center, post_circle.radius, 1e10,  # Infinite mass
                config.min_separation
            )
            ball.position = new_ball_pos
            
            collision = Collision(
                type="ball_post",
                entity1="ball",
                entity2=f"{goal_side}_{post_pos}",
                position=collision_pos,
                normal=normal
            )
            collisions.append(collision)
            event_log.add(GameEvent.collision_post(
                "ball", goal_side, post_pos, collision_pos, sim_time
            ))
        
        # Players vs post
        for player in players:
            player_circle = Circle(player.position, player.radius)
            result = circle_circle_collision(player_circle, post_circle)
            
            if result is not None:
                normal, penetration = result
                collision_pos = player.position - normal * (player.radius - penetration / 2)
                
                vel_toward = player.velocity.dot(-normal)
                if vel_toward > 0:
                    player.velocity = wall_impulse_response(
                        player.velocity, -normal, config.post_restitution
                    )
                    player.is_sleeping = False
                    player.sleep_timer = 0.0
                
                new_player_pos, _ = separate_circles(
                    player.position, player.radius, player.mass,
                    post_circle.center, post_circle.radius, 1e10,
                    config.min_separation
                )
                player.position = new_player_pos
                
                collision = Collision(
                    type="player_post",
                    entity1=player.id,
                    entity2=f"{goal_side}_{post_pos}",
                    position=collision_pos,
                    normal=normal
                )
                collisions.append(collision)
                event_log.add(GameEvent.collision_post(
                    player.id, goal_side, post_pos, collision_pos, sim_time
                ))
    
    return collisions


def resolve_all_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    max_iterations: int = 4
) -> List[Collision]:
    """
    Resolve all collisions with iteration for stability.
    
    Multiple iterations handle chain reactions and ensure separation.
    """
    all_collisions: List[Collision] = []
    
    for _ in range(max_iterations):
        collisions: List[Collision] = []
        
        collisions.extend(resolve_ball_player_collisions(
            ball, players, config, event_log, sim_time
        ))
        collisions.extend(resolve_player_player_collisions(
            players, config, event_log, sim_time
        ))
        collisions.extend(resolve_wall_collisions(
            ball, players, arena, config, event_log, sim_time
        ))
        collisions.extend(resolve_post_collisions(
            ball, players, arena, config, event_log, sim_time
        ))
        
        all_collisions.extend(collisions)
        
        if not collisions:
            break
    
    return all_collisions
