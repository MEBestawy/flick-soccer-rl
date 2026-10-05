"""
Collision detection and resolution (optimized hot path).

Same collision response equations as before, with:
- float wall segment tests (no Vec2 alloc per check)
- skip sleeping bodies where safe
- optional event logging
- only re-iterate when dynamics actually changed
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .geometry import Vec2, Circle, circle_circle_collision
from .config import PhysicsConfig
from .models import Player, Ball
from .arena import Arena
from .physics import (
    impulse_response,
    wall_impulse_response,
    separate_circles,
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


# Cached wall data: (x1,y1,x2,y2,dx,dy,len_sq,name)
_WallCache = Tuple[float, float, float, float, float, float, float, str]


def _wall_cache(arena: Arena) -> List[_WallCache]:
    cached = getattr(arena, "_fast_walls", None)
    if cached is not None:
        return cached  # type: ignore[return-value]
    out: List[_WallCache] = []
    for seg in arena.walls:
        x1, y1 = seg.start.x, seg.start.y
        x2, y2 = seg.end.x, seg.end.y
        dx, dy = x2 - x1, y2 - y1
        out.append((x1, y1, x2, y2, dx, dy, dx * dx + dy * dy, arena.get_wall_name(seg)))
    arena._fast_walls = out  # type: ignore[attr-defined]
    return out


def _closest_on_seg(
    px: float, py: float, x1: float, y1: float, dx: float, dy: float, len_sq: float
) -> Tuple[float, float]:
    if len_sq < 1e-10:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return x1 + dx * t, y1 + dy * t


def _seg_normal(dx: float, dy: float, len_sq: float) -> Tuple[float, float]:
    length = math.sqrt(len_sq) if len_sq > 1e-10 else 1.0
    return -dy / length, dx / length


def resolve_ball_player_collisions(
    ball: Ball,
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> List[Collision]:
    collisions, _ = _resolve_ball_player_collisions(
        ball, players, config, event_log, sim_time, log_events=log_events
    )
    return collisions


def _resolve_ball_player_collisions(
    ball: Ball,
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> Tuple[List[Collision], bool]:
    collisions: List[Collision] = []
    changed = False
    if ball.is_sleeping:
        # Sleeping ball cannot start a new contact without someone else moving;
        # still check awake players (they may run into the ball).
        pass

    ball_circle = Circle(ball.position, ball.radius)

    for player in players:
        if ball.is_sleeping and player.is_sleeping:
            continue
        player_circle = Circle(player.position, player.radius)
        result = circle_circle_collision(ball_circle, player_circle)
        if result is None:
            continue

        normal, penetration = result
        collision_pos = ball.position + normal * (ball.radius - penetration / 2)

        new_ball_vel, new_player_vel = impulse_response(
            ball.position,
            ball.velocity,
            ball.mass,
            player.position,
            player.velocity,
            player.mass,
            normal,
            config.restitution,
            config.collision_friction,
        )
        ball.velocity = new_ball_vel
        player.velocity = new_player_vel
        ball.is_sleeping = False
        ball.sleep_timer = 0.0
        player.is_sleeping = False
        player.sleep_timer = 0.0

        new_ball_pos, new_player_pos = separate_circles(
            ball.position,
            ball.radius,
            ball.mass,
            player.position,
            player.radius,
            player.mass,
            config.min_separation,
        )
        ball.position = new_ball_pos
        player.position = new_player_pos
        # Refresh circle center for subsequent players in this pass
        ball_circle = Circle(ball.position, ball.radius)

        collisions.append(
            Collision(
                type="ball_player",
                entity1="ball",
                entity2=player.id,
                position=collision_pos,
                normal=normal,
            )
        )
        if log_events:
            event_log.add(
                GameEvent.collision_ball_player(player.id, collision_pos, sim_time)
            )
        changed = True

    return collisions, changed


def resolve_player_player_collisions(
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> List[Collision]:
    collisions, _ = _resolve_player_player_collisions(
        players, config, event_log, sim_time, log_events=log_events
    )
    return collisions


def _resolve_player_player_collisions(
    players: List[Player],
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> Tuple[List[Collision], bool]:
    collisions: List[Collision] = []
    changed = False
    n = len(players)

    for i in range(n):
        p1 = players[i]
        for j in range(i + 1, n):
            p2 = players[j]
            if p1.is_sleeping and p2.is_sleeping:
                continue

            result = circle_circle_collision(
                Circle(p1.position, p1.radius), Circle(p2.position, p2.radius)
            )
            if result is None:
                continue

            normal, penetration = result
            collision_pos = p1.position + normal * (p1.radius - penetration / 2)

            new_vel1, new_vel2 = impulse_response(
                p1.position,
                p1.velocity,
                p1.mass,
                p2.position,
                p2.velocity,
                p2.mass,
                normal,
                config.player_player_restitution,
                config.collision_friction,
            )
            p1.velocity = new_vel1
            p2.velocity = new_vel2
            p1.is_sleeping = False
            p1.sleep_timer = 0.0
            p2.is_sleeping = False
            p2.sleep_timer = 0.0

            new_pos1, new_pos2 = separate_circles(
                p1.position,
                p1.radius,
                p1.mass,
                p2.position,
                p2.radius,
                p2.mass,
                config.min_separation,
            )
            p1.position = new_pos1
            p2.position = new_pos2

            collisions.append(
                Collision(
                    type="player_player",
                    entity1=p1.id,
                    entity2=p2.id,
                    position=collision_pos,
                    normal=normal,
                )
            )
            if log_events:
                event_log.add(
                    GameEvent.collision_player_player(
                        p1.id, p2.id, collision_pos, sim_time
                    )
                )
            changed = True

    return collisions, changed


def resolve_wall_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> List[Collision]:
    collisions, _ = _resolve_wall_collisions(
        ball, players, arena, config, event_log, sim_time, log_events=log_events
    )
    return collisions


def _resolve_wall_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> Tuple[List[Collision], bool]:
    collisions: List[Collision] = []
    changed = False
    walls = _wall_cache(arena)
    min_sep = config.min_separation
    wall_rest = config.wall_restitution

    def hit_wall(
        px: float,
        py: float,
        vx: float,
        vy: float,
        radius: float,
        entity_id: str,
    ) -> Tuple[float, float, float, float, bool]:
        nonlocal changed
        local_changed = False
        for x1, y1, _x2, _y2, dx, dy, len_sq, wall_name in walls:
            cx, cy = _closest_on_seg(px, py, x1, y1, dx, dy, len_sq)
            ddx = px - cx
            ddy = py - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist >= radius:
                continue
            if dist < 1e-10:
                nx, ny = _seg_normal(dx, dy, len_sq)
            else:
                nx, ny = ddx / dist, ddy / dist
            vel_toward = vx * nx + vy * ny
            if vel_toward < 0.0:
                vel = wall_impulse_response(Vec2(vx, vy), Vec2(nx, ny), wall_rest)
                vx, vy = vel.x, vel.y
                local_changed = True
                changed = True
            # Separate
            required = radius + min_sep
            dist_n = (px - cx) * nx + (py - cy) * ny
            if dist_n < required:
                push = required - dist_n
                px += nx * push
                py += ny * push
                local_changed = True
            collisions.append(
                Collision(
                    type="ball_wall" if entity_id == "ball" else "player_wall",
                    entity1=entity_id,
                    entity2=wall_name,
                    position=Vec2(cx, cy),
                    normal=Vec2(nx, ny),
                )
            )
            if log_events:
                event_log.add(
                    GameEvent.collision_wall(
                        entity_id, wall_name, Vec2(cx, cy), sim_time
                    )
                )
        return px, py, vx, vy, local_changed

    if not ball.is_sleeping:
        bx, by, bvx, bvy, ch = hit_wall(
            ball.position.x,
            ball.position.y,
            ball.velocity.x,
            ball.velocity.y,
            ball.radius,
            "ball",
        )
        ball.position = Vec2(bx, by)
        ball.velocity = Vec2(bvx, bvy)
        if ch:
            ball.is_sleeping = False
            ball.sleep_timer = 0.0

    for player in players:
        if player.is_sleeping:
            continue
        px, py, vx, vy, ch = hit_wall(
            player.position.x,
            player.position.y,
            player.velocity.x,
            player.velocity.y,
            player.radius,
            player.id,
        )
        player.position = Vec2(px, py)
        player.velocity = Vec2(vx, vy)
        if ch:
            player.is_sleeping = False
            player.sleep_timer = 0.0

    return collisions, changed


def resolve_post_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> List[Collision]:
    collisions, _ = _resolve_post_collisions(
        ball, players, arena, config, event_log, sim_time, log_events=log_events
    )
    return collisions


def _resolve_post_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    *,
    log_events: bool = True,
) -> Tuple[List[Collision], bool]:
    collisions: List[Collision] = []
    changed = False

    for post_circle, goal_side, post_pos in arena.posts:
        if not ball.is_sleeping:
            result = circle_circle_collision(
                Circle(ball.position, ball.radius), post_circle
            )
            if result is not None:
                normal, penetration = result
                collision_pos = ball.position - normal * (ball.radius - penetration / 2)
                vel_toward = ball.velocity.dot(-normal)
                if vel_toward > 0:
                    ball.velocity = wall_impulse_response(
                        ball.velocity, -normal, config.post_restitution
                    )
                    ball.is_sleeping = False
                    ball.sleep_timer = 0.0
                    changed = True
                new_ball_pos, _ = separate_circles(
                    ball.position,
                    ball.radius,
                    ball.mass,
                    post_circle.center,
                    post_circle.radius,
                    1e10,
                    config.min_separation,
                )
                ball.position = new_ball_pos
                collisions.append(
                    Collision(
                        type="ball_post",
                        entity1="ball",
                        entity2=f"{goal_side}_{post_pos}",
                        position=collision_pos,
                        normal=normal,
                    )
                )
                if log_events:
                    event_log.add(
                        GameEvent.collision_post(
                            "ball", goal_side, post_pos, collision_pos, sim_time
                        )
                    )
                changed = True

        for player in players:
            if player.is_sleeping:
                continue
            result = circle_circle_collision(
                Circle(player.position, player.radius), post_circle
            )
            if result is None:
                continue
            normal, penetration = result
            collision_pos = player.position - normal * (player.radius - penetration / 2)
            vel_toward = player.velocity.dot(-normal)
            if vel_toward > 0:
                player.velocity = wall_impulse_response(
                    player.velocity, -normal, config.post_restitution
                )
                player.is_sleeping = False
                player.sleep_timer = 0.0
                changed = True
            new_player_pos, _ = separate_circles(
                player.position,
                player.radius,
                player.mass,
                post_circle.center,
                post_circle.radius,
                1e10,
                config.min_separation,
            )
            player.position = new_player_pos
            collisions.append(
                Collision(
                    type="player_post",
                    entity1=player.id,
                    entity2=f"{goal_side}_{post_pos}",
                    position=collision_pos,
                    normal=normal,
                )
            )
            if log_events:
                event_log.add(
                    GameEvent.collision_post(
                        player.id, goal_side, post_pos, collision_pos, sim_time
                    )
                )
            changed = True

    return collisions, changed


def resolve_all_collisions(
    ball: Ball,
    players: List[Player],
    arena: Arena,
    config: PhysicsConfig,
    event_log: EventLog,
    sim_time: float,
    max_iterations: int = 4,
    *,
    log_events: bool = True,
) -> List[Collision]:
    """
    Resolve all collisions with iteration for stability.

    Multiple iterations handle chain reactions and ensure separation.
    """
    all_collisions: List[Collision] = []

    for _ in range(max_iterations):
        collisions: List[Collision] = []
        changed = False

        c, ch = _resolve_ball_player_collisions(
            ball, players, config, event_log, sim_time, log_events=log_events
        )
        collisions.extend(c)
        changed = changed or ch

        c, ch = _resolve_player_player_collisions(
            players, config, event_log, sim_time, log_events=log_events
        )
        collisions.extend(c)
        changed = changed or ch

        c, ch = _resolve_wall_collisions(
            ball, players, arena, config, event_log, sim_time, log_events=log_events
        )
        collisions.extend(c)
        changed = changed or ch

        c, ch = _resolve_post_collisions(
            ball, players, arena, config, event_log, sim_time, log_events=log_events
        )
        collisions.extend(c)
        changed = changed or ch

        all_collisions.extend(collisions)

        if not changed:
            break

    return all_collisions
