"""
Bridge to the Rust flick_physics extension.

Falls back to None if the extension is not installed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .geometry import Vec2
from .config import PhysicsConfig, SimConfig
from .models import GameState, Player, Ball, Team
from .arena import Arena
from .actions import FlickAction

try:
    import flick_physics as _fp  # type: ignore
except ImportError:
    _fp = None

def _rust_enabled() -> bool:
    import os
    if _fp is None:
        return False
    return os.environ.get("FLICK_PHYSICS_RUST", "1") != "0"


# Back-compat attribute; prefer _rust_enabled() for live toggles.
RUST_AVAILABLE = _fp is not None


def pack_body(x: float, y: float, vx: float, vy: float, r: float, m: float, sleeping: bool, sleep_timer: float) -> List[float]:
    return [x, y, vx, vy, r, m, 1.0 if sleeping else 0.0, sleep_timer]


def pack_ball(ball: Ball) -> List[float]:
    return pack_body(
        ball.position.x,
        ball.position.y,
        ball.velocity.x,
        ball.velocity.y,
        ball.radius,
        ball.mass,
        ball.is_sleeping,
        ball.sleep_timer,
    )


def pack_player(p: Player) -> List[float]:
    return pack_body(
        p.position.x,
        p.position.y,
        p.velocity.x,
        p.velocity.y,
        p.radius,
        p.mass,
        p.is_sleeping,
        p.sleep_timer,
    )


def unpack_into_ball(data: List[float], ball: Ball) -> None:
    ball.position = Vec2(data[0], data[1])
    ball.velocity = Vec2(data[2], data[3])
    ball.radius = data[4]
    ball.mass = data[5]
    ball.is_sleeping = data[6] > 0.5
    ball.sleep_timer = data[7]


def unpack_into_player(data: List[float], player: Player) -> None:
    player.position = Vec2(data[0], data[1])
    player.velocity = Vec2(data[2], data[3])
    player.radius = data[4]
    player.mass = data[5]
    player.is_sleeping = data[6] > 0.5
    player.sleep_timer = data[7]


def pack_walls(arena: Arena) -> List[float]:
    out: List[float] = []
    for seg in arena.walls:
        out.extend([seg.start.x, seg.start.y, seg.end.x, seg.end.y])
    return out


def pack_posts(arena: Arena) -> List[float]:
    out: List[float] = []
    for circle, _side, _pos in arena.posts:
        out.extend([circle.center.x, circle.center.y, circle.radius])
    return out


def pack_cfg(config: SimConfig) -> List[float]:
    p = config.physics
    return [
        p.timestep,
        float(p.max_substeps),
        p.player_drag,
        p.ball_drag,
        p.restitution,
        p.player_player_restitution,
        p.wall_restitution,
        p.post_restitution,
        p.collision_friction,
        p.max_launch_speed,
        p.min_launch_speed,
        p.sleep_threshold,
        p.sleep_time_required,
        p.ccd_threshold,
        p.min_separation,
        p.ball_radius,
        config.max_simulation_time,
        p.goal_y_min,
        p.goal_y_max,
        p.pitch_width,
    ]


# Cached geometry packs keyed by PhysicsConfig identity
_geom_cache: dict[int, Tuple[List[float], List[float]]] = {}


def _arena_geom(arena: Arena) -> Tuple[List[float], List[float]]:
    key = id(arena)
    cached = _geom_cache.get(key)
    if cached is not None:
        return cached
    packed = (pack_walls(arena), pack_posts(arena))
    _geom_cache[key] = packed
    return packed


def rust_simulate_flick(
    state: GameState,
    action: FlickAction,
    arena: Arena,
    config: SimConfig,
) -> Optional[Tuple[float, Optional[Team], int]]:
    """
    Run physics for one flick via Rust.

    Mutates ``state`` ball/players/simulation_time/last_touch in place.

    Returns:
        (sim_time, scoring_team_or_None, last_touch_player_index) or None if Rust unavailable.
    """
    if not _rust_enabled() or _fp is None:
        return None

    launch_idx = None
    for i, p in enumerate(state.players):
        if p.id == action.player_id:
            launch_idx = i
            break
    if launch_idx is None:
        return None

    walls, posts = _arena_geom(arena)
    players_flat: List[float] = []
    for p in state.players:
        players_flat.extend(pack_player(p))

    direction = action.normalized_direction()
    bout, pout, sim_time, goal_code, last_touch = _fp.simulate_flick(
        pack_ball(state.ball),
        players_flat,
        walls,
        posts,
        pack_cfg(config),
        launch_idx,
        direction.x,
        direction.y,
        action.power,
    )

    unpack_into_ball(bout, state.ball)
    for i, p in enumerate(state.players):
        unpack_into_player(pout[i * 8 : (i + 1) * 8], p)

    state.simulation_time = sim_time
    if 0 <= last_touch < len(state.players):
        touched = state.players[last_touch]
        state.last_touch_team = touched.team
        state.last_touch_player = touched.id

    scoring: Optional[Team] = None
    if goal_code == 1:
        scoring = Team.B
    elif goal_code == 2:
        scoring = Team.A

    return sim_time, scoring, last_touch
