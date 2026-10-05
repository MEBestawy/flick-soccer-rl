"""Canonical L→R observations from the acting team's perspective."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sim.config import SimConfig
from sim.models import GameState, Player, Team

from .config import RLConfig

_TEACHER = None


def _shoot_teacher(config: SimConfig):
    global _TEACHER
    from .teachers import ShootAtGoalTeacher

    if _TEACHER is None or getattr(_TEACHER, "config", None) is not config:
        _TEACHER = ShootAtGoalTeacher(config, power=0.88)
    return _TEACHER


def _mirror_x(x: float, pitch_width: float) -> float:
    return pitch_width - x


def _norm_pos(x: float, y: float, pitch_w: float, pitch_h: float) -> Tuple[float, float]:
    nx = (x - pitch_w / 2.0) / (pitch_w / 2.0)
    ny = (y - pitch_h / 2.0) / (pitch_h / 2.0)
    return float(nx), float(ny)


def _norm_vel(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    scale = max(max_speed, 1e-6)
    return float(vx / scale), float(vy / scale)


def _ordered_team_players(state: GameState, team: Team) -> List[Player]:
    return sorted(state.get_team_players(team), key=lambda p: p.id)


def teacher_hint_features(
    state: GameState,
    perspective_team: Team,
    config: SimConfig,
    rl_config: RLConfig,
    *,
    use_search: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """One-hot player + (cos, sin, power) hint (teacher or ExIt-lite search)."""
    if use_search:
        from .search_labels import search_improved_hint

        return search_improved_hint(
            state,
            perspective_team,
            sim_config=config,
            rl_config=rl_config,
            rng=rng,
        )
    teacher = _shoot_teacher(config)
    action = teacher.act(state, perspective_team)
    oh = np.zeros(rl_config.num_players, dtype=np.float32)
    idx = int(np.clip(action.player_index, 0, rl_config.num_players - 1))
    oh[idx] = 1.0
    d = action.direction_unit
    return np.concatenate(
        [oh, np.asarray([d[0], d[1], float(action.power)], dtype=np.float32)]
    )


def observation_from_state(
    state: GameState,
    perspective_team: Team,
    config: SimConfig | None = None,
    rl_config: RLConfig | None = None,
    *,
    include_teacher_hint: bool = True,
    search_hint_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Flat float observation from `perspective_team`'s view.

    Always attacks LEFT→RIGHT (+x). Team B states are horizontally mirrored.
    Trailing dims encode the shoot-teacher suggestion for residual policies.
    """
    cfg = config or SimConfig.default()
    rl = rl_config or RLConfig()
    physics = cfg.physics
    w, h = physics.pitch_width, physics.pitch_height
    mirror = perspective_team == Team.B

    def map_pos(x: float, y: float) -> Tuple[float, float]:
        if mirror:
            x = _mirror_x(x, w)
        return _norm_pos(x, y, w, h)

    def map_vel(vx: float, vy: float) -> Tuple[float, float]:
        if mirror:
            vx = -vx
        return _norm_vel(vx, vy, rl.max_speed_norm)

    ball = state.ball
    bx, by = map_pos(ball.position.x, ball.position.y)
    bvx, bvy = map_vel(ball.velocity.x, ball.velocity.y)

    friendly = _ordered_team_players(state, perspective_team)
    opponents = _ordered_team_players(state, perspective_team.opponent)

    def sort_key(p: Player) -> Tuple[float, float, str]:
        x = _mirror_x(p.position.x, w) if mirror else p.position.x
        return (x, p.position.y, p.id)

    friendly = sorted(friendly, key=sort_key)
    opponents = sorted(opponents, key=sort_key)

    while len(friendly) < rl.num_players:
        friendly.append(friendly[-1] if friendly else state.players[0])
    while len(opponents) < rl.num_players:
        opponents.append(opponents[-1] if opponents else state.players[0])
    friendly = friendly[: rl.num_players]
    opponents = opponents[: rl.num_players]

    feats: List[float] = [bx, by, bvx, bvy]

    core_with_geom = rl.obs_dim - (rl.teacher_hint_dim if include_teacher_hint else 0)
    use_geom = core_with_geom >= 55  # 47 + 8 geometry

    if use_geom:
        # Ball → opponent goal (always +x in canonical frame)
        goal_x, goal_y = 1.0, 0.0
        feats.extend([goal_x - bx, goal_y - by])

        def _nearest_rel(players: List[Player]) -> Tuple[float, float, float]:
            best = (0.0, 0.0, 2.0)
            best_d = 1e9
            for p in players:
                px, py = map_pos(p.position.x, p.position.y)
                dx, dy = px - bx, py - by
                d = float(np.hypot(dx, dy))
                if d < best_d:
                    best_d = d
                    best = (dx, dy, d)
            return best

        fr = _nearest_rel(friendly)
        op = _nearest_rel(opponents)
        feats.extend([fr[0], fr[1], fr[2], op[0], op[1], op[2]])

    for p in friendly:
        px, py = map_pos(p.position.x, p.position.y)
        pvx, pvy = map_vel(p.velocity.x, p.velocity.y)
        feats.extend([px, py, pvx, pvy])
    for p in opponents:
        px, py = map_pos(p.position.x, p.position.y)
        pvx, pvy = map_vel(p.velocity.x, p.velocity.y)
        feats.extend([px, py, pvx, pvy])

    own_score = state.score_a if perspective_team == Team.A else state.score_b
    opp_score = state.score_b if perspective_team == Team.A else state.score_a
    feats.append(own_score / float(cfg.goals_to_win))
    feats.append(opp_score / float(cfg.goals_to_win))

    match_frac = 1.0
    if cfg.match_time_limit > 0:
        match_frac = max(0.0, 1.0 - state.match_time / cfg.match_time_limit)
    feats.append(match_frac)

    arr = np.asarray(feats, dtype=np.float32)
    if include_teacher_hint:
        use_search = False
        if search_hint_prob > 0.0:
            r = rng if rng is not None else np.random.default_rng()
            use_search = bool(r.random() < search_hint_prob)
        hint = teacher_hint_features(
            state,
            perspective_team,
            cfg,
            rl,
            use_search=use_search,
            rng=rng,
        )
        arr = np.concatenate([arr, hint])
    assert arr.shape == (rl.obs_dim,), f"obs dim {arr.shape} != {rl.obs_dim}"
    return arr


def decode_teacher_hint(
    obs: np.ndarray, rl_config: Optional[RLConfig] = None
) -> Tuple[int, float, float]:
    """Return (player_index, angle, power) from trailing teacher hint."""
    rl = rl_config or RLConfig()
    hint = np.asarray(obs[..., -rl.teacher_hint_dim :], dtype=np.float64)
    oh = hint[: rl.num_players]
    cos_a, sin_a, power = hint[rl.num_players : rl.num_players + 3]
    return int(np.argmax(oh)), float(np.arctan2(sin_a, cos_a)), float(power)


def friendly_player_ids(state: GameState, perspective_team: Team) -> List[str]:
    """Player ids in the same order used by the observation / action head."""
    from sim.config import PhysicsConfig

    physics_w = PhysicsConfig().pitch_width
    mirror = perspective_team == Team.B
    players = list(state.get_team_players(perspective_team))

    def sort_key(p: Player) -> Tuple[float, float, str]:
        x = _mirror_x(p.position.x, physics_w) if mirror else p.position.x
        return (x, p.position.y, p.id)

    ordered = sorted(players, key=sort_key)
    while len(ordered) < 5:
        ordered.append(ordered[-1])
    return [p.id for p in ordered[:5]]
