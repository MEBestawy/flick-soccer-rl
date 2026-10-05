"""Canonical L→R observations from the acting team's perspective."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sim.config import SimConfig
from sim.models import GameState, Player, Team

from .config import RLConfig


def _mirror_x(x: float, pitch_width: float) -> float:
    return pitch_width - x


def _norm_pos(x: float, y: float, pitch_w: float, pitch_h: float) -> Tuple[float, float]:
    # Center at 0, extents ±1
    nx = (x - pitch_w / 2.0) / (pitch_w / 2.0)
    ny = (y - pitch_h / 2.0) / (pitch_h / 2.0)
    return float(nx), float(ny)


def _norm_vel(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    scale = max(max_speed, 1e-6)
    return float(vx / scale), float(vy / scale)


def _ordered_team_players(state: GameState, team: Team) -> List[Player]:
    """Stable order: sort by id so indices map consistently after mirror."""
    return sorted(state.get_team_players(team), key=lambda p: p.id)


def observation_from_state(
    state: GameState,
    perspective_team: Team,
    config: SimConfig | None = None,
    rl_config: RLConfig | None = None,
) -> np.ndarray:
    """
    Flat float observation from `perspective_team`'s view.

    Always attacks LEFT→RIGHT (+x). Team B states are horizontally mirrored
    and friendly/opponent slots swapped accordingly.
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

    friendly_team = perspective_team
    opponent_team = perspective_team.opponent
    friendly = _ordered_team_players(state, friendly_team)
    opponents = _ordered_team_players(state, opponent_team)

    # After mirror, re-sort by mirrored x then y then id for spatial stability
    def sort_key(p: Player) -> Tuple[float, float, str]:
        x = _mirror_x(p.position.x, w) if mirror else p.position.x
        return (x, p.position.y, p.id)

    friendly = sorted(friendly, key=sort_key)
    opponents = sorted(opponents, key=sort_key)

    # Pad / truncate to exactly 5
    while len(friendly) < rl.num_players:
        friendly.append(friendly[-1] if friendly else state.players[0])
    while len(opponents) < rl.num_players:
        opponents.append(opponents[-1] if opponents else state.players[0])
    friendly = friendly[: rl.num_players]
    opponents = opponents[: rl.num_players]

    feats: List[float] = [bx, by, bvx, bvy]
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
    assert arr.shape == (rl.obs_dim,), f"obs dim {arr.shape} != {rl.obs_dim}"
    return arr


def friendly_player_ids(state: GameState, perspective_team: Team) -> List[str]:
    """Player ids in the same order used by the observation / action head."""
    physics_w = 120.0  # only used for sort consistency; real width from positions
    # Recompute same ordering as observation_from_state
    cfg_w = None
    try:
        # Infer pitch width from ball/players if needed — use standard
        from sim.config import PhysicsConfig

        physics_w = PhysicsConfig().pitch_width
    except Exception:
        physics_w = 120.0

    mirror = perspective_team == Team.B
    players = list(state.get_team_players(perspective_team))

    def sort_key(p: Player) -> Tuple[float, float, str]:
        x = _mirror_x(p.position.x, physics_w) if mirror else p.position.x
        return (x, p.position.y, p.id)

    ordered = sorted(players, key=sort_key)
    while len(ordered) < 5:
        ordered.append(ordered[-1])
    return [p.id for p in ordered[:5]]
