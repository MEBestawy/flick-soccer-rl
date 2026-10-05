"""One-ply action improvement search (ExIt-lite) for better training labels."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sim.config import SimConfig
from sim.models import GameState, Team

from .actions import RLAction
from .config import RLConfig
from .env import FlickRLEnv
from .rewards import ball_potential
from .teachers import ShootAtGoalTeacher


def _score_outcome(
    prev: GameState,
    nxt: GameState,
    team: Team,
    sim_config: SimConfig,
    rl_config: RLConfig,
) -> float:
    own_prev = prev.score_a if team == Team.A else prev.score_b
    own_nxt = nxt.score_a if team == Team.A else nxt.score_b
    opp_prev = prev.score_b if team == Team.A else prev.score_a
    opp_nxt = nxt.score_b if team == Team.A else nxt.score_a
    scored = max(0, own_nxt - own_prev)
    conceded = max(0, opp_nxt - opp_prev)
    phi0 = ball_potential(prev, team, sim_config)
    phi1 = ball_potential(nxt, team, sim_config)
    touch = 0.0
    if (
        nxt.last_touch_team == team
        and nxt.last_touch_player is not None
        and nxt.last_touch_player != prev.last_touch_player
    ):
        touch = 1.0
    return (
        10.0 * scored
        - 8.0 * conceded
        + 2.0 * max(0.0, phi1 - phi0)
        + 0.5 * touch
    )


def _inject_state(env: FlickRLEnv, state: GameState) -> None:
    live = state.clone()
    env.inner.simulator._state = live
    env.inner._state = live
    env.inner._done = False
    env._state = live
    env._turns = 0


def improve_action(
    state: GameState,
    team: Team,
    base: RLAction,
    *,
    sim_config: SimConfig,
    rl_config: RLConfig,
    n_candidates: int = 12,
    angle_noise: float = 0.45,
    power_noise: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[RLAction, float]:
    """
    Evaluate noisy variants of ``base`` with a one-flick rollout; return the best.

    Uses a temporary FlickRLEnv so physics matches training.
    """
    rng = rng or np.random.default_rng()
    env = FlickRLEnv(sim_config, rl_config)
    env.rl_config.easy_scenario_prob = 0.0
    env._search_hint_prob = 0.0

    candidates: List[RLAction] = [base]
    base_ang = float(np.arctan2(base.direction_raw[1], base.direction_raw[0]))
    for _ in range(max(0, n_candidates - 1)):
        ang = base_ang + float(rng.normal(0.0, angle_noise))
        pidx = int(base.player_index)
        if rng.random() < 0.25:
            pidx = int(rng.integers(0, rl_config.num_players))
        pow_ = float(np.clip(base.power + rng.normal(0.0, power_noise), 0.15, 1.0))
        candidates.append(
            RLAction(
                player_index=pidx,
                direction_raw=np.array([np.cos(ang), np.sin(ang)], dtype=np.float64),
                power=pow_,
            )
        )

    best_a = base
    best_s = -1e18
    for a in candidates:
        _inject_state(env, state)
        prev = env._state.clone()  # type: ignore[union-attr]
        result = env.step(a, reward_perspective=team, include_turn_penalty=False)
        assert env.state is not None
        s = _score_outcome(prev, env.state, team, sim_config, rl_config)
        if result.terminated and result.info.get("goals_scored", 0) > 0:
            s += 5.0
        if s > best_s:
            best_s = s
            best_a = a
    return best_a, float(best_s)


def search_improved_hint(
    state: GameState,
    team: Team,
    *,
    sim_config: SimConfig,
    rl_config: RLConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return teacher-hint features for the search-improved action."""
    teacher = ShootAtGoalTeacher(sim_config, power=0.88)
    base = teacher.act(state, team)
    best, _ = improve_action(
        state,
        team,
        base,
        sim_config=sim_config,
        rl_config=rl_config,
        n_candidates=rl_config.search_candidates,
        angle_noise=rl_config.search_angle_noise,
        power_noise=rl_config.search_power_noise,
        rng=rng,
    )
    oh = np.zeros(rl_config.num_players, dtype=np.float32)
    oh[int(np.clip(best.player_index, 0, rl_config.num_players - 1))] = 1.0
    d = best.direction_unit
    return np.concatenate(
        [oh, np.asarray([d[0], d[1], float(best.power)], dtype=np.float32)]
    )
