"""Sparse goal rewards + potential-based ball-progress shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sim.config import SimConfig
from sim.models import GameState, Team

from .config import RLConfig
from .observations import observation_from_state


@dataclass
class RewardBreakdown:
    total: float
    goal: float
    concede: float
    win: float
    loss: float
    turn: float
    shaping: float


def ball_potential(state: GameState, team: Team, config: SimConfig) -> float:
    """
    Phi(s) = normalized ball x from team's attacking perspective.
    Team always attacks toward +1 after canonicalization.
    """
    w = config.physics.pitch_width
    x = state.ball.position.x
    if team == Team.B:
        x = w - x
    # map [0,w] -> [-1,1]
    return (x - w / 2.0) / (w / 2.0)


def compute_transition_reward(
    prev: GameState,
    nxt: GameState,
    acting_team: Team,
    config: SimConfig,
    rl_config: RLConfig,
    *,
    shaping_scale: Optional[float] = None,
) -> RewardBreakdown:
    """
    Reward from the acting team's perspective for one completed turn.

    Scoring: +goal_reward / -concede_penalty
    Terminal: +win_reward / -loss_penalty when match ends
    Shaping: scale * (gamma * Phi' - Phi)
    """
    own_prev = prev.score_a if acting_team == Team.A else prev.score_b
    opp_prev = prev.score_b if acting_team == Team.A else prev.score_a
    own_nxt = nxt.score_a if acting_team == Team.A else nxt.score_b
    opp_nxt = nxt.score_b if acting_team == Team.A else nxt.score_a

    scored = max(0, own_nxt - own_prev)
    conceded = max(0, opp_nxt - opp_prev)

    goal_r = rl_config.goal_reward * scored
    concede_r = -rl_config.concede_penalty * conceded

    win_r = 0.0
    loss_r = 0.0
    winner = nxt.winner(config.goals_to_win)
    if winner == acting_team:
        win_r = rl_config.win_reward
    elif winner == acting_team.opponent:
        loss_r = -rl_config.loss_penalty

    turn_r = -rl_config.turn_penalty

    scale = rl_config.shaping_scale if shaping_scale is None else shaping_scale
    phi = ball_potential(prev, acting_team, config)
    phi_n = ball_potential(nxt, acting_team, config)
    shaping = scale * (rl_config.gamma * phi_n - phi)

    total = goal_r + concede_r + win_r + loss_r + turn_r + shaping
    return RewardBreakdown(
        total=total,
        goal=goal_r,
        concede=concede_r,
        win=win_r,
        loss=loss_r,
        turn=turn_r,
        shaping=shaping,
    )
