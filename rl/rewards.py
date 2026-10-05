"""Sparse goal rewards + potential-based ball-progress shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sim.config import SimConfig
from sim.models import GameState, Team

from .config import RLConfig


@dataclass
class RewardBreakdown:
    total: float
    goal: float
    concede: float
    win: float
    loss: float
    turn: float
    shaping: float
    touch: float
    progress: float


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
    perspective: Team,
    config: SimConfig,
    rl_config: RLConfig,
    *,
    shaping_scale: Optional[float] = None,
    include_turn_penalty: bool = True,
    truncated: bool = False,
    flicked_player_id: Optional[str] = None,
) -> RewardBreakdown:
    """
    Reward from ``perspective`` team's point of view for one completed turn.

    Used both for the acting learner and to re-attribute opponent turns so
    concedes / opp own-goals / ball movement credit the learner correctly.
    """
    own_prev = prev.score_a if perspective == Team.A else prev.score_b
    opp_prev = prev.score_b if perspective == Team.A else prev.score_a
    own_nxt = nxt.score_a if perspective == Team.A else nxt.score_b
    opp_nxt = nxt.score_b if perspective == Team.A else nxt.score_a

    scored = max(0, own_nxt - own_prev)
    conceded = max(0, opp_nxt - opp_prev)

    # Only full goal credit if our team last-touched (not opp own-goals).
    goal_r = 0.0
    if scored:
        if nxt.last_touch_team == perspective:
            goal_r = rl_config.goal_reward * scored
        else:
            goal_r = rl_config.own_goal_gift * scored

    concede_r = -rl_config.concede_penalty * conceded

    win_r = 0.0
    loss_r = 0.0
    winner = nxt.winner(config.goals_to_win)
    if winner == perspective:
        win_r = rl_config.win_reward
    elif winner == perspective.opponent:
        loss_r = -rl_config.loss_penalty
    elif truncated and scored == 0 and conceded == 0:
        # Score-based terminal on turn-cap so returns aren't pure turn tax.
        if own_nxt > opp_nxt:
            win_r = rl_config.win_reward * 0.5
        elif own_nxt < opp_nxt:
            loss_r = -rl_config.loss_penalty * 0.5

    turn_r = -rl_config.turn_penalty if include_turn_penalty else 0.0

    scale = rl_config.shaping_scale if shaping_scale is None else shaping_scale
    phi = ball_potential(prev, perspective, config)
    phi_n = ball_potential(nxt, perspective, config)
    shaping = scale * (rl_config.gamma * phi_n - phi)

    # Dense forward progress (non-telescoping): only reward positive ball advance.
    progress = rl_config.progress_coef * max(0.0, phi_n - phi)

    # Bonus when our team newly touches the ball this turn.
    touch = 0.0
    if (
        nxt.last_touch_team == perspective
        and nxt.last_touch_player is not None
        and nxt.last_touch_player != prev.last_touch_player
    ):
        touch = rl_config.touch_bonus
        # Extra if the touch advanced the ball toward goal.
        if phi_n > phi + 0.02:
            touch += rl_config.touch_bonus

    # Shot quality: ball velocity toward opponent goal after our involvement.
    shot = 0.0
    approach = 0.0
    if include_turn_penalty:  # learner's own action
        bx = nxt.ball.velocity.x
        if perspective == Team.B:
            bx = -bx
        if bx > 5.0 and nxt.last_touch_team == perspective:
            shot = rl_config.shot_bonus * min(1.0, bx / 40.0)

        if flicked_player_id is not None:
            p0 = prev.get_player(flicked_player_id)
            p1 = nxt.get_player(flicked_player_id)
            if p0 is not None and p1 is not None:
                d0 = (prev.ball.position - p0.position).length()
                # Did the flicked player move toward where the ball was?
                d1 = (prev.ball.position - p1.position).length()
                if d1 < d0 - 1.5:
                    approach = rl_config.approach_bonus * min(1.0, (d0 - d1) / 20.0)

    total = (
        goal_r
        + concede_r
        + win_r
        + loss_r
        + turn_r
        + shaping
        + touch
        + progress
        + shot
        + approach
    )
    return RewardBreakdown(
        total=total,
        goal=goal_r,
        concede=concede_r,
        win=win_r,
        loss=loss_r,
        turn=turn_r,
        shaping=shaping,
        touch=touch + shot + approach,
        progress=progress,
    )
