"""
Geometric aiming helpers for marble / billiard-style flicks.

Ghost-ball method: to send the ball in direction D, aim the player at the
contact point on the far side of the ball (center - D_hat * (r_p + r_b)).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..actions import FlickAction
from ..config import SimConfig
from ..geometry import Vec2
from ..models import GameState, Team
from ..simulator import GameSimulator


def launch_speed(config: SimConfig, power: float) -> float:
    physics = config.physics
    p = max(0.0, min(1.0, power))
    return physics.min_launch_speed + (
        physics.max_launch_speed - physics.min_launch_speed
    ) * p


def approx_travel(config: SimConfig, *, power: float, drag: float) -> float:
    """Integrated distance under exponential drag ≈ v0 / drag."""
    speed = launch_speed(config, power)
    if drag <= 1e-9:
        return speed
    return speed / drag


def player_reach_bands(config: SimConfig) -> Dict[str, float]:
    """World-unit travel estimates for soft / medium / max flicks."""
    physics = config.physics
    return {
        "soft_0.35": approx_travel(config, power=0.35, drag=physics.player_drag),
        "medium_0.6": approx_travel(config, power=0.6, drag=physics.player_drag),
        "strong_0.85": approx_travel(config, power=0.85, drag=physics.player_drag),
        "max_1.0": approx_travel(config, power=1.0, drag=physics.player_drag),
        "ball_after_strong_hit": approx_travel(
            config, power=1.0, drag=physics.ball_drag
        ),
    }


def ghost_ball_aim(
    player_pos: Vec2,
    player_radius: float,
    ball_pos: Vec2,
    ball_radius: float,
    desired_ball_direction: Vec2,
) -> Optional[Vec2]:
    """
    Return player launch direction so a central collision sends the ball
    along desired_ball_direction. None if geometry is degenerate.
    """
    ball_dir = desired_ball_direction.normalized()
    if ball_dir.length() < 1e-9:
        return None

    # Contact: player center should arrive at ball_center - ball_dir * (rp+rb)
    contact_center = ball_pos - ball_dir * (player_radius + ball_radius)
    aim = contact_center - player_pos
    if aim.length() < 1e-9:
        # Already overlapping along the line — push along ball_dir
        return ball_dir
    return aim.normalized()


def goal_targets(config: SimConfig, team: Team) -> Dict[str, Vec2]:
    """Named world targets for ball-aim choices."""
    physics = config.physics
    cy = physics.pitch_height / 2.0
    mouth_half = physics.goal_height / 2.0 * 0.85
    if team == Team.A:
        # Attack right
        gx = physics.pitch_width
        clear_x = physics.pitch_width * 0.75
        safe_away = Vec2(physics.pitch_width * 0.55, cy)
        return {
            "opp_goal_center": Vec2(gx, cy),
            "opp_goal_near_top": Vec2(gx, cy + mouth_half),
            "opp_goal_near_bottom": Vec2(gx, cy - mouth_half),
            "upfield_center": Vec2(clear_x, cy),
            "upfield_top": Vec2(clear_x, physics.pitch_height * 0.78),
            "upfield_bottom": Vec2(clear_x, physics.pitch_height * 0.22),
            "park_safe": safe_away,
            "corner_top": Vec2(physics.pitch_width * 0.92, physics.pitch_height * 0.92),
            "corner_bottom": Vec2(physics.pitch_width * 0.92, physics.pitch_height * 0.08),
        }
    gx = 0.0
    clear_x = physics.pitch_width * 0.25
    safe_away = Vec2(physics.pitch_width * 0.45, cy)
    return {
        "opp_goal_center": Vec2(gx, cy),
        "opp_goal_near_top": Vec2(gx, cy + mouth_half),
        "opp_goal_near_bottom": Vec2(gx, cy - mouth_half),
        "upfield_center": Vec2(clear_x, cy),
        "upfield_top": Vec2(clear_x, physics.pitch_height * 0.78),
        "upfield_bottom": Vec2(clear_x, physics.pitch_height * 0.22),
        "park_safe": safe_away,
        "corner_top": Vec2(physics.pitch_width * 0.08, physics.pitch_height * 0.92),
        "corner_bottom": Vec2(physics.pitch_width * 0.08, physics.pitch_height * 0.08),
    }


def own_goal_center(config: SimConfig, team: Team) -> Vec2:
    physics = config.physics
    cy = physics.pitch_height / 2.0
    if team == Team.A:
        return Vec2(0.0, cy)
    return Vec2(physics.pitch_width, cy)


def action_to_send_ball(
    state: GameState,
    config: SimConfig,
    player_id: str,
    ball_target: Vec2,
    power: float,
) -> Optional[FlickAction]:
    """Build a FlickAction that aims to send the ball toward ball_target."""
    player = state.get_player(player_id)
    if player is None:
        return None
    desired = ball_target - state.ball.position
    if desired.length() < 1e-6:
        desired = Vec2(1.0, 0.0) if player.team == Team.A else Vec2(-1.0, 0.0)
    direction = ghost_ball_aim(
        player.position,
        player.radius,
        state.ball.position,
        state.ball.radius,
        desired,
    )
    if direction is None:
        return None
    return FlickAction(player_id=player_id, direction=direction, power=power)


def score_outcome(
    before: GameState,
    after: GameState,
    team: Team,
    config: SimConfig,
) -> float:
    """Heuristic value of a settled state from `team`'s perspective."""
    my_before = before.score_a if team == Team.A else before.score_b
    opp_before = before.score_b if team == Team.A else before.score_a
    my_after = after.score_a if team == Team.A else after.score_b
    opp_after = after.score_b if team == Team.A else after.score_a

    score = 0.0
    scored = my_after - my_before
    conceded = opp_after - opp_before
    score += 1000.0 * scored
    score -= 1200.0 * conceded  # conceding is slightly worse than scoring is good

    if after.winner(config.goals_to_win) == team:
        score += 5000.0
    elif after.winner(config.goals_to_win) == team.opponent:
        score -= 5000.0

    own = own_goal_center(config, team)
    opp = goal_targets(config, team)["opp_goal_center"]
    ball = after.ball.position

    # Prefer ball closer to opp goal, farther from own goal
    score += 8.0 * ((before.ball.position - opp).length() - (ball - opp).length())
    score += 10.0 * ((ball - own).length() - (before.ball.position - own).length())

    # Bonus if ball is in opponent mouth Y corridor and deep in attack
    physics = config.physics
    if physics.goal_y_min < ball.y < physics.goal_y_max:
        score += 3.0
        attack_depth = ball.x / physics.pitch_width
        if team == Team.B:
            attack_depth = 1.0 - attack_depth
        score += 6.0 * attack_depth

    # Slight preference for keeping a player covering own goal
    cover = False
    for p in after.get_team_players(team):
        if (p.position - own).length() < physics.pitch_width * 0.22:
            cover = True
            break
    if cover:
        score += 1.5
    else:
        score -= 2.5

    return score


def evaluate_action(
    state: GameState,
    action: FlickAction,
    config: SimConfig,
    team: Team,
) -> Tuple[float, Optional[GameState]]:
    """Simulate one flick until rest; return (score, final_state)."""
    if not action.is_valid(state):
        return (-1e9, None)
    result = GameSimulator.simulate_action(state, action, config)
    if not result.success or result.final_state is None:
        return (-1e9, None)
    return score_outcome(state, result.final_state, team, config), result.final_state


def search_best_action(
    state: GameState,
    config: SimConfig,
    team: Team,
    player_ids: List[str],
    target_keys: List[str],
    powers: List[float],
) -> Optional[FlickAction]:
    """
    Brute-force a small candidate set with ghost-ball aims + short simulation.
    """
    targets = goal_targets(config, team)
    best_action: Optional[FlickAction] = None
    best_score = -1e18

    for pid in player_ids:
        for tkey in target_keys:
            target = targets.get(tkey)
            if target is None:
                continue
            for power in powers:
                action = action_to_send_ball(state, config, pid, target, power)
                if action is None:
                    continue
                value, _ = evaluate_action(state, action, config, team)
                if value > best_score:
                    best_score = value
                    best_action = action

    return best_action
