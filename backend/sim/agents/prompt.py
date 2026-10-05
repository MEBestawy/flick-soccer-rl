"""
Build normalized match observations for AI agents (especially Jev).

All spatial quantities are normalized before being placed in the prompt:
  - Positions: pitch center → (0, 0), half-extents → ±1
  - Radii / sizes: divided by pitch half-diagonal
  - Drag coefficients: relative to player_drag (player_drag → 1.0)
  - Distances: divided by pitch half-diagonal (same size scale)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..config import SimConfig
from ..geometry import Vec2
from ..models import GameState, Player, Team
from .aiming import player_reach_bands


def _half_extents(config: SimConfig) -> tuple[float, float, float]:
    physics = config.physics
    half_w = physics.pitch_width / 2.0
    half_h = physics.pitch_height / 2.0
    half_diag = (half_w**2 + half_h**2) ** 0.5
    return half_w, half_h, half_diag


def normalize_position(x: float, y: float, config: SimConfig) -> tuple[float, float]:
    """Map world (x, y) to normalized coords in roughly [-1, 1]."""
    half_w, half_h, _ = _half_extents(config)
    nx = (x - half_w) / half_w
    ny = (y - half_h) / half_h
    return nx, ny


def normalize_size(radius: float, config: SimConfig) -> float:
    """Normalize a radius/length by pitch half-diagonal."""
    _, _, half_diag = _half_extents(config)
    return radius / half_diag


def normalize_drag(drag: float, config: SimConfig) -> float:
    """Normalize drag relative to player_drag (players → 1.0)."""
    ref = config.physics.player_drag
    if ref <= 0:
        return drag
    return drag / ref


def _bearing_deg(from_pos: Vec2, to_pos: Vec2) -> Optional[float]:
    """Compass bearing in degrees: 0=+X/east, 90=+Y/north."""
    delta = to_pos - from_pos
    if delta.length() < 1e-9:
        return None
    return (math.degrees(math.atan2(delta.y, delta.x)) + 360.0) % 360.0


def _compass_label(bearing_deg: Optional[float]) -> str:
    if bearing_deg is None:
        return "n/a"
    labels = [
        "E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW",
        "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE",
    ]
    idx = int((bearing_deg + 11.25) // 22.5) % 16
    return labels[idx]


def _approx_max_travel(config: SimConfig, power: float = 1.0) -> float:
    """
    Approximate max travel distance for a flicked player.

    Under exponential drag, integrated distance → launch_speed / drag.
    """
    physics = config.physics
    speed = physics.min_launch_speed + (
        physics.max_launch_speed - physics.min_launch_speed
    ) * power
    if physics.player_drag <= 1e-9:
        return speed
    return speed / physics.player_drag


def _ball_zone(ball_nx: float, acting_team: Team) -> str:
    """Where the ball is relative to the acting team's attack direction."""
    # Team A attacks +x; Team B attacks -x. Remap so +1 = deep in attack.
    attack_x = ball_nx if acting_team == Team.A else -ball_nx
    if attack_x < -0.45:
        return "defensive_third (near YOUR goal — DANGER)"
    if attack_x > 0.45:
        return "attacking_third (near OPPONENT goal — chance to score)"
    return "middle_third"


def _player_tactics(
    player: Player,
    ball_pos: Vec2,
    own_goal: Vec2,
    opp_goal: Vec2,
    config: SimConfig,
    max_travel: float,
    bands: Dict[str, float],
) -> Dict[str, Any]:
    to_ball = ball_pos - player.position
    dist_ball = to_ball.length()
    dist_own = (player.position - own_goal).length()
    dist_opp = (player.position - opp_goal).length()
    ball_to_own = (ball_pos - own_goal).length()
    ball_to_opp = (ball_pos - opp_goal).length()

    # Is player roughly between ball and own goal? (defensive covering)
    ball_to_own_goal = own_goal - ball_pos
    player_from_ball = player.position - ball_pos
    between_own = False
    if ball_to_own_goal.length() > 1e-6 and player_from_ball.length() > 1e-6:
        proj = player_from_ball.dot(ball_to_own_goal.normalized())
        between_own = 0.0 < proj < ball_to_own_goal.length()

    # Is player on the attacking side of the ball (behind ball for a shot)?
    attack_dir = (opp_goal - own_goal).normalized()
    behind_for_shot = player_from_ball.dot(attack_dir) < -0.15 * dist_ball

    contact_gap = max(0.0, dist_ball - player.radius - config.physics.ball_radius)
    can_reach_soft = contact_gap <= bands["soft_0.35"]
    can_reach_medium = contact_gap <= bands["medium_0.6"]
    can_reach_strong = contact_gap <= bands["strong_0.85"]
    can_reach_max = contact_gap <= bands["max_1.0"]

    # Minimum power (0..1) likely needed to close the contact gap
    min_power_needed = 1.0
    for pwr, key in ((0.35, "soft_0.35"), (0.6, "medium_0.6"), (0.85, "strong_0.85"), (1.0, "max_1.0")):
        if contact_gap <= bands[key]:
            min_power_needed = pwr
            break

    bearing_ball = _bearing_deg(player.position, ball_pos)
    through_goal = ball_pos + (opp_goal - ball_pos).normalized() * 0.01
    bearing_through_goal = _bearing_deg(player.position, through_goal)
    away_own = ball_pos + (ball_pos - own_goal).normalized() * 0.01
    bearing_clear = _bearing_deg(player.position, away_own)

    # Shot quality: alignment of (player→ball) with (ball→opp goal)
    to_ball_n = to_ball.normalized()
    to_goal_n = (opp_goal - ball_pos).normalized()
    shot_alignment = to_ball_n.dot(to_goal_n) if to_ball.length() > 1e-6 else 0.0

    return {
        "distance_to_ball_normalized": round(normalize_size(dist_ball, config), 4),
        "contact_gap_normalized": round(normalize_size(contact_gap, config), 4),
        "contact_gap_world": round(contact_gap, 3),
        "bearing_to_ball_deg": None if bearing_ball is None else round(bearing_ball, 1),
        "bearing_to_ball_compass": _compass_label(bearing_ball),
        "aim_through_ball_to_opp_goal_compass": _compass_label(bearing_through_goal),
        "aim_clear_ball_from_own_goal_compass": _compass_label(bearing_clear),
        "distance_to_own_goal_normalized": round(normalize_size(dist_own, config), 4),
        "distance_to_opp_goal_normalized": round(normalize_size(dist_opp, config), 4),
        "ball_closer_to_own_goal_than_player": ball_to_own < dist_own,
        "between_ball_and_own_goal": between_own,
        "behind_ball_for_attacking_shot": behind_for_shot,
        "shot_alignment_player_ball_goal": round(shot_alignment, 3),
        "can_likely_reach_ball_soft_flick": can_reach_soft,
        "can_likely_reach_ball_medium_flick": can_reach_medium,
        "can_likely_reach_ball_strong_flick": can_reach_strong,
        "can_likely_reach_ball_max_flick": can_reach_max,
        "min_power_likely_to_reach_ball": min_power_needed if can_reach_max else None,
        "reach_radii_normalized": {
            "soft": round(normalize_size(bands["soft_0.35"], config), 4),
            "medium": round(normalize_size(bands["medium_0.6"], config), 4),
            "strong": round(normalize_size(bands["strong_0.85"], config), 4),
            "max": round(normalize_size(bands["max_1.0"], config), 4),
        },
        "ball_distance_to_own_goal_normalized": round(
            normalize_size(ball_to_own, config), 4
        ),
        "ball_distance_to_opp_goal_normalized": round(
            normalize_size(ball_to_opp, config), 4
        ),
    }


def build_normalized_observation(
    state: GameState,
    config: SimConfig,
    *,
    acting_team: Team,
    acting_label: str = "Jev",
    opponent_label: str = "Opponent",
) -> Dict[str, Any]:
    """
    Structured, normalized observation for prompts / agents.

    Scores are labelled from the acting agent's perspective so Jev always
    knows which score is its own.
    """
    physics = config.physics
    half_w, half_h, half_diag = _half_extents(config)
    bands = player_reach_bands(config)
    max_travel = bands["max_1.0"]

    my_score = state.score_a if acting_team == Team.A else state.score_b
    opp_score = state.score_b if acting_team == Team.A else state.score_a
    goals_needed = max(0, config.goals_to_win - my_score)
    opp_goals_needed = max(0, config.goals_to_win - opp_score)

    left_posts = [
        normalize_position(0.0, physics.goal_y_min, config),
        normalize_position(0.0, physics.goal_y_max, config),
    ]
    right_posts = [
        normalize_position(physics.pitch_width, physics.goal_y_min, config),
        normalize_position(physics.pitch_width, physics.goal_y_max, config),
    ]

    if acting_team == Team.A:
        own_goal_posts = left_posts
        opp_goal_posts = right_posts
        attack_direction = "+x (toward +1 / right)"
        defend_direction = "-x (toward -1 / left)"
        own_goal = Vec2(0.0, physics.pitch_height / 2.0)
        opp_goal = Vec2(physics.pitch_width, physics.pitch_height / 2.0)
    else:
        own_goal_posts = right_posts
        opp_goal_posts = left_posts
        attack_direction = "-x (toward -1 / left)"
        defend_direction = "+x (toward +1 / right)"
        own_goal = Vec2(physics.pitch_width, physics.pitch_height / 2.0)
        opp_goal = Vec2(0.0, physics.pitch_height / 2.0)

    ball_nx, ball_ny = normalize_position(
        state.ball.position.x, state.ball.position.y, config
    )
    ball_pos = state.ball.position
    ball_to_own = (ball_pos - own_goal).length()
    ball_to_opp = (ball_pos - opp_goal).length()

    players_payload: List[Dict[str, Any]] = []
    for player in sorted(state.players, key=lambda p: p.id):
        px, py = normalize_position(player.position.x, player.position.y, config)
        relation = (
            acting_label if player.team == acting_team else opponent_label
        )
        tactics = _player_tactics(
            player, ball_pos, own_goal, opp_goal, config, max_travel, bands
        )
        players_payload.append(
            {
                "id": player.id,
                "side": relation,
                "team": player.team.value,
                "controllable_this_turn": (
                    player.team == acting_team
                    and state.current_team == acting_team
                ),
                "position_normalized": {
                    "x": round(px, 4),
                    "y": round(py, 4),
                },
                "radius_normalized": round(normalize_size(player.radius, config), 4),
                "mass": player.mass,
                "linear_drag_normalized": round(
                    normalize_drag(physics.player_drag, config), 4
                ),
                **tactics,
            }
        )

    mine = [p for p in players_payload if p["side"] == acting_label]
    nearest_mine = min(mine, key=lambda p: p["distance_to_ball_normalized"]) if mine else None
    opp_players = [p for p in players_payload if p["side"] == opponent_label]
    nearest_opp = (
        min(opp_players, key=lambda p: p["distance_to_ball_normalized"])
        if opp_players
        else None
    )
    reachable = [
        p["id"]
        for p in mine
        if p["controllable_this_turn"] and p["can_likely_reach_ball_max_flick"]
    ]
    reachable_soft = [
        p["id"]
        for p in mine
        if p["controllable_this_turn"] and p["can_likely_reach_ball_soft_flick"]
    ]

    # Opponent pieces that could reach the ball on their next turn
    opp_threat_reach = [
        p["id"]
        for p in opp_players
        if p["can_likely_reach_ball_max_flick"]
    ]

    # Open shooting lane: few bodies near the segment ball→opp goal
    lane_clear = True
    to_opp = opp_goal - ball_pos
    lane_len = to_opp.length()
    if lane_len > 1e-6:
        lane_n = to_opp.normalized()
        blockers = 0
        for p in state.players:
            rel = p.position - ball_pos
            along = rel.dot(lane_n)
            if along < 0 or along > lane_len:
                continue
            perp = (rel - lane_n * along).length()
            if perp < p.radius + state.ball.radius + 1.5:
                blockers += 1
        lane_clear = blockers <= 1  # ball itself / soft clutter ok

    danger = "low"
    if ball_to_own < physics.pitch_width * 0.28:
        danger = "high"
    elif ball_to_own < physics.pitch_width * 0.42:
        danger = "medium"

    scoring_chance = "low"
    if ball_to_opp < physics.pitch_width * 0.28 and lane_clear:
        scoring_chance = "high"
    elif ball_to_opp < physics.pitch_width * 0.42:
        scoring_chance = "medium"

    score_diff = my_score - opp_score
    if score_diff > 0:
        game_script = "LEADING — protect the lead; do not gift counters; still punish open chances."
    elif score_diff < 0:
        game_script = "TRAILING — prioritize creating shots; take calculated risks; do not park aimlessly."
    else:
        game_script = "TIED — balance defense and clinical attacks; win the next high-quality chance."

    return {
        "objectives": {
            "primary": (
                f"You are a RUTHLESS expert agent for {acting_label}. "
                f"First to {config.goals_to_win} wins. Thoroughly defeat {opponent_label}. "
                f"You need {goals_needed} more goal(s); they need {opp_goals_needed}."
            ),
            "score": (
                f"SCORE GOALS for {acting_label} by sending the ball through the "
                f"OPPONENT goal mouth (posts listed below)."
            ),
            "defend": (
                f"PREVENT GOALS against {acting_label}. Stop the ball from entering "
                f"YOUR goal mouth. Clear dangerous balls away from your goal BEFORE "
                f"{opponent_label} can take a clean shot."
            ),
            "plan_ahead": (
                "Think 2–3 turns ahead like billiards: this flick should leave the ball "
                "where YOUR next flick (or a forced weak opponent reply) creates a score "
                "or removes danger. Prefer setups that leave a teammate behind the ball "
                "for the following attack. Never leave your goal empty if the ball is "
                "in your half."
            ),
            "anti_patterns": [
                "Do NOT march idle players across the pitch if they cannot contact the ball this turn.",
                "Do NOT flick toward the opponent half just to 'advance' empty pieces.",
                "Do NOT hit the ball toward your own goal.",
                "A flick that never touches the ball is usually wasted.",
                "Do NOT max-power every shot — overhit balls bounce unpredictably and concede counters.",
            ],
            "game_script": game_script,
        },
        "coordinate_system": {
            "description": (
                "Positions are normalized so pitch center is (0,0). "
                "X ∈ [-1, 1] left→right, Y ∈ [-1, 1] bottom→top. "
                "Sizes and distances are ÷ pitch_half_diagonal. "
                "Drag is relative to player_drag (players = 1.0). "
                "Compass bearings: 0°=+X/east, 90°=+Y/north, 180°=west, 270°=south."
            ),
            "pitch_half_width_world": half_w,
            "pitch_half_height_world": half_h,
            "pitch_half_diagonal_world": round(half_diag, 4),
            "attack_direction_for_acting_side": attack_direction,
            "defend_direction_for_acting_side": defend_direction,
        },
        "field": {
            "width_normalized": 2.0,
            "height_normalized": 2.0,
            "aspect_ratio_world": round(physics.pitch_width / physics.pitch_height, 4),
            "world_width": physics.pitch_width,
            "world_height": physics.pitch_height,
        },
        "goals": {
            "mouth_height_normalized": round(
                physics.goal_height / physics.pitch_height, 4
            ),
            "mouth_y_range_normalized": [
                round(normalize_position(0.0, physics.goal_y_min, config)[1], 4),
                round(normalize_position(0.0, physics.goal_y_max, config)[1], 4),
            ],
            "depth_normalized": round(normalize_size(physics.goal_width, config), 4),
            "post_radius_normalized": round(
                normalize_size(physics.goal_post_radius, config), 4
            ),
            "scoring_rule": (
                "A goal is scored when the BALL CENTER crosses the goal line "
                "between the two goal posts (inside the mouth Y range). "
                "Hitting a post is a bounce, not a goal."
            ),
            "own_goal_posts_normalized": [
                {"x": round(p[0], 4), "y": round(p[1], 4)} for p in own_goal_posts
            ],
            "opponent_goal_posts_normalized": [
                {"x": round(p[0], 4), "y": round(p[1], 4)} for p in opp_goal_posts
            ],
            "own_goal_center_normalized": {
                "x": round(normalize_position(own_goal.x, own_goal.y, config)[0], 4),
                "y": round(normalize_position(own_goal.x, own_goal.y, config)[1], 4),
            },
            "opponent_goal_center_normalized": {
                "x": round(normalize_position(opp_goal.x, opp_goal.y, config)[0], 4),
                "y": round(normalize_position(opp_goal.x, opp_goal.y, config)[1], 4),
            },
        },
        "score": {
            f"{acting_label}_score": my_score,
            f"{opponent_label}_score": opp_score,
            "goals_to_win": config.goals_to_win,
            "note": (
                f"{acting_label}_score is YOUR score. "
                f"{opponent_label}_score is the enemy score. "
                f"You win by reaching {config.goals_to_win} first."
            ),
        },
        "turn": {
            "phase": state.phase.name,
            "turn_number": state.turn_number,
            "current_team": state.current_team.value,
            "acting_side": acting_label,
            "acting_team_id": acting_team.value,
            "one_flick_only": True,
        },
        "situation": {
            "ball_zone": _ball_zone(ball_nx, acting_team),
            "danger_to_own_goal": danger,
            "scoring_chance": scoring_chance,
            "shot_lane_to_opp_goal_mostly_clear": lane_clear,
            "ball_distance_to_own_goal_normalized": round(
                normalize_size(ball_to_own, config), 4
            ),
            "ball_distance_to_opp_goal_normalized": round(
                normalize_size(ball_to_opp, config), 4
            ),
            "nearest_own_player_to_ball": nearest_mine["id"] if nearest_mine else None,
            "nearest_opponent_to_ball": nearest_opp["id"] if nearest_opp else None,
            "own_players_that_can_reach_ball_this_turn": reachable,
            "own_players_that_can_reach_ball_soft": reachable_soft,
            "opponent_players_that_could_reach_ball_next": opp_threat_reach,
            "approx_max_player_travel_normalized": round(
                normalize_size(max_travel, config), 4
            ),
            "approx_max_ball_travel_normalized": round(
                normalize_size(bands["ball_after_strong_hit"], config), 4
            ),
            "reach_bands_world": {k: round(v, 2) for k, v in bands.items()},
            "reach_bands_normalized": {
                k: round(normalize_size(v, config), 4) for k, v in bands.items()
            },
            "priority_hint": (
                "If danger_to_own_goal is high OR an opponent can reach the ball next: "
                "clear/widen the ball from your goal NOW. If scoring_chance is high and "
                "lane is clear: shoot at the mouth (center or far post). Otherwise: "
                "contact the ball to advance it to a teammate's striking line for next turn."
            ),
        },
        "strategy_doctrine": {
            "identity": (
                f"{acting_label} plays like an elite Subbuteo/billiards assassin — "
                "clinical, patient, and merciless on mistakes."
            ),
            "principles": [
                "Ball > bodies. Only relocate a piece without ball contact to block a "
                "critical shot lane or re-cover an empty goal.",
                "Win the contact geometry: strike from behind the ball toward the target "
                "(ghost-ball thinking). Glancing hits deflect; centered hits transfer hard.",
                "Build 2-move attacks: touch → leave ball for a better-aligned teammate → score.",
                "When leading, starve counters: clear wide/upfield, keep a sentry near your goal.",
                "When trailing, manufacture shots: pull defenders, then smash toward the mouth.",
                "Use walls as cushions for bank shots into the mouth or for safe clearances.",
                "Never gift the opponent a free hit on a ball sitting in your box.",
            ],
        },
        "mechanics": {
            "summary": (
                "Turn-based marble football (Subbuteo + billiards + air hockey). "
                "You do NOT run or dribble. Each turn you slingshot ONE disc; rigid-body "
                "physics resolves collisions until sleep; then the opponent flicks."
            ),
            "marble_behavior": (
                "Discs behave like heavy air-hockey mallets; the ball is a light marble. "
                "A fast heavy player smashing a still ball LAUNCHES the ball and keeps some "
                "momentum. A fast ball hitting a player REDIRECTS/ slows a lot and barely "
                "moves the player. Glancing contacts create billiard-like deflections. "
                "Energy bleeds via drag and inelastic collisions — nothing slides forever."
            ),
            "reach_model": (
                "Under drag, a flicked player's travel distance ≈ launch_speed / player_drag. "
                f"Approx reach at soft/medium/strong/max power (world units): "
                f"{bands['soft_0.35']:.1f} / {bands['medium_0.6']:.1f} / "
                f"{bands['strong_0.85']:.1f} / {bands['max_1.0']:.1f}. "
                f"A strongly struck ball can travel ~{bands['ball_after_strong_hit']:.1f} world units. "
                "If contact_gap > max reach, that player CANNOT hit the ball this turn."
            ),
            "contact_model": (
                "Useful plays almost always COLLIDE with the ball. Aim so your disc's path "
                "intersects the ball; the impulse sends the ball roughly along the contact normal "
                "(away from your disc). To shoot at a target, approach from the opposite side "
                "(ghost ball behind the real ball)."
            ),
            "aiming": (
                "Launch direction = travel direction after release (opposite drag-back). "
                "Power 0..1 scales launch speed. Soft = precision; max = long clearance/shot."
            ),
            "walls_and_posts": (
                "Bank shots off walls/posts are legal and strong. Posts bounce; only a ball "
                "center crossing the goal line inside the mouth scores."
            ),
        },
        "physics": {
            "player_radius_normalized": round(
                normalize_size(physics.player_radius, config), 4
            ),
            "ball_radius_normalized": round(
                normalize_size(physics.ball_radius, config), 4
            ),
            "player_mass": physics.player_mass,
            "ball_mass": physics.ball_mass,
            "player_linear_drag_normalized": 1.0,
            "ball_linear_drag_normalized": round(
                normalize_drag(physics.ball_drag, config), 4
            ),
            "player_drag_world": physics.player_drag,
            "ball_drag_world": physics.ball_drag,
            "max_launch_speed_world": physics.max_launch_speed,
            "note": (
                "Higher drag means faster deceleration. "
                "Players (drag=1.0 normalized) stop much sooner than the ball "
                f"(drag={normalize_drag(physics.ball_drag, config):.3f} normalized). "
                "Therefore: strike the ball; do not expect a player to escort it across the pitch."
            ),
        },
        "ball": {
            "position_normalized": {
                "x": round(ball_nx, 4),
                "y": round(ball_ny, 4),
            },
            "radius_normalized": round(normalize_size(state.ball.radius, config), 4),
            "mass": state.ball.mass,
            "linear_drag_normalized": round(
                normalize_drag(physics.ball_drag, config), 4
            ),
        },
        "players": players_payload,
    }


def observation_to_prompt_text(observation: Dict[str, Any]) -> str:
    """Render observation dict as a compact, readable state string for Jev."""
    lines: List[str] = []
    obj = observation["objectives"]
    lines.append("FLICK FOOTBALL — NORMALIZED MATCH STATE")
    lines.append(f"OBJECTIVE: {obj['primary']}")
    lines.append(f"  • {obj['score']}")
    lines.append(f"  • {obj['defend']}")
    lines.append(f"  • PLAN AHEAD: {obj['plan_ahead']}")
    lines.append(f"  • SCRIPT: {obj['game_script']}")
    lines.append("AVOID:")
    for anti in obj["anti_patterns"]:
        lines.append(f"  - {anti}")
    lines.append("")

    doctrine = observation["strategy_doctrine"]
    lines.append(f"DOCTRINE: {doctrine['identity']}")
    for principle in doctrine["principles"]:
        lines.append(f"  - {principle}")
    lines.append("")

    mech = observation["mechanics"]
    lines.append(f"MECHANICS: {mech['summary']}")
    lines.append(f"  Marble physics: {mech['marble_behavior']}")
    lines.append(f"  Reach model: {mech['reach_model']}")
    lines.append(f"  Contact: {mech['contact_model']}")
    lines.append(f"  Aiming: {mech['aiming']}")
    lines.append(f"  Walls/posts: {mech['walls_and_posts']}")
    lines.append("")

    cs = observation["coordinate_system"]
    lines.append(cs["description"])
    lines.append(f"Attack (score this way): {cs['attack_direction_for_acting_side']}")
    lines.append(f"Defend (protect this way): {cs['defend_direction_for_acting_side']}")
    lines.append("")

    field = observation["field"]
    lines.append(
        f"Field (normalized extents): width={field['width_normalized']}, "
        f"height={field['height_normalized']} "
        f"(world {field['world_width']} x {field['world_height']}, "
        f"aspect {field['aspect_ratio_world']})."
    )

    goals = observation["goals"]
    lines.append(f"Scoring rule: {goals['scoring_rule']}")
    lines.append(
        f"Goal mouth height (fraction of field height): {goals['mouth_height_normalized']}; "
        f"mouth Y range normalized: {goals['mouth_y_range_normalized']}; "
        f"goal depth (size-normalized): {goals['depth_normalized']}; "
        f"post radius (size-normalized): {goals['post_radius_normalized']}."
    )
    lines.append(f"YOUR goal posts (defend): {goals['own_goal_posts_normalized']}")
    lines.append(f"YOUR goal center: {goals['own_goal_center_normalized']}")
    lines.append(
        f"OPPONENT goal posts (score here): {goals['opponent_goal_posts_normalized']}"
    )
    lines.append(f"OPPONENT goal center: {goals['opponent_goal_center_normalized']}")
    lines.append("")

    score = observation["score"]
    score_keys = [k for k in score if k.endswith("_score")]
    score_bits = [f"{k}={score[k]}" for k in score_keys]
    lines.append(
        "SCORE: "
        + ", ".join(score_bits)
        + f" (first to {score['goals_to_win']}). {score['note']}"
    )

    turn = observation["turn"]
    lines.append(
        f"Turn {turn['turn_number']} | phase={turn['phase']} | "
        f"acting_side={turn['acting_side']} (team {turn['acting_team_id']}) | "
        f"ONE flick only={turn['one_flick_only']}"
    )
    lines.append("")

    sit = observation["situation"]
    lines.append("SITUATION:")
    lines.append(f"  Ball zone: {sit['ball_zone']}")
    lines.append(f"  Danger to YOUR goal: {sit['danger_to_own_goal']}")
    lines.append(f"  Chance to score now: {sit['scoring_chance']}")
    lines.append(
        f"  Shot lane to opp goal mostly clear: {sit['shot_lane_to_opp_goal_mostly_clear']}"
    )
    lines.append(
        f"  Ball→your goal (normalized): {sit['ball_distance_to_own_goal_normalized']}; "
        f"Ball→opp goal (normalized): {sit['ball_distance_to_opp_goal_normalized']}"
    )
    lines.append(
        f"  Nearest YOUR player to ball: {sit['nearest_own_player_to_ball']}; "
        f"nearest OPPONENT to ball: {sit['nearest_opponent_to_ball']}"
    )
    lines.append(
        f"  YOUR players that can REACH ball (max): "
        f"{sit['own_players_that_can_reach_ball_this_turn']}; "
        f"soft-reach: {sit['own_players_that_can_reach_ball_soft']}"
    )
    lines.append(
        f"  OPPONENT players that could reach ball next turn: "
        f"{sit['opponent_players_that_could_reach_ball_next']}"
    )
    lines.append(
        f"  Reach bands (normalized) soft/med/strong/max/ball: "
        f"{sit['reach_bands_normalized']}"
    )
    lines.append(f"  Priority hint: {sit['priority_hint']}")
    lines.append("")

    phys = observation["physics"]
    lines.append(
        f"Friction/drag (normalized to player_drag): "
        f"players={phys['player_linear_drag_normalized']}, "
        f"ball={phys['ball_linear_drag_normalized']}. {phys['note']}"
    )
    lines.append(
        f"Masses: player={phys['player_mass']}, ball={phys['ball_mass']}. "
        f"Radii (size-normalized): player={phys['player_radius_normalized']}, "
        f"ball={phys['ball_radius_normalized']}. "
        f"Max launch speed (world): {phys['max_launch_speed_world']}."
    )
    lines.append("")

    ball = observation["ball"]
    lines.append(
        f"BALL position_normalized=({ball['position_normalized']['x']}, "
        f"{ball['position_normalized']['y']}), "
        f"radius_normalized={ball['radius_normalized']}, "
        f"mass={ball['mass']}, "
        f"linear_drag_normalized={ball['linear_drag_normalized']}"
    )
    lines.append("")
    lines.append(
        "PLAYERS (sorted by usefulness). For YOUR [CAN FLICK] pieces, respect REACH "
        "bands vs contact_gap — if gap > max reach, they cannot hit the ball:"
    )
    # Sort: controllable reachable first, then by distance to ball
    ordered = sorted(
        observation["players"],
        key=lambda p: (
            0 if p["controllable_this_turn"] and p["can_likely_reach_ball_max_flick"] else 1,
            0 if p["controllable_this_turn"] else 2,
            p["distance_to_ball_normalized"],
        ),
    )
    for p in ordered:
        flag = " [CAN FLICK]" if p["controllable_this_turn"] else ""
        reach = ""
        if p["controllable_this_turn"]:
            if p["can_likely_reach_ball_soft_flick"]:
                reach = " REACH:soft+"
            elif p["can_likely_reach_ball_medium_flick"]:
                reach = " REACH:medium+"
            elif p["can_likely_reach_ball_strong_flick"]:
                reach = " REACH:strong+"
            elif p["can_likely_reach_ball_max_flick"]:
                reach = " REACH:max-only"
            else:
                reach = " REACH:impossible"
        pos = p["position_normalized"]
        min_p = p.get("min_power_likely_to_reach_ball")
        lines.append(
            f"  - {p['id']} ({p['side']}/team {p['team']}){flag}{reach}: "
            f"pos=({pos['x']}, {pos['y']}), "
            f"gap_to_contact={p['contact_gap_normalized']} "
            f"(world {p['contact_gap_world']}), "
            f"min_power≈{min_p}, "
            f"shot_align={p['shot_alignment_player_ball_goal']}, "
            f"bearing_to_ball={p['bearing_to_ball_compass']}, "
            f"behind_ball_for_shot={p['behind_ball_for_attacking_shot']}, "
            f"covers_own_goal={p['between_ball_and_own_goal']}"
        )

    lines.append("")
    lines.append(
        "DECISION RULES: Pick ONE reachable [CAN FLICK] player with the best geometry "
        "(high shot_align / behind_ball for attacks; covers_own_goal for defense). "
        "Choose where the BALL should go (opp goal / upfield clear / safe park). "
        "Power just enough to reach + finish — soft when close and aligned, strong when "
        "clearing danger or shooting from range. Plan the leave for your next turn. "
        "Empty relocation marches are amateur — crush the human with marble geometry."
    )
    return "\n".join(lines)
