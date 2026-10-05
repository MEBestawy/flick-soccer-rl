"""Tests for ghost-ball aiming and candidate search."""

from sim import FlickAction, HeadlessEnv, Team, Vec2
from sim.agents.aiming import (
    action_to_send_ball,
    evaluate_action,
    ghost_ball_aim,
    player_reach_bands,
    search_best_action,
)


def test_ghost_ball_aims_behind_ball_for_rightward_shot() -> None:
    # Player left of ball; want ball to go right → aim further left through ball
    direction = ghost_ball_aim(
        player_pos=Vec2(0.0, 0.0),
        player_radius=2.0,
        ball_pos=Vec2(10.0, 0.0),
        ball_radius=1.0,
        desired_ball_direction=Vec2(1.0, 0.0),
    )
    assert direction is not None
    assert direction.x > 0.9


def test_reach_bands_ordered() -> None:
    env = HeadlessEnv()
    bands = player_reach_bands(env.config)
    assert bands["soft_0.35"] < bands["medium_0.6"] < bands["strong_0.85"] < bands["max_1.0"]
    assert bands["ball_after_strong_hit"] > bands["max_1.0"]


def test_action_to_send_ball_valid() -> None:
    env = HeadlessEnv()
    state = env.reset(Team.A)
    # A5 is usually near center on kickoff
    action = action_to_send_ball(
        state,
        env.config,
        "A5",
        Vec2(env.config.physics.pitch_width, env.config.physics.pitch_height / 2),
        0.7,
    )
    assert action is not None
    assert action.is_valid(state)


def test_search_finds_improving_action() -> None:
    env = HeadlessEnv()
    state = env.reset(Team.A)
    action = search_best_action(
        state,
        env.config,
        Team.A,
        player_ids=["A5", "A4"],
        target_keys=["opp_goal_center", "upfield_center"],
        powers=[0.6, 0.9],
    )
    assert action is not None
    score, final = evaluate_action(state, action, env.config, Team.A)
    assert final is not None
    assert score > -1e8
