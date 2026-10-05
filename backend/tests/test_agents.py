"""Tests for agent strategy pattern and normalized Jev prompts."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from sim import FlickAction, HeadlessEnv, Team, Vec2
from sim.agents import (
    Agent,
    HeuristicAgent,
    JevAgent,
    MatchRunner,
    RandomAgent,
    available_agents,
    build_normalized_observation,
    create_agent,
    normalize_drag,
    normalize_position,
    normalize_size,
    observation_to_prompt_text,
    register_agent,
)
from sim.agents.jev_client import JevClient, JevClientError


class StubAgent(Agent):
    name = "stub"

    def __init__(self, team: Team, player_id: str) -> None:
        super().__init__(team)
        self.player_id = player_id

    def select_action(self, state):  # type: ignore[no-untyped-def]
        return FlickAction(
            player_id=self.player_id,
            direction=Vec2(1.0, 0.0) if self.team == Team.A else Vec2(-1.0, 0.0),
            power=0.5,
        )


def test_normalize_position_center_is_zero() -> None:
    env = HeadlessEnv()
    state = env.reset()
    cfg = env.config
    nx, ny = normalize_position(
        cfg.physics.pitch_width / 2,
        cfg.physics.pitch_height / 2,
        cfg,
    )
    assert abs(nx) < 1e-9
    assert abs(ny) < 1e-9
    assert state.ball.position.x == pytest.approx(cfg.physics.pitch_width / 2)


def test_normalize_corners_near_unit() -> None:
    env = HeadlessEnv()
    cfg = env.config
    nx, ny = normalize_position(0.0, 0.0, cfg)
    assert nx == pytest.approx(-1.0)
    assert ny == pytest.approx(-1.0)
    nx2, ny2 = normalize_position(
        cfg.physics.pitch_width, cfg.physics.pitch_height, cfg
    )
    assert nx2 == pytest.approx(1.0)
    assert ny2 == pytest.approx(1.0)


def test_normalize_drag_players_are_one() -> None:
    cfg = HeadlessEnv().config
    assert normalize_drag(cfg.physics.player_drag, cfg) == pytest.approx(1.0)
    assert normalize_drag(cfg.physics.ball_drag, cfg) < 1.0


def test_normalize_size_positive() -> None:
    cfg = HeadlessEnv().config
    assert normalize_size(cfg.physics.player_radius, cfg) > 0
    assert normalize_size(cfg.physics.ball_radius, cfg) < normalize_size(
        cfg.physics.player_radius, cfg
    )


def test_prompt_includes_required_fields() -> None:
    env = HeadlessEnv()
    state = env.reset()
    obs = build_normalized_observation(
        state,
        env.config,
        acting_team=Team.B,
        acting_label="Jev",
        opponent_label="Opponent",
    )
    text = observation_to_prompt_text(obs)

    assert "Jev_score" in obs["score"]
    assert "Opponent_score" in obs["score"]
    assert obs["score"]["Jev_score"] == state.score_b
    assert obs["score"]["Opponent_score"] == state.score_a

    assert "players" in obs
    assert len(obs["players"]) == 10
    assert "ball" in obs
    assert "linear_drag_normalized" in obs["ball"]
    assert "own_goal_posts_normalized" in obs["goals"]
    assert "opponent_goal_posts_normalized" in obs["goals"]
    assert "field" in obs
    assert "objectives" in obs
    assert "situation" in obs
    assert "strategy_doctrine" in obs
    assert "SCORE GOALS" in obs["objectives"]["score"]
    assert "PREVENT GOALS" in obs["objectives"]["defend"]
    assert "plan_ahead" in obs["objectives"]
    assert "reach_bands_normalized" in obs["situation"]
    assert "marble_behavior" in obs["mechanics"]
    assert "reach_model" in obs["mechanics"]

    # Per-player tactical fields
    sample = obs["players"][0]
    assert "distance_to_ball_normalized" in sample
    assert "bearing_to_ball_compass" in sample
    assert "can_likely_reach_ball_max_flick" in sample
    assert "aim_through_ball_to_opp_goal_compass" in sample
    assert "reach_radii_normalized" in sample
    assert "shot_alignment_player_ball_goal" in sample

    # Prompt text must mention the key concepts
    assert "Jev_score" in text or "Jev" in text
    assert "Opponent" in text
    assert "BALL" in text
    assert "linear_drag_normalized" in text
    assert "goal posts" in text.lower() or "Goal" in text
    assert "normalized" in text.lower()
    assert "SCORE GOALS" in text
    assert "PREVENT GOALS" in text
    assert "march" in text.lower() or "AVOID" in text
    assert "REACH" in text
    assert "Danger to YOUR goal" in text
    assert "Marble physics" in text or "marble" in text.lower()
    assert "DOCTRINE" in text
    assert "PLAN AHEAD" in text


def test_create_agent_registry() -> None:
    assert "random" in available_agents()
    assert "heuristic" in available_agents()
    assert "jev" in available_agents()
    agent = create_agent("random", Team.A)
    assert isinstance(agent, RandomAgent)
    assert agent.team == Team.A


def test_register_custom_agent() -> None:
    register_agent(
        "stub_factory",
        lambda team, config: StubAgent(team, "A1" if team == Team.A else "B1"),
    )
    agent = create_agent("stub_factory", Team.A)
    assert agent.name == "stub"


def test_random_and_heuristic_produce_valid_actions() -> None:
    env = HeadlessEnv()
    state = env.reset(Team.A)
    for AgentCls in (RandomAgent, HeuristicAgent):
        agent = AgentCls(Team.A)
        action = agent.select_action(state)
        assert action.is_valid(state)


def test_match_runner_strategy_swappable() -> None:
    result = MatchRunner(
        team_a=RandomAgent(Team.A, seed=1),
        team_b=HeuristicAgent(Team.B),
        max_turns=12,
    ).play()
    assert result.turns > 0
    assert result.score_a >= 0
    assert result.score_b >= 0
    assert len(result.history) == result.turns


def test_jev_agent_uses_stub_client() -> None:
    """JevAgent maps typed answers → FlickAction without a real network call."""

    class FakeClient(JevClient):
        def __init__(self) -> None:  # noqa: D107
            # Bypass env key requirement
            self.api_key = "test"
            self.endpoint = "http://example.invalid"
            self.model = "jev-latest"
            self.timeout = 1.0

        def decide(self, state, questions):  # type: ignore[no-untyped-def]
            players = list(questions["player"]["criteria"].keys())
            assert "intent" in questions
            assert "strategy" in questions
            assert "ball_target" in questions
            return {
                "strategy": {
                    "choice": "build_two_move",
                    "probabilities": {"build_two_move": 1.0},
                },
                "intent": {
                    "choice": "advance_ball",
                    "probabilities": {"advance_ball": 1.0},
                },
                "player": {"choice": players[0], "probabilities": {players[0]: 1.0}},
                "ball_target": {
                    "choice": "opp_goal_center",
                    "probabilities": {"opp_goal_center": 1.0},
                },
                "power": {"score": 2.0},
            }

    env = HeadlessEnv()
    state = env.reset(Team.B)
    # Force Team B turn
    if state.current_team != Team.B:
        # Take a dummy Team A action first if needed
        a = HeuristicAgent(Team.A)
        state, *_ = env.step(a.select_action(state))

    agent = JevAgent(Team.B, client=FakeClient(), enable_search=False)  # type: ignore[arg-type]
    action = agent.select_action(env.state)  # type: ignore[arg-type]
    assert action.player_id.startswith("B")
    assert 0.0 <= action.power <= 1.0
    assert action.direction.length() > 0.9
    assert agent.last_prompt is not None
    assert "normalized" in agent.last_prompt.lower()
    assert "DOCTRINE" in agent.last_prompt


def test_jev_agent_falls_back_on_api_error() -> None:
    class BoomClient(JevClient):
        def __init__(self) -> None:
            self.api_key = "test"
            self.endpoint = "http://example.invalid"
            self.model = "jev-latest"
            self.timeout = 1.0

        def decide(self, state, questions):  # type: ignore[no-untyped-def]
            raise JevClientError("network down")

    env = HeadlessEnv()
    state = env.reset(Team.A)
    agent = JevAgent(Team.A, client=BoomClient())  # type: ignore[arg-type]
    action = agent.select_action(state)
    assert action.is_valid(state)
