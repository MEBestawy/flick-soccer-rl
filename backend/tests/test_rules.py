"""Tests for game rules."""

import pytest
from sim.geometry import Vec2
from sim.config import SimConfig
from sim.models import GameState, GamePhase, Team, Player, Ball
from sim.arena import Arena
from sim.events import EventLog
from sim.rules import (
    check_goal, handle_goal, reset_after_goal,
    check_game_over, get_winner, advance_turn
)


class TestGoalDetection:
    """Tests for goal detection."""
    
    def test_no_goal_center(self):
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        
        ball = Ball(position=Vec2(60.0, 36.0))  # Center
        
        result = check_goal(ball, arena)
        assert result is None
    
    def test_goal_left(self):
        """Ball in left goal = Team B scores."""
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        
        # Ball inside left goal
        ball = Ball(position=Vec2(-1.0, 36.0))
        
        result = check_goal(ball, arena)
        assert result == Team.B
    
    def test_goal_right(self):
        """Ball in right goal = Team A scores."""
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        
        # Ball inside right goal
        ball = Ball(position=Vec2(config.physics.pitch_width + 1.0, 36.0))
        
        result = check_goal(ball, arena)
        assert result == Team.A
    
    def test_no_goal_outside_posts(self):
        """Ball past goal line but outside posts = no goal."""
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        
        # Ball past left goal line but above goal mouth
        ball = Ball(position=Vec2(-1.0, config.physics.goal_y_max + 5.0))
        
        result = check_goal(ball, arena)
        assert result is None


class TestGoalHandling:
    """Tests for goal handling."""
    
    def test_handle_goal_updates_score(self):
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        event_log = EventLog()
        
        # Create minimal state
        state = GameState(
            phase=GamePhase.SIMULATING,
            current_team=Team.A,
            turn_number=1,
            players=[],
            ball=Ball(position=Vec2(0.0, 0.0)),
            score_a=0,
            score_b=0,
        )
        
        handle_goal(state, Team.A, arena, config, event_log, 1.0)
        
        assert state.score_a == 1
        assert state.score_b == 0
        assert state.phase == GamePhase.GOAL
    
    def test_handle_goal_logs_event(self):
        config = SimConfig.default()
        arena = Arena.create(config.physics)
        event_log = EventLog()
        
        state = GameState(
            phase=GamePhase.SIMULATING,
            current_team=Team.A,
            turn_number=1,
            players=[],
            ball=Ball(position=Vec2(0.0, 0.0)),
        )
        state.last_touch_player = "A1"
        
        handle_goal(state, Team.A, arena, config, event_log, 2.5)
        
        assert len(event_log) == 1
        assert event_log.events[0].type.name == "GOAL_SCORED"
        assert event_log.events[0].time == 2.5


class TestResetAfterGoal:
    """Tests for position reset after goal."""
    
    def test_reset_positions(self):
        config = SimConfig.default()
        physics = config.physics
        arena = Arena.create(physics)
        
        # Create state with players
        players = [
            Player(id=f"A{i+1}", team=Team.A, position=Vec2(50.0, 50.0))
            for i in range(5)
        ] + [
            Player(id=f"B{i+1}", team=Team.B, position=Vec2(70.0, 20.0))
            for i in range(5)
        ]
        
        state = GameState(
            phase=GamePhase.GOAL,
            current_team=Team.A,
            turn_number=1,
            players=players,
            ball=Ball(position=Vec2(-5.0, 36.0)),  # In goal
            score_a=1,
            score_b=0,
        )
        
        reset_after_goal(state, arena, config)
        
        # Ball should be at center
        assert abs(state.ball.position.x - physics.pitch_width / 2) < 10
        
        # All players should be within pitch
        for player in state.players:
            assert 0 <= player.position.x <= physics.pitch_width
            assert 0 <= player.position.y <= physics.pitch_height


class TestGameOver:
    """Tests for game over conditions."""
    
    def test_not_over_at_start(self):
        config = SimConfig.default()
        
        state = GameState(
            phase=GamePhase.AIMING,
            current_team=Team.A,
            turn_number=1,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
            score_a=0,
            score_b=0,
        )
        
        assert not check_game_over(state, config)
    
    def test_game_over_team_a_wins(self):
        config = SimConfig.default()
        
        state = GameState(
            phase=GamePhase.AIMING,
            current_team=Team.A,
            turn_number=10,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
            score_a=config.goals_to_win,
            score_b=0,
        )
        
        assert check_game_over(state, config)
        assert get_winner(state, config) == Team.A
    
    def test_game_over_team_b_wins(self):
        config = SimConfig.default()
        
        state = GameState(
            phase=GamePhase.AIMING,
            current_team=Team.B,
            turn_number=15,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
            score_a=1,
            score_b=config.goals_to_win,
        )
        
        assert check_game_over(state, config)
        assert get_winner(state, config) == Team.B
    
    def test_game_over_max_turns(self):
        config = SimConfig.default()
        
        state = GameState(
            phase=GamePhase.AIMING,
            current_team=Team.A,
            turn_number=config.max_turns * 2,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
            score_a=2,
            score_b=1,
        )
        
        assert check_game_over(state, config)
        assert get_winner(state, config) == Team.A  # Higher score wins


class TestTurnAdvance:
    """Tests for turn advancement."""
    
    def test_advance_turn(self):
        state = GameState(
            phase=GamePhase.TURN_END,
            current_team=Team.A,
            turn_number=1,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
        )
        
        advance_turn(state)
        
        assert state.current_team == Team.B
        assert state.turn_number == 2
        assert state.turn_time == 0.0
    
    def test_advance_turn_alternates(self):
        state = GameState(
            phase=GamePhase.TURN_END,
            current_team=Team.B,
            turn_number=4,
            players=[],
            ball=Ball(position=Vec2(60.0, 36.0)),
        )
        
        advance_turn(state)
        
        assert state.current_team == Team.A
