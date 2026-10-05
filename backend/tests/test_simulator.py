"""Tests for game simulator."""

import pytest
from sim import (
    GameSimulator, GameState, FlickAction, SimConfig,
    Team, GamePhase, Vec2
)
from sim.actions import ActionError


class TestSimulatorBasics:
    """Basic simulator tests."""
    
    def test_create_simulator(self):
        sim = GameSimulator()
        assert sim.state is None
    
    def test_new_game(self):
        sim = GameSimulator()
        state = sim.new_game()
        
        assert state is not None
        assert state.phase == GamePhase.KICKOFF
        assert state.current_team == Team.A
        assert state.turn_number == 1
        assert len(state.players) == 10  # 5 per team
        assert state.score_a == 0
        assert state.score_b == 0
    
    def test_new_game_team_b_starts(self):
        sim = GameSimulator()
        state = sim.new_game(Team.B)
        
        assert state.current_team == Team.B
    
    def test_player_positions(self):
        sim = GameSimulator()
        state = sim.new_game()
        
        team_a = state.get_team_players(Team.A)
        team_b = state.get_team_players(Team.B)
        
        assert len(team_a) == 5
        assert len(team_b) == 5
        
        # Team A should be on left side
        for p in team_a:
            assert p.position.x < 70  # Less than center + some margin
        
        # Team B should be on right side
        for p in team_b:
            assert p.position.x > 50  # Greater than center - some margin
    
    def test_ball_at_center(self):
        sim = GameSimulator()
        state = sim.new_game()
        
        # Ball should be near center
        assert abs(state.ball.position.x - 60.0) < 10
        assert abs(state.ball.position.y - 36.0) < 10


class TestActions:
    """Action execution tests."""
    
    def test_valid_action(self):
        sim = GameSimulator()
        sim.new_game()
        
        # Get a controllable player
        controllable = sim.state.get_controllable_players()
        assert len(controllable) > 0
        
        player = controllable[0]
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action)
        
        assert result.success
        assert result.final_state is not None
        assert result.frames is not None
        assert len(result.frames) > 0
    
    def test_invalid_player(self):
        sim = GameSimulator()
        sim.new_game()
        
        action = FlickAction(
            player_id="INVALID",
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action)
        
        assert not result.success
        assert result.error == ActionError.PLAYER_NOT_FOUND
    
    def test_wrong_team(self):
        sim = GameSimulator()
        sim.new_game(Team.A)
        
        # Try to flick Team B player
        team_b_players = sim.state.get_team_players(Team.B)
        player = team_b_players[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action)
        
        assert not result.success
        assert result.error == ActionError.WRONG_TEAM
    
    def test_invalid_power(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=1.5  # Invalid: > 1
        )
        
        result = sim.execute_action(action)
        
        assert not result.success
        assert result.error == ActionError.POWER_OUT_OF_RANGE
    
    def test_zero_direction(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(0.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action)
        
        assert not result.success
        assert result.error == ActionError.ZERO_DIRECTION


class TestTurnTransitions:
    """Turn and phase transition tests."""
    
    def test_turn_advances_after_action(self):
        sim = GameSimulator()
        sim.new_game(Team.A)
        
        initial_turn = sim.state.turn_number
        initial_team = sim.state.current_team
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.3
        )
        
        result = sim.execute_action(action)
        
        assert result.success
        # Turn should advance
        assert sim.state.turn_number == initial_turn + 1
        # Team should switch
        assert sim.state.current_team == Team.B
    
    def test_simulation_settles(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.2  # Low power = faster settling
        )
        
        result = sim.execute_action(action)
        
        assert result.success
        # All should be at rest after simulation
        assert result.final_state.all_at_rest(sim.config.physics.sleep_threshold)


class TestDeterminism:
    """Determinism tests."""
    
    def test_same_action_same_result(self):
        """Same initial state + action = same result."""
        config = SimConfig.default()
        
        # Run 1
        sim1 = GameSimulator(config)
        state1 = sim1.new_game()
        action = FlickAction(
            player_id="A1",
            direction=Vec2(1.0, 0.5),
            power=0.7
        )
        result1 = sim1.execute_action(action)
        
        # Run 2
        sim2 = GameSimulator(config)
        state2 = sim2.new_game()
        result2 = sim2.execute_action(action)
        
        # States should be identical
        final1 = result1.final_state
        final2 = result2.final_state
        
        assert final1.ball.position.x == final2.ball.position.x
        assert final1.ball.position.y == final2.ball.position.y
        
        for p1 in final1.players:
            p2 = final2.get_player(p1.id)
            assert p1.position.x == p2.position.x
            assert p1.position.y == p2.position.y
    
    def test_clone_independence(self):
        """Cloned states should be independent."""
        sim = GameSimulator()
        state = sim.new_game()
        
        cloned = state.clone()
        
        # Modify original
        state.score_a = 5
        state.ball.position = Vec2(0.0, 0.0)
        
        # Clone should be unchanged
        assert cloned.score_a == 0
        assert cloned.ball.position.x != 0.0


class TestGoalScoring:
    """Goal scoring tests."""
    
    def test_goal_detection(self):
        """Test that goals are detected properly."""
        sim = GameSimulator()
        sim.new_game()
        
        # Move ball into goal area
        sim.state.ball.position = Vec2(-1.0, 36.0)  # Inside left goal
        sim.state.ball.is_sleeping = False
        sim.state.ball.velocity = Vec2(-10.0, 0.0)
        
        # The goal should be detected in simulation
        # We'll test this indirectly through the rules module
        from sim.rules import check_goal
        from sim.arena import Arena
        
        arena = Arena.create(sim.config.physics)
        scoring_team = check_goal(sim.state.ball, arena)
        
        assert scoring_team == Team.B  # B scores on A's goal


class TestFrameCapture:
    """Frame capture tests."""
    
    def test_frames_captured(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action, capture_frames=True)
        
        assert result.frames is not None
        assert len(result.frames) > 1
        
        # Frames should be ordered by time
        for i in range(1, len(result.frames)):
            assert result.frames[i].time >= result.frames[i-1].time
    
    def test_no_frames_when_disabled(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.3
        )
        
        result = sim.execute_action(action, capture_frames=False)
        
        assert result.frames is None


class TestEvents:
    """Event logging tests."""
    
    def test_events_logged(self):
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = sim.execute_action(action)
        
        assert result.events is not None
        assert len(result.events) > 0
        
        # Should have flick event
        event_types = [e.type.name for e in result.events]
        assert "FLICK" in event_types


class TestFunctionalAPI:
    """Functional API tests."""
    
    def test_create_initial_state(self):
        state = GameSimulator.create_initial_state()
        
        assert state is not None
        assert state.phase == GamePhase.KICKOFF
        assert len(state.players) == 10
    
    def test_simulate_action_static(self):
        state = GameSimulator.create_initial_state()
        
        action = FlickAction(
            player_id="A1",
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        result = GameSimulator.simulate_action(state, action)
        
        assert result.success
        assert result.final_state is not None
        
        # Original state should be unchanged
        assert state.turn_number == 1
        assert state.phase == GamePhase.KICKOFF
