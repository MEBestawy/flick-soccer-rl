"""Stress tests for simulation robustness."""

import random
import time
import pytest
from sim import GameSimulator, FlickAction, HeadlessEnv, SimConfig, Vec2, Team


class TestStress:
    """Stress and robustness tests."""
    
    def test_many_rapid_actions(self):
        """Test many rapid consecutive actions."""
        env = HeadlessEnv()
        env.reset()
        
        for _ in range(50):
            if env.is_done():
                break
            
            action = env.random_action()
            if action is None:
                break
            
            state, reward, done, info = env.step(action)
            assert info["valid"]
    
    def test_max_power_shots(self):
        """Test that max power shots don't cause issues."""
        sim = GameSimulator()
        sim.new_game()
        
        for _ in range(10):
            controllable = sim.state.get_controllable_players()
            if not controllable:
                break
            
            player = controllable[0]
            direction = Vec2(
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            )
            
            action = FlickAction(
                player_id=player.id,
                direction=direction,
                power=1.0  # Max power
            )
            
            result = sim.execute_action(action)
            
            assert result.success
            # All objects should eventually settle
            assert sim.state.all_at_rest(sim.config.physics.sleep_threshold)
    
    def test_zero_power_shots(self):
        """Test that zero/minimal power shots work."""
        sim = GameSimulator()
        sim.new_game()
        
        controllable = sim.state.get_controllable_players()
        player = controllable[0]
        
        action = FlickAction(
            player_id=player.id,
            direction=Vec2(1.0, 0.0),
            power=0.0  # Zero power
        )
        
        result = sim.execute_action(action)
        
        assert result.success
    
    def test_diagonal_directions(self):
        """Test various diagonal direction vectors."""
        sim = GameSimulator()
        
        directions = [
            Vec2(1.0, 1.0),
            Vec2(-1.0, 1.0),
            Vec2(1.0, -1.0),
            Vec2(-1.0, -1.0),
            Vec2(0.5, 0.866),  # 60 degrees
            Vec2(0.866, 0.5),  # 30 degrees
        ]
        
        for direction in directions:
            sim.new_game()
            controllable = sim.state.get_controllable_players()
            player = controllable[0]
            
            action = FlickAction(
                player_id=player.id,
                direction=direction,
                power=0.5
            )
            
            result = sim.execute_action(action)
            assert result.success
    
    def test_no_penetration_after_many_collisions(self):
        """Test that objects don't penetrate after many collisions."""
        env = HeadlessEnv()
        env.reset()
        
        for _ in range(30):
            if env.is_done():
                break
            
            action = env.random_action()
            if action is None:
                break
            
            state, _, _, info = env.step(action)
            
            if not info["valid"]:
                continue
            
            # Check no penetration
            config = env.config.physics
            
            # Ball-player distances
            for player in state.players:
                dist = (state.ball.position - player.position).length()
                min_dist = state.ball.radius + player.radius - 0.5  # Allow small tolerance
                # Objects should be separated or nearly separated
                assert dist >= min_dist or state.ball.is_sleeping
            
            # Player-player distances
            for i, p1 in enumerate(state.players):
                for p2 in state.players[i+1:]:
                    dist = (p1.position - p2.position).length()
                    min_dist = p1.radius + p2.radius - 0.5
                    assert dist >= min_dist or (p1.is_sleeping and p2.is_sleeping)
    
    def test_positions_stay_in_bounds(self):
        """Test that all objects stay within arena bounds."""
        env = HeadlessEnv()
        config = env.config.physics
        
        for _ in range(5):  # Multiple games
            env.reset()
            
            for _ in range(20):  # Multiple turns
                if env.is_done():
                    break
                
                action = env.random_action()
                if action is None:
                    break
                
                state, _, _, info = env.step(action)
                
                if not info["valid"]:
                    continue
                
                # Ball should be within reasonable bounds
                # (can be in goals, so extend bounds slightly)
                assert -config.goal_width - 1 <= state.ball.position.x <= config.pitch_width + config.goal_width + 1
                assert -1 <= state.ball.position.y <= config.pitch_height + 1
                
                # Players should stay in bounds
                for player in state.players:
                    assert -config.goal_width - 1 <= player.position.x <= config.pitch_width + config.goal_width + 1
                    assert -1 <= player.position.y <= config.pitch_height + 1
    
    def test_simulation_time_bounded(self):
        """Test that simulation time is reasonable."""
        sim = GameSimulator()
        
        times = []
        for _ in range(10):
            sim.new_game()
            
            controllable = sim.state.get_controllable_players()
            player = controllable[0]
            
            action = FlickAction(
                player_id=player.id,
                direction=Vec2(1.0, 0.0),
                power=1.0
            )
            
            start = time.perf_counter()
            result = sim.execute_action(action)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            
            assert result.success
        
        avg_time = sum(times) / len(times)
        # Should be fast - under 500ms average (allowing for CI variability)
        assert avg_time < 0.5, f"Average simulation time too slow: {avg_time:.3f}s"
    
    def test_determinism_over_many_games(self):
        """Test determinism across multiple identical games."""
        config = SimConfig.default()
        
        for _ in range(5):
            # Run same sequence twice
            actions = []
            
            # First run
            sim1 = GameSimulator(config)
            sim1.new_game(Team.A)
            
            for _ in range(10):
                controllable = sim1.state.get_controllable_players()
                if not controllable:
                    break
                
                player = controllable[0]
                action = FlickAction(
                    player_id=player.id,
                    direction=Vec2(0.7, 0.3),
                    power=0.6
                )
                actions.append(action)
                sim1.execute_action(action)
            
            final1 = sim1.state
            
            # Second run with same actions
            sim2 = GameSimulator(config)
            sim2.new_game(Team.A)
            
            for action in actions:
                sim2.execute_action(action)
            
            final2 = sim2.state
            
            # Should be identical
            assert final1.score_a == final2.score_a
            assert final1.score_b == final2.score_b
            assert final1.turn_number == final2.turn_number
            
            assert abs(final1.ball.position.x - final2.ball.position.x) < 1e-10
            assert abs(final1.ball.position.y - final2.ball.position.y) < 1e-10


class TestEdgeCases:
    """Edge case tests."""
    
    def test_action_when_game_over(self):
        """Test action after game is over."""
        sim = GameSimulator()
        sim.new_game()
        
        # Force game over
        sim.state.score_a = sim.config.goals_to_win
        sim.state.phase = GamePhase.GAME_OVER
        
        controllable = sim.state.get_controllable_players()
        
        # Should have no controllable players when game is over
        assert len(controllable) == 0
    
    def test_clone_deep_copy(self):
        """Test that clone creates fully independent copy."""
        sim = GameSimulator()
        state = sim.new_game()
        
        cloned = state.clone()
        
        # Modify original
        state.ball.position = Vec2(999.0, 999.0)
        state.players[0].position = Vec2(888.0, 888.0)
        state.score_a = 99
        
        # Clone should be unchanged
        assert cloned.ball.position.x != 999.0
        assert cloned.players[0].position.x != 888.0
        assert cloned.score_a != 99


from sim.models import GamePhase
