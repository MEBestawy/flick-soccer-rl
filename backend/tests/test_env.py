"""Tests for headless environment."""

import pytest
from sim import HeadlessEnv, FlickAction, Team, GamePhase, Vec2


class TestHeadlessEnv:
    """Tests for HeadlessEnv class."""
    
    def test_create_env(self):
        env = HeadlessEnv()
        assert env.state is None
        assert env.is_done()
    
    def test_reset(self):
        env = HeadlessEnv()
        state = env.reset()
        
        assert state is not None
        assert not env.is_done()
        assert state.phase == GamePhase.KICKOFF
    
    def test_reset_with_team(self):
        env = HeadlessEnv()
        state = env.reset(starting_team=Team.B)
        
        assert state.current_team == Team.B
    
    def test_step(self):
        env = HeadlessEnv()
        env.reset()
        
        actions = env.get_legal_actions()
        assert len(actions) > 0
        
        action = actions[0]
        state, reward, done, info = env.step(action)
        
        assert state is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "valid" in info
        assert info["valid"]
    
    def test_invalid_step_before_reset(self):
        env = HeadlessEnv()
        
        action = FlickAction(
            player_id="A1",
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        with pytest.raises(RuntimeError):
            env.step(action)
    
    def test_get_legal_actions(self):
        env = HeadlessEnv()
        env.reset()
        
        actions = env.get_legal_actions()
        
        assert len(actions) > 0
        
        # All actions should be valid
        for action in actions:
            error = action.validate(env.state)
            assert error is None
    
    def test_get_controllable_players(self):
        env = HeadlessEnv()
        env.reset(Team.A)
        
        players = env.get_controllable_players()
        
        assert len(players) == 5
        assert all(p.startswith("A") for p in players)
    
    def test_get_observation(self):
        env = HeadlessEnv()
        env.reset()
        
        obs = env.get_observation()
        
        assert "phase" in obs
        assert "current_team" in obs
        assert "ball" in obs
        assert "players" in obs
        assert len(obs["players"]) == 10
    
    def test_get_flat_observation(self):
        env = HeadlessEnv()
        env.reset()
        
        obs = env.get_flat_observation()
        
        assert isinstance(obs, list)
        assert all(isinstance(x, float) for x in obs)
        # Should have: team(1) + scores(2) + ball(4) + 10 players * 5 = 57
        assert len(obs) == 57
    
    def test_random_action(self):
        env = HeadlessEnv()
        env.reset()
        
        action = env.random_action()
        
        assert action is not None
        assert action.is_valid(env.state)
    
    def test_render_ascii(self):
        env = HeadlessEnv()
        env.reset()
        
        ascii_art = env.render_ascii()
        
        assert isinstance(ascii_art, str)
        assert len(ascii_art) > 0
        assert "Turn" in ascii_art
    
    def test_full_game(self):
        """Test playing a complete game."""
        env = HeadlessEnv()
        env.reset()
        
        turns = 0
        max_turns = 200  # Safety limit
        
        while not env.is_done() and turns < max_turns:
            action = env.random_action()
            if action is None:
                break
            
            state, reward, done, info = env.step(action)
            turns += 1
        
        # Game should eventually end
        # (either by goals or max turns)
        assert turns > 0


class TestRewards:
    """Tests for reward calculation."""
    
    def test_invalid_action_penalty(self):
        env = HeadlessEnv()
        env.reset(Team.A)
        
        # Invalid action (wrong team)
        action = FlickAction(
            player_id="B1",
            direction=Vec2(1.0, 0.0),
            power=0.5
        )
        
        state, reward, done, info = env.step(action)
        
        assert reward < 0  # Penalty
        assert not info["valid"]
    
    def test_valid_action_no_goal(self):
        env = HeadlessEnv()
        env.reset()
        
        action = env.random_action()
        state, reward, done, info = env.step(action)
        
        # No goal = 0 reward typically
        assert info["valid"]
