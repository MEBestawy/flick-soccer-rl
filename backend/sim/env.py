"""
Headless environment for RL/AI agents.

Provides a gym-like interface for running games without any UI.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import random

from .geometry import Vec2
from .config import SimConfig
from .models import GameState, GamePhase, Team
from .actions import FlickAction, ActionResult, get_legal_actions
from .simulator import GameSimulator
from .events import GameEvent


class HeadlessEnv:
    """
    Headless game environment for AI/RL training.
    
    Provides:
    - Gym-like reset/step interface
    - Observation helpers
    - Action space helpers
    
    Example:
        env = HeadlessEnv()
        state = env.reset()
        
        while not env.is_done():
            actions = env.get_legal_actions()
            action = random.choice(actions)
            state, reward, done, info = env.step(action)
    """
    
    def __init__(self, config: Optional[SimConfig] = None) -> None:
        """
        Initialize environment.
        
        Args:
            config: Simulation config. Uses default if not provided.
        """
        self.config = config or SimConfig.default()
        self.simulator = GameSimulator(self.config)
        self._state: Optional[GameState] = None
        self._done: bool = True
        self._turn_count: int = 0
    
    @property
    def state(self) -> Optional[GameState]:
        """Current game state."""
        return self._state
    
    def reset(self, starting_team: Team = Team.A) -> GameState:
        """
        Reset environment to initial state.
        
        Args:
            starting_team: Team to kick off
        
        Returns:
            Initial game state
        """
        self._state = self.simulator.new_game(starting_team)
        self._done = False
        self._turn_count = 0
        return self._state.clone()
    
    def step(self, action: FlickAction) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Execute one complete turn.
        
        Args:
            action: The flick action to execute
        
        Returns:
            Tuple of (state, reward, done, info)
            - state: New game state
            - reward: Reward for the action (goal scored = +1, conceded = -1)
            - done: Whether game is over
            - info: Additional information dict
        """
        if self._state is None or self._done:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        acting_team = self._state.current_team
        state = self._state
        old_score_a = state.score_a
        old_score_b = state.score_b
        
        # Execute action
        result = self.simulator.execute_action(action, capture_frames=False)
        
        if not result.success:
            # Invalid action - return current state with negative reward
            return (
                state.clone(),
                -0.1,  # Small penalty for invalid action
                self._done,
                {"error": result.error_message, "valid": False}
            )
        
        final_state = result.final_state
        if final_state is None:
            raise RuntimeError("Action succeeded but no final state returned")
        
        self._state = final_state
        self._turn_count += 1
        
        # Calculate reward (from perspective of acting team)
        reward = 0.0
        if acting_team == Team.A:
            reward += (final_state.score_a - old_score_a)
            reward -= (final_state.score_b - old_score_b)
        else:
            reward += (final_state.score_b - old_score_b)
            reward -= (final_state.score_a - old_score_a)
        
        # Check if done
        self._done = final_state.phase == GamePhase.GAME_OVER
        
        info = {
            "valid": True,
            "turn_count": self._turn_count,
            "events": [e.type.name for e in (result.events or [])],
            "simulation_time": result.simulated_time,
        }
        
        return final_state.clone(), reward, self._done, info
    
    def is_done(self) -> bool:
        """Check if game is over."""
        return self._done
    
    def get_legal_actions(
        self,
        direction_samples: int = 8,
        power_samples: int = 3
    ) -> List[FlickAction]:
        """
        Get list of legal actions for current state.
        
        Args:
            direction_samples: Number of direction angles to sample
            power_samples: Number of power levels to sample
        
        Returns:
            List of valid FlickAction objects
        """
        if self._state is None:
            return []
        return get_legal_actions(self._state, direction_samples, power_samples)
    
    def get_controllable_players(self) -> List[str]:
        """Get IDs of players that can be controlled."""
        if self._state is None:
            return []
        return [p.id for p in self._state.get_controllable_players()]
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get observation dict for current state.
        
        Returns a simplified observation suitable for ML models.
        """
        if self._state is None:
            return {}
        
        state = self._state
        
        # Normalize positions to 0-1
        w = self.config.physics.pitch_width
        h = self.config.physics.pitch_height
        
        obs = {
            "phase": state.phase.name,
            "current_team": state.current_team.value,
            "turn_number": state.turn_number,
            "score": [state.score_a, state.score_b],
            "ball": {
                "position": [state.ball.position.x / w, state.ball.position.y / h],
                "velocity": [state.ball.velocity.x, state.ball.velocity.y],
            },
            "players": {},
        }
        
        for player in state.players:
            obs["players"][player.id] = {
                "team": player.team.value,
                "position": [player.position.x / w, player.position.y / h],
                "velocity": [player.velocity.x, player.velocity.y],
            }
        
        return obs
    
    def get_flat_observation(self) -> List[float]:
        """
        Get flattened observation vector.
        
        Useful for neural network input.
        """
        if self._state is None:
            return []
        
        state = self._state
        w = self.config.physics.pitch_width
        h = self.config.physics.pitch_height
        
        obs: List[float] = []
        
        # Team indicator (1 for A, -1 for B)
        obs.append(1.0 if state.current_team == Team.A else -1.0)
        
        # Scores (normalized)
        obs.append(state.score_a / self.config.goals_to_win)
        obs.append(state.score_b / self.config.goals_to_win)
        
        # Ball
        obs.append(state.ball.position.x / w)
        obs.append(state.ball.position.y / h)
        obs.append(state.ball.velocity.x / self.config.physics.max_launch_speed)
        obs.append(state.ball.velocity.y / self.config.physics.max_launch_speed)
        
        # Players (sorted by ID for consistency)
        for player in sorted(state.players, key=lambda p: p.id):
            team_indicator = 1.0 if player.team == Team.A else -1.0
            obs.append(team_indicator)
            obs.append(player.position.x / w)
            obs.append(player.position.y / h)
            obs.append(player.velocity.x / self.config.physics.max_launch_speed)
            obs.append(player.velocity.y / self.config.physics.max_launch_speed)
        
        return obs
    
    def random_action(self) -> Optional[FlickAction]:
        """Get a random legal action."""
        actions = self.get_legal_actions()
        if not actions:
            return None
        return random.choice(actions)
    
    def render_ascii(self) -> str:
        """
        Render current state as ASCII art.
        
        Useful for debugging.
        """
        if self._state is None:
            return "No game in progress"
        
        state = self._state
        w = int(self.config.physics.pitch_width)
        h = int(self.config.physics.pitch_height)
        
        # Scale down for display
        scale = 2
        dw = w // scale
        dh = h // scale
        
        # Initialize grid
        grid = [['.' for _ in range(dw)] for _ in range(dh)]
        
        # Draw boundaries
        for x in range(dw):
            grid[0][x] = '-'
            grid[dh-1][x] = '-'
        for y in range(dh):
            grid[y][0] = '|'
            grid[y][dw-1] = '|'
        
        # Draw goals
        goal_y_min = int(self.config.physics.goal_y_min) // scale
        goal_y_max = int(self.config.physics.goal_y_max) // scale
        for y in range(goal_y_min, goal_y_max):
            if 0 <= y < dh:
                grid[y][0] = '['
                grid[y][dw-1] = ']'
        
        # Draw ball
        bx = int(state.ball.position.x) // scale
        by = int(state.ball.position.y) // scale
        if 0 <= bx < dw and 0 <= by < dh:
            grid[dh - 1 - by][bx] = 'o'
        
        # Draw players
        for player in state.players:
            px = int(player.position.x) // scale
            py = int(player.position.y) // scale
            if 0 <= px < dw and 0 <= py < dh:
                char = player.id[0].lower() if player.team == Team.A else player.id[0].upper()
                grid[dh - 1 - py][px] = char
        
        # Build string
        lines = [''.join(row) for row in grid]
        header = f"Turn {state.turn_number} | {state.current_team.value}'s turn | Score: {state.score_a}-{state.score_b}"
        return header + '\n' + '\n'.join(lines)
