"""
Main game simulator.

Orchestrates physics, collisions, rules, and events into a complete
game simulation. This is the primary interface for running games.
"""

from __future__ import annotations
import time
from typing import List, Optional, Callable

from .geometry import Vec2
from .config import SimConfig, PhysicsConfig
from .models import GameState, GamePhase, Team, Player, Ball, Frame
from .actions import FlickAction, ActionResult, ActionError
from .physics import PhysicsEngine, compute_required_substeps
from .collisions import resolve_all_collisions
from .arena import Arena
from .rules import (
    check_goal, handle_goal, reset_after_goal,
    check_game_over, get_winner, advance_turn, transition_phase
)
from .events import GameEvent, EventLog, EventType
from .formations import get_kickoff_positions
from . import rust_bridge


class GameSimulator:
    """
    Complete game simulator.
    
    Provides both stateful and functional APIs for running games.
    
    Stateful API:
        sim = GameSimulator()
        sim.new_game()
        result = sim.execute_action(action)
    
    Functional API:
        state = GameSimulator.create_initial_state(config)
        new_state, frames, events = GameSimulator.simulate_action(state, action, config)
    """
    
    def __init__(self, config: Optional[SimConfig] = None) -> None:
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration. Uses default if not provided.
        """
        self.config = config or SimConfig.default()
        self.physics = PhysicsEngine(self.config.physics)
        self.arena = Arena.create(self.config.physics)
        self.event_log = EventLog()
        
        self._state: Optional[GameState] = None
    
    @property
    def state(self) -> Optional[GameState]:
        """Current game state (read-only access)."""
        return self._state
    
    def new_game(self, starting_team: Team = Team.A) -> GameState:
        """
        Start a new game.
        
        Args:
            starting_team: Team to kick off
        
        Returns:
            Initial game state
        """
        self.event_log.clear()
        self._state = self._create_initial_state(starting_team)
        self.event_log.add(GameEvent.match_start())
        self.event_log.add(GameEvent.turn_start(starting_team, 1))
        return self._state.clone()
    
    def _create_initial_state(self, starting_team: Team) -> GameState:
        """Create initial game state."""
        config = self.config
        physics = config.physics
        
        # Get kickoff positions
        kicking_side = "left" if starting_team == Team.A else "right"
        team_a_pos, team_b_pos, ball_pos = get_kickoff_positions(physics, kicking_side)
        
        # Create players
        players: List[Player] = []
        
        for i, pos in enumerate(team_a_pos):
            players.append(Player(
                id=f"A{i+1}",
                team=Team.A,
                position=pos,
                velocity=Vec2.zero(),
                radius=physics.player_radius,
                mass=physics.player_mass,
                is_sleeping=True,
            ))
        
        for i, pos in enumerate(team_b_pos):
            players.append(Player(
                id=f"B{i+1}",
                team=Team.B,
                position=pos,
                velocity=Vec2.zero(),
                radius=physics.player_radius,
                mass=physics.player_mass,
                is_sleeping=True,
            ))
        
        # Create ball
        ball = Ball(
            position=ball_pos,
            velocity=Vec2.zero(),
            radius=physics.ball_radius,
            mass=physics.ball_mass,
            is_sleeping=True,
        )
        
        return GameState(
            phase=GamePhase.KICKOFF,
            current_team=starting_team,
            turn_number=1,
            players=players,
            ball=ball,
        )
    
    def execute_action(
        self,
        action: FlickAction,
        capture_frames: bool = True
    ) -> ActionResult:
        """
        Execute an action in the current game.
        
        Args:
            action: The flick action to execute
            capture_frames: Whether to capture animation frames
        
        Returns:
            ActionResult with final state, frames, and events
        """
        if self._state is None:
            return ActionResult.failure(
                ActionError.WRONG_PHASE,
                "No game in progress. Call new_game() first."
            )
        
        # Validate action
        error = action.validate(self._state)
        if error is not None:
            return ActionResult.invalid_action(error)
        
        # Capture start state
        start_state = self._state.clone()
        start_time = time.perf_counter()
        
        # Clear events for this action
        action_events = EventLog()
        
        # Execute
        frames = self._simulate_action(
            action, action_events, capture_frames
        )
        
        end_time = time.perf_counter()
        
        # Build result
        result = ActionResult(
            success=True,
            start_state=start_state,
            final_state=self._state.clone(),
            frames=frames if capture_frames else None,
            events=action_events.events,
            simulation_duration=end_time - start_time,
            simulated_time=self._state.simulation_time,
        )
        
        # Add events to main log
        for event in action_events.events:
            self.event_log.add(event)
        
        return result
    
    def _simulate_action(
        self,
        action: FlickAction,
        event_log: EventLog,
        capture_frames: bool
    ) -> List[Frame]:
        """Run physics simulation for an action."""
        state = self._state
        if state is None:
            return []
        
        config = self.config
        physics_config = config.physics
        dt = physics_config.timestep
        
        # Get player and apply impulse
        player = state.get_player(action.player_id)
        if player is None:
            return []
        
        # Log flick event
        event_log.add(GameEvent.flick(
            action.player_id,
            player.team,
            action.direction,
            action.power,
            0.0
        ))
        
        # Transition to simulating
        state.phase = GamePhase.SIMULATING
        state.simulation_time = 0.0
        
        # Track touches
        state.last_touch_team = player.team
        state.last_touch_player = player.id
        
        frames: List[Frame] = []
        sim_time = 0.0
        goal_scored = False
        scoring_team = None
        log_events = capture_frames  # RL path skips event spam

        # Fast Rust path (no frame capture). Same physics/rules outcomes.
        if rust_bridge._rust_enabled() and not capture_frames:
            rust_result = rust_bridge.rust_simulate_flick(state, action, self.arena, config)
            if rust_result is not None:
                sim_time, scoring_team, _ = rust_result
                goal_scored = scoring_team is not None
                if goal_scored and scoring_team is not None:
                    handle_goal(
                        state, scoring_team, self.arena, config, event_log, sim_time
                    )
                else:
                    event_log.add(GameEvent.all_at_rest(sim_time))
                # Fall through to post-simulation state handling
                if goal_scored:
                    if check_game_over(state, config):
                        winner = get_winner(state, config)
                        state.phase = GamePhase.GAME_OVER
                        event_log.add(GameEvent.match_end(
                            winner, state.score_a, state.score_b
                        ))
                    else:
                        reset_after_goal(state, self.arena, config)
                        state.phase = GamePhase.KICKOFF
                        event_log.add(GameEvent.turn_start(
                            state.current_team, state.turn_number
                        ))
                else:
                    event_log.add(GameEvent.turn_end(
                        state.current_team, state.turn_number
                    ))
                    advance_turn(state)
                    state.phase = GamePhase.AIMING
                    event_log.add(GameEvent.turn_start(
                        state.current_team, state.turn_number
                    ))
                return frames

        self.physics.launch_player(
            player,
            action.normalized_direction(),
            action.power
        )
        
        # Frame capture
        frame_interval = 1.0 / config.frame_capture_rate
        next_frame_time = 0.0
        
        if capture_frames:
            frames.append(Frame.from_state(state, 0.0))
        
        # Simulation loop
        max_sim_time = config.max_simulation_time
        
        while sim_time < max_sim_time:
            # Compute substeps for fast objects
            max_speed = max(
                state.ball.velocity.length(),
                max((p.velocity.length() for p in state.players), default=0.0)
            )
            substeps = compute_required_substeps(
                Vec2(max_speed, 0), physics_config.ball_radius, dt, physics_config
            )
            sub_dt = dt / substeps
            
            # Physics substeps
            for _ in range(substeps):
                # Step physics
                self.physics.step_ball(state.ball, sub_dt)
                for p in state.players:
                    self.physics.step_player(p, sub_dt)
                
                # Resolve collisions
                collisions = resolve_all_collisions(
                    state.ball, state.players, self.arena,
                    physics_config, event_log, sim_time,
                    log_events=log_events,
                )
                
                # Track ball touches
                for c in collisions:
                    if c.type == "ball_player" and c.entity2 is not None:
                        touched_player = state.get_player(c.entity2)
                        if touched_player:
                            state.last_touch_team = touched_player.team
                            state.last_touch_player = touched_player.id
            
            sim_time += dt
            state.simulation_time = sim_time
            
            # Capture frame
            if capture_frames and sim_time >= next_frame_time:
                frames.append(Frame.from_state(state, sim_time))
                next_frame_time += frame_interval
            
            # Check for goal
            scoring_team = check_goal(state.ball, self.arena)
            if scoring_team is not None:
                handle_goal(
                    state, scoring_team, self.arena, config, event_log, sim_time
                )
                goal_scored = True
                break
            
            # Check if all at rest
            if state.all_at_rest(physics_config.sleep_threshold):
                event_log.add(GameEvent.all_at_rest(sim_time))
                break
        
        # Final frame
        if capture_frames and (not frames or frames[-1].time < sim_time):
            frames.append(Frame.from_state(state, sim_time))
        
        # Handle post-simulation state
        if goal_scored:
            # Check for game over
            if check_game_over(state, config):
                winner = get_winner(state, config)
                state.phase = GamePhase.GAME_OVER
                event_log.add(GameEvent.match_end(
                    winner, state.score_a, state.score_b
                ))
            else:
                # Reset for next kickoff
                reset_after_goal(state, self.arena, config)
                state.phase = GamePhase.KICKOFF
                event_log.add(GameEvent.turn_start(
                    state.current_team, state.turn_number
                ))
        else:
            # Normal turn end
            event_log.add(GameEvent.turn_end(
                state.current_team, state.turn_number
            ))
            advance_turn(state)
            state.phase = GamePhase.AIMING
            event_log.add(GameEvent.turn_start(
                state.current_team, state.turn_number
            ))
        
        return frames
    
    def simulate_until_rest(
        self,
        state: GameState,
        capture_frames: bool = False
    ) -> tuple[GameState, List[Frame]]:
        """
        Continue simulation until all objects are at rest.
        
        Useful for testing or continuing paused simulations.
        """
        state = state.clone()
        physics_config = self.config.physics
        dt = physics_config.timestep
        
        frames: List[Frame] = []
        frame_interval = 1.0 / self.config.frame_capture_rate
        next_frame_time = state.simulation_time
        
        event_log = EventLog()  # Temporary log
        max_time = self.config.max_simulation_time
        
        while state.simulation_time < max_time:
            self.physics.step_ball(state.ball, dt)
            for player in state.players:
                self.physics.step_player(player, dt)
            
            resolve_all_collisions(
                state.ball, state.players, self.arena,
                physics_config, event_log, state.simulation_time
            )
            
            state.simulation_time += dt
            
            if capture_frames and state.simulation_time >= next_frame_time:
                frames.append(Frame.from_state(state, state.simulation_time))
                next_frame_time += frame_interval
            
            if state.all_at_rest(physics_config.sleep_threshold):
                break
        
        return state, frames
    
    # Class-level functional API
    @staticmethod
    def create_initial_state(config: Optional[SimConfig] = None) -> GameState:
        """Create initial game state (functional API)."""
        sim = GameSimulator(config)
        return sim.new_game()
    
    @staticmethod
    def simulate_action(
        state: GameState,
        action: FlickAction,
        config: Optional[SimConfig] = None
    ) -> ActionResult:
        """
        Simulate action on state (functional API).
        
        Does not modify input state.
        """
        sim = GameSimulator(config)
        sim._state = state.clone()
        return sim.execute_action(action)
    
    def get_events(self) -> List[GameEvent]:
        """Get all events from current game."""
        return self.event_log.events
    
    def reset(self) -> None:
        """Reset simulator state."""
        self._state = None
        self.event_log.clear()
