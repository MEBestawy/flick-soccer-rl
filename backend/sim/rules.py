"""
Game rules and state transitions.

Handles:
- Goal detection
- Turn transitions
- Win conditions
- Phase management
"""

from __future__ import annotations
from typing import Optional, Tuple

from .geometry import Vec2
from .config import SimConfig
from .models import GameState, GamePhase, Team, Player, Ball
from .arena import Arena
from .events import GameEvent, EventLog
from .formations import get_goal_reset_positions


def check_goal(
    ball: Ball,
    arena: Arena,
) -> Optional[Team]:
    """
    Check if a goal has been scored.
    
    Returns:
        Team that scored (opposite of goal side), or None
    """
    # Check left goal (Team B scores on Team A)
    if arena.left_goal.ball_in_goal(ball.position):
        return Team.B
    
    # Check right goal (Team A scores on Team B)
    if arena.right_goal.ball_in_goal(ball.position):
        return Team.A
    
    return None


def handle_goal(
    state: GameState,
    scoring_team: Team,
    arena: Arena,
    config: SimConfig,
    event_log: EventLog,
    sim_time: float
) -> None:
    """
    Handle a goal being scored.
    
    Updates score, logs event, transitions to GOAL phase.
    """
    # Update score
    if scoring_team == Team.A:
        state.score_a += 1
    else:
        state.score_b += 1
    
    # Log event
    event_log.add(GameEvent.goal_scored(
        scoring_team,
        state.last_touch_player,
        sim_time
    ))
    
    # Transition to GOAL phase
    state.phase = GamePhase.GOAL


def reset_after_goal(
    state: GameState,
    arena: Arena,
    config: SimConfig,
) -> None:
    """
    Reset positions after a goal.
    
    Team that was scored on gets kickoff.
    """
    # Determine who kicks off (team that was scored on)
    if state.score_a > state.score_b:
        # Team A just scored, Team B kicks off
        kicking_side = "right"
    else:
        # Team B just scored, Team A kicks off
        kicking_side = "left"
    
    # Get reset positions
    team_a_pos, team_b_pos, ball_pos = get_goal_reset_positions(
        config.physics, kicking_side
    )
    
    # Reset player positions
    team_a_players = state.get_team_players(Team.A)
    team_b_players = state.get_team_players(Team.B)
    
    for i, player in enumerate(team_a_players):
        if i < len(team_a_pos):
            player.position = team_a_pos[i]
            player.velocity = Vec2.zero()
            player.is_sleeping = True
            player.sleep_timer = 0.0
    
    for i, player in enumerate(team_b_players):
        if i < len(team_b_pos):
            player.position = team_b_pos[i]
            player.velocity = Vec2.zero()
            player.is_sleeping = True
            player.sleep_timer = 0.0
    
    # Reset ball
    state.ball.position = ball_pos
    state.ball.velocity = Vec2.zero()
    state.ball.is_sleeping = True
    state.ball.sleep_timer = 0.0
    
    # Set kicking team
    state.current_team = Team.B if kicking_side == "right" else Team.A
    
    # Clear touch tracking
    state.last_touch_team = None
    state.last_touch_player = None


def check_game_over(state: GameState, config: SimConfig) -> bool:
    """Check if game is over."""
    return (
        state.score_a >= config.goals_to_win or
        state.score_b >= config.goals_to_win or
        state.turn_number >= config.max_turns * 2  # Both teams
    )


def get_winner(state: GameState, config: SimConfig) -> Optional[Team]:
    """Get the winning team, or None if tie/ongoing."""
    if state.score_a >= config.goals_to_win:
        return Team.A
    if state.score_b >= config.goals_to_win:
        return Team.B
    if state.turn_number >= config.max_turns * 2:
        # Max turns reached, winner by score
        if state.score_a > state.score_b:
            return Team.A
        if state.score_b > state.score_a:
            return Team.B
    return None


def advance_turn(state: GameState) -> None:
    """Advance to next team's turn."""
    state.current_team = state.current_team.opponent
    state.turn_number += 1
    state.turn_time = 0.0


def transition_phase(
    state: GameState,
    new_phase: GamePhase,
    event_log: EventLog,
    sim_time: float
) -> None:
    """Transition to a new game phase."""
    old_phase = state.phase
    state.phase = new_phase
    event_log.add(GameEvent.phase_change(
        old_phase.name, new_phase.name, sim_time
    ))
