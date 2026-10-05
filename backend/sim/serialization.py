"""
JSON serialization for game state.

Provides conversion between game objects and JSON-compatible dictionaries.
Used for API responses, replays, and persistence.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from .geometry import Vec2
from .models import GameState, GamePhase, Team, Player, Ball, Frame
from .actions import FlickAction
from .events import GameEvent, EventType


def serialize_vec2(v: Vec2) -> Dict[str, float]:
    """Serialize Vec2 to dict."""
    return {"x": v.x, "y": v.y}


def deserialize_vec2(d: Dict[str, float]) -> Vec2:
    """Deserialize Vec2 from dict."""
    return Vec2(d["x"], d["y"])


def serialize_player(player: Player) -> Dict[str, Any]:
    """Serialize Player to dict."""
    return {
        "id": player.id,
        "team": player.team.value,
        "position": serialize_vec2(player.position),
        "velocity": serialize_vec2(player.velocity),
        "radius": player.radius,
        "mass": player.mass,
        "is_sleeping": player.is_sleeping,
    }


def deserialize_player(d: Dict[str, Any]) -> Player:
    """Deserialize Player from dict."""
    return Player(
        id=d["id"],
        team=Team(d["team"]),
        position=deserialize_vec2(d["position"]),
        velocity=deserialize_vec2(d["velocity"]),
        radius=d.get("radius", 2.3),
        mass=d.get("mass", 4.0),
        is_sleeping=d.get("is_sleeping", False),
    )


def serialize_ball(ball: Ball) -> Dict[str, Any]:
    """Serialize Ball to dict."""
    return {
        "position": serialize_vec2(ball.position),
        "velocity": serialize_vec2(ball.velocity),
        "radius": ball.radius,
        "mass": ball.mass,
        "is_sleeping": ball.is_sleeping,
    }


def deserialize_ball(d: Dict[str, Any]) -> Ball:
    """Deserialize Ball from dict."""
    return Ball(
        position=deserialize_vec2(d["position"]),
        velocity=deserialize_vec2(d["velocity"]),
        radius=d.get("radius", 1.35),
        mass=d.get("mass", 1.0),
        is_sleeping=d.get("is_sleeping", False),
    )


def serialize_state(state: GameState) -> Dict[str, Any]:
    """
    Serialize complete game state to dict.
    
    This is the primary serialization method for API responses.
    """
    return {
        "phase": state.phase.name,
        "current_team": state.current_team.value,
        "turn_number": state.turn_number,
        "players": [serialize_player(p) for p in state.players],
        "ball": serialize_ball(state.ball),
        "score_a": state.score_a,
        "score_b": state.score_b,
        "match_time": state.match_time,
        "turn_time": state.turn_time,
        "simulation_time": state.simulation_time,
        "last_touch_team": state.last_touch_team.value if state.last_touch_team else None,
        "last_touch_player": state.last_touch_player,
    }


def deserialize_state(d: Dict[str, Any]) -> GameState:
    """Deserialize game state from dict."""
    return GameState(
        phase=GamePhase[d["phase"]],
        current_team=Team(d["current_team"]),
        turn_number=d["turn_number"],
        players=[deserialize_player(p) for p in d["players"]],
        ball=deserialize_ball(d["ball"]),
        score_a=d.get("score_a", 0),
        score_b=d.get("score_b", 0),
        match_time=d.get("match_time", 0.0),
        turn_time=d.get("turn_time", 0.0),
        simulation_time=d.get("simulation_time", 0.0),
        last_touch_team=Team(d["last_touch_team"]) if d.get("last_touch_team") else None,
        last_touch_player=d.get("last_touch_player"),
    )


def serialize_frame(frame: Frame) -> Dict[str, Any]:
    """Serialize animation frame to dict."""
    return {
        "time": frame.time,
        "ball_position": serialize_vec2(frame.ball_position),
        "ball_velocity": serialize_vec2(frame.ball_velocity),
        "player_positions": {
            pid: serialize_vec2(pos) 
            for pid, pos in frame.player_positions.items()
        },
        "player_velocities": {
            pid: serialize_vec2(vel)
            for pid, vel in frame.player_velocities.items()
        },
    }


def deserialize_frame(d: Dict[str, Any]) -> Frame:
    """Deserialize animation frame from dict."""
    return Frame(
        time=d["time"],
        ball_position=deserialize_vec2(d["ball_position"]),
        ball_velocity=deserialize_vec2(d["ball_velocity"]),
        player_positions={
            pid: deserialize_vec2(pos)
            for pid, pos in d["player_positions"].items()
        },
        player_velocities={
            pid: deserialize_vec2(vel)
            for pid, vel in d["player_velocities"].items()
        },
    )


def serialize_event(event: GameEvent) -> Dict[str, Any]:
    """Serialize game event to dict."""
    return {
        "type": event.type.name,
        "time": event.time,
        "data": event.data,
    }


def deserialize_event(d: Dict[str, Any]) -> GameEvent:
    """Deserialize game event from dict."""
    return GameEvent(
        type=EventType[d["type"]],
        time=d["time"],
        data=d.get("data", {}),
    )


def serialize_action(action: FlickAction) -> Dict[str, Any]:
    """Serialize flick action to dict."""
    return {
        "player_id": action.player_id,
        "direction": serialize_vec2(action.direction),
        "power": action.power,
    }


def deserialize_action(d: Dict[str, Any]) -> FlickAction:
    """Deserialize flick action from dict."""
    return FlickAction(
        player_id=d["player_id"],
        direction=deserialize_vec2(d["direction"]),
        power=d["power"],
    )


# Replay format
def serialize_replay(
    states: List[GameState],
    actions: List[FlickAction],
    all_frames: List[List[Frame]],
    events: List[GameEvent]
) -> Dict[str, Any]:
    """Serialize complete game replay."""
    return {
        "version": "1.0",
        "initial_state": serialize_state(states[0]) if states else None,
        "actions": [serialize_action(a) for a in actions],
        "frames": [
            [serialize_frame(f) for f in turn_frames]
            for turn_frames in all_frames
        ],
        "events": [serialize_event(e) for e in events],
        "final_state": serialize_state(states[-1]) if states else None,
    }
