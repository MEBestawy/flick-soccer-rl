"""Tests for serialization."""

import pytest
from sim import (
    GameSimulator, GameState, FlickAction, Vec2, Team, GamePhase
)
from sim.models import Player, Ball, Frame
from sim.events import GameEvent, EventType
from sim.serialization import (
    serialize_vec2, deserialize_vec2,
    serialize_player, deserialize_player,
    serialize_ball, deserialize_ball,
    serialize_state, deserialize_state,
    serialize_frame, deserialize_frame,
    serialize_event, deserialize_event,
    serialize_action, deserialize_action,
)


class TestVec2Serialization:
    """Tests for Vec2 serialization."""
    
    def test_serialize(self):
        v = Vec2(3.5, -2.1)
        d = serialize_vec2(v)
        
        assert d == {"x": 3.5, "y": -2.1}
    
    def test_deserialize(self):
        d = {"x": 1.0, "y": 2.0}
        v = deserialize_vec2(d)
        
        assert v.x == 1.0
        assert v.y == 2.0
    
    def test_roundtrip(self):
        original = Vec2(123.456, -789.012)
        deserialized = deserialize_vec2(serialize_vec2(original))
        
        assert original.x == deserialized.x
        assert original.y == deserialized.y


class TestPlayerSerialization:
    """Tests for Player serialization."""
    
    def test_serialize(self):
        player = Player(
            id="A1",
            team=Team.A,
            position=Vec2(10.0, 20.0),
            velocity=Vec2(1.0, 2.0),
            radius=2.3,
            mass=4.0,
            is_sleeping=False,
        )
        
        d = serialize_player(player)
        
        assert d["id"] == "A1"
        assert d["team"] == "A"
        assert d["position"]["x"] == 10.0
        assert d["velocity"]["y"] == 2.0
        assert d["radius"] == 2.3
        assert d["is_sleeping"] == False
    
    def test_roundtrip(self):
        original = Player(
            id="B3",
            team=Team.B,
            position=Vec2(50.0, 30.0),
            velocity=Vec2(-5.0, 3.0),
        )
        
        deserialized = deserialize_player(serialize_player(original))
        
        assert original.id == deserialized.id
        assert original.team == deserialized.team
        assert original.position.x == deserialized.position.x
        assert original.velocity.y == deserialized.velocity.y


class TestBallSerialization:
    """Tests for Ball serialization."""
    
    def test_serialize(self):
        ball = Ball(
            position=Vec2(60.0, 36.0),
            velocity=Vec2(0.0, 0.0),
            radius=1.35,
            mass=1.0,
        )
        
        d = serialize_ball(ball)
        
        assert d["position"]["x"] == 60.0
        assert d["radius"] == 1.35
    
    def test_roundtrip(self):
        original = Ball(
            position=Vec2(55.0, 40.0),
            velocity=Vec2(10.0, -5.0),
        )
        
        deserialized = deserialize_ball(serialize_ball(original))
        
        assert original.position.x == deserialized.position.x
        assert original.velocity.y == deserialized.velocity.y


class TestStateSerialization:
    """Tests for GameState serialization."""
    
    def test_serialize_initial_state(self):
        sim = GameSimulator()
        state = sim.new_game()
        
        d = serialize_state(state)
        
        assert d["phase"] == "KICKOFF"
        assert d["current_team"] == "A"
        assert d["turn_number"] == 1
        assert len(d["players"]) == 10
        assert "ball" in d
        assert d["score_a"] == 0
        assert d["score_b"] == 0
    
    def test_roundtrip(self):
        sim = GameSimulator()
        original = sim.new_game()
        
        # Make some changes
        original.score_a = 2
        original.turn_number = 5
        
        deserialized = deserialize_state(serialize_state(original))
        
        assert original.phase == deserialized.phase
        assert original.current_team == deserialized.current_team
        assert original.turn_number == deserialized.turn_number
        assert original.score_a == deserialized.score_a
        assert len(original.players) == len(deserialized.players)


class TestFrameSerialization:
    """Tests for Frame serialization."""
    
    def test_serialize(self):
        frame = Frame(
            time=1.5,
            ball_position=Vec2(65.0, 40.0),
            ball_velocity=Vec2(5.0, 0.0),
            player_positions={"A1": Vec2(20.0, 30.0)},
            player_velocities={"A1": Vec2(0.0, 0.0)},
        )
        
        d = serialize_frame(frame)
        
        assert d["time"] == 1.5
        assert d["ball_position"]["x"] == 65.0
        assert "A1" in d["player_positions"]
    
    def test_roundtrip(self):
        original = Frame(
            time=2.0,
            ball_position=Vec2(70.0, 35.0),
            ball_velocity=Vec2(-3.0, 2.0),
            player_positions={
                "A1": Vec2(10.0, 20.0),
                "B1": Vec2(100.0, 50.0),
            },
            player_velocities={
                "A1": Vec2(0.0, 0.0),
                "B1": Vec2(1.0, 1.0),
            },
        )
        
        deserialized = deserialize_frame(serialize_frame(original))
        
        assert original.time == deserialized.time
        assert original.ball_position.x == deserialized.ball_position.x
        assert len(original.player_positions) == len(deserialized.player_positions)


class TestEventSerialization:
    """Tests for GameEvent serialization."""
    
    def test_serialize(self):
        event = GameEvent.goal_scored(Team.A, "A1", 5.5)
        
        d = serialize_event(event)
        
        assert d["type"] == "GOAL_SCORED"
        assert d["time"] == 5.5
        assert d["data"]["scoring_team"] == "A"
    
    def test_roundtrip(self):
        original = GameEvent.collision_ball_player("A1", Vec2(50.0, 30.0), 1.2)
        
        deserialized = deserialize_event(serialize_event(original))
        
        assert original.type == deserialized.type
        assert original.time == deserialized.time
        assert original.data["player_id"] == deserialized.data["player_id"]


class TestActionSerialization:
    """Tests for FlickAction serialization."""
    
    def test_serialize(self):
        action = FlickAction(
            player_id="A1",
            direction=Vec2(0.8, 0.6),
            power=0.75,
        )
        
        d = serialize_action(action)
        
        assert d["player_id"] == "A1"
        assert d["direction"]["x"] == 0.8
        assert d["power"] == 0.75
    
    def test_roundtrip(self):
        original = FlickAction(
            player_id="B3",
            direction=Vec2(-1.0, 0.5),
            power=0.5,
        )
        
        deserialized = deserialize_action(serialize_action(original))
        
        assert original.player_id == deserialized.player_id
        assert original.direction.x == deserialized.direction.x
        assert original.power == deserialized.power
