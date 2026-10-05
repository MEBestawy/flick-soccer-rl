"""Tests for collision detection and resolution."""

import pytest
from sim.geometry import Vec2
from sim.config import PhysicsConfig
from sim.models import Player, Ball, Team
from sim.arena import Arena
from sim.events import EventLog
from sim.collisions import (
    resolve_ball_player_collisions,
    resolve_player_player_collisions,
    resolve_wall_collisions,
    resolve_post_collisions,
    resolve_all_collisions,
)


class TestBallPlayerCollisions:
    """Tests for ball-player collisions."""
    
    def test_no_collision(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        ball = Ball(position=Vec2(60.0, 36.0))
        players = [
            Player(id="A1", team=Team.A, position=Vec2(20.0, 36.0))
        ]
        
        collisions = resolve_ball_player_collisions(
            ball, players, config, event_log, 0.0
        )
        
        assert len(collisions) == 0
    
    def test_collision_detected(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        # Ball and player overlapping
        ball = Ball(position=Vec2(20.0, 36.0), velocity=Vec2(10.0, 0.0))
        ball.radius = config.ball_radius
        
        player = Player(
            id="A1",
            team=Team.A,
            position=Vec2(20.0 + config.ball_radius + config.player_radius - 0.5, 36.0),
            velocity=Vec2(-5.0, 0.0)
        )
        player.radius = config.player_radius
        
        collisions = resolve_ball_player_collisions(
            ball, [player], config, event_log, 0.0
        )
        
        assert len(collisions) == 1
        assert collisions[0].type == "ball_player"
    
    def test_collision_separates_objects(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        # Ball and player overlapping significantly
        ball = Ball(position=Vec2(20.0, 36.0))
        ball.radius = config.ball_radius
        
        player = Player(
            id="A1",
            team=Team.A,
            position=Vec2(20.0 + 2.0, 36.0),  # Close together
        )
        player.radius = config.player_radius
        
        resolve_ball_player_collisions(ball, [player], config, event_log, 0.0)
        
        # Should be separated
        dist = (ball.position - player.position).length()
        min_dist = ball.radius + player.radius
        assert dist >= min_dist - 0.1  # Allow small tolerance
    
    def test_collision_event_logged(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        ball = Ball(position=Vec2(20.0, 36.0), velocity=Vec2(10.0, 0.0))
        ball.radius = config.ball_radius
        
        player = Player(
            id="A1",
            team=Team.A,
            position=Vec2(20.0 + 2.0, 36.0),
        )
        player.radius = config.player_radius
        
        resolve_ball_player_collisions(ball, [player], config, event_log, 1.5)
        
        assert len(event_log) == 1
        event = event_log.events[0]
        assert event.type.name == "COLLISION_BALL_PLAYER"
        assert event.time == 1.5


class TestPlayerPlayerCollisions:
    """Tests for player-player collisions."""
    
    def test_no_collision(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        players = [
            Player(id="A1", team=Team.A, position=Vec2(20.0, 36.0)),
            Player(id="A2", team=Team.A, position=Vec2(40.0, 36.0)),
        ]
        
        collisions = resolve_player_player_collisions(
            players, config, event_log, 0.0
        )
        
        assert len(collisions) == 0
    
    def test_collision_detected(self):
        config = PhysicsConfig()
        event_log = EventLog()
        
        players = [
            Player(
                id="A1", team=Team.A,
                position=Vec2(20.0, 36.0),
                velocity=Vec2(10.0, 0.0)
            ),
            Player(
                id="A2", team=Team.A,
                position=Vec2(20.0 + config.player_radius * 1.5, 36.0),
                velocity=Vec2(-5.0, 0.0)
            ),
        ]
        for p in players:
            p.radius = config.player_radius
        
        collisions = resolve_player_player_collisions(
            players, config, event_log, 0.0
        )
        
        assert len(collisions) == 1
        assert collisions[0].type == "player_player"


class TestWallCollisions:
    """Tests for wall collisions."""
    
    def test_no_collision(self):
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        ball = Ball(position=Vec2(60.0, 36.0))  # Center of pitch
        players = []
        
        collisions = resolve_wall_collisions(
            ball, players, arena, config, event_log, 0.0
        )
        
        assert len(collisions) == 0
    
    def test_ball_wall_collision(self):
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        # Ball near top wall
        ball = Ball(
            position=Vec2(60.0, config.pitch_height - 0.5),
            velocity=Vec2(0.0, 10.0)
        )
        ball.radius = config.ball_radius
        
        collisions = resolve_wall_collisions(
            ball, [], arena, config, event_log, 0.0
        )
        
        assert len(collisions) == 1
        assert "ball" in collisions[0].entity1
    
    def test_wall_collision_reflects_velocity(self):
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        # Ball moving into top wall
        ball = Ball(
            position=Vec2(60.0, config.pitch_height - 0.5),
            velocity=Vec2(0.0, 10.0)
        )
        ball.radius = config.ball_radius
        
        resolve_wall_collisions(ball, [], arena, config, event_log, 0.0)
        
        # Velocity should be reflected (pointing down)
        assert ball.velocity.y < 0


class TestGoalPostCollisions:
    """Tests for goal post collisions."""
    
    def test_no_collision(self):
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        ball = Ball(position=Vec2(60.0, 36.0))  # Center
        
        collisions = resolve_post_collisions(
            ball, [], arena, config, event_log, 0.0
        )
        
        assert len(collisions) == 0
    
    def test_ball_post_collision(self):
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        # Ball near left goal post
        goal_y = config.goal_y_max
        ball = Ball(
            position=Vec2(0.5, goal_y),
            velocity=Vec2(-10.0, 0.0)
        )
        ball.radius = config.ball_radius
        
        collisions = resolve_post_collisions(
            ball, [], arena, config, event_log, 0.0
        )
        
        assert len(collisions) == 1
        assert "ball" in collisions[0].entity1


class TestResolveAllCollisions:
    """Tests for comprehensive collision resolution."""
    
    def test_chain_collisions(self):
        """Test that multiple iterations handle chain reactions."""
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        # Set up potential chain: ball -> player -> player
        ball = Ball(
            position=Vec2(30.0, 36.0),
            velocity=Vec2(20.0, 0.0)
        )
        ball.radius = config.ball_radius
        
        players = [
            Player(
                id="A1", team=Team.A,
                position=Vec2(30.0 + config.ball_radius + config.player_radius - 0.5, 36.0),
            ),
            Player(
                id="A2", team=Team.A,
                position=Vec2(30.0 + config.ball_radius + 3 * config.player_radius - 1.0, 36.0),
            ),
        ]
        for p in players:
            p.radius = config.player_radius
        
        collisions = resolve_all_collisions(
            ball, players, arena, config, event_log, 0.0
        )
        
        # Should have resolved ball-A1 collision at minimum
        assert len(collisions) >= 1
    
    def test_iteration_limit(self):
        """Test that iteration is bounded."""
        config = PhysicsConfig()
        arena = Arena.create(config)
        event_log = EventLog()
        
        # Normal case shouldn't hit iteration limit
        ball = Ball(position=Vec2(60.0, 36.0))
        
        # This should complete without infinite loop
        collisions = resolve_all_collisions(
            ball, [], arena, config, event_log, 0.0, max_iterations=4
        )
        
        assert isinstance(collisions, list)
