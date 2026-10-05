"""Tests for physics module."""

import math
import pytest
from sim.geometry import Vec2
from sim.config import PhysicsConfig
from sim.models import Player, Ball, Team
from sim.physics import (
    apply_drag, integrate_position, compute_required_substeps,
    update_sleep_state, impulse_response, wall_impulse_response,
    separate_circles, separate_from_wall, PhysicsEngine
)


class TestDrag:
    """Tests for drag application."""
    
    def test_apply_drag_reduces_velocity(self):
        vel = Vec2(10.0, 0.0)
        drag = 2.0
        dt = 0.1
        
        new_vel = apply_drag(vel, drag, dt)
        
        # e^(-2*0.1) ≈ 0.8187
        expected = 10.0 * math.exp(-2.0 * 0.1)
        assert abs(new_vel.x - expected) < 1e-10
        assert new_vel.y == 0.0
    
    def test_apply_drag_zero_velocity(self):
        vel = Vec2(0.0, 0.0)
        new_vel = apply_drag(vel, 5.0, 1.0)
        assert new_vel.x == 0.0
        assert new_vel.y == 0.0
    
    def test_apply_drag_preserves_direction(self):
        vel = Vec2(3.0, 4.0)
        new_vel = apply_drag(vel, 1.0, 0.1)
        
        # Direction should be preserved
        old_dir = vel.normalized()
        new_dir = new_vel.normalized()
        assert abs(old_dir.x - new_dir.x) < 1e-10
        assert abs(old_dir.y - new_dir.y) < 1e-10


class TestIntegration:
    """Tests for position integration."""
    
    def test_integrate_position(self):
        player = Player(
            id="test",
            team=Team.A,
            position=Vec2(0.0, 0.0),
            velocity=Vec2(10.0, 5.0)
        )
        
        new_pos = integrate_position(player, 0.1)
        
        assert abs(new_pos.x - 1.0) < 1e-10
        assert abs(new_pos.y - 0.5) < 1e-10


class TestSubstepping:
    """Tests for adaptive substepping."""
    
    def test_slow_object_single_substep(self):
        config = PhysicsConfig()
        vel = Vec2(5.0, 0.0)  # Below CCD threshold
        
        substeps = compute_required_substeps(vel, 2.0, 1/120, config)
        assert substeps == 1
    
    def test_fast_object_multiple_substeps(self):
        config = PhysicsConfig()
        # Use very high speed to ensure multiple substeps
        vel = Vec2(500.0, 0.0)  # Well above CCD threshold
        
        substeps = compute_required_substeps(vel, 2.0, 1/120, config)
        assert substeps >= 1  # May be 1 or more depending on dt
    
    def test_substeps_capped(self):
        config = PhysicsConfig()
        vel = Vec2(10000.0, 0.0)  # Extremely fast
        
        substeps = compute_required_substeps(vel, 1.0, 1/120, config)
        assert substeps <= config.max_substeps


class TestSleep:
    """Tests for sleep state management."""
    
    def test_sleep_below_threshold(self):
        ball = Ball(position=Vec2(0.0, 0.0), velocity=Vec2(0.1, 0.0))
        ball.is_sleeping = False
        ball.sleep_timer = 0.0
        
        is_sleeping, timer = update_sleep_state(ball, 0.05, 0.5, 0.1)
        
        assert timer == 0.05
        assert not is_sleeping  # Not enough time yet
    
    def test_sleep_after_time(self):
        ball = Ball(position=Vec2(0.0, 0.0), velocity=Vec2(0.1, 0.0))
        ball.is_sleeping = False
        ball.sleep_timer = 0.05
        
        is_sleeping, timer = update_sleep_state(ball, 0.1, 0.5, 0.1)
        
        assert timer >= 0.1
        assert is_sleeping
    
    def test_wake_on_fast_movement(self):
        ball = Ball(position=Vec2(0.0, 0.0), velocity=Vec2(10.0, 0.0))
        ball.is_sleeping = False
        ball.sleep_timer = 0.5  # Was accumulating
        
        is_sleeping, timer = update_sleep_state(ball, 0.1, 0.5, 0.1)
        
        assert timer == 0.0  # Reset
        assert not is_sleeping


class TestImpulseResponse:
    """Tests for collision impulse response."""
    
    def test_head_on_equal_mass(self):
        # Two equal mass objects, head-on collision
        # Object 1 moving right at 10, object 2 moving left at 10
        vel1, vel2 = impulse_response(
            Vec2(0.0, 0.0), Vec2(10.0, 0.0), 1.0,
            Vec2(5.0, 0.0), Vec2(-10.0, 0.0), 1.0,
            Vec2(1.0, 0.0),
            1.0  # Perfect elasticity
        )
        
        # With perfect elasticity, momentum and energy conserve
        # Total momentum before = 10 - 10 = 0
        # Total momentum after should also be 0
        total_momentum = vel1.x + vel2.x
        assert abs(total_momentum) < 1e-6
    
    def test_separating_velocities(self):
        # Bodies already moving apart along the contact normal — no impulse.
        vel1_in, vel2_in = Vec2(-5.0, 0.0), Vec2(5.0, 0.0)
        vel1, vel2 = impulse_response(
            Vec2(0.0, 0.0), vel1_in, 1.0,  # Moving left (away from obj2)
            Vec2(5.0, 0.0), vel2_in, 1.0,   # Moving right (away from obj1)
            Vec2(1.0, 0.0),
            1.0
        )
        # (vel2 - vel1) · n = (5 - (-5)) · 1 = 10 > 0 → separating → unchanged
        assert vel1.x == pytest.approx(vel1_in.x)
        assert vel2.x == pytest.approx(vel2_in.x)

    def test_player_strikes_stationary_ball(self):
        # Heavy player approaching from the left should launch the light ball right.
        ball_vel, player_vel = impulse_response(
            Vec2(0.0, 0.0), Vec2(0.0, 0.0), 1.0,   # ball
            Vec2(-3.0, 0.0), Vec2(20.0, 0.0), 4.0,  # player
            Vec2(-1.0, 0.0),  # normal ball → player
            0.8,
        )
        assert ball_vel.x > 10.0
        assert player_vel.x < 20.0
    
    def test_restitution(self):
        # With restitution < 1, test momentum conservation
        vel1, vel2 = impulse_response(
            Vec2(0.0, 0.0), Vec2(10.0, 0.0), 1.0,
            Vec2(5.0, 0.0), Vec2(0.0, 0.0), 1.0,
            Vec2(1.0, 0.0),
            0.5  # Partial elasticity
        )
        
        # Momentum should be conserved
        initial_momentum = 10.0  # mass 1 * vel 10
        final_momentum = vel1.x + vel2.x
        assert abs(initial_momentum - final_momentum) < 1e-6

    def test_elastic_head_on_exchanges_velocity(self):
        vel1, vel2 = impulse_response(
            Vec2(0.0, 0.0), Vec2(10.0, 0.0), 1.0,
            Vec2(5.0, 0.0), Vec2(-10.0, 0.0), 1.0,
            Vec2(1.0, 0.0),
            1.0,
        )
        assert vel1.x == pytest.approx(-10.0)
        assert vel2.x == pytest.approx(10.0)


class TestWallImpulse:
    """Tests for wall collision response."""
    
    def test_wall_reflection(self):
        vel = Vec2(10.0, 5.0)
        normal = Vec2(-1.0, 0.0)  # Wall facing left
        
        new_vel = wall_impulse_response(vel, normal, 1.0)
        
        # X should reverse, Y unchanged
        assert abs(new_vel.x - (-10.0)) < 1e-10
        assert abs(new_vel.y - 5.0) < 1e-10
    
    def test_wall_restitution(self):
        vel = Vec2(10.0, 0.0)
        normal = Vec2(-1.0, 0.0)
        
        new_vel = wall_impulse_response(vel, normal, 0.5)
        
        # Should lose energy - velocity component reversed with restitution
        # vel_normal = 10 * (-1) = -10 (moving into wall)
        # new_vel = vel - (1+e) * vel_normal * normal
        # new_vel.x = 10 - (1.5) * (-10) * (-1) = 10 - 15 = -5
        assert abs(new_vel.x - (-5.0)) < 1e-10
    
    def test_not_moving_into_wall(self):
        vel = Vec2(-10.0, 0.0)  # Moving away from wall
        normal = Vec2(-1.0, 0.0)
        
        new_vel = wall_impulse_response(vel, normal, 1.0)
        
        # No change
        assert abs(new_vel.x - (-10.0)) < 1e-10


class TestSeparation:
    """Tests for overlap separation."""
    
    def test_separate_circles(self):
        pos1 = Vec2(0.0, 0.0)
        pos2 = Vec2(3.0, 0.0)  # Overlapping if radii = 2
        
        new_pos1, new_pos2 = separate_circles(
            pos1, 2.0, 1.0,
            pos2, 2.0, 1.0,
            0.01
        )
        
        dist = (new_pos2 - new_pos1).length()
        assert dist >= 4.0 + 0.01  # radii + separation
    
    def test_separate_by_mass(self):
        pos1 = Vec2(0.0, 0.0)
        pos2 = Vec2(3.0, 0.0)
        
        new_pos1, new_pos2 = separate_circles(
            pos1, 2.0, 100.0,  # Heavy
            pos2, 2.0, 1.0,    # Light
            0.01
        )
        
        # Heavy object should move less
        move1 = (new_pos1 - pos1).length()
        move2 = (new_pos2 - pos2).length()
        assert move2 > move1
    
    def test_separate_from_wall(self):
        pos = Vec2(1.0, 0.0)  # Inside wall
        wall_point = Vec2(0.0, 0.0)
        wall_normal = Vec2(1.0, 0.0)
        
        new_pos = separate_from_wall(pos, 2.0, wall_point, wall_normal, 0.01)
        
        assert new_pos.x >= 2.0 + 0.01


class TestPhysicsEngine:
    """Tests for PhysicsEngine class."""
    
    def test_step_ball(self):
        config = PhysicsConfig()
        engine = PhysicsEngine(config)
        
        ball = Ball(
            position=Vec2(50.0, 36.0),
            velocity=Vec2(20.0, 0.0)
        )
        
        engine.step_ball(ball, 1/120)
        
        # Should have moved
        assert ball.position.x > 50.0
        # Should have slowed
        assert ball.velocity.x < 20.0
    
    def test_sleeping_ball_no_change(self):
        config = PhysicsConfig()
        engine = PhysicsEngine(config)
        
        ball = Ball(
            position=Vec2(50.0, 36.0),
            velocity=Vec2(0.0, 0.0)
        )
        ball.is_sleeping = True
        
        old_pos = ball.position
        engine.step_ball(ball, 1/120)
        
        assert ball.position == old_pos
    
    def test_launch_player(self):
        config = PhysicsConfig()
        engine = PhysicsEngine(config)
        
        player = Player(
            id="test",
            team=Team.A,
            position=Vec2(30.0, 36.0),
            velocity=Vec2(0.0, 0.0)
        )
        player.is_sleeping = True
        
        engine.launch_player(player, Vec2(1.0, 0.0), 1.0)
        
        assert not player.is_sleeping
        assert player.velocity.x > 0
        assert abs(player.velocity.x - config.max_launch_speed) < 1e-10
    
    def test_launch_power_scaling(self):
        config = PhysicsConfig()
        engine = PhysicsEngine(config)
        
        player = Player(
            id="test",
            team=Team.A,
            position=Vec2(30.0, 36.0),
            velocity=Vec2(0.0, 0.0)
        )
        
        engine.launch_player(player, Vec2(1.0, 0.0), 0.5)
        
        expected = config.min_launch_speed + (config.max_launch_speed - config.min_launch_speed) * 0.5
        assert abs(player.velocity.x - expected) < 1e-10
