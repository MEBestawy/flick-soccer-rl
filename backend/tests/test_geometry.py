"""Tests for geometry module."""

import math
import pytest
from sim.geometry import (
    Vec2, Circle, Segment,
    circle_circle_collision, circle_segment_collision,
    ray_circle_intersection, ray_segment_intersection, clamp
)


class TestVec2:
    """Tests for Vec2 class."""
    
    def test_creation(self):
        v = Vec2(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0
    
    def test_add(self):
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0
    
    def test_sub(self):
        v1 = Vec2(5.0, 7.0)
        v2 = Vec2(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0
    
    def test_mul(self):
        v = Vec2(2.0, 3.0)
        result = v * 2.0
        assert result.x == 4.0
        assert result.y == 6.0
    
    def test_rmul(self):
        v = Vec2(2.0, 3.0)
        result = 3.0 * v
        assert result.x == 6.0
        assert result.y == 9.0
    
    def test_div(self):
        v = Vec2(6.0, 8.0)
        result = v / 2.0
        assert result.x == 3.0
        assert result.y == 4.0
    
    def test_neg(self):
        v = Vec2(3.0, -4.0)
        result = -v
        assert result.x == -3.0
        assert result.y == 4.0
    
    def test_dot(self):
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        assert v1.dot(v2) == 11.0  # 1*3 + 2*4
    
    def test_cross(self):
        v1 = Vec2(1.0, 0.0)
        v2 = Vec2(0.0, 1.0)
        assert v1.cross(v2) == 1.0  # 1*1 - 0*0
    
    def test_length(self):
        v = Vec2(3.0, 4.0)
        assert v.length() == 5.0
    
    def test_length_squared(self):
        v = Vec2(3.0, 4.0)
        assert v.length_squared() == 25.0
    
    def test_normalized(self):
        v = Vec2(3.0, 4.0)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 1e-10
        assert abs(n.x - 0.6) < 1e-10
        assert abs(n.y - 0.8) < 1e-10
    
    def test_normalized_zero(self):
        v = Vec2(0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0
    
    def test_rotate(self):
        v = Vec2(1.0, 0.0)
        rotated = v.rotate(math.pi / 2)  # 90 degrees
        assert abs(rotated.x) < 1e-10
        assert abs(rotated.y - 1.0) < 1e-10
    
    def test_reflect(self):
        v = Vec2(1.0, -1.0)
        normal = Vec2(0.0, 1.0)
        reflected = v.reflect(normal)
        assert abs(reflected.x - 1.0) < 1e-10
        assert abs(reflected.y - 1.0) < 1e-10
    
    def test_lerp(self):
        v1 = Vec2(0.0, 0.0)
        v2 = Vec2(10.0, 20.0)
        mid = v1.lerp(v2, 0.5)
        assert mid.x == 5.0
        assert mid.y == 10.0
    
    def test_distance_to(self):
        v1 = Vec2(0.0, 0.0)
        v2 = Vec2(3.0, 4.0)
        assert v1.distance_to(v2) == 5.0
    
    def test_from_angle(self):
        v = Vec2.from_angle(0.0, 2.0)
        assert abs(v.x - 2.0) < 1e-10
        assert abs(v.y) < 1e-10
        
        v = Vec2.from_angle(math.pi / 2, 1.0)
        assert abs(v.x) < 1e-10
        assert abs(v.y - 1.0) < 1e-10
    
    def test_to_tuple(self):
        v = Vec2(1.0, 2.0)
        assert v.to_tuple() == (1.0, 2.0)
    
    def test_from_tuple(self):
        v = Vec2.from_tuple((3.0, 4.0))
        assert v.x == 3.0
        assert v.y == 4.0


class TestCircle:
    """Tests for Circle class."""
    
    def test_contains(self):
        c = Circle(Vec2(0.0, 0.0), 5.0)
        assert c.contains(Vec2(0.0, 0.0))
        assert c.contains(Vec2(3.0, 4.0))
        assert not c.contains(Vec2(6.0, 0.0))
    
    def test_intersects_circle(self):
        c1 = Circle(Vec2(0.0, 0.0), 2.0)
        c2 = Circle(Vec2(3.0, 0.0), 2.0)
        assert c1.intersects_circle(c2)  # Overlapping
        
        c3 = Circle(Vec2(5.0, 0.0), 2.0)
        assert not c1.intersects_circle(c3)  # Not touching
    
    def test_distance_to_circle(self):
        c1 = Circle(Vec2(0.0, 0.0), 2.0)
        c2 = Circle(Vec2(5.0, 0.0), 2.0)
        assert c1.distance_to_circle(c2) == 1.0  # 5 - 2 - 2


class TestSegment:
    """Tests for Segment class."""
    
    def test_direction(self):
        s = Segment(Vec2(0.0, 0.0), Vec2(3.0, 4.0))
        d = s.direction
        assert d.x == 3.0
        assert d.y == 4.0
    
    def test_length(self):
        s = Segment(Vec2(0.0, 0.0), Vec2(3.0, 4.0))
        assert s.length == 5.0
    
    def test_closest_point(self):
        s = Segment(Vec2(0.0, 0.0), Vec2(10.0, 0.0))
        
        # Point on segment
        assert s.closest_point(Vec2(5.0, 5.0)) == Vec2(5.0, 0.0)
        
        # Point before start
        assert s.closest_point(Vec2(-5.0, 0.0)) == Vec2(0.0, 0.0)
        
        # Point after end
        assert s.closest_point(Vec2(15.0, 0.0)) == Vec2(10.0, 0.0)
    
    def test_distance_to_point(self):
        s = Segment(Vec2(0.0, 0.0), Vec2(10.0, 0.0))
        assert s.distance_to_point(Vec2(5.0, 3.0)) == 3.0
    
    def test_intersects_circle(self):
        s = Segment(Vec2(0.0, 0.0), Vec2(10.0, 0.0))
        c = Circle(Vec2(5.0, 2.0), 3.0)
        assert s.intersects_circle(c)
        
        c2 = Circle(Vec2(5.0, 5.0), 2.0)
        assert not s.intersects_circle(c2)


class TestCollisionDetection:
    """Tests for collision detection functions."""
    
    def test_circle_circle_no_collision(self):
        c1 = Circle(Vec2(0.0, 0.0), 2.0)
        c2 = Circle(Vec2(10.0, 0.0), 2.0)
        result = circle_circle_collision(c1, c2)
        assert result is None
    
    def test_circle_circle_collision(self):
        c1 = Circle(Vec2(0.0, 0.0), 2.0)
        c2 = Circle(Vec2(3.0, 0.0), 2.0)
        result = circle_circle_collision(c1, c2)
        assert result is not None
        normal, penetration = result
        assert abs(normal.x - 1.0) < 1e-10
        assert abs(normal.y) < 1e-10
        assert abs(penetration - 1.0) < 1e-10  # 2+2-3 = 1
    
    def test_circle_segment_no_collision(self):
        c = Circle(Vec2(0.0, 5.0), 2.0)
        s = Segment(Vec2(0.0, 0.0), Vec2(10.0, 0.0))
        result = circle_segment_collision(c, s)
        assert result is None
    
    def test_circle_segment_collision(self):
        c = Circle(Vec2(5.0, 1.0), 2.0)
        s = Segment(Vec2(0.0, 0.0), Vec2(10.0, 0.0))
        result = circle_segment_collision(c, s)
        assert result is not None
        normal, penetration = result
        assert abs(normal.y - 1.0) < 1e-10
        assert abs(penetration - 1.0) < 1e-10  # 2 - 1
    
    def test_ray_circle_intersection(self):
        origin = Vec2(0.0, 0.0)
        direction = Vec2(1.0, 0.0)
        circle = Circle(Vec2(5.0, 0.0), 1.0)
        
        t = ray_circle_intersection(origin, direction, circle)
        assert t is not None
        assert abs(t - 4.0) < 1e-10  # Hit at x=4 (circle edge)
    
    def test_ray_circle_miss(self):
        origin = Vec2(0.0, 0.0)
        direction = Vec2(1.0, 0.0)
        circle = Circle(Vec2(5.0, 10.0), 1.0)  # Above ray
        
        t = ray_circle_intersection(origin, direction, circle)
        assert t is None
    
    def test_ray_segment_intersection(self):
        origin = Vec2(0.0, 0.0)
        direction = Vec2(1.0, 1.0).normalized()
        segment = Segment(Vec2(5.0, 0.0), Vec2(5.0, 10.0))  # Vertical line at x=5
        
        t = ray_segment_intersection(origin, direction, segment)
        assert t is not None
    
    def test_ray_segment_miss(self):
        origin = Vec2(0.0, 0.0)
        direction = Vec2(-1.0, 0.0)  # Going left
        segment = Segment(Vec2(5.0, 0.0), Vec2(5.0, 10.0))  # To the right
        
        t = ray_segment_intersection(origin, direction, segment)
        assert t is None


class TestClamp:
    """Tests for clamp function."""
    
    def test_clamp_in_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0
    
    def test_clamp_below(self):
        assert clamp(-5.0, 0.0, 10.0) == 0.0
    
    def test_clamp_above(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0
