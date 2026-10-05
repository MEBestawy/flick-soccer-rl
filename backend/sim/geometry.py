"""
2D geometry primitives and utilities.

Provides vector math, collision detection helpers, and geometric primitives.
All operations are designed for deterministic, numerically stable physics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True, slots=True)
class Vec2:
    """Immutable 2D vector with common operations."""
    
    x: float
    y: float
    
    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)
    
    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)
    
    def __abs__(self) -> float:
        return self.length()
    
    def dot(self, other: Vec2) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: Vec2) -> float:
        """2D cross product (scalar z-component)."""
        return self.x * other.y - self.y * other.x
    
    def length(self) -> float:
        """Magnitude of vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self) -> float:
        """Squared magnitude (avoids sqrt for comparisons)."""
        return self.x * self.x + self.y * self.y
    
    def normalized(self) -> Vec2:
        """Unit vector in same direction. Returns zero vector if length is zero."""
        length = self.length()
        if length < 1e-10:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)
    
    def rotate(self, angle: float) -> Vec2:
        """Rotate vector by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def reflect(self, normal: Vec2) -> Vec2:
        """Reflect vector across a normal."""
        n = normal.normalized()
        return self - 2 * self.dot(n) * n
    
    def project_onto(self, other: Vec2) -> Vec2:
        """Project this vector onto another."""
        other_len_sq = other.length_squared()
        if other_len_sq < 1e-10:
            return Vec2(0.0, 0.0)
        return other * (self.dot(other) / other_len_sq)
    
    def perpendicular(self) -> Vec2:
        """Return perpendicular vector (90 degrees counterclockwise)."""
        return Vec2(-self.y, self.x)
    
    def lerp(self, other: Vec2, t: float) -> Vec2:
        """Linear interpolation to other vector."""
        return Vec2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )
    
    def distance_to(self, other: Vec2) -> float:
        """Distance to another point."""
        return (self - other).length()
    
    def angle(self) -> float:
        """Angle of vector from positive x-axis (radians)."""
        return math.atan2(self.y, self.x)
    
    @staticmethod
    def from_angle(angle: float, length: float = 1.0) -> Vec2:
        """Create vector from angle and length."""
        return Vec2(math.cos(angle) * length, math.sin(angle) * length)
    
    @staticmethod
    def zero() -> Vec2:
        """Zero vector."""
        return Vec2(0.0, 0.0)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float]) -> Vec2:
        """Create from tuple."""
        return Vec2(t[0], t[1])


@dataclass(frozen=True, slots=True)
class Circle:
    """Circle defined by center and radius."""
    
    center: Vec2
    radius: float
    
    def contains(self, point: Vec2) -> bool:
        """Check if point is inside circle."""
        return (point - self.center).length_squared() <= self.radius * self.radius
    
    def intersects_circle(self, other: Circle) -> bool:
        """Check if circles overlap."""
        dist_sq = (self.center - other.center).length_squared()
        combined_radius = self.radius + other.radius
        return dist_sq <= combined_radius * combined_radius
    
    def distance_to_circle(self, other: Circle) -> float:
        """Distance between circle surfaces (negative if overlapping)."""
        return (self.center - other.center).length() - self.radius - other.radius


@dataclass(frozen=True, slots=True)
class Segment:
    """Line segment defined by two endpoints."""
    
    start: Vec2
    end: Vec2
    
    @property
    def direction(self) -> Vec2:
        """Direction vector (not normalized)."""
        return self.end - self.start
    
    @property
    def length(self) -> float:
        """Length of segment."""
        return self.direction.length()
    
    @property
    def normal(self) -> Vec2:
        """Normal vector (perpendicular, pointing left)."""
        d = self.direction.normalized()
        return Vec2(-d.y, d.x)
    
    def closest_point(self, point: Vec2) -> Vec2:
        """Find closest point on segment to given point."""
        v = self.direction
        len_sq = v.length_squared()
        if len_sq < 1e-10:
            return self.start
        
        t = max(0.0, min(1.0, (point - self.start).dot(v) / len_sq))
        return self.start + v * t
    
    def distance_to_point(self, point: Vec2) -> float:
        """Distance from point to segment."""
        return (point - self.closest_point(point)).length()
    
    def intersects_circle(self, circle: Circle) -> bool:
        """Check if segment intersects circle."""
        return self.distance_to_point(circle.center) <= circle.radius


def circle_circle_collision(
    c1: Circle,
    c2: Circle
) -> Optional[Tuple[Vec2, float]]:
    """
    Detect collision between two circles.
    
    Returns:
        Tuple of (collision_normal, penetration_depth) if colliding,
        None otherwise. Normal points from c1 to c2.
    """
    delta = c2.center - c1.center
    dist = delta.length()
    combined_radius = c1.radius + c2.radius
    
    if dist >= combined_radius:
        return None
    
    if dist < 1e-10:
        # Circles are at same position, use arbitrary normal
        return (Vec2(1.0, 0.0), combined_radius)
    
    normal = delta / dist
    penetration = combined_radius - dist
    return (normal, penetration)


def circle_segment_collision(
    circle: Circle,
    segment: Segment
) -> Optional[Tuple[Vec2, float]]:
    """
    Detect collision between circle and line segment.
    
    Returns:
        Tuple of (collision_normal, penetration_depth) if colliding,
        None otherwise. Normal points from segment toward circle.
    """
    closest = segment.closest_point(circle.center)
    delta = circle.center - closest
    dist = delta.length()
    
    if dist >= circle.radius:
        return None
    
    if dist < 1e-10:
        # Circle center on segment, use segment normal
        return (segment.normal, circle.radius)
    
    normal = delta / dist
    penetration = circle.radius - dist
    return (normal, penetration)


def ray_circle_intersection(
    origin: Vec2,
    direction: Vec2,
    circle: Circle
) -> Optional[float]:
    """
    Find first intersection of ray with circle.
    
    Args:
        origin: Ray start point
        direction: Ray direction (should be normalized)
        circle: Circle to test against
    
    Returns:
        Distance along ray to intersection, or None if no intersection
    """
    oc = origin - circle.center
    a = direction.dot(direction)
    b = 2.0 * oc.dot(direction)
    c = oc.dot(oc) - circle.radius * circle.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return None
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    
    # Return smallest positive t
    if t1 >= 0:
        return t1
    if t2 >= 0:
        return t2
    return None


def ray_segment_intersection(
    origin: Vec2,
    direction: Vec2,
    segment: Segment
) -> Optional[float]:
    """
    Find intersection of ray with line segment.
    
    Returns:
        Distance along ray to intersection, or None if no intersection
    """
    v1 = origin - segment.start
    v2 = segment.end - segment.start
    v3 = Vec2(-direction.y, direction.x)
    
    dot = v2.dot(v3)
    if abs(dot) < 1e-10:
        return None  # Parallel
    
    t1 = v2.cross(v1) / dot
    t2 = v1.dot(v3) / dot
    
    if t1 >= 0 and 0 <= t2 <= 1:
        return t1
    return None


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))
