#!/usr/bin/env python3
"""
Benchmark script for soccer simulation.

Measures performance of the physics simulation under various conditions.
"""

import sys
import time
import statistics
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sim import GameSimulator, FlickAction, HeadlessEnv, SimConfig, Vec2, Team


def benchmark_single_turn(iterations: int = 100) -> dict:
    """Benchmark single turn simulation."""
    print(f"\n=== Benchmarking single turn ({iterations} iterations) ===")
    
    times = []
    frame_counts = []
    
    for i in range(iterations):
        sim = GameSimulator()
        sim.new_game()
        
        action = FlickAction(
            player_id="A1",
            direction=Vec2(1.0, 0.5),
            power=0.7
        )
        
        start = time.perf_counter()
        result = sim.execute_action(action, capture_frames=True)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed * 1000)  # Convert to ms
        frame_counts.append(len(result.frames or []))
    
    results = {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
        "avg_frames": statistics.mean(frame_counts),
    }
    
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Std:    {results['std_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Avg frames: {results['avg_frames']:.1f}")
    
    return results


def benchmark_full_game(games: int = 10) -> dict:
    """Benchmark complete games."""
    print(f"\n=== Benchmarking full games ({games} games) ===")
    
    game_times = []
    turn_counts = []
    
    for g in range(games):
        env = HeadlessEnv()
        env.reset()
        
        turns = 0
        start = time.perf_counter()
        
        while not env.is_done() and turns < 200:
            action = env.random_action()
            if action is None:
                break
            env.step(action)
            turns += 1
        
        elapsed = time.perf_counter() - start
        game_times.append(elapsed)
        turn_counts.append(turns)
    
    results = {
        "mean_game_s": statistics.mean(game_times),
        "mean_turns": statistics.mean(turn_counts),
        "mean_turn_ms": statistics.mean(game_times) / statistics.mean(turn_counts) * 1000,
        "total_turns": sum(turn_counts),
        "total_time_s": sum(game_times),
    }
    
    print(f"  Mean game time: {results['mean_game_s']:.2f} s")
    print(f"  Mean turns:     {results['mean_turns']:.1f}")
    print(f"  Mean turn time: {results['mean_turn_ms']:.2f} ms")
    print(f"  Turns/second:   {results['total_turns'] / results['total_time_s']:.1f}")
    
    return results


def benchmark_high_speed_collisions(iterations: int = 50) -> dict:
    """Benchmark high-speed collision scenarios."""
    print(f"\n=== Benchmarking high-speed collisions ({iterations} iterations) ===")
    
    times = []
    
    for _ in range(iterations):
        sim = GameSimulator()
        sim.new_game()
        
        # Max power shot toward dense area
        action = FlickAction(
            player_id="A1",
            direction=Vec2(1.0, 0.0),
            power=1.0
        )
        
        start = time.perf_counter()
        result = sim.execute_action(action, capture_frames=False)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed * 1000)
    
    results = {
        "mean_ms": statistics.mean(times),
        "max_ms": max(times),
    }
    
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    
    return results


def benchmark_no_frames(iterations: int = 100) -> dict:
    """Benchmark simulation without frame capture."""
    print(f"\n=== Benchmarking without frame capture ({iterations} iterations) ===")
    
    times_with_frames = []
    times_without_frames = []
    
    for _ in range(iterations):
        sim = GameSimulator()
        sim.new_game()
        
        action = FlickAction(
            player_id="A1",
            direction=Vec2(0.8, 0.6),
            power=0.5
        )
        
        # With frames
        start = time.perf_counter()
        sim.execute_action(action, capture_frames=True)
        times_with_frames.append((time.perf_counter() - start) * 1000)
        
        # Reset and without frames
        sim.new_game()
        start = time.perf_counter()
        sim.execute_action(action, capture_frames=False)
        times_without_frames.append((time.perf_counter() - start) * 1000)
    
    results = {
        "with_frames_ms": statistics.mean(times_with_frames),
        "without_frames_ms": statistics.mean(times_without_frames),
        "speedup": statistics.mean(times_with_frames) / statistics.mean(times_without_frames),
    }
    
    print(f"  With frames:    {results['with_frames_ms']:.2f} ms")
    print(f"  Without frames: {results['without_frames_ms']:.2f} ms")
    print(f"  Speedup:        {results['speedup']:.2f}x")
    
    return results


def benchmark_state_cloning(iterations: int = 1000) -> dict:
    """Benchmark state cloning performance."""
    print(f"\n=== Benchmarking state cloning ({iterations} iterations) ===")
    
    sim = GameSimulator()
    state = sim.new_game()
    
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        cloned = state.clone()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000000)  # Convert to microseconds
    
    results = {
        "mean_us": statistics.mean(times),
        "clones_per_sec": 1000000 / statistics.mean(times),
    }
    
    print(f"  Mean clone time: {results['mean_us']:.2f} µs")
    print(f"  Clones/second:   {results['clones_per_sec']:.0f}")
    
    return results


def main():
    print("=" * 60)
    print("Soccer Simulation Benchmark")
    print("=" * 60)
    
    all_results = {}
    
    all_results["single_turn"] = benchmark_single_turn()
    all_results["full_game"] = benchmark_full_game()
    all_results["high_speed"] = benchmark_high_speed_collisions()
    all_results["no_frames"] = benchmark_no_frames()
    all_results["cloning"] = benchmark_state_cloning()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Single turn:        {all_results['single_turn']['mean_ms']:.2f} ms avg")
    print(f"Turns/second:       {all_results['full_game']['total_turns'] / all_results['full_game']['total_time_s']:.1f}")
    print(f"Without frames:     {all_results['no_frames']['without_frames_ms']:.2f} ms avg")
    print(f"State clone:        {all_results['cloning']['mean_us']:.2f} µs avg")
    
    return all_results


if __name__ == "__main__":
    main()
