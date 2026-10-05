#!/usr/bin/env python3
"""Peak SPS benchmark for the flick soccer simulator."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sim import FlickAction, GameSimulator, HeadlessEnv, SimConfig, Vec2
from sim.rust_bridge import RUST_AVAILABLE, _rust_enabled


def turn_sps(config: SimConfig, capture: bool, n: int = 300, warmup: int = 20) -> float:
    sim = GameSimulator(config)
    action = FlickAction(player_id="A1", direction=Vec2(0.9, 0.3), power=0.75)
    for _ in range(warmup):
        sim.new_game()
        sim.execute_action(action, capture_frames=capture)
    t0 = time.perf_counter()
    for _ in range(n):
        sim.new_game()
        sim.execute_action(action, capture_frames=capture)
    return n / (time.perf_counter() - t0)


def env_sps(config: SimConfig, n: int = 400, warmup: int = 20) -> float:
    env = HeadlessEnv(config)
    env.reset()
    for _ in range(warmup):
        a = env.random_action()
        if a is None:
            env.reset()
            continue
        env.step(a)
        if env.is_done():
            env.reset()
    t0 = time.perf_counter()
    steps = 0
    while steps < n:
        a = env.random_action()
        if a is None:
            env.reset()
            continue
        env.step(a)
        steps += 1
        if env.is_done():
            env.reset()
    return steps / (time.perf_counter() - t0)


def _worker_batch(n: int) -> float:
    # Workers inherit FLICK_PHYSICS_RUST from parent env
    return env_sps(SimConfig.default(), n=n, warmup=10)


def main() -> None:
    print("=== Simulator peak SPS ===")
    print(f"Rust extension installed: {RUST_AVAILABLE}")
    print(f"Rust enabled (FLICK_PHYSICS_RUST): {_rust_enabled()}\n")

    for label, cfg in [
        ("default/UI physics", SimConfig.default()),
        ("fast physics (old train)", SimConfig.fast()),
    ]:
        print(f"{label}:")
        print(f"  single-turn no frames:   {turn_sps(cfg, False, n=250):7.1f} turns/s")
        print(f"  single-turn with frames: {turn_sps(cfg, True, n=150):7.1f} turns/s")
        print(f"  HeadlessEnv random:      {env_sps(cfg, n=300):7.1f} steps/s")
        print()

    if RUST_AVAILABLE:
        print("Python-only fallback (FLICK_PHYSICS_RUST=0):")
        os.environ["FLICK_PHYSICS_RUST"] = "0"
        print(f"  HeadlessEnv random:      {env_sps(SimConfig.default(), n=200):7.1f} steps/s")
        os.environ["FLICK_PHYSICS_RUST"] = "1"
        print()

    workers = 8
    per = 200
    print(f"Multiprocess HeadlessEnv (UI physics, {workers} workers × {per} steps):")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker_batch, [per] * workers))
    wall = time.perf_counter() - t0
    total = workers * per
    print(f"  aggregate: {total / wall:7.1f} steps/s")
    print(f"  per-worker mean: {sum(results) / len(results):.1f} steps/s")
    print(f"  wall: {wall:.2f}s")


if __name__ == "__main__":
    main()
