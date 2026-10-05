#!/usr/bin/env python3
"""Benchmark RL env throughput and CPU vs MPS PPO update."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from _rl_path import setup_paths

setup_paths()

from rl.actions import RLAction
from rl.config import RLConfig
from rl.device_utils import resolve_device
from rl.env import FlickRLEnv
from rl.model import ActorCritic
from rl.opponents import HeuristicRLOpponent, RandomOpponent
from sim.config import SimConfig


def bench_env(num_envs: int, steps: int) -> float:
    cfg = RLConfig(num_envs=num_envs)
    sim = SimConfig.default()
    envs = [FlickRLEnv(sim, cfg) for _ in range(num_envs)]
    opp = RandomOpponent(seed=0)
    for e in envs:
        e.reset()
    t0 = time.time()
    n = 0
    while n < steps:
        for e in envs:
            assert e.state is not None
            a = opp.act(e.state, e.state.current_team)
            r = e.step(a)
            n += 1
            if r.terminated or r.truncated:
                e.reset()
    return n / (time.time() - t0)


def bench_ppo(device_name: str, batch: int = 2048) -> float:
    device = resolve_device(device_name)
    cfg = RLConfig()
    model = ActorCritic(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = torch.randn(batch, cfg.obs_dim, device=device)
    t0 = time.time()
    for _ in range(20):
        sample = model.act(obs)
        loss = -(sample.log_prob.mean() + sample.value.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return 20 / (time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-steps", type=int, default=2000)
    args = p.parse_args()

    print("=== Env throughput (random actions) ===")
    for n in (1, 4, 8, 16):
        sps = bench_env(n, args.env_steps)
        print(f"  num_envs={n:2d}  {sps:.0f} steps/sec")

    print("\n=== PPO forward/backward updates/sec ===")
    for d in ("cpu", "mps"):
        try:
            ups = bench_ppo(d)
            print(f"  device={d:3s}  {ups:.1f} updates/sec")
        except Exception as exc:
            print(f"  device={d:3s}  skipped ({exc})")


if __name__ == "__main__":
    main()
