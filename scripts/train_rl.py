#!/usr/bin/env python3
"""Train a PPO self-play policy for flick soccer."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from _rl_path import setup_paths

ROOT = setup_paths()

from sim.config import SimConfig

from rl.config import RLConfig
from rl.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train flick-soccer PPO agent")
    p.add_argument("--run-name", default="experiment_01")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Subprocess env workers (default: 0 sequential; with --aggressive: =num-envs)",
    )
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--fast-sim", action="store_true", default=True)
    p.add_argument("--full-sim", action="store_true", help="Use default (slower) physics")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument(
        "--aggressive",
        action="store_true",
        help="Max-throughput preset: parallel workers, fewer evals, all CPU cores",
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cores = os.cpu_count() or 8
    num_envs = args.num_envs
    num_workers = 0 if args.num_workers is None else args.num_workers
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every
    eval_every = args.eval_every
    eval_games = args.eval_games

    if args.aggressive:
        # Saturate CPU with one worker per env; prefer core count.
        num_envs = max(num_envs, cores)
        if args.num_workers is None:
            num_workers = num_envs
        rollout_steps = max(rollout_steps, 512)
        if checkpoint_every is None:
            checkpoint_every = 100_000
        if eval_every is None:
            eval_every = 50_000
        if eval_games is None:
            eval_games = 8
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = RLConfig(
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        num_envs=num_envs,
        num_workers=num_workers,
        total_steps=args.total_steps,
        use_curriculum=not args.no_curriculum,
    )
    if rollout_steps is not None:
        cfg.rollout_steps = rollout_steps
    if checkpoint_every is not None:
        cfg.checkpoint_every = checkpoint_every
    if eval_every is not None:
        cfg.eval_every = eval_every
    if eval_games is not None:
        cfg.eval_games = eval_games
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    sim = SimConfig.default() if args.full_sim else SimConfig.fast()
    trainer = Trainer(cfg, sim_config=sim, runs_root=args.runs_root)
    if args.resume is not None:
        trainer.resume(args.resume)
    try:
        trainer.train(total_steps=args.total_steps)
    finally:
        if trainer.parallel_envs is not None:
            trainer.parallel_envs.close()


if __name__ == "__main__":
    main()
