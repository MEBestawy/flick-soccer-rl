#!/usr/bin/env python3
"""Train a PPO self-play policy for flick soccer."""

from __future__ import annotations

import argparse
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
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--fast-sim", action="store_true", default=True)
    p.add_argument("--full-sim", action="store_true", help="Use default (slower) physics")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = RLConfig(
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        use_curriculum=not args.no_curriculum,
    )
    if args.rollout_steps is not None:
        cfg.rollout_steps = args.rollout_steps
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    sim = SimConfig.default() if args.full_sim else SimConfig.fast()
    trainer = Trainer(cfg, sim_config=sim, runs_root=args.runs_root)
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.train(total_steps=args.total_steps)


if __name__ == "__main__":
    main()
