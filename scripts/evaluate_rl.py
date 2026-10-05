#!/usr/bin/env python3
"""Evaluate a checkpoint against a heuristic (or random) opponent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _rl_path import setup_paths

ROOT = setup_paths()

from sim.config import SimConfig

from rl.checkpoints import load_checkpoint
from rl.config import RLConfig
from rl.evaluation import evaluate_vs_opponent
from rl.model import ActorCritic
from rl.opponents import HeuristicRLOpponent, RandomOpponent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = RLConfig.from_dict(blob.get("config", {}))
    model = ActorCritic(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    sim = SimConfig.default()
    if args.opponent == "heuristic":
        opp = HeuristicRLOpponent(sim, noise=0.05)
    else:
        opp = RandomOpponent(seed=0)

    stats = evaluate_vs_opponent(
        model,
        opp,
        rl_config=cfg,
        sim_config=sim,
        games=args.games,
    )
    print(json.dumps(stats, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
