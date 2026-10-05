#!/usr/bin/env python3
"""
Play a checkpoint vs heuristic and export a replay JSON.

Optionally print a path the frontend/replay tools can load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _rl_path import setup_paths

ROOT = setup_paths()

from sim.config import SimConfig
from sim.models import Team

from rl.checkpoints import load_checkpoint
from rl.config import RLConfig
from rl.model import ActorCritic
from rl.opponents import HeuristicRLOpponent
from rl.replay_export import export_match_replay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--team", choices=["A", "B"], default="A")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cpu")
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
    opp = HeuristicRLOpponent(sim, noise=0.0)
    out = args.out or (
        ROOT / "runs" / "watch" / f"{args.checkpoint.stem}_seed_{args.seed}.json"
    )
    payload = export_match_replay(
        out,
        model,
        opp,
        rl_config=cfg,
        sim_config=sim,
        seed=args.seed,
        policy_team=Team.A if args.team == "A" else Team.B,
        meta={"checkpoint": str(args.checkpoint)},
    )
    print(json.dumps({k: payload[k] for k in ("seed", "score_a", "score_b", "winner", "turns")}, indent=2))
    print(f"Replay written: {out}")
    print("Load this JSON with scripts/replay_game.py or the frontend replay viewer.")


if __name__ == "__main__":
    main()
