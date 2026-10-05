#!/usr/bin/env python3
"""Head-to-head comparison of two checkpoints."""

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
from rl.evaluation import play_match, update_elo
from rl.model import ActorCritic
from rl.opponents import PolicyOpponent


def load_policy(path: Path, device: torch.device) -> tuple[ActorCritic, RLConfig]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = RLConfig.from_dict(blob.get("config", {}))
    model = ActorCritic(cfg).to(device)
    load_checkpoint(path, model, device=device)
    model.eval()
    return model, cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model_a, cfg = load_policy(args.a, device)
    model_b, _ = load_policy(args.b, device)
    sim = SimConfig.default()
    opp_b = PolicyOpponent(model_b, cfg, sim, deterministic=True, label="B")

    a_wins = b_wins = draws = 0
    goals_a = goals_b = 0
    for i in range(args.games):
        team = Team.A if i % 2 == 0 else Team.B
        m = play_match(
            model_a,
            opp_b,
            rl_config=cfg,
            sim_config=sim,
            seed=20_000 + i,
            policy_team=team,
            deterministic=True,
        )
        if team == Team.A:
            sa, sb = m.score_a, m.score_b
        else:
            sa, sb = m.score_b, m.score_a
        goals_a += sa
        goals_b += sb
        if sa > sb:
            a_wins += 1
        elif sb > sa:
            b_wins += 1
        else:
            draws += 1

    n = max(1, args.games)
    score = a_wins / n + 0.5 * draws / n
    elo_a, elo_b = update_elo(1000.0, 1000.0, score)
    result = {
        "a": str(args.a),
        "b": str(args.b),
        "games": args.games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "a_win_rate": a_wins / n,
        "elo_delta_est": elo_a - elo_b,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
