"""Historical opponent checkpoint pool with lazy loading."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sim.config import SimConfig

from .config import RLConfig
from .model import ActorCritic
from .opponents import HeuristicRLOpponent, Opponent, PolicyOpponent, RandomOpponent


@dataclass
class PoolEntry:
    path: str
    step: int
    elo: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at_step: int = 0


class OpponentPool:
    def __init__(
        self,
        root: Path,
        rl_config: RLConfig,
        sim_config: SimConfig,
        device: torch.device,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rl_config = rl_config
        self.sim_config = sim_config
        self.device = device
        self.entries: List[PoolEntry] = []
        self._cache: OrderedDict[str, ActorCritic] = OrderedDict()
        self._cache_limit = 4
        self.meta_path = self.root / "pool.json"
        self._load_meta()

    def _load_meta(self) -> None:
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text())
            self.entries = [PoolEntry(**e) for e in data.get("entries", [])]

    def save_meta(self) -> None:
        payload = {"entries": [asdict(e) for e in self.entries]}
        self.meta_path.write_text(json.dumps(payload, indent=2))

    def add_snapshot(self, model: ActorCritic, step: int) -> PoolEntry:
        path = self.root / f"policy_{step:09d}.pt"
        torch.save({"model": model.state_dict(), "step": step}, path)
        entry = PoolEntry(
            path=str(path),
            step=step,
            elo=self.rl_config.elo_initial,
            created_at_step=step,
        )
        self.entries.append(entry)
        self._trim()
        self.save_meta()
        return entry

    def _trim(self) -> None:
        max_n = self.rl_config.max_pool_size
        if len(self.entries) <= max_n:
            return
        # Keep oldest, newest, and highest Elo
        by_elo = sorted(self.entries, key=lambda e: e.elo, reverse=True)
        keep = {id(self.entries[0]), id(self.entries[-1])}
        for e in by_elo[: max(3, max_n // 3)]:
            keep.add(id(e))
        # Fill with recent
        for e in reversed(self.entries):
            if len(keep) >= max_n:
                break
            keep.add(id(e))
        self.entries = [e for e in self.entries if id(e) in keep]

    def _load_model(self, path: str) -> ActorCritic:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        model = ActorCritic(self.rl_config).to(self.device)
        blob = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(blob["model"])
        model.eval()
        self._cache[path] = model
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return model

    def sample_opponent(self, step: int, current: ActorCritic) -> Opponent:
        h, hist, curr = self.rl_config.opponent_mix(step)
        r = random.random()
        if r < h:
            return HeuristicRLOpponent(self.sim_config, seed=step)
        if r < h + hist and self.entries:
            entry = random.choice(self.entries)
            model = self._load_model(entry.path)
            return PolicyOpponent(
                model,
                self.rl_config,
                self.sim_config,
                deterministic=False,
                label=f"hist_{entry.step}",
                checkpoint_path=entry.path,
            )
        return PolicyOpponent(
            current,
            self.rl_config,
            self.sim_config,
            deterministic=False,
            label="current",
        )

    def update_elo(self, entry_step: Optional[int], won: bool, drew: bool) -> None:
        # Simplified: update historical entry if matched
        if entry_step is None:
            return
        for e in self.entries:
            if e.step == entry_step:
                if drew:
                    e.draws += 1
                elif won:
                    e.wins += 1
                else:
                    e.losses += 1
                break
        self.save_meta()
