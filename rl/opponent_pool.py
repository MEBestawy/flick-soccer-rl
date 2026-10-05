"""Historical opponent checkpoint pool with Elo-weighted (PfSP-lite) sampling."""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from sim.config import SimConfig

from .config import RLConfig
from .model import ActorCritic
from .opponents import HeuristicRLOpponent, Opponent, PolicyOpponent


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
        by_elo = sorted(self.entries, key=lambda e: e.elo, reverse=True)
        keep = {id(self.entries[0]), id(self.entries[-1])}
        for e in by_elo[: max(3, max_n // 3)]:
            keep.add(id(e))
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

    def _sample_historical(self) -> PoolEntry:
        """Prefer stronger historical opponents (PfSP-lite softmax over Elo)."""
        assert self.entries
        temp = max(1e-3, float(self.rl_config.hist_elo_softmax_temp))
        elos = [e.elo for e in self.entries]
        max_e = max(elos)
        weights = [math.exp((e - max_e) / temp) for e in elos]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for entry, w in zip(self.entries, weights):
            acc += w
            if r <= acc:
                return entry
        return self.entries[-1]

    def sample_opponent(self, step: int, current: ActorCritic) -> Opponent:
        h, hist, curr = self.rl_config.opponent_mix(step)
        # Renormalize if pool empty (fold hist into current)
        if not self.entries:
            hist = 0.0
            s = h + curr
            h, curr = (h / s, curr / s) if s > 0 else (0.5, 0.5)
        r = random.random()
        if r < h:
            return HeuristicRLOpponent(self.sim_config, seed=step)
        if r < h + hist and self.entries:
            entry = self._sample_historical()
            model = self._load_model(entry.path)
            # Match current teacher_mix so hist opponents behave consistently
            if hasattr(current, "teacher_mix"):
                model.set_teacher_mix(current.teacher_mix)
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
        if entry_step is None:
            return
        for e in self.entries:
            if e.step == entry_step:
                if drew:
                    e.draws += 1
                    score = 0.5
                elif won:
                    e.wins += 1
                    score = 1.0
                else:
                    e.losses += 1
                    score = 0.0
                # Opponent Elo rises when they beat the learner
                k = self.rl_config.elo_k
                expected = 1.0 / (
                    1.0
                    + 10
                    ** ((self.rl_config.elo_initial - e.elo) / 400.0)
                )
                e.elo += k * (score - expected)
                break
        self.save_meta()
