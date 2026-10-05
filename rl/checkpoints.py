"""Atomic, resumable training checkpoints."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import RLConfig
from .model import ActorCritic


def _atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: RLConfig,
    global_step: int,
    update: int,
    best_elo: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "global_step": global_step,
        "update": update,
        "best_elo": best_elo,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "extra": extra or {},
    }
    if torch.backends.mps.is_available():
        try:
            payload["rng"]["mps"] = torch.mps.get_rng_state()  # type: ignore[attr-defined]
        except Exception:
            pass
    _atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    blob = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    rng = blob.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    return blob


def save_config_json(path: Path, config: RLConfig, extra: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rl": config.to_dict(), **(extra or {})}
    path.write_text(json.dumps(data, indent=2, default=str))
