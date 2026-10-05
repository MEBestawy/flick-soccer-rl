"""RL policy agent backed by a trained PPO checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..actions import FlickAction
from ..config import SimConfig
from ..models import GameState, Team
from .base import Agent

# Shared loaded policies: checkpoint path -> (model, rl_config)
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _repo_root() -> Path:
    # backend/sim/agents/rl_agent.py -> repo root
    return Path(__file__).resolve().parents[3]


def _ensure_rl_importable() -> Path:
    root = _repo_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def resolve_rl_checkpoint(explicit: Optional[str] = None) -> Path:
    """
    Resolve the primary / best RL checkpoint.

    Priority:
      1. explicit path argument
      2. RL_CHECKPOINT env var
      3. runs/current_best/checkpoints/best.pt
      4. newest runs/*/checkpoints/best.pt by mtime
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RL checkpoint not found: {path}")
        return path

    env = os.environ.get("RL_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_file():
            # Allow paths relative to repo root
            alt = _repo_root() / env
            path = alt.resolve() if alt.is_file() else path
        if not path.is_file():
            raise FileNotFoundError(f"RL_CHECKPOINT not found: {env}")
        return path

    root = _ensure_rl_importable()
    current_best = root / "runs" / "current_best" / "checkpoints" / "best.pt"
    if current_best.is_file():
        return current_best

    candidates = sorted(
        (root / "runs").glob("*/checkpoints/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer real training runs over the current_best symlink-style folder already checked
    candidates = [p for p in candidates if "current_best" not in p.parts]
    if not candidates:
        raise FileNotFoundError(
            "No RL checkpoint found. Train first or set RL_CHECKPOINT to a .pt file."
        )
    return candidates[0]


def resolve_rl_first_checkpoint(explicit: Optional[str] = None) -> Path:
    """
    Resolve the first logged training run checkpoint (rl-first).

    Priority:
      1. explicit path argument
      2. RL_FIRST_CHECKPOINT env var
      3. runs/parallel_smoke/checkpoints/best.pt (first completed logged run)
      4. oldest runs/*/checkpoints/best.pt by mtime
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RL-first checkpoint not found: {path}")
        return path

    env = os.environ.get("RL_FIRST_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RL_FIRST_CHECKPOINT not found: {path}")
        return path

    root = _ensure_rl_importable()
    smoke = root / "runs" / "parallel_smoke" / "checkpoints" / "best.pt"
    if smoke.is_file():
        return smoke

    candidates = sorted(
        (root / "runs").glob("*/checkpoints/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No RL-first checkpoint found. Set RL_FIRST_CHECKPOINT to a .pt file."
        )
    return candidates[0]


def _load_policy(checkpoint: Path) -> Tuple[Any, Any]:
    key = str(checkpoint)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    _ensure_rl_importable()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the RL agent. "
            "Install with: cd backend && uv sync --extra rl"
        ) from exc

    from rl.checkpoints import load_checkpoint
    from rl.config import RLConfig
    from rl.model import ActorCritic

    device = torch.device("cpu")
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = RLConfig.from_dict(blob.get("config", {}))
    model = ActorCritic(cfg).to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    # Play the learned policy, not the teacher prior used during early training.
    extra = blob.get("extra") or {}
    mix = extra.get("teacher_mix")
    if mix is None:
        mix = 0.0
    if hasattr(model, "set_teacher_mix"):
        model.set_teacher_mix(float(mix))
    _MODEL_CACHE[key] = (model, cfg)
    return model, cfg


class RLAgent(Agent):
    """
    Play using a trained PPO ActorCritic checkpoint.

    Uses deterministic actions by default for stable playtesting.
    """

    name = "rl"

    def __init__(
        self,
        team: Team,
        config: Optional[SimConfig] = None,
        *,
        checkpoint: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        deterministic: bool = True,
        name: str = "rl",
    ) -> None:
        super().__init__(team, config)
        self.name = name
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        elif name == "rl-first":
            self.checkpoint_path = resolve_rl_first_checkpoint(checkpoint)
        else:
            self.checkpoint_path = resolve_rl_checkpoint(checkpoint)
        self.deterministic = deterministic
        self._model, self._rl_config = _load_policy(self.checkpoint_path)

    def select_action(self, state: GameState) -> FlickAction:
        from rl.actions import rl_action_to_flick
        from rl.observations import observation_from_state

        obs = observation_from_state(
            state, self.team, self.config, self._rl_config
        )
        rl_action = self._model.act_numpy(obs, deterministic=self.deterministic)
        return rl_action_to_flick(rl_action, state, self.team)

    def __repr__(self) -> str:
        return (
            f"RLAgent(team={self.team.value}, name={self.name!r}, "
            f"checkpoint={self.checkpoint_path.name!r})"
        )
