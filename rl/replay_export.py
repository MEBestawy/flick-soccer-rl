"""Export evaluation games for later visual playback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from sim.config import SimConfig
from sim.models import Team

from .config import RLConfig
from .evaluation import play_match
from .model import ActorCritic
from .opponents import Opponent


def export_match_replay(
    path: Path,
    policy: ActorCritic,
    opponent: Opponent,
    *,
    rl_config: RLConfig,
    sim_config: SimConfig,
    seed: int,
    policy_team: Team = Team.A,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    match = play_match(
        policy,
        opponent,
        rl_config=rl_config,
        sim_config=sim_config,
        seed=seed,
        policy_team=policy_team,
        deterministic=True,
        record=True,
    )
    payload = {
        "seed": seed,
        "policy_team": policy_team.value,
        "opponent": opponent.name,
        "score_a": match.score_a,
        "score_b": match.score_b,
        "winner": match.winner,
        "turns": match.turns,
        "actions": match.actions,
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload
