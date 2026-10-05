"""RL action <-> FlickAction conversion (with perspective un-mirror)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sim.actions import FlickAction
from sim.geometry import Vec2
from sim.models import GameState, Team

from .observations import friendly_player_ids


@dataclass
class RLAction:
    player_index: int  # 0..4 in observation order
    direction_raw: np.ndarray  # shape (2,) pre-normalization sample
    power: float  # [0, 1]

    @property
    def direction_unit(self) -> np.ndarray:
        v = np.asarray(self.direction_raw, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float64)
        return v / n


def rl_action_to_flick(
    rl_action: RLAction,
    state: GameState,
    perspective_team: Team,
) -> FlickAction:
    """Map RL action into world-space FlickAction for the acting team."""
    ids = friendly_player_ids(state, perspective_team)
    idx = int(np.clip(rl_action.player_index, 0, len(ids) - 1))
    player_id = ids[idx]

    d = rl_action.direction_unit
    dx, dy = float(d[0]), float(d[1])
    # Observation is mirrored for Team B; un-mirror direction for world coords
    if perspective_team == Team.B:
        dx = -dx

    power = float(np.clip(rl_action.power, 0.0, 1.0))
    return FlickAction(
        player_id=player_id,
        direction=Vec2(dx, dy),
        power=power,
    )


def pack_action_arrays(
    player_idx: int,
    direction_raw: np.ndarray,
    power: float,
) -> np.ndarray:
    """Pack into length-4 array: [player, dx, dy, power]."""
    return np.asarray(
        [player_idx, direction_raw[0], direction_raw[1], power],
        dtype=np.float32,
    )


def unpack_action_array(arr: np.ndarray) -> RLAction:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return RLAction(
        player_index=int(round(a[0])),
        direction_raw=a[1:3].copy(),
        power=float(a[3]),
    )
