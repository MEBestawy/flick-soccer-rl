"""RL environment: one step = one complete settled game turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sim import HeadlessEnv, Team
from sim.config import SimConfig
from sim.models import GamePhase, GameState

from .actions import RLAction, rl_action_to_flick
from .config import RLConfig
from .observations import observation_from_state
from .rewards import compute_transition_reward


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class FlickRLEnv:
    """
    Gymnasium-style wrapper around HeadlessEnv.

    The policy always sees a canonical L→R observation for the current team.
    """

    def __init__(
        self,
        sim_config: Optional[SimConfig] = None,
        rl_config: Optional[RLConfig] = None,
        *,
        shaping_scale: Optional[float] = None,
    ) -> None:
        self.sim_config = sim_config or SimConfig.default()
        self.rl_config = rl_config or RLConfig()
        self._shaping_override = shaping_scale
        self.inner = HeadlessEnv(self.sim_config)
        self._state: Optional[GameState] = None
        self._turns = 0
        self._episode_reward = 0.0

    @property
    def state(self) -> Optional[GameState]:
        return self._state

    @property
    def acting_team(self) -> Team:
        assert self._state is not None
        return self._state.current_team

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            rng = np.random.default_rng(seed)
            start = Team.A if rng.random() < 0.5 else Team.B
        else:
            start = Team.A
        self._state = self.inner.reset(start)
        self._turns = 0
        self._episode_reward = 0.0
        obs = observation_from_state(
            self._state, self._state.current_team, self.sim_config, self.rl_config
        )
        return obs, {"team": self._state.current_team.value, "seed": seed}

    def step(self, action: RLAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() first")

        prev = self._state.clone()
        acting = prev.current_team
        flick = rl_action_to_flick(action, prev, acting)

        nxt, _legacy_r, done, info = self.inner.step(flick)
        self._state = nxt
        self._turns += 1

        scale = self._shaping_override
        breakdown = compute_transition_reward(
            prev,
            nxt,
            acting,
            self.sim_config,
            self.rl_config,
            shaping_scale=scale,
        )
        reward = breakdown.total
        self._episode_reward += reward

        truncated = self._turns >= self.rl_config.max_turns_per_game
        terminated = bool(done) or nxt.phase == GamePhase.GAME_OVER
        if truncated and not terminated:
            terminated = False

        if terminated or truncated:
            obs = observation_from_state(
                nxt, acting, self.sim_config, self.rl_config
            )
        else:
            obs = observation_from_state(
                nxt, nxt.current_team, self.sim_config, self.rl_config
            )

        out_info = {
            **info,
            "acting_team": acting.value,
            "next_team": nxt.current_team.value,
            "reward_breakdown": breakdown.__dict__,
            "score_a": nxt.score_a,
            "score_b": nxt.score_b,
            "turns": self._turns,
            "episode_reward": self._episode_reward,
            "flick_player": flick.player_id,
            "flick_power": flick.power,
        }
        return StepResult(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=out_info,
        )


class SelfPlayEnv:
    """
    Full-game env where both sides are controlled externally.

    Used when training: learner acts for `learner_team` (or both if self-play),
    opponent policy provides the other team's action.
    """

    def __init__(
        self,
        sim_config: Optional[SimConfig] = None,
        rl_config: Optional[RLConfig] = None,
    ) -> None:
        self.env = FlickRLEnv(sim_config, rl_config)
        self.rl_config = rl_config or RLConfig()
        self.sim_config = self.env.sim_config

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(seed)

    def step(self, action: RLAction) -> StepResult:
        return self.env.step(action)
