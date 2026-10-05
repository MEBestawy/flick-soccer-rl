"""RL environment: one step = one complete settled game turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sim import HeadlessEnv, Team
from sim.config import SimConfig
from sim.geometry import Vec2
from sim.models import GamePhase, GameState

from .actions import RLAction, rl_action_to_flick
from .config import RLConfig
from .observations import observation_from_state
from .rewards import RewardBreakdown, compute_transition_reward


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
        # Sync first-to-N from RL curriculum defaults.
        self.sim_config.goals_to_win = self.rl_config.goals_to_win
        self._shaping_override = shaping_scale
        self._search_hint_prob = 0.0
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

    def apply_match_rules(self, goals_to_win: int, max_turns: int) -> None:
        """Update first-to-N / turn cap (curriculum)."""
        self.rl_config.max_turns_per_game = max_turns
        self.sim_config.goals_to_win = goals_to_win
        self.inner.config.goals_to_win = goals_to_win

    def _bind_live_state(self) -> GameState:
        """Share the simulator's live state (HeadlessEnv/new_game return clones)."""
        live = self.inner.simulator._state
        assert live is not None
        self.inner._state = live
        self._state = live
        return live

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            rng = np.random.default_rng(seed)
            start = Team.A if rng.random() < 0.5 else Team.B
        else:
            rng = np.random.default_rng()
            start = Team.A
        self.inner.reset(start)
        self._bind_live_state()
        self.inner._done = False
        self._turns = 0
        self._episode_reward = 0.0

        easy = False
        if rng.random() < self.rl_config.easy_scenario_prob:
            easy = self._place_easy_ball(rng)

        assert self._state is not None
        obs = observation_from_state(
            self._state,
            self._state.current_team,
            self.sim_config,
            self.rl_config,
            search_hint_prob=getattr(self, "_search_hint_prob", 0.0),
            rng=rng,
        )
        return obs, {
            "team": self._state.current_team.value,
            "seed": seed,
            "easy_scenario": easy,
        }

    def _place_easy_ball(self, rng: np.random.Generator) -> bool:
        """Put the ball just ahead of a random player on the kicking team."""
        assert self._state is not None
        team = self._state.current_team
        own = [p for p in self._state.players if p.team == team]
        if not own:
            return False
        player = own[int(rng.integers(0, len(own)))]
        # Attack direction in world coords
        attack = 1.0 if team == Team.A else -1.0
        gap = player.radius + self._state.ball.radius + 0.8
        bx = player.position.x + attack * gap
        by = player.position.y + float(rng.uniform(-0.5, 0.5))
        w = self.sim_config.physics.pitch_width
        h = self.sim_config.physics.pitch_height
        bx = float(np.clip(bx, 2.0, w - 2.0))
        by = float(np.clip(by, 2.0, h - 2.0))
        self._state.ball.position = Vec2(bx, by)
        self._state.ball.velocity = Vec2(0.0, 0.0)
        self._state.ball.is_sleeping = True
        self._state.ball.sleep_timer = 1.0
        return True

    def step(
        self,
        action: RLAction,
        *,
        reward_perspective: Optional[Team] = None,
        include_turn_penalty: bool = True,
    ) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() first")

        prev = self._state.clone()
        acting = prev.current_team
        perspective = reward_perspective if reward_perspective is not None else acting
        flick = rl_action_to_flick(action, prev, acting)

        _nxt_clone, _legacy_r, done, info = self.inner.step(flick)
        nxt = self._bind_live_state()
        self._turns += 1

        truncated = self._turns >= self.rl_config.max_turns_per_game
        terminated = bool(done) or nxt.phase == GamePhase.GAME_OVER

        scale = self._shaping_override
        breakdown = compute_transition_reward(
            prev,
            nxt,
            perspective,
            self.sim_config,
            self.rl_config,
            shaping_scale=scale,
            include_turn_penalty=include_turn_penalty,
            truncated=truncated and not terminated,
            flicked_player_id=flick.player_id if include_turn_penalty else None,
        )
        reward = breakdown.total
        self._episode_reward += reward

        search_p = getattr(self, "_search_hint_prob", 0.0)
        if terminated or truncated:
            obs = observation_from_state(
                nxt,
                acting,
                self.sim_config,
                self.rl_config,
                search_hint_prob=0.0,
            )
        else:
            obs = observation_from_state(
                nxt,
                nxt.current_team,
                self.sim_config,
                self.rl_config,
                search_hint_prob=search_p,
            )

        out_info = {
            **info,
            "acting_team": acting.value,
            "perspective": perspective.value,
            "next_team": nxt.current_team.value,
            "reward_breakdown": breakdown.__dict__,
            "score_a": nxt.score_a,
            "score_b": nxt.score_b,
            "turns": self._turns,
            "episode_reward": self._episode_reward,
            "flick_player": flick.player_id,
            "flick_power": flick.power,
            "goals_scored": int(
                (nxt.score_a - prev.score_a) if perspective == Team.A else (nxt.score_b - prev.score_b)
            ),
            "goals_conceded": int(
                (nxt.score_b - prev.score_b) if perspective == Team.A else (nxt.score_a - prev.score_a)
            ),
        }
        return StepResult(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=out_info,
        )

    def credit_opponent_step(
        self,
        prev: GameState,
        nxt: GameState,
        learner: Team,
        *,
        truncated: bool,
    ) -> RewardBreakdown:
        """Learner-perspective reward for a turn the opponent just took."""
        scale = self._shaping_override
        return compute_transition_reward(
            prev,
            nxt,
            learner,
            self.sim_config,
            self.rl_config,
            shaping_scale=scale,
            include_turn_penalty=False,
            truncated=truncated,
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
