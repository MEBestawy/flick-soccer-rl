"""Evaluation matches, Elo updates, tournaments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sim.config import SimConfig
from sim.models import Team
from sim.serialization import serialize_action, serialize_state

from .actions import RLAction
from .config import RLConfig
from .env import FlickRLEnv
from .model import ActorCritic
from .observations import observation_from_state
from .opponents import HeuristicRLOpponent, Opponent, PolicyOpponent


@dataclass
class MatchResult:
    score_a: int
    score_b: int
    turns: int
    winner: Optional[str]
    actions: List[Dict]
    seed: int
    goals_on_policy_turn: int = 0
    goals_on_opp_turn: int = 0


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
    """score_a: 1 win, 0.5 draw, 0 loss for player A."""
    ea = _elo_expected(ra, rb)
    eb = 1.0 - ea
    return ra + k * (score_a - ea), rb + k * ((1.0 - score_a) - eb)


def play_match(
    policy: ActorCritic,
    opponent: Opponent,
    *,
    rl_config: RLConfig,
    sim_config: SimConfig,
    seed: int,
    policy_team: Team = Team.A,
    deterministic: bool = True,
    record: bool = False,
) -> MatchResult:
    env = FlickRLEnv(sim_config, rl_config)
    env.apply_match_rules(sim_config.goals_to_win, rl_config.max_turns_per_game)
    # Never use training easy-scenarios / search labels during evaluation.
    env.rl_config.easy_scenario_prob = 0.0
    env._search_hint_prob = 0.0
    env.reset(seed=seed)
    actions: List[Dict] = []
    goals_on_policy_turn = 0
    goals_on_opp_turn = 0

    while True:
        assert env.state is not None
        team = env.state.current_team
        prev_a, prev_b = env.state.score_a, env.state.score_b
        if team == policy_team:
            o = observation_from_state(
                env.state,
                team,
                sim_config,
                rl_config,
                search_hint_prob=0.0,
            )
            action = policy.act_numpy(o, deterministic=deterministic)
        else:
            action = opponent.act(env.state, team)

        if record:
            from .actions import rl_action_to_flick

            flick = rl_action_to_flick(action, env.state, team)
            actions.append(
                {
                    "team": team.value,
                    "action": serialize_action(flick),
                }
            )

        result = env.step(action)
        st = env.state
        assert st is not None
        da = st.score_a - prev_a
        db = st.score_b - prev_b
        if team == policy_team:
            scored = da if policy_team == Team.A else db
            if scored > 0:
                goals_on_policy_turn += scored
        else:
            # Goals for either side on opponent's turn.
            goals_on_opp_turn += da + db

        if result.terminated or result.truncated:
            winner = None
            w = st.winner(sim_config.goals_to_win)
            if w is not None:
                winner = w.value
            m = MatchResult(
                score_a=st.score_a,
                score_b=st.score_b,
                turns=result.info.get("turns", 0),
                winner=winner,
                actions=actions,
                seed=seed,
                goals_on_policy_turn=goals_on_policy_turn,
                goals_on_opp_turn=goals_on_opp_turn,
            )
            return m


def evaluate_vs_opponent(
    policy: ActorCritic,
    opponent: Opponent,
    *,
    rl_config: RLConfig,
    sim_config: SimConfig,
    games: int,
    seeds: Optional[List[int]] = None,
    policy_team: Team = Team.A,
) -> Dict[str, float]:
    seeds = seeds or list(range(10_000, 10_000 + games))
    wins = draws = losses = 0
    gf = ga = 0
    turns = 0
    goals_on_policy = 0
    goals_on_opp = 0
    for i in range(games):
        seed = seeds[i % len(seeds)]
        # Alternate sides
        team = policy_team if i % 2 == 0 else policy_team.opponent
        m = play_match(
            policy,
            opponent,
            rl_config=rl_config,
            sim_config=sim_config,
            seed=seed,
            policy_team=team,
            deterministic=True,
        )
        if team == Team.A:
            my_s, opp_s = m.score_a, m.score_b
        else:
            my_s, opp_s = m.score_b, m.score_a
        gf += my_s
        ga += opp_s
        turns += m.turns
        goals_on_policy += m.goals_on_policy_turn
        goals_on_opp += m.goals_on_opp_turn
        if my_s > opp_s:
            wins += 1
        elif my_s < opp_s:
            losses += 1
        else:
            draws += 1

    n = max(1, games)
    return {
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
        "goals_for": gf / n,
        "goals_against": ga / n,
        "goal_diff": (gf - ga) / n,
        "avg_turns": turns / n,
        "goals_on_policy_turn": goals_on_policy / n,
        "goals_on_opp_turn": goals_on_opp / n,
    }
