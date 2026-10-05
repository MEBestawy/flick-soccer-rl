"""
Jev-powered nightmare flick agent.

Pipeline:
  1. Build a rich normalized tactical prompt (reach, threats, marble physics, doctrine).
  2. Ask Jev for strategy / intent / player / ball_target / power.
  3. Convert to a precise ghost-ball FlickAction.
  4. Run a short 1-ply simulation search over nearby candidates and pick the best.

Falls back to HeuristicAgent if the API fails.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..actions import FlickAction
from ..config import SimConfig
from ..models import GameState, Team
from .aiming import (
    action_to_send_ball,
    evaluate_action,
    goal_targets,
    search_best_action,
)
from .base import Agent
from .heuristic_agent import HeuristicAgent
from .jev_client import JevClient, JevClientError
from .prompt import build_normalized_observation, observation_to_prompt_text

POWER_CRITERIA: List[str] = [
    "feather tap — only when already touching / tiny gap",
    "soft controlled marble nudge",
    "medium strike — default scoring / advancing contact",
    "strong drive — long clearance or distant shot",
    "maximum power — last-resort long bomb / desperate clear",
]

BALL_TARGETS: Dict[str, str] = {
    "opp_goal_center": "Send ball at OPPONENT goal center — primary scoring finish.",
    "opp_goal_near_top": "Send ball at OPPONENT near-top of mouth — far-post / angle finish.",
    "opp_goal_near_bottom": "Send ball at OPPONENT near-bottom of mouth — far-post / angle finish.",
    "upfield_center": "Advance/clear ball upfield through center toward attack.",
    "upfield_top": "Advance/clear ball upfield toward the top flank (safer bank).",
    "upfield_bottom": "Advance/clear ball upfield toward the bottom flank (safer bank).",
    "park_safe": "Park the ball in a safer midfield spot away from your goal.",
    "corner_top": "Pin the ball high in the attack corner to set a next-turn shot.",
    "corner_bottom": "Pin the ball low in the attack corner to set a next-turn shot.",
}


def _power_from_score(score_value: float) -> float:
    max_idx = float(len(POWER_CRITERIA) - 1)
    t = max(0.0, min(1.0, float(score_value) / max_idx))
    return 0.15 + 0.85 * t


class JevAgent(Agent):
    """Hard Jev agent: typed decisions + ghost-ball aim + 1-ply search."""

    name = "jev"

    def __init__(
        self,
        team: Team,
        config: Optional[SimConfig] = None,
        *,
        client: Optional[JevClient] = None,
        acting_label: str = "Jev",
        opponent_label: str = "Opponent",
        fallback: Optional[Agent] = None,
        model: str = "jev-latest",
        enable_search: bool = True,
    ) -> None:
        super().__init__(team, config)
        self.acting_label = acting_label
        self.opponent_label = opponent_label
        self._client = client
        self._model = model
        self._fallback = fallback or HeuristicAgent(team, config)
        self.enable_search = enable_search
        self.last_raw_answers: Optional[Dict[str, Any]] = None
        self.last_prompt: Optional[str] = None
        self._last_obs: Optional[Dict[str, Any]] = None
        self.last_search_score: Optional[float] = None

    def _get_client(self) -> JevClient:
        if self._client is None:
            self._client = JevClient(model=self._model)
        return self._client

    def build_prompt(self, state: GameState) -> str:
        obs = build_normalized_observation(
            state,
            self.config,
            acting_team=self.team,
            acting_label=self.acting_label,
            opponent_label=self.opponent_label,
        )
        self._last_obs = obs
        return observation_to_prompt_text(obs)

    def _player_blurb(self, pid: str) -> str:
        if not self._last_obs:
            return f"Flick player {pid}."
        for p in self._last_obs["players"]:
            if p["id"] != pid:
                continue
            if p["can_likely_reach_ball_soft_flick"]:
                reach = "REACH soft+ (precise)"
            elif p["can_likely_reach_ball_medium_flick"]:
                reach = "REACH medium+"
            elif p["can_likely_reach_ball_strong_flick"]:
                reach = "REACH strong+"
            elif p["can_likely_reach_ball_max_flick"]:
                reach = "REACH max-only"
            else:
                reach = "CANNOT reach ball"
            return (
                f"{pid}: gap={p['contact_gap_normalized']} (world {p['contact_gap_world']}), "
                f"{reach}, min_power≈{p['min_power_likely_to_reach_ball']}, "
                f"shot_align={p['shot_alignment_player_ball_goal']}, "
                f"behind_for_shot={p['behind_ball_for_attacking_shot']}, "
                f"covers_goal={p['between_ball_and_own_goal']}."
            )
        return f"Flick player {pid}."

    def _questions(self, controllable_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        sit = (self._last_obs or {}).get("situation", {})
        reachable = set(sit.get("own_players_that_can_reach_ball_this_turn") or [])
        danger = sit.get("danger_to_own_goal", "low")
        scoring = sit.get("scoring_chance", "low")
        lane = sit.get("shot_lane_to_opp_goal_mostly_clear", False)
        threats = sit.get("opponent_players_that_could_reach_ball_next") or []

        player_criteria = {
            pid: self._player_blurb(pid)
            + (
                " PREFERRED contact piece."
                if pid in reachable
                else " Avoid unless emergency cover — cannot contact ball."
            )
            for pid in controllable_ids
        }

        return {
            "strategy": {
                "type": "choice",
                "instructions": (
                    f"You are {self.acting_label}, a nightmare marble-football tactician. "
                    f"Danger={danger}, scoring_chance={scoring}, lane_clear={lane}, "
                    f"opponent_threats_next={threats}. "
                    "Choose the multi-turn strategic frame for this decision."
                ),
                "criteria": {
                    "finish_now": (
                        "A high-quality shot exists — prioritize scoring this flick."
                    ),
                    "build_two_move": (
                        "Set up a 2-move combo: move ball onto a better striking line "
                        "for your next turn while denying the opponent a free hit."
                    ),
                    "emergency_clear": (
                        "Ball is dangerous — clear it far from your goal immediately."
                    ),
                    "starve_counter": (
                        "Protect a lead / deny counters — safe park or wide clear, keep cover."
                    ),
                    "press_for_equalizer": (
                        "Trailing or need a goal — manufacture pressure and a shot lane."
                    ),
                },
            },
            "intent": {
                "type": "choice",
                "instructions": (
                    f"Pick the concrete intent of THIS flick for {self.acting_label}. "
                    "Remember: score AND prevent goals. Prefer ball contact."
                ),
                "criteria": {
                    "shoot_at_goal": "Strike to score now (mouth / far post).",
                    "advance_ball": "Contact to advance ball into a better attack spot.",
                    "clear_danger": "Clear ball away from your goal / upfield.",
                    "setup_touch": "Soft setup touch leaving a teammate a cleaner next shot.",
                    "block_or_cover": "Only if needed: cover goal / block lane without ball contact.",
                },
            },
            "player": {
                "type": "choice",
                "instructions": (
                    "Pick exactly ONE controllable player. Strongly prefer REACHABLE "
                    "pieces with high shot_align for attacks, or covers_goal for defense. "
                    "Never march a far piece into empty space."
                ),
                "criteria": player_criteria,
            },
            "ball_target": {
                "type": "choice",
                "instructions": (
                    "Choose where the BALL should travel after contact (not where the "
                    "player ends). Scoring targets beat vague advances when the lane is open. "
                    "Clear/park targets when danger is high."
                ),
                "criteria": dict(BALL_TARGETS),
            },
            "power": {
                "type": "score",
                "instructions": (
                    "Select power using contact_gap / min_power and intent. "
                    "Close aligned shots → soft/medium. Danger clears & long shots → strong/max. "
                    "Do not overhit when precision scoring is available."
                ),
                "criteria": POWER_CRITERIA,
            },
        }

    def select_action(self, state: GameState) -> FlickAction:
        controllable = [
            p for p in state.get_controllable_players() if p.team == self.team
        ]
        if not controllable:
            raise RuntimeError(f"No controllable players for team {self.team.value}")

        controllable_ids = [p.id for p in controllable]
        prompt = self.build_prompt(state)
        self.last_prompt = prompt

        try:
            answers = self._get_client().decide(
                prompt, self._questions(controllable_ids)
            )
            self.last_raw_answers = answers
            seed = self._action_from_answers(answers, controllable_ids, state)
            if self.enable_search:
                return self._refine_with_search(state, seed, answers)
            return seed
        except (JevClientError, KeyError, ValueError, TypeError) as exc:
            self.last_raw_answers = {"error": str(exc)}
            if self.enable_search:
                refined = self._search_only(state, controllable_ids)
                if refined is not None:
                    return refined
            return self._fallback.select_action(state)

    def _pick_player(
        self,
        player_ans: Dict[str, Any],
        controllable_ids: Sequence[str],
    ) -> str:
        player_id = str(player_ans["choice"])
        if player_id in controllable_ids:
            chosen = player_id
        else:
            probs = player_ans.get("probabilities") or {}
            ranked = sorted(
                (
                    (pid, float(p))
                    for pid, p in probs.items()
                    if pid in controllable_ids
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if not ranked:
                raise ValueError(f"Illegal player choice from Jev: {player_id}")
            chosen = ranked[0][0]

        if self._last_obs:
            reachable = set(
                self._last_obs["situation"].get(
                    "own_players_that_can_reach_ball_this_turn"
                )
                or []
            )
            if reachable and chosen not in reachable:
                probs = player_ans.get("probabilities") or {}
                ranked = sorted(
                    (
                        (pid, float(probs.get(pid, 0.0)))
                        for pid in controllable_ids
                        if pid in reachable
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if ranked:
                    chosen = ranked[0][0]
        return chosen

    def _action_from_answers(
        self,
        answers: Dict[str, Any],
        controllable_ids: List[str],
        state: GameState,
    ) -> FlickAction:
        player_id = self._pick_player(answers["player"], controllable_ids)
        target_key = str(answers["ball_target"]["choice"])
        if target_key not in BALL_TARGETS:
            # Fall back to highest-prob legal target
            probs = answers["ball_target"].get("probabilities") or {}
            ranked = sorted(
                ((k, float(v)) for k, v in probs.items() if k in BALL_TARGETS),
                key=lambda item: item[1],
                reverse=True,
            )
            target_key = ranked[0][0] if ranked else "opp_goal_center"

        power = _power_from_score(float(answers["power"]["score"]))
        # Respect per-player min power to reach when available
        if self._last_obs:
            for p in self._last_obs["players"]:
                if p["id"] == player_id and p.get("min_power_likely_to_reach_ball") is not None:
                    power = max(power, float(p["min_power_likely_to_reach_ball"]) * 0.95)
                    break

        targets = goal_targets(self.config, self.team)
        target = targets.get(target_key, targets["opp_goal_center"])
        action = action_to_send_ball(state, self.config, player_id, target, power)
        if action is None or not action.is_valid(state):
            raise ValueError("Could not build ghost-ball action from Jev answers")
        return action

    def _refine_with_search(
        self,
        state: GameState,
        seed: FlickAction,
        answers: Dict[str, Any],
    ) -> FlickAction:
        """Evaluate seed + nearby target/power variants; keep the best simulation."""
        sit = (self._last_obs or {}).get("situation", {})
        reachable = list(
            sit.get("own_players_that_can_reach_ball_this_turn")
            or [seed.player_id]
        )
        if seed.player_id not in reachable:
            reachable = [seed.player_id, *reachable]

        intent = str((answers.get("intent") or {}).get("choice", ""))
        strategy = str((answers.get("strategy") or {}).get("choice", ""))

        if intent == "shoot_at_goal" or strategy == "finish_now":
            target_keys = [
                "opp_goal_center",
                "opp_goal_near_top",
                "opp_goal_near_bottom",
                "corner_top",
                "corner_bottom",
            ]
            powers = [0.45, 0.65, 0.85, 1.0]
        elif intent == "clear_danger" or strategy == "emergency_clear":
            target_keys = [
                "upfield_center",
                "upfield_top",
                "upfield_bottom",
                "park_safe",
            ]
            powers = [0.7, 0.85, 1.0]
        elif strategy == "build_two_move" or intent == "setup_touch":
            target_keys = [
                "corner_top",
                "corner_bottom",
                "upfield_top",
                "upfield_bottom",
                "opp_goal_center",
            ]
            powers = [0.35, 0.55, 0.75]
        else:
            target_keys = [
                "opp_goal_center",
                "opp_goal_near_top",
                "opp_goal_near_bottom",
                "upfield_center",
                "upfield_top",
                "upfield_bottom",
                "park_safe",
            ]
            powers = [0.4, 0.6, 0.8, 1.0]

        # Always include seed power
        powers = sorted(set(powers + [round(seed.power, 2)]))

        best = seed
        best_score, _ = evaluate_action(state, seed, self.config, self.team)
        searched = search_best_action(
            state,
            self.config,
            self.team,
            player_ids=reachable[:4],
            target_keys=target_keys,
            powers=powers,
        )
        if searched is not None:
            score, _ = evaluate_action(state, searched, self.config, self.team)
            if score > best_score:
                best = searched
                best_score = score

        self.last_search_score = best_score
        return best

    def _search_only(
        self,
        state: GameState,
        controllable_ids: List[str],
    ) -> Optional[FlickAction]:
        sit = (self._last_obs or {}).get("situation", {})
        reachable = list(
            sit.get("own_players_that_can_reach_ball_this_turn") or controllable_ids
        )
        action = search_best_action(
            state,
            self.config,
            self.team,
            player_ids=reachable[:5],
            target_keys=list(BALL_TARGETS.keys()),
            powers=[0.4, 0.6, 0.85, 1.0],
        )
        if action is not None:
            self.last_search_score, _ = evaluate_action(
                state, action, self.config, self.team
            )
        return action
