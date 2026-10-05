"""Multiprocess flick-soccer env workers for parallel rollouts."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sim.config import SimConfig
from sim.models import Team

from .actions import RLAction
from .config import RLConfig
from .env import FlickRLEnv
from .model import ActorCritic
from .observations import observation_from_state
from .opponents import HeuristicRLOpponent, PolicyOpponent


@dataclass
class WorkerStepResult:
    obs: np.ndarray
    reward: float
    done: float
    ep_finished: bool
    ep_reward: float
    ep_turns: float
    score_a: int
    score_b: int
    both: bool
    learner_team: str


def _worker_main(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    sim_config: SimConfig,
    rl_config: RLConfig,
    worker_id: int,
) -> None:
    parent_remote.close()
    torch.set_num_threads(1)

    env = FlickRLEnv(sim_config, rl_config)
    heuristic = HeuristicRLOpponent(sim_config, seed=rl_config.seed + worker_id)
    current_model = ActorCritic(rl_config)
    current_model.eval()
    hist_cache: Dict[str, ActorCritic] = {}

    both = False
    learner_team = Team.A
    opponent: Any = heuristic
    ep_return = 0.0

    def _load_hist(path: str) -> ActorCritic:
        if path not in hist_cache:
            model = ActorCritic(rl_config)
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["model"])
            model.eval()
            hist_cache[path] = model
            if len(hist_cache) > 4:
                hist_cache.pop(next(iter(hist_cache)))
        return hist_cache[path]

    def _obs_for(team: Team) -> np.ndarray:
        assert env.state is not None
        return observation_from_state(
            env.state,
            team,
            sim_config,
            rl_config,
            search_hint_prob=env._search_hint_prob,
        )

    def _sync_to_learner() -> np.ndarray:
        guard = 0
        while env.state is not None and guard < 50:
            team = env.state.current_team
            if both or team == learner_team:
                return _obs_for(team)
            action = opponent.act(env.state, team)
            result = env.step(action)
            if result.terminated or result.truncated:
                env.reset()
                return _sync_to_learner()
            guard += 1
        assert env.state is not None
        return _obs_for(env.state.current_team)

    def _configure_opponent(spec: Dict[str, Any]) -> None:
        nonlocal both, learner_team, opponent
        kind = spec["kind"]
        if kind == "heuristic":
            both = False
            learner_team = Team.A if spec.get("learner_team", "A") == "A" else Team.B
            opponent = heuristic
        elif kind == "current":
            both = True
            learner_team = Team.A
            opponent = PolicyOpponent(
                current_model, rl_config, sim_config, deterministic=False, label="current"
            )
        elif kind == "hist":
            both = False
            learner_team = Team.A if spec.get("learner_team", "A") == "A" else Team.B
            model = _load_hist(spec["path"])
            opponent = PolicyOpponent(
                model,
                rl_config,
                sim_config,
                deterministic=False,
                label=f"hist_{spec.get('step', 0)}",
            )
        else:
            raise ValueError(f"Unknown opponent kind: {kind}")

    def _episode_stats() -> Tuple[float, float, int, int]:
        st = env.state
        assert st is not None
        return ep_return, float(env._turns), int(st.score_a), int(st.score_b)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                _configure_opponent(data["opp"])
                env._shaping_override = data.get("shaping")
                gtw = int(data["opp"].get("goals_to_win", rl_config.goals_to_win))
                mt = int(data["opp"].get("max_turns", rl_config.max_turns_per_game))
                env.apply_match_rules(gtw, mt)
                env.reset(seed=data.get("seed"))
                ep_return = 0.0
                obs = _sync_to_learner()
                remote.send(obs)
            elif cmd == "sync_weights":
                current_model.load_state_dict(data)
                current_model.eval()
                remote.send(True)
            elif cmd == "set_curriculum":
                env.rl_config.easy_scenario_prob = float(data.get("easy_prob", 0.0))
                env._search_hint_prob = float(data.get("search_prob", 0.0))
                if "teacher_mix" in data and hasattr(current_model, "set_teacher_mix"):
                    current_model.set_teacher_mix(float(data["teacher_mix"]))
                    for m in hist_cache.values():
                        if hasattr(m, "set_teacher_mix"):
                            m.set_teacher_mix(float(data["teacher_mix"]))
                remote.send(True)
            elif cmd == "step":
                env._shaping_override = data.get("shaping")
                action = RLAction(
                    player_index=int(data["player"]),
                    direction_raw=np.asarray(data["direction"], dtype=np.float64),
                    power=float(data["power"]),
                )
                result = env.step(action)
                reward = float(result.reward)
                ep_return += reward
                done = bool(result.terminated or result.truncated)
                ep_finished = done
                ep_reward = ep_turns = 0.0
                score_a = score_b = 0

                if done:
                    ep_reward, ep_turns, score_a, score_b = _episode_stats()
                    obs = result.obs
                else:
                    if not both and env.state is not None:
                        while (
                            env.state is not None
                            and env.state.current_team != learner_team
                        ):
                            opp_action = opponent.act(env.state, env.state.current_team)
                            opp_res = env.step(
                                opp_action,
                                reward_perspective=learner_team,
                                include_turn_penalty=False,
                            )
                            reward += float(opp_res.reward)
                            ep_return += float(opp_res.reward)
                            if opp_res.terminated or opp_res.truncated:
                                done = True
                                ep_finished = True
                                ep_reward, ep_turns, score_a, score_b = _episode_stats()
                                break
                    if not done and env.state is not None:
                        team = env.state.current_team
                        obs = _obs_for(team)
                    else:
                        obs = result.obs

                remote.send(
                    WorkerStepResult(
                        obs=obs,
                        reward=reward,
                        done=float(done),
                        ep_finished=ep_finished,
                        ep_reward=ep_reward,
                        ep_turns=ep_turns,
                        score_a=score_a,
                        score_b=score_b,
                        both=both,
                        learner_team=learner_team.value,
                    )
                )
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        remote.close()


class ParallelFlickEnvs:
    """One subprocess per env; steps fan out across CPU cores."""

    def __init__(
        self,
        num_envs: int,
        sim_config: SimConfig,
        rl_config: RLConfig,
    ) -> None:
        self.num_envs = num_envs
        self.sim_config = sim_config
        self.rl_config = rl_config
        ctx = mp.get_context("spawn")
        self._remotes: List[mp.connection.Connection] = []
        self._processes: List[mp.Process] = []
        for i in range(num_envs):
            parent_remote, worker_remote = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main,
                args=(worker_remote, parent_remote, sim_config, rl_config, i),
                daemon=True,
            )
            proc.start()
            worker_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(proc)

    def sync_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        for remote in self._remotes:
            remote.send(("sync_weights", cpu_sd))
        for remote in self._remotes:
            remote.recv()

    def set_curriculum(
        self,
        *,
        easy_prob: float,
        search_prob: float,
        teacher_mix: float,
    ) -> None:
        payload = {
            "easy_prob": easy_prob,
            "search_prob": search_prob,
            "teacher_mix": teacher_mix,
        }
        for remote in self._remotes:
            remote.send(("set_curriculum", payload))
        for remote in self._remotes:
            remote.recv()

    def reset_one(self, i: int, opp: Dict[str, Any], seed: Optional[int], shaping: Optional[float]) -> np.ndarray:
        self._remotes[i].send(("reset", {"opp": opp, "seed": seed, "shaping": shaping}))
        return self._remotes[i].recv()

    def step(
        self,
        players: np.ndarray,
        directions: np.ndarray,
        powers: np.ndarray,
        shaping: Optional[float],
    ) -> List[WorkerStepResult]:
        for i, remote in enumerate(self._remotes):
            remote.send(
                (
                    "step",
                    {
                        "player": int(players[i]),
                        "direction": directions[i],
                        "power": float(powers[i]),
                        "shaping": shaping,
                    },
                )
            )
        return [remote.recv() for remote in self._remotes]

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                remote.close()
            except OSError:
                pass
        for proc in self._processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
