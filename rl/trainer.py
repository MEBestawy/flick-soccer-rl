"""Main PPO self-play trainer."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sim.config import SimConfig
from sim.models import Team

from .actions import RLAction
from .buffer import RolloutBuffer
from .checkpoints import load_checkpoint, save_checkpoint, save_config_json
from .config import RLConfig
from .device_utils import resolve_device
from .env import FlickRLEnv
from .evaluation import evaluate_vs_opponent, update_elo
from .metrics import MetricsLogger
from .model import ActorCritic
from .observations import observation_from_state
from .opponent_pool import OpponentPool
from .opponents import HeuristicRLOpponent, Opponent, PolicyOpponent
from .parallel_env import ParallelFlickEnvs
from .ppo import ppo_update
from .replay_export import export_match_replay


def _git_meta() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        meta["git_commit"] = commit
        meta["git_dirty"] = dirty
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"] = None
    return meta


class Trainer:
    def __init__(
        self,
        rl_config: Optional[RLConfig] = None,
        sim_config: Optional[SimConfig] = None,
        runs_root: Optional[Path] = None,
    ) -> None:
        self.rl_config = rl_config or RLConfig()
        # Fast settling thresholds keep training on CPU/Rosetta practical.
        self.sim_config = sim_config or SimConfig.fast()
        self.device = resolve_device(self.rl_config.device)
        self.runs_root = Path(runs_root or Path(__file__).resolve().parents[1] / "runs")
        self.run_dir = self.runs_root / self.rl_config.run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model = ActorCritic(self.rl_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.rl_config.learning_rate
        )
        self.global_step = 0
        self.update = 0
        self.best_elo = self.rl_config.elo_initial
        self.current_elo = self.rl_config.elo_initial

        self.pool = OpponentPool(
            self.run_dir / "opponents",
            self.rl_config,
            self.sim_config,
            self.device,
        )
        self.metrics = MetricsLogger(self.run_dir)

        self._use_parallel = self.rl_config.num_workers >= 2
        self.parallel_envs: Optional[ParallelFlickEnvs] = None
        self.envs: List[FlickRLEnv] = []
        if self._use_parallel:
            # Keep main-process torch from oversubscribing against env workers.
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
            self.parallel_envs = ParallelFlickEnvs(
                self.rl_config.num_envs, self.sim_config, self.rl_config
            )
            atexit.register(self.parallel_envs.close)
        else:
            self.envs = [
                FlickRLEnv(self.sim_config, self.rl_config)
                for _ in range(self.rl_config.num_envs)
            ]
        self._opponents: List[Opponent] = []
        self._learner_team: List[Team] = []
        self._obs: List[np.ndarray] = []
        self._ep_returns = np.zeros(self.rl_config.num_envs, dtype=np.float32)
        self._both_flags = [False] * self.rl_config.num_envs

        save_config_json(
            self.run_dir / "config.json",
            self.rl_config,
            extra={
                "sim_physics": self.sim_config.physics.__dict__,
                "goals_to_win": self.sim_config.goals_to_win,
                "model_params": self.model.parameter_count(),
                "device": str(self.device),
                "parallel_workers": self.rl_config.num_workers if self._use_parallel else 0,
                **_git_meta(),
            },
        )

    def resume(self, path: Path) -> None:
        print(f"Resuming run from {path}")
        blob = load_checkpoint(path, self.model, self.optimizer, self.device)
        self.global_step = int(blob.get("global_step", 0))
        self.update = int(blob.get("update", 0))
        self.best_elo = float(blob.get("best_elo", self.best_elo))
        if "config" in blob:
            self.rl_config = RLConfig.from_dict(blob["config"])
        print(
            f"  environment steps={self.global_step}  PPO update={self.update}  "
            f"best_elo={self.best_elo:.1f}"
        )

    def _opp_to_spec(self, opp: Opponent) -> Dict[str, Any]:
        if opp.name == "heuristic":
            team = Team.A if np.random.rand() < 0.5 else Team.B
            return {"kind": "heuristic", "learner_team": team.value}
        if opp.name == "current":
            return {"kind": "current"}
        if isinstance(opp, PolicyOpponent) and opp.checkpoint_path:
            team = Team.A if np.random.rand() < 0.5 else Team.B
            step = 0
            if opp.name.startswith("hist_"):
                try:
                    step = int(opp.name.split("_", 1)[1])
                except ValueError:
                    step = 0
            return {
                "kind": "hist",
                "path": opp.checkpoint_path,
                "step": step,
                "learner_team": team.value,
            }
        # Fallback: treat unknown policy labels as current self-play
        return {"kind": "current"}

    def _reset_env(self, i: int, seed: Optional[int] = None) -> None:
        opp = self.pool.sample_opponent(self.global_step, self.model)
        self._opponents[i] = opp
        shaping = self.rl_config.shaping_scale_at(self.global_step)

        if self._use_parallel:
            assert self.parallel_envs is not None
            spec = self._opp_to_spec(opp)
            both = spec["kind"] == "current"
            self._both_flags[i] = both
            if both:
                self._learner_team[i] = Team.A
            else:
                self._learner_team[i] = (
                    Team.A if spec.get("learner_team", "A") == "A" else Team.B
                )
            self._obs[i] = self.parallel_envs.reset_one(i, spec, seed, shaping)
            self._ep_returns[i] = 0.0
            return

        # When opponent is current policy, learn from both sides
        if opp.name == "current":
            self._learner_team[i] = Team.A  # unused flag: both
            both = True
        else:
            both = False
            self._learner_team[i] = Team.A if np.random.rand() < 0.5 else Team.B
        self._both_flags[i] = both
        obs, _ = self.envs[i].reset(seed=seed)
        # Auto-play until learner to move (unless both)
        self._obs[i] = self._sync_to_learner(i, both)
        self._ep_returns[i] = 0.0

    def _both_sides(self, i: int) -> bool:
        if self._use_parallel:
            return self._both_flags[i]
        return self._opponents[i].name == "current"

    def _sync_to_learner(self, i: int, both: bool) -> np.ndarray:
        env = self.envs[i]
        guard = 0
        while env.state is not None and guard < 50:
            team = env.state.current_team
            if both or team == self._learner_team[i]:
                return observation_from_state(
                    env.state, team, self.sim_config, self.rl_config
                )
            # Opponent acts
            action = self._opponents[i].act(env.state, team)
            result = env.step(action)
            if result.terminated or result.truncated:
                obs, _ = env.reset()
                return self._sync_to_learner(i, both)
            guard += 1
        assert env.state is not None
        return observation_from_state(
            env.state, env.state.current_team, self.sim_config, self.rl_config
        )

    def _init_envs(self) -> None:
        n = self.rl_config.num_envs
        self._opponents = [HeuristicRLOpponent(self.sim_config) for _ in range(n)]
        self._learner_team = [Team.A for _ in range(n)]
        self._obs = [np.zeros(self.rl_config.obs_dim, dtype=np.float32) for _ in range(n)]
        for i in range(n):
            self._reset_env(i, seed=self.rl_config.seed + i)

    def _record_episode_outcome(
        self,
        *,
        both: bool,
        team: Team,
        score_a: int,
        score_b: int,
        ep_reward: float,
        ep_turns: float,
        ep_rewards: List[float],
        ep_lens: List[float],
        tallies: Dict[str, float],
    ) -> None:
        if both:
            if score_a > score_b:
                tallies["wins"] += 0.5
                tallies["losses"] += 0.5
            elif score_b > score_a:
                tallies["wins"] += 0.5
                tallies["losses"] += 0.5
            else:
                tallies["draws"] += 1
            tallies["goals_for"] += (score_a + score_b) / 2
            tallies["goals_against"] += (score_a + score_b) / 2
        else:
            my = score_a if team == Team.A else score_b
            opp = score_b if team == Team.A else score_a
            tallies["goals_for"] += my
            tallies["goals_against"] += opp
            if my > opp:
                tallies["wins"] += 1
            elif my < opp:
                tallies["losses"] += 1
            else:
                tallies["draws"] += 1
        ep_rewards.append(ep_reward)
        ep_lens.append(ep_turns)

    def collect_rollout(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if self._use_parallel:
            return self._collect_rollout_parallel(buffer)
        return self._collect_rollout_serial(buffer)

    def _collect_rollout_parallel(self, buffer: RolloutBuffer) -> Dict[str, float]:
        assert self.parallel_envs is not None
        self.model.train()
        self.parallel_envs.sync_weights(self.model.state_dict())
        ep_rewards: List[float] = []
        ep_lens: List[float] = []
        tallies = {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
        }
        t0 = time.time()

        for _t in range(self.rl_config.rollout_steps):
            obs_batch = np.stack(self._obs, axis=0)
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample = self.model.act(obs_t, deterministic=False)

            players = sample.player.cpu().numpy()
            angles = sample.angle.cpu().numpy()
            powers = sample.power.cpu().numpy()
            dirs = sample.direction_raw.cpu().numpy()
            logps = sample.log_prob.cpu().numpy()
            values = sample.value.cpu().numpy()
            shaping = self.rl_config.shaping_scale_at(self.global_step)

            results = self.parallel_envs.step(players, dirs, powers, shaping)
            rewards = np.zeros(self.rl_config.num_envs, dtype=np.float32)
            dones = np.zeros(self.rl_config.num_envs, dtype=np.float32)

            for i, result in enumerate(results):
                rewards[i] = result.reward
                dones[i] = result.done
                self._ep_returns[i] += result.reward
                self.global_step += 1
                self._obs[i] = result.obs
                self._both_flags[i] = result.both
                self._learner_team[i] = (
                    Team.A if result.learner_team == "A" else Team.B
                )

                if result.ep_finished:
                    self._record_episode_outcome(
                        both=result.both,
                        team=self._learner_team[i],
                        score_a=result.score_a,
                        score_b=result.score_b,
                        ep_reward=float(self._ep_returns[i]),
                        ep_turns=result.ep_turns,
                        ep_rewards=ep_rewards,
                        ep_lens=ep_lens,
                        tallies=tallies,
                    )
                    self._reset_env(i)

            buffer.add(
                obs_batch,
                players,
                angles,
                powers,
                logps,
                rewards,
                dones,
                values,
            )

        obs_batch = np.stack(self._obs, axis=0)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            last_values = self.model.act(obs_t).value.cpu().numpy()
        last_dones = np.zeros(self.rl_config.num_envs, dtype=np.float32)
        buffer.compute_gae(last_values, last_dones)

        elapsed = max(1e-6, time.time() - t0)
        n = max(1, len(ep_rewards))
        return {
            "game/reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "game/length": float(np.mean(ep_lens)) if ep_lens else 0.0,
            "game/win_rate": tallies["wins"] / n if ep_rewards else 0.0,
            "game/draw_rate": tallies["draws"] / n if ep_rewards else 0.0,
            "game/loss_rate": tallies["losses"] / n if ep_rewards else 0.0,
            "game/goals_for": tallies["goals_for"] / n if ep_rewards else 0.0,
            "game/goals_against": tallies["goals_against"] / n if ep_rewards else 0.0,
            "perf/env_steps_per_sec": self.rl_config.rollout_steps
            * self.rl_config.num_envs
            / elapsed,
        }

    def _collect_rollout_serial(self, buffer: RolloutBuffer) -> Dict[str, float]:
        self.model.train()
        ep_rewards: List[float] = []
        ep_lens: List[float] = []
        tallies = {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
        }
        t0 = time.time()

        for _t in range(self.rl_config.rollout_steps):
            obs_batch = np.stack(self._obs, axis=0)
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample = self.model.act(obs_t, deterministic=False)

            players = sample.player.cpu().numpy()
            angles = sample.angle.cpu().numpy()
            powers = sample.power.cpu().numpy()
            dirs = sample.direction_raw.cpu().numpy()
            logps = sample.log_prob.cpu().numpy()
            values = sample.value.cpu().numpy()

            rewards = np.zeros(self.rl_config.num_envs, dtype=np.float32)
            dones = np.zeros(self.rl_config.num_envs, dtype=np.float32)

            for i in range(self.rl_config.num_envs):
                env = self.envs[i]
                assert env.state is not None
                action = RLAction(
                    player_index=int(players[i]),
                    direction_raw=dirs[i],
                    power=float(powers[i]),
                )
                env._shaping_override = self.rl_config.shaping_scale_at(self.global_step)
                result = env.step(action)
                rewards[i] = result.reward
                self._ep_returns[i] += result.reward
                done = result.terminated or result.truncated
                dones[i] = float(done)
                self.global_step += 1

                if done:
                    st = env.state
                    assert st is not None
                    self._record_episode_outcome(
                        both=self._both_sides(i),
                        team=self._learner_team[i],
                        score_a=st.score_a,
                        score_b=st.score_b,
                        ep_reward=float(self._ep_returns[i]),
                        ep_turns=float(result.info.get("turns", 0)),
                        ep_rewards=ep_rewards,
                        ep_lens=ep_lens,
                        tallies=tallies,
                    )
                    self._reset_env(i)
                else:
                    both = self._both_sides(i)
                    if not both and env.state is not None:
                        while (
                            env.state is not None
                            and env.state.current_team != self._learner_team[i]
                        ):
                            opp_action = self._opponents[i].act(
                                env.state, env.state.current_team
                            )
                            opp_res = env.step(opp_action)
                            if opp_res.terminated or opp_res.truncated:
                                st = env.state
                                assert st is not None
                                self._record_episode_outcome(
                                    both=False,
                                    team=self._learner_team[i],
                                    score_a=st.score_a,
                                    score_b=st.score_b,
                                    ep_reward=float(self._ep_returns[i]),
                                    ep_turns=float(opp_res.info.get("turns", 0)),
                                    ep_rewards=ep_rewards,
                                    ep_lens=ep_lens,
                                    tallies=tallies,
                                )
                                dones[i] = 1.0
                                self._reset_env(i)
                                break
                    if env.state is not None and dones[i] == 0:
                        team = env.state.current_team
                        self._obs[i] = observation_from_state(
                            env.state, team, self.sim_config, self.rl_config
                        )

            buffer.add(
                obs_batch,
                players,
                angles,
                powers,
                logps,
                rewards,
                dones,
                values,
            )

        obs_batch = np.stack(self._obs, axis=0)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            last_values = self.model.act(obs_t).value.cpu().numpy()
        last_dones = np.zeros(self.rl_config.num_envs, dtype=np.float32)
        buffer.compute_gae(last_values, last_dones)

        elapsed = max(1e-6, time.time() - t0)
        n = max(1, len(ep_rewards))
        return {
            "game/reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "game/length": float(np.mean(ep_lens)) if ep_lens else 0.0,
            "game/win_rate": tallies["wins"] / n if ep_rewards else 0.0,
            "game/draw_rate": tallies["draws"] / n if ep_rewards else 0.0,
            "game/loss_rate": tallies["losses"] / n if ep_rewards else 0.0,
            "game/goals_for": tallies["goals_for"] / n if ep_rewards else 0.0,
            "game/goals_against": tallies["goals_against"] / n if ep_rewards else 0.0,
            "perf/env_steps_per_sec": self.rl_config.rollout_steps
            * self.rl_config.num_envs
            / elapsed,
        }

    def train(self, total_steps: Optional[int] = None) -> None:
        total = total_steps or self.rl_config.total_steps
        self._init_envs()
        buffer = RolloutBuffer(self.rl_config, self.rl_config.num_envs)
        workers = self.rl_config.num_workers if self._use_parallel else 0
        print(
            f"Training {self.rl_config.run_name} on {self.device} | "
            f"params={self.model.parameter_count()} | envs={self.rl_config.num_envs} | "
            f"workers={workers}"
        )

        while self.global_step < total:
            buffer.reset()
            # refresh shaping on envs
            for e in self.envs:
                e._shaping_override = self.rl_config.shaping_scale_at(self.global_step)

            roll_stats = self.collect_rollout(buffer)
            batch = buffer.get(self.device)
            t0 = time.time()
            ppo_stats = ppo_update(self.model, self.optimizer, batch, self.rl_config)
            ppo_time = time.time() - t0
            self.update += 1

            stats = {
                **{f"train/{k}": v for k, v in ppo_stats.items() if k != "n_updates"},
                **roll_stats,
                "train/learning_rate": self.rl_config.learning_rate,
                "perf/ppo_update_sec": ppo_time,
                "eval/elo": self.current_elo,
            }
            self.metrics.log(self.global_step, stats)

            print(
                f"Step {self.global_step:,}  "
                f"SPS {roll_stats['perf/env_steps_per_sec']:.0f}  "
                f"Reward {roll_stats['game/reward']:+.3f}  "
                f"Win {roll_stats['game/win_rate']:.0%}  "
                f"Entropy {ppo_stats['entropy']:.2f}  "
                f"Elo {self.current_elo:.0f}  "
                f"Best {self.best_elo:.0f}"
            )

            if self.global_step % self.rl_config.pool_snapshot_every < (
                self.rl_config.rollout_steps * self.rl_config.num_envs
            ):
                self.pool.add_snapshot(self.model, self.global_step)

            if self.global_step % self.rl_config.checkpoint_every < (
                self.rl_config.rollout_steps * self.rl_config.num_envs
            ):
                self._save_all()

            if self.global_step % self.rl_config.eval_every < (
                self.rl_config.rollout_steps * self.rl_config.num_envs
            ):
                self._evaluate_and_maybe_best()

            if self.global_step % self.rl_config.replay_every < (
                self.rl_config.rollout_steps * self.rl_config.num_envs
            ):
                self._export_replays()

        self._save_all()
        self._evaluate_and_maybe_best()
        self.metrics.close()
        print("Training complete.")

    def _save_all(self) -> None:
        save_checkpoint(
            self.ckpt_dir / "latest.pt",
            model=self.model,
            optimizer=self.optimizer,
            config=self.rl_config,
            global_step=self.global_step,
            update=self.update,
            best_elo=self.best_elo,
        )
        save_checkpoint(
            self.ckpt_dir / f"step_{self.global_step:09d}.pt",
            model=self.model,
            optimizer=self.optimizer,
            config=self.rl_config,
            global_step=self.global_step,
            update=self.update,
            best_elo=self.best_elo,
        )

    def _evaluate_and_maybe_best(self) -> None:
        self.model.eval()
        heur = HeuristicRLOpponent(self.sim_config, noise=0.05)
        stats = evaluate_vs_opponent(
            self.model,
            heur,
            rl_config=self.rl_config,
            sim_config=self.sim_config,
            games=self.rl_config.eval_games,
            seeds=self.rl_config.eval_seeds,
        )
        # Elo vs heuristic baseline 1000
        score = stats["win_rate"] + 0.5 * stats["draw_rate"]
        self.current_elo, _ = update_elo(
            self.current_elo, self.rl_config.elo_initial, score, self.rl_config.elo_k
        )
        self.metrics.log(
            self.global_step,
            {
                "eval/win_rate_heuristic": stats["win_rate"],
                "eval/goal_diff": stats["goal_diff"],
                "eval/avg_turns": stats["avg_turns"],
                "eval/elo": self.current_elo,
            },
        )
        print(
            f"  Eval vs heuristic: win={stats['win_rate']:.0%}  "
            f"GD={stats['goal_diff']:+.2f}  Elo={self.current_elo:.0f}"
        )
        if self.current_elo > self.best_elo + 5:
            self.best_elo = self.current_elo
            save_checkpoint(
                self.ckpt_dir / "best.pt",
                model=self.model,
                optimizer=self.optimizer,
                config=self.rl_config,
                global_step=self.global_step,
                update=self.update,
                best_elo=self.best_elo,
            )
            print(f"  New best.pt (Elo {self.best_elo:.0f})")

    def _export_replays(self) -> None:
        self.model.eval()
        out_dir = self.run_dir / "replays" / f"step_{self.global_step:09d}"
        heur = HeuristicRLOpponent(self.sim_config, noise=0.0)
        for seed in self.rl_config.eval_seeds[:3]:
            export_match_replay(
                out_dir / f"vs_heuristic_seed_{seed}.json",
                self.model,
                heur,
                rl_config=self.rl_config,
                sim_config=self.sim_config,
                seed=seed,
                meta={"step": self.global_step},
            )
