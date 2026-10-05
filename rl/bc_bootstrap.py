"""Behavioral cloning / DAgger bootstrap from shoot teacher."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from sim.config import SimConfig

from .config import RLConfig
from .env import FlickRLEnv
from .model import ActorCritic
from .observations import observation_from_state
from .opponents import HeuristicRLOpponent, Opponent
from .teachers import ShootAtGoalTeacher


def behavioral_clone_bootstrap(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    rl_config: RLConfig,
    sim_config: SimConfig,
    device: torch.device,
    updates: int = 400,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    DAgger-style clone: on each state, train toward the shoot teacher label while
    executing a mix of teacher / learner actions so the obs distribution matches
    deployment.
    """
    env = FlickRLEnv(sim_config, rl_config)
    env.rl_config.easy_scenario_prob = 0.15
    shoot = ShootAtGoalTeacher(sim_config, power=0.9)
    heur = HeuristicRLOpponent(sim_config, noise=0.05, seed=rl_config.seed)
    env.reset(seed=rl_config.seed)
    rng = np.random.default_rng(rl_config.seed + 7)

    model.train()
    total_loss = 0.0
    total_lp = 0.0

    for i in range(updates):
        # Anneal toward more learner rollouts (classic DAgger).
        teacher_exec_p = max(0.2, 0.9 - 0.7 * (i / max(1, updates - 1)))
        use_shoot = (i % 4) != 0  # mostly shoot teacher labels
        labeler: Opponent = shoot if use_shoot else heur

        obs_l: List[np.ndarray] = []
        players: List[int] = []
        angles: List[float] = []
        powers: List[float] = []

        while len(obs_l) < batch_size:
            if env.state is None:
                env.reset()
            assert env.state is not None
            team = env.state.current_team
            obs = observation_from_state(env.state, team, env.sim_config, env.rl_config)
            label = labeler.act(env.state, team)
            ang = float(np.arctan2(label.direction_raw[1], label.direction_raw[0]))
            obs_l.append(obs)
            players.append(int(label.player_index))
            angles.append(ang)
            powers.append(float(label.power))

            # Execute mix of teacher label vs current policy (DAgger).
            if rng.random() < teacher_exec_p:
                execute = label
            else:
                with torch.no_grad():
                    execute = model.act_numpy(obs, deterministic=False)

            result = env.step(execute)
            if result.terminated or result.truncated:
                env.reset()

        obs_t = torch.as_tensor(
            np.stack(obs_l[:batch_size]), dtype=torch.float32, device=device
        )
        players_t = torch.as_tensor(
            players[:batch_size], dtype=torch.long, device=device
        )
        angles_t = torch.as_tensor(
            angles[:batch_size], dtype=torch.float32, device=device
        )
        powers_t = torch.as_tensor(
            powers[:batch_size], dtype=torch.float32, device=device
        )

        logp, entropy, _values = model.evaluate_actions(
            obs_t, players_t, angles_t, powers_t
        )
        loss = -logp.mean() - 0.002 * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), rl_config.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.item())
        total_lp += float(logp.mean().item())
        if (i + 1) % 200 == 0:
            print(
                f"  BC/DAgger {i+1}/{updates}  "
                f"loss={total_loss/(i+1):.3f}  logp={total_lp/(i+1):.3f}  "
                f"exec_teacher_p={teacher_exec_p:.2f}"
            )

    return {
        "bc_loss": total_loss / max(1, updates),
        "bc_logp": total_lp / max(1, updates),
    }
