#!/usr/bin/env python3
"""
Sanity / overfit test: PPO should learn to shoot an open goal.

Exits 0 if mean reward improves meaningfully over training.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from _rl_path import setup_paths

ROOT = setup_paths()

from rl.buffer import RolloutBuffer
from rl.config import RLConfig
from rl.device_utils import resolve_device
from rl.model import ActorCritic
from rl.ppo import ppo_update
from rl.sanity_env import OpenGoalSanityEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO open-goal sanity check")
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = RLConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        minibatch_size=128,
        update_epochs=4,
        learning_rate=3e-4,
        entropy_coef=0.02,
        total_steps=args.steps,
        device=args.device,
        seed=args.seed,
        max_turns_per_game=8,
        shaping_scale=0.1,
        shaping_scale_final=0.05,
    )
    device = resolve_device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    envs = [OpenGoalSanityEnv(rl_config=cfg) for _ in range(cfg.num_envs)]
    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=args.seed + i)
        obs_list.append(o)

    buffer = RolloutBuffer(cfg, cfg.num_envs)
    global_step = 0
    ep_rewards: list[float] = []
    window: list[float] = []
    early_mean = None
    t0 = time.time()

    print(
        f"Sanity open-goal PPO | params={model.parameter_count()} | "
        f"device={device} | steps={args.steps}"
    )

    while global_step < args.steps:
        buffer.reset()
        for _ in range(cfg.rollout_steps):
            obs_batch = np.stack(obs_list, axis=0)
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                sample = model.act(obs_t, deterministic=False)

            players = sample.player.cpu().numpy()
            angles = sample.angle.cpu().numpy()
            powers = sample.power.cpu().numpy()
            dirs = sample.direction_raw.cpu().numpy()
            logps = sample.log_prob.cpu().numpy()
            values = sample.value.cpu().numpy()
            rewards = np.zeros(cfg.num_envs, dtype=np.float32)
            dones = np.zeros(cfg.num_envs, dtype=np.float32)

            for i in range(cfg.num_envs):
                from rl.actions import RLAction

                action = RLAction(
                    player_index=int(players[i]),
                    direction_raw=dirs[i],
                    power=float(powers[i]),
                )
                result = envs[i].step(action)
                rewards[i] = result.reward
                done = result.terminated or result.truncated
                dones[i] = float(done)
                global_step += 1
                if done:
                    ep_rewards.append(float(result.info.get("episode_reward", result.reward)))
                    window.append(ep_rewards[-1])
                    if len(window) > 50:
                        window.pop(0)
                    o, _ = envs[i].reset(seed=args.seed + global_step + i)
                    obs_list[i] = o
                else:
                    obs_list[i] = result.obs

            buffer.add(
                obs_batch, players, angles, powers, logps, rewards, dones, values
            )

        obs_batch = np.stack(obs_list, axis=0)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            last_v = model.act(obs_t).value.cpu().numpy()
        buffer.compute_gae(last_v, np.zeros(cfg.num_envs, dtype=np.float32))
        stats = ppo_update(model, opt, buffer.get(device), cfg)

        mean_r = float(np.mean(window)) if window else 0.0
        if early_mean is None and len(ep_rewards) >= 20:
            early_mean = float(np.mean(ep_rewards[:20]))
        sps = global_step / max(1e-6, time.time() - t0)
        print(
            f"Step {global_step:,}  reward {mean_r:+.3f}  "
            f"entropy {stats['entropy']:.2f}  SPS {sps:.0f}  "
            f"episodes {len(ep_rewards)}"
        )

    late = float(np.mean(ep_rewards[-30:])) if len(ep_rewards) >= 30 else float(
        np.mean(ep_rewards) if ep_rewards else 0.0
    )
    early = early_mean if early_mean is not None else 0.0
    print(f"\nEarly mean reward: {early:+.3f}")
    print(f"Late mean reward:  {late:+.3f}")

    # Deterministic eval score rate
    scores = 0
    trials = 20
    for i in range(trials):
        env = OpenGoalSanityEnv(rl_config=cfg)
        o, _ = env.reset(seed=10_000 + i)
        scored = False
        for _ in range(cfg.max_turns_per_game):
            a = model.act_numpy(o, deterministic=True)
            r = env.step(a)
            o = r.obs
            if env.state and env.state.score_a > 0:
                scored = True
                break
            if r.terminated or r.truncated:
                break
        scores += int(scored)
    rate = scores / trials
    print(f"Deterministic score rate: {rate:.0%} ({scores}/{trials})")

    ok = late > early + 0.05 or rate >= 0.4
    if ok:
        print("SANITY PASS")
        raise SystemExit(0)
    print("SANITY FAIL — PPO did not clearly improve")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
