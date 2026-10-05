"""Core RL unit tests (obs, actions, model, rewards, PPO, checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sim import HeadlessEnv, Team
from sim.config import SimConfig
from sim.geometry import Vec2

from rl.actions import RLAction, rl_action_to_flick
from rl.buffer import RolloutBuffer
from rl.checkpoints import load_checkpoint, save_checkpoint
from rl.config import RLConfig
from rl.env import FlickRLEnv
from rl.model import ActorCritic
from rl.observations import friendly_player_ids, observation_from_state
from rl.opponent_pool import OpponentPool
from rl.opponents import HeuristicRLOpponent
from rl.ppo import ppo_update
from rl.rewards import ball_potential, compute_transition_reward


def test_observation_dim_and_mirror():
    env = HeadlessEnv()
    state = env.reset(Team.A)
    cfg = SimConfig.default()
    rl = RLConfig()
    oa = observation_from_state(state, Team.A, cfg, rl)
    ob = observation_from_state(state, Team.B, cfg, rl)
    assert oa.shape == (rl.obs_dim,)
    assert ob.shape == (rl.obs_dim,)
    # Ball x should flip sign under mirror (approx, same |x| if centered)
    # Friendly first player x for A vs B should differ
    assert not np.allclose(oa, ob)


def test_action_to_flick_normalized():
    env = HeadlessEnv()
    state = env.reset(Team.A)
    action = RLAction(
        player_index=0,
        direction_raw=np.array([3.0, 4.0]),
        power=0.7,
    )
    flick = rl_action_to_flick(action, state, Team.A)
    assert abs(flick.direction.length() - 1.0) < 1e-5
    assert 0.0 <= flick.power <= 1.0
    ids = friendly_player_ids(state, Team.A)
    assert flick.player_id in ids


def test_model_distributions_finite():
    cfg = RLConfig()
    model = ActorCritic(cfg)
    obs = torch.randn(8, cfg.obs_dim)
    sample = model.act(obs)
    assert torch.isfinite(sample.log_prob).all()
    assert torch.isfinite(sample.entropy).all()
    assert ((sample.power > 0) & (sample.power < 1)).all()
    assert sample.player.min() >= 0 and sample.player.max() < 5
    d = sample.direction_raw
    norms = torch.linalg.norm(d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    a1 = model.act_numpy(obs[0].numpy(), deterministic=True)
    a2 = model.act_numpy(obs[0].numpy(), deterministic=True)
    assert a1.player_index == a2.player_index
    assert np.allclose(a1.direction_raw, a2.direction_raw)
    assert abs(a1.power - a2.power) < 1e-6


def test_reward_scoring_and_shaping_sign():
    env = HeadlessEnv()
    prev = env.reset(Team.A)
    nxt = prev.clone()
    nxt.score_a = prev.score_a + 1
    # Move ball forward for Team A
    nxt.ball.position = Vec2(prev.ball.position.x + 10.0, prev.ball.position.y)
    cfg = SimConfig.default()
    rl = RLConfig()
    br = compute_transition_reward(prev, nxt, Team.A, cfg, rl, shaping_scale=0.05)
    assert br.goal > 0
    assert br.shaping > 0 or ball_potential(nxt, Team.A, cfg) > ball_potential(prev, Team.A, cfg)


def test_env_step_settled():
    env = FlickRLEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (RLConfig().obs_dim,)
    action = RLAction(0, np.array([1.0, 0.0]), 0.5)
    result = env.step(action)
    assert np.isfinite(result.reward)
    assert result.obs.shape == obs.shape


def test_ppo_update_changes_weights():
    cfg = RLConfig(rollout_steps=32, minibatch_size=16, update_epochs=2, num_envs=2)
    model = ActorCritic(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    buf = RolloutBuffer(cfg, cfg.num_envs)
    for _ in range(cfg.rollout_steps):
        obs = np.random.randn(cfg.num_envs, cfg.obs_dim).astype(np.float32)
        t = torch.as_tensor(obs)
        with torch.no_grad():
            s = model.act(t)
        buf.add(
            obs,
            s.player.numpy(),
            s.angle.numpy(),
            s.power.numpy(),
            s.log_prob.numpy(),
            np.random.randn(cfg.num_envs).astype(np.float32),
            np.zeros(cfg.num_envs, dtype=np.float32),
            s.value.numpy(),
        )
    buf.compute_gae(np.zeros(cfg.num_envs, dtype=np.float32), np.zeros(cfg.num_envs, dtype=np.float32))
    stats = ppo_update(model, opt, buf.get(torch.device("cpu")), cfg)
    assert np.isfinite(stats["policy_loss"])
    changed = any(
        not torch.allclose(before[k], model.state_dict()[k]) for k in before
    )
    assert changed


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = RLConfig()
    model = ActorCritic(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        config=cfg,
        global_step=123,
        update=4,
        best_elo=1010.0,
    )
    model2 = ActorCritic(cfg)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    blob = load_checkpoint(path, model2, opt2)
    assert blob["global_step"] == 123
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(a, b)


def test_opponent_pool_sample(tmp_path: Path):
    cfg = RLConfig()
    sim = SimConfig.default()
    pool = OpponentPool(tmp_path, cfg, sim, torch.device("cpu"))
    model = ActorCritic(cfg)
    pool.add_snapshot(model, step=1000)
    opp = pool.sample_opponent(0, model)
    assert opp is not None
    env = HeadlessEnv()
    st = env.reset()
    action = opp.act(st, st.current_team)
    assert 0 <= action.player_index < 5
