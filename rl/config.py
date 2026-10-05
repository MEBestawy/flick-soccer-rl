"""Central RL hyperparameters and training knobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RLConfig:
    # Observation / action
    num_players: int = 5
    obs_dim: int = 47  # ball4 + friendly20 + opp20 + scores2 + time1
    max_speed_norm: float = 120.0

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 4

    # Network
    hidden_sizes: tuple[int, ...] = (256, 256, 128)

    # Rewards
    goal_reward: float = 1.0
    concede_penalty: float = 1.0
    win_reward: float = 1.0
    loss_penalty: float = 1.0
    turn_penalty: float = 0.001
    shaping_scale: float = 0.05
    shaping_scale_final: float = 0.02
    shaping_anneal_steps: int = 1_000_000

    # Training
    num_envs: int = 8
    # 0/1 = in-process sequential envs; >=2 = subprocess workers (one per env).
    num_workers: int = 0
    total_steps: int = 1_000_000
    seed: int = 42
    device: str = "auto"  # auto | cpu | mps
    max_turns_per_game: int = 200

    # Checkpoint / eval
    checkpoint_every: int = 50_000
    eval_every: int = 25_000
    eval_games: int = 20
    replay_every: int = 50_000
    max_pool_size: int = 20
    pool_snapshot_every: int = 25_000

    # Curriculum opponent mix: (heuristic, historical, current)
    curriculum_stage1_steps: int = 250_000
    curriculum_stage2_steps: int = 1_000_000
    stage1_mix: tuple[float, float, float] = (0.55, 0.20, 0.25)
    stage2_mix: tuple[float, float, float] = (0.25, 0.35, 0.40)
    stage3_mix: tuple[float, float, float] = (0.10, 0.30, 0.60)
    use_curriculum: bool = True

    # Eval seeds for progress comparison
    eval_seeds: List[int] = field(
        default_factory=lambda: [1001, 1002, 1003, 1004, 1005]
    )

    # Elo
    elo_initial: float = 1000.0
    elo_k: float = 32.0

    run_name: str = "experiment_01"

    def shaping_scale_at(self, step: int) -> float:
        if self.shaping_anneal_steps <= 0:
            return self.shaping_scale_final
        t = min(1.0, step / float(self.shaping_anneal_steps))
        return self.shaping_scale * (1.0 - t) + self.shaping_scale_final * t

    def opponent_mix(self, step: int) -> tuple[float, float, float]:
        if not self.use_curriculum:
            return self.stage3_mix
        if step < self.curriculum_stage1_steps:
            return self.stage1_mix
        if step < self.curriculum_stage2_steps:
            return self.stage2_mix
        return self.stage3_mix

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
