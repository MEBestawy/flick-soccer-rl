"""Foundation RL config: annealable teacher + league self-play ready."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class RLConfig:
    # Observation / action
    num_players: int = 5
    # 47 core + 8 geometry (goal vec + nearest friendly/opp) + 8 teacher/search hint
    obs_dim: int = 63
    teacher_hint_dim: int = 8
    max_speed_norm: float = 120.0

    # PPO
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.15
    learning_rate: float = 1.2e-4
    lr_final: float = 3e-5
    entropy_coef: float = 0.012
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 1024
    minibatch_size: int = 512
    update_epochs: int = 3
    target_kl: float = 0.015
    bc_coef: float = 0.15
    bc_coef_final: float = 0.0
    bc_anneal_steps: int = 200_000
    # Critical: train mix=0 heads to imitate teacher/search so anneal transfers.
    free_bc_coef: float = 0.4
    free_bc_coef_final: float = 0.1
    free_bc_anneal_steps: int = 450_000

    # Network
    hidden_sizes: tuple[int, ...] = (512, 512, 256)
    angle_log_std_min: float = -1.8
    angle_log_std_max: float = -0.2
    max_angle_residual: float = 1.2
    teacher_player_bias: float = 3.5
    # 1 = full teacher residual prior; 0 = pure learned policy
    teacher_mix_start: float = 1.0
    teacher_mix_final: float = 0.0
    teacher_mix_hold_steps: int = 50_000
    teacher_mix_anneal_steps: int = 300_000
    dual_value: bool = True

    # Rewards
    goal_reward: float = 10.0
    concede_penalty: float = 8.0
    win_reward: float = 4.0
    loss_penalty: float = 4.0
    turn_penalty: float = 0.0001
    shaping_scale: float = 0.04
    shaping_scale_final: float = 0.005
    shaping_anneal_steps: int = 300_000
    touch_bonus: float = 0.45
    progress_coef: float = 0.04
    shot_bonus: float = 0.35
    approach_bonus: float = 0.04
    own_goal_gift: float = 0.02
    easy_scenario_prob: float = 0.08
    easy_scenario_anneal_steps: int = 150_000

    # ExIt-lite search labels (overwrite hint with improved flick)
    # Keep modest — each search does N one-flick rollouts.
    search_label_prob: float = 0.06
    search_label_prob_final: float = 0.15
    search_anneal_steps: int = 300_000
    search_candidates: int = 6
    search_angle_noise: float = 0.45
    search_power_noise: float = 0.20

    bc_bootstrap_updates: int = 0
    bc_bootstrap_batch: int = 256
    bc_aux_updates: int = 0
    bc_aux_until_steps: int = 0

    # Training
    num_envs: int = 8
    num_workers: int = 0
    total_steps: int = 1_000_000
    seed: int = 42
    device: str = "auto"
    max_turns_per_game: int = 80
    goals_to_win: int = 1
    late_goals_to_win: int = 2
    late_max_turns: int = 120
    curriculum_phase_steps: int = 400_000

    checkpoint_every: int = 50_000
    eval_every: int = 25_000
    eval_games: int = 24
    replay_every: int = 100_000
    max_pool_size: int = 24
    pool_snapshot_every: int = 25_000

    # League mix: (heuristic, historical, current/self)
    curriculum_stage1_steps: int = 80_000
    curriculum_stage2_steps: int = 250_000
    stage1_mix: tuple[float, float, float] = (0.40, 0.25, 0.35)
    stage2_mix: tuple[float, float, float] = (0.15, 0.40, 0.45)
    stage3_mix: tuple[float, float, float] = (0.08, 0.42, 0.50)
    use_curriculum: bool = True
    # Prefer stronger historical opponents (PfSP-lite)
    hist_elo_softmax_temp: float = 0.5

    eval_seeds: List[int] = field(
        default_factory=lambda: [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
    )
    elo_initial: float = 1000.0
    elo_k: float = 32.0
    run_name: str = "experiment_01"

    def _anneal(self, start: float, final: float, steps: int, step: int) -> float:
        if steps <= 0:
            return final
        t = min(1.0, step / float(steps))
        return start * (1.0 - t) + final * t

    def shaping_scale_at(self, step: int) -> float:
        return self._anneal(
            self.shaping_scale, self.shaping_scale_final, self.shaping_anneal_steps, step
        )

    def bc_coef_at(self, step: int) -> float:
        return self._anneal(self.bc_coef, self.bc_coef_final, self.bc_anneal_steps, step)

    def teacher_mix_at(self, step: int) -> float:
        if step < self.teacher_mix_hold_steps:
            return self.teacher_mix_start
        return self._anneal(
            self.teacher_mix_start,
            self.teacher_mix_final,
            self.teacher_mix_anneal_steps,
            step - self.teacher_mix_hold_steps,
        )

    def free_bc_coef_at(self, step: int) -> float:
        return self._anneal(
            self.free_bc_coef,
            self.free_bc_coef_final,
            self.free_bc_anneal_steps,
            step,
        )

    def search_prob_at(self, step: int) -> float:
        return self._anneal(
            self.search_label_prob,
            self.search_label_prob_final,
            self.search_anneal_steps,
            step,
        )

    def easy_prob_at(self, step: int) -> float:
        return self._anneal(
            self.easy_scenario_prob, 0.0, self.easy_scenario_anneal_steps, step
        )

    def learning_rate_at(self, step: int, total: int) -> float:
        total = max(total, 1)
        t = min(1.0, step / float(total))
        # Cosine decay
        import math

        return self.lr_final + 0.5 * (self.learning_rate - self.lr_final) * (
            1.0 + math.cos(math.pi * t)
        )

    def opponent_mix(self, step: int) -> tuple[float, float, float]:
        if not self.use_curriculum:
            return self.stage3_mix
        if step < self.curriculum_stage1_steps:
            return self.stage1_mix
        if step < self.curriculum_stage2_steps:
            return self.stage2_mix
        return self.stage3_mix

    def match_rules_at(self, step: int) -> Tuple[int, int]:
        if step < self.curriculum_phase_steps:
            return self.goals_to_win, self.max_turns_per_game
        return self.late_goals_to_win, self.late_max_turns

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
