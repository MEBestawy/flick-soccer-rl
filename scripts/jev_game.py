#!/usr/bin/env python3
"""
Play a headless match with a Jev agent (strategy-pattern).

Examples:
  # Human-side heuristic vs Jev
  uv run python scripts/jev_game.py

  # Jev vs Jev
  uv run python scripts/jev_game.py --team-a jev --team-b jev

  # Print the normalized prompt Jev would see for kickoff
  uv run python scripts/jev_game.py --show-prompt-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

from sim import Team  # noqa: E402
from sim.agents import (  # noqa: E402
    JevAgent,
    MatchRunner,
    available_agents,
    create_agent,
    observation_to_prompt_text,
    build_normalized_observation,
)
from sim.env import HeadlessEnv  # noqa: E402


def show_prompt(team: Team = Team.B) -> None:
    env = HeadlessEnv()
    state = env.reset(Team.A)
    obs = build_normalized_observation(
        state,
        env.config,
        acting_team=team,
        acting_label="Jev",
        opponent_label="Opponent",
    )
    print(observation_to_prompt_text(obs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play flick football with Jev")
    parser.add_argument(
        "--team-a",
        default="heuristic",
        help=f"Strategy for Team A (human yellow). Options: {available_agents()}",
    )
    parser.add_argument(
        "--team-b",
        default="jev",
        help=f"Strategy for Team B (pink). Default: jev. Options: {available_agents()}",
    )
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument(
        "--show-prompt-only",
        action="store_true",
        help="Print normalized Jev prompt for kickoff and exit",
    )
    parser.add_argument(
        "--verbose-prompt",
        action="store_true",
        help="Print Jev prompt each Jev turn",
    )
    args = parser.parse_args()

    if args.show_prompt_only:
        show_prompt(Team.B)
        return

    team_a = create_agent(args.team_a, Team.A)
    team_b = create_agent(args.team_b, Team.B)

    # Label Jev clearly in prompts when present.
    if isinstance(team_a, JevAgent):
        team_a.acting_label = "Jev"
        team_a.opponent_label = "Opponent"
    if isinstance(team_b, JevAgent):
        team_b.acting_label = "Jev"
        team_b.opponent_label = "Opponent"

    print("=" * 56)
    print(f"Team A ({team_a.name}) vs Team B ({team_b.name})")
    print("First to 3 goals · strategy pattern agents")
    print("=" * 56)

    runner = MatchRunner(team_a=team_a, team_b=team_b, max_turns=args.max_turns)
    # Stream turns manually for nicer logs
    state = runner.env.reset(Team.A)
    team_a.on_match_start(state)
    team_b.on_match_start(state)

    turns = 0
    while not runner.env.is_done() and turns < args.max_turns:
        agent = runner.agent_for(state.current_team)
        if args.verbose_prompt and isinstance(agent, JevAgent):
            print("\n--- Jev prompt ---")
            print(agent.build_prompt(state))
            print("--- end prompt ---\n")

        action = agent.select_action(state)
        old_a, old_b = state.score_a, state.score_b
        state, reward, done, info = runner.env.step(action)
        turns += 1

        print(
            f"Turn {turns:3d} | {agent.name:10s} team {agent.team.value} | "
            f"{action.player_id} dir=({action.direction.x:+.2f},{action.direction.y:+.2f}) "
            f"power={action.power:.2f} | score {state.score_a}-{state.score_b} "
            f"| reward={reward:+.1f}"
        )
        if state.score_a > old_a or state.score_b > old_b:
            print(f"  ⚽ GOAL! {state.score_a}-{state.score_b}")
        if done:
            break

    print()
    print("=" * 56)
    print(f"Final: Team A {state.score_a} - {state.score_b} Team B  ({turns} turns)")
    winner = state.winner(runner.config.goals_to_win)
    if winner:
        print(f"Winner: Team {winner.value}")
    else:
        print("No winner yet (max turns or draw)")
    print("=" * 56)


if __name__ == "__main__":
    main()
