#!/usr/bin/env python3
"""
Play a random game and display results.

Useful for testing and demonstration.
"""

import sys
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sim import HeadlessEnv, Team, GamePhase


def play_random_game(
    verbose: bool = True,
    max_turns: int = 200,
    show_ascii: bool = False
) -> dict:
    """
    Play a random game.
    
    Args:
        verbose: Print turn-by-turn updates
        max_turns: Maximum turns before stopping
        show_ascii: Show ASCII representation each turn
    
    Returns:
        Game statistics
    """
    env = HeadlessEnv()
    state = env.reset()
    
    if verbose:
        print("=" * 50)
        print("Starting Random Game")
        print("=" * 50)
        print(f"Team A (Yellow) vs Team B (Pink)")
        print(f"First to {env.config.goals_to_win} goals wins")
        print()
    
    turn = 0
    goals_a = []
    goals_b = []
    
    while not env.is_done() and turn < max_turns:
        current_team = state.current_team
        
        action = env.random_action()
        if action is None:
            if verbose:
                print("No valid actions available!")
            break
        
        old_score_a = state.score_a
        old_score_b = state.score_b
        
        state, reward, done, info = env.step(action)
        turn += 1
        
        if verbose:
            # Check for goals
            if state.score_a > old_score_a:
                goals_a.append(turn)
                print(f"⚽ GOAL! Team A scores! ({state.score_a}-{state.score_b})")
            if state.score_b > old_score_b:
                goals_b.append(turn)
                print(f"⚽ GOAL! Team B scores! ({state.score_a}-{state.score_b})")
            
            if turn % 10 == 0:
                print(f"Turn {turn}: Score {state.score_a}-{state.score_b}")
            
            if show_ascii and turn % 5 == 0:
                print(env.render_ascii())
                print()
    
    # Final results
    stats = {
        "turns": turn,
        "score_a": state.score_a,
        "score_b": state.score_b,
        "winner": None,
        "goals_a_turns": goals_a,
        "goals_b_turns": goals_b,
    }
    
    if state.score_a > state.score_b:
        stats["winner"] = "A"
    elif state.score_b > state.score_a:
        stats["winner"] = "B"
    
    if verbose:
        print()
        print("=" * 50)
        print("Game Over!")
        print("=" * 50)
        print(f"Final Score: Team A {state.score_a} - {state.score_b} Team B")
        
        if stats["winner"]:
            print(f"Winner: Team {stats['winner']}!")
        else:
            print("Result: Draw")
        
        print(f"Total Turns: {turn}")
        print()
        
        if goals_a:
            print(f"Team A goals on turns: {goals_a}")
        if goals_b:
            print(f"Team B goals on turns: {goals_b}")
    
    return stats


def play_multiple_games(count: int, verbose: bool = False) -> dict:
    """Play multiple games and aggregate statistics."""
    print(f"Playing {count} random games...")
    
    results = {
        "total_games": count,
        "team_a_wins": 0,
        "team_b_wins": 0,
        "draws": 0,
        "total_turns": 0,
        "total_goals": 0,
    }
    
    for i in range(count):
        stats = play_random_game(verbose=verbose)
        
        results["total_turns"] += stats["turns"]
        results["total_goals"] += stats["score_a"] + stats["score_b"]
        
        if stats["winner"] == "A":
            results["team_a_wins"] += 1
        elif stats["winner"] == "B":
            results["team_b_wins"] += 1
        else:
            results["draws"] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{count} games")
    
    results["avg_turns"] = results["total_turns"] / count
    results["avg_goals"] = results["total_goals"] / count
    
    print()
    print("=" * 50)
    print(f"Results from {count} games:")
    print("=" * 50)
    print(f"Team A wins: {results['team_a_wins']} ({100*results['team_a_wins']/count:.1f}%)")
    print(f"Team B wins: {results['team_b_wins']} ({100*results['team_b_wins']/count:.1f}%)")
    print(f"Draws:       {results['draws']} ({100*results['draws']/count:.1f}%)")
    print(f"Avg turns:   {results['avg_turns']:.1f}")
    print(f"Avg goals:   {results['avg_goals']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Play random soccer games")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="Number of games to play")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--ascii", action="store_true",
                        help="Show ASCII representation")
    parser.add_argument("--max-turns", type=int, default=200,
                        help="Maximum turns per game")
    
    args = parser.parse_args()
    
    if args.count == 1:
        play_random_game(
            verbose=True,
            max_turns=args.max_turns,
            show_ascii=args.ascii
        )
    else:
        play_multiple_games(args.count, verbose=args.verbose)


if __name__ == "__main__":
    main()
