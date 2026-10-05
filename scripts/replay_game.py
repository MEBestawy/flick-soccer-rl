#!/usr/bin/env python3
"""
Replay a saved game from JSON.

Can also generate sample replays for testing.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sim import GameSimulator, FlickAction, Vec2, Team
from sim.serialization import (
    serialize_state, serialize_action, serialize_frame, serialize_event,
    deserialize_state, deserialize_action, serialize_replay
)


def generate_sample_replay(output_path: Optional[str] = None) -> dict:
    """
    Generate a sample replay by playing a short game.
    
    Returns:
        Replay data as dict
    """
    print("Generating sample replay...")
    
    sim = GameSimulator()
    initial_state = sim.new_game()
    
    states = [initial_state.clone()]
    actions = []
    all_frames = []
    all_events = []
    
    # Play a few turns
    for turn in range(10):
        controllable = sim.state.get_controllable_players()
        if not controllable:
            break
        
        # Make a somewhat directed action
        player = controllable[0]
        
        # Aim toward ball
        ball_pos = sim.state.ball.position
        player_pos = sim.state.get_player(player.id).position
        direction = ball_pos - player_pos
        
        if direction.length() < 0.1:
            direction = Vec2(1.0, 0.0)
        
        action = FlickAction(
            player_id=player.id,
            direction=direction,
            power=0.5 + (turn % 3) * 0.2
        )
        
        result = sim.execute_action(action, capture_frames=True)
        
        if result.success:
            actions.append(action)
            states.append(result.final_state.clone())
            all_frames.append(result.frames or [])
            all_events.extend(result.events or [])
        
        if sim.state.phase.name == "GAME_OVER":
            break
    
    replay = serialize_replay(states, actions, all_frames, all_events)
    replay["metadata"] = {
        "turns": len(actions),
        "final_score": f"{states[-1].score_a}-{states[-1].score_b}",
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(replay, f, indent=2)
        print(f"Saved replay to {output_path}")
    
    return replay


def load_replay(path: str) -> dict:
    """Load replay from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def print_replay_info(replay: dict) -> None:
    """Print information about a replay."""
    print("=" * 50)
    print("Replay Information")
    print("=" * 50)
    
    print(f"Version: {replay.get('version', 'unknown')}")
    
    if "metadata" in replay:
        meta = replay["metadata"]
        print(f"Turns: {meta.get('turns', 'unknown')}")
        print(f"Final Score: {meta.get('final_score', 'unknown')}")
    
    print(f"Actions: {len(replay.get('actions', []))}")
    
    total_frames = sum(len(f) for f in replay.get("frames", []))
    print(f"Total Frames: {total_frames}")
    
    print(f"Events: {len(replay.get('events', []))}")
    
    # Initial state info
    if replay.get("initial_state"):
        init = replay["initial_state"]
        print(f"\nInitial State:")
        print(f"  Current Team: {init['current_team']}")
        print(f"  Players: {len(init['players'])}")
    
    # Final state info
    if replay.get("final_state"):
        final = replay["final_state"]
        print(f"\nFinal State:")
        print(f"  Phase: {final['phase']}")
        print(f"  Score: {final['score_a']}-{final['score_b']}")
        print(f"  Turn: {final['turn_number']}")


def replay_to_console(replay: dict, detailed: bool = False) -> None:
    """
    Replay game to console.
    
    Args:
        replay: Replay data
        detailed: Show detailed frame information
    """
    print("\n" + "=" * 50)
    print("Replaying Game")
    print("=" * 50 + "\n")
    
    actions = replay.get("actions", [])
    frames_by_turn = replay.get("frames", [])
    
    for i, action in enumerate(actions):
        print(f"Turn {i+1}: Player {action['player_id']} flicks")
        print(f"  Direction: ({action['direction']['x']:.2f}, {action['direction']['y']:.2f})")
        print(f"  Power: {action['power']:.2f}")
        
        if detailed and i < len(frames_by_turn):
            turn_frames = frames_by_turn[i]
            print(f"  Frames: {len(turn_frames)}")
            
            if turn_frames:
                first = turn_frames[0]
                last = turn_frames[-1]
                print(f"  Ball start: ({first['ball_position']['x']:.1f}, {first['ball_position']['y']:.1f})")
                print(f"  Ball end: ({last['ball_position']['x']:.1f}, {last['ball_position']['y']:.1f})")
        
        print()
    
    if replay.get("final_state"):
        final = replay["final_state"]
        print(f"Final Score: {final['score_a']}-{final['score_b']}")


def verify_replay(replay: dict) -> bool:
    """
    Verify replay by re-simulating and comparing results.
    
    Returns:
        True if replay is valid and deterministic
    """
    print("\nVerifying replay determinism...")
    
    if not replay.get("initial_state") or not replay.get("actions"):
        print("  Error: Missing initial state or actions")
        return False
    
    # Re-simulate
    sim = GameSimulator()
    sim.new_game()
    
    actions = [deserialize_action(a) for a in replay["actions"]]
    
    for i, action in enumerate(actions):
        result = sim.execute_action(action, capture_frames=False)
        if not result.success:
            print(f"  Error: Action {i+1} failed: {result.error_message}")
            return False
    
    # Compare final states
    expected = replay.get("final_state")
    actual = sim.state
    
    if expected:
        if actual.score_a != expected["score_a"]:
            print(f"  Error: Score A mismatch: {actual.score_a} vs {expected['score_a']}")
            return False
        if actual.score_b != expected["score_b"]:
            print(f"  Error: Score B mismatch: {actual.score_b} vs {expected['score_b']}")
            return False
        
        # Check ball position
        expected_ball = expected["ball"]["position"]
        if abs(actual.ball.position.x - expected_ball["x"]) > 0.01:
            print(f"  Error: Ball X mismatch")
            return False
    
    print("  Replay verified successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Replay soccer games")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate sample replay")
    gen_parser.add_argument("-o", "--output", help="Output file path")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show replay info")
    info_parser.add_argument("file", help="Replay file path")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play replay to console")
    play_parser.add_argument("file", help="Replay file path")
    play_parser.add_argument("-d", "--detailed", action="store_true",
                             help="Show detailed frame info")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify replay determinism")
    verify_parser.add_argument("file", help="Replay file path")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        output = args.output or "sample_replay.json"
        generate_sample_replay(output)
    
    elif args.command == "info":
        replay = load_replay(args.file)
        print_replay_info(replay)
    
    elif args.command == "play":
        replay = load_replay(args.file)
        replay_to_console(replay, detailed=args.detailed)
    
    elif args.command == "verify":
        replay = load_replay(args.file)
        success = verify_replay(replay)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
