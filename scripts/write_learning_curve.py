#!/usr/bin/env python3
"""Generate / refresh the live learning-curve HTML for a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _rl_path import setup_paths

ROOT = setup_paths()

from rl.learning_curve import write_learning_curve


def main() -> None:
    p = argparse.ArgumentParser(description="Write learning_curve.html for a run")
    p.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Run directory containing metrics.csv (default: newest under runs/)",
    )
    p.add_argument("--refresh-seconds", type=int, default=10)
    args = p.parse_args()

    run = args.run_dir
    if run is None:
        runs = sorted(
            (ROOT / "runs").glob("*"),
            key=lambda d: (d / "metrics.csv").stat().st_mtime
            if (d / "metrics.csv").exists()
            else 0,
            reverse=True,
        )
        runs = [d for d in runs if d.is_dir() and (d / "metrics.csv").exists()]
        if not runs:
            raise SystemExit("No runs with metrics.csv found")
        run = runs[0]

    out = write_learning_curve(run, refresh_seconds=args.refresh_seconds)
    print(json.dumps({"run": str(run), "learning_curve": str(out)}, indent=2))


if __name__ == "__main__":
    main()
