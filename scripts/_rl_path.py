"""Shared sys.path bootstrap for RL scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    backend = root / "backend"
    for p in (root, backend):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root
