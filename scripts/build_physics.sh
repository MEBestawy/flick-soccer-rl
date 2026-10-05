#!/usr/bin/env bash
# Build the Rust flick_physics extension into the active Python env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend/native/flick_physics"

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

if [[ -x "$ROOT/backend/.venv/bin/python" ]]; then
  # Prefer project venv so the extension lands where training runs.
  # shellcheck source=/dev/null
  source "$ROOT/backend/.venv/bin/activate"
fi

if ! command -v maturin >/dev/null 2>&1; then
  python -m pip install maturin
fi

if ! command -v rustc >/dev/null 2>&1; then
  echo "rustc not found. Install from https://rustup.rs" >&2
  exit 1
fi

maturin develop --release
cd "$ROOT/backend"
python -c "from sim.rust_bridge import RUST_AVAILABLE; assert RUST_AVAILABLE; print('flick_physics OK')"
