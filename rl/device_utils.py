"""Apple Silicon device selection."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS unavailable, falling back to CPU")
        return torch.device("cpu")
    # auto: prefer CPU for tiny nets + CPU sim unless user forces mps
    # Empirically CPU often wins for small PPO batches on M-series.
    return torch.device("cpu")
