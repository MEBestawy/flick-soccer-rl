"""TensorBoard + CSV metrics logging."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.tb_dir))
        self.csv_path = self.run_dir / "metrics.csv"
        self._fieldnames: Optional[list[str]] = None
        self._start = time.time()

    def log(self, step: int, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.writer.add_scalar(k, v, step)
        row = {"step": step, **values}
        write_header = not self.csv_path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        # Grow columns if needed
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        with self.csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
