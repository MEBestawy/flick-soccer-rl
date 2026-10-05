#!/usr/bin/env python3
"""Generate a simple HTML+CSV training report for a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _rl_path import setup_paths

ROOT = setup_paths()

from rl.learning_curve import load_metrics_rows, write_learning_curve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    run = args.run_dir
    report_dir = run / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics_rows(run / "metrics.csv")
    curve_path = write_learning_curve(run)

    # Keep a snapshot copy inside report/ for archival browsing.
    snapshot = report_dir / "learning_curve.html"
    snapshot.write_text(curve_path.read_text(encoding="utf-8"), encoding="utf-8")

    replays = (
        sorted((run / "replays").glob("**/*.json")) if (run / "replays").exists() else []
    )
    replay_links = "".join(
        f'<li><a href="../{r.relative_to(run)}">{r.relative_to(run)}</a></li>'
        for r in replays[:30]
    )

    cfg_path = run / "config.json"
    cfg_snip = cfg_path.read_text()[:2000] if cfg_path.exists() else "{}"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Training report — {run.name}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; background: #f7f5f0; color: #1c1c1c; }}
h1 {{ font-size: 1.8rem; }}
section {{ margin: 1.5rem 0; padding-bottom: 1rem; border-bottom: 1px solid #ccc; }}
a {{ color: #1a5f2a; }}
code {{ font-size: 0.85rem; }}
</style></head><body>
<h1>Training report: {run.name}</h1>
<p>Metrics rows: {len(rows)} · Replays: {len(replays)}</p>
<section>
  <h2>Learning curve</h2>
  <p>Live chart (auto-refreshes during training):
    <a href="../learning_curve.html">learning_curve.html</a>
  </p>
  <p>Snapshot: <a href="learning_curve.html">report/learning_curve.html</a></p>
  <iframe src="learning_curve.html" style="width:100%;height:920px;border:1px solid #ccc;border-radius:8px;background:#fff"></iframe>
</section>
<section><h2>Replays</h2><ul>{replay_links or '<li>None yet</li>'}</ul></section>
<section><h2>Config (truncated)</h2><pre><code>{cfg_snip}</code></pre></section>
</body></html>
"""
    out = report_dir / "index.html"
    out.write_text(html)
    summary = {
        "run": str(run),
        "metric_rows": len(rows),
        "replays": len(replays),
        "report": str(out),
        "learning_curve": str(curve_path),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
