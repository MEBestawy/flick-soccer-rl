#!/usr/bin/env python3
"""Generate a simple HTML+CSV training report for a run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _rl_path import setup_paths

ROOT = setup_paths()


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def series(rows: list[dict], key: str) -> list[tuple[int, float]]:
    out = []
    for r in rows:
        if key not in r or r[key] in ("", None):
            continue
        try:
            out.append((int(float(r["step"])), float(r[key])))
        except (ValueError, KeyError):
            continue
    return out


def sparkline_svg(points: list[tuple[int, float]], w: int = 640, h: int = 120) -> str:
    if len(points) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1.0

    def px(x: float, y: float) -> str:
        X = (x - xmin) / (xmax - xmin) * (w - 20) + 10
        Y = h - 10 - (y - ymin) / (ymax - ymin) * (h - 20)
        return f"{X:.1f},{Y:.1f}"

    poly = " ".join(px(x, y) for x, y in points)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<polyline fill="none" stroke="#1a5f2a" stroke-width="2" points="{poly}"/>'
        f"</svg>"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    run = args.run_dir
    report_dir = run / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(run / "metrics.csv")
    charts = [
        ("eval/elo", "Elo"),
        ("eval/win_rate_heuristic", "Win rate vs heuristic"),
        ("game/reward", "Episode reward"),
        ("train/entropy", "Entropy"),
        ("perf/env_steps_per_sec", "Env steps/sec"),
        ("eval/goal_diff", "Goal differential"),
    ]

    sections = []
    for key, title in charts:
        pts = series(rows, key)
        if not pts:
            continue
        sections.append(
            f"<section><h2>{title}</h2>{sparkline_svg(pts)}"
            f"<p>Last: {pts[-1][1]:.4f} @ step {pts[-1][0]}</p></section>"
        )

    replays = sorted((run / "replays").glob("**/*.json")) if (run / "replays").exists() else []
    replay_links = "".join(
        f"<li><a href=\"../{r.relative_to(run)}\">{r.relative_to(run)}</a></li>"
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
svg {{ background: #fff; border: 1px solid #ddd; }}
code {{ font-size: 0.85rem; }}
</style></head><body>
<h1>Training report: {run.name}</h1>
<p>Metrics rows: {len(rows)} · Replays: {len(replays)}</p>
{''.join(sections)}
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
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
