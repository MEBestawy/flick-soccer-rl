"""Live learning-curve HTML regenerated from metrics.csv during training."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

Curve = Tuple[str, str, str]  # (metrics key, title, stroke color)

DEFAULT_CURVES: Sequence[Curve] = (
    ("eval/elo", "Eval Elo vs heuristic", "#1a5f2a"),
    ("eval/win_rate_heuristic", "Eval win rate vs heuristic", "#0b6e4f"),
    ("eval/goal_diff", "Eval goal differential", "#2a6f97"),
    ("game/reward", "Train episode reward", "#9b2226"),
    ("game/win_rate", "Train win rate", "#bb3e03"),
    ("train/entropy", "Policy entropy", "#5c4d7a"),
    ("perf/env_steps_per_sec", "Env steps / sec", "#415a77"),
)


def load_metrics_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series(rows: Iterable[dict], key: str) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in rows:
        if key not in r or r[key] in ("", None):
            continue
        try:
            out.append((int(float(r["step"])), float(r[key])))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def sparkline_svg(
    points: List[Tuple[int, float]],
    *,
    stroke: str = "#1a5f2a",
    w: int = 720,
    h: int = 160,
) -> str:
    if len(points) < 2:
        return (
            f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<text x="12" y="{h // 2}" fill="#888" font-size="14">'
            f"Need more points…</text></svg>"
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1.0

    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 28
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def px(x: float, y: float) -> str:
        X = pad_l + (x - xmin) / (xmax - xmin) * plot_w
        Y = pad_t + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h
        return f"{X:.1f},{Y:.1f}"

    poly = " ".join(px(x, y) for x, y in points)
    y0 = pad_t + plot_h
    x0 = pad_l
    grid = (
        f'<line x1="{x0}" y1="{pad_t}" x2="{x0}" y2="{y0}" stroke="#ddd"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#ddd"/>'
    )
    labels = (
        f'<text x="{x0}" y="{h - 8}" fill="#666" font-size="11">{xmin:,}</text>'
        f'<text x="{x0 + plot_w}" y="{h - 8}" fill="#666" font-size="11" '
        f'text-anchor="end">{xmax:,}</text>'
        f'<text x="8" y="{pad_t + 10}" fill="#666" font-size="11">{ymax:.3g}</text>'
        f'<text x="8" y="{y0}" fill="#666" font-size="11">{ymin:.3g}</text>'
    )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"{grid}{labels}"
        f'<polyline fill="none" stroke="{stroke}" stroke-width="2.2" '
        f'points="{poly}"/>'
        f"</svg>"
    )


def render_learning_curve_html(
    run_name: str,
    rows: List[dict],
    *,
    curves: Sequence[Curve] = DEFAULT_CURVES,
    refresh_seconds: int = 10,
) -> str:
    sections: List[str] = []
    for key, title, color in curves:
        pts = series(rows, key)
        if not pts:
            continue
        last_step, last_val = pts[-1]
        sections.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f"{sparkline_svg(pts, stroke=color)}"
            f"<p class='meta'>n={len(pts)} · last {last_val:.4f} @ step {last_step:,} "
            f"<code>{html.escape(key)}</code></p>"
            "</section>"
        )

    body = "".join(sections) or (
        "<p class='empty'>Waiting for metrics… training will fill this page.</p>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="{int(refresh_seconds)}"/>
  <title>Learning curve — {html.escape(run_name)}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --ink: #1c1c1c;
      --muted: #5c5c5c;
      --card: #ffffff;
      --line: #d8d2c4;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      background: radial-gradient(1200px 600px at 10% -10%, #fff8e7, var(--bg));
      color: var(--ink);
    }}
    main {{ max-width: 860px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
    h1 {{ font-size: 1.85rem; margin: 0 0 0.35rem; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); margin: 0 0 1.5rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem 1rem 0.75rem;
      margin: 0 0 1rem;
      box-shadow: 0 1px 0 rgba(0,0,0,0.03);
    }}
    h2 {{ font-size: 1.05rem; margin: 0 0 0.6rem; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin: 0.55rem 0 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; }}
    .empty {{ color: var(--muted); }}
    .badge {{
      display: inline-block;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #3d5a40;
      background: #e4efdf;
      border: 1px solid #c5d6bf;
      border-radius: 999px;
      padding: 0.2rem 0.55rem;
      margin-bottom: 0.75rem;
    }}
  </style>
</head>
<body>
  <main>
    <div class="badge">Live · auto-refresh {int(refresh_seconds)}s</div>
    <h1>Learning curve</h1>
    <p class="sub">{html.escape(run_name)} · {len(rows)} metric rows</p>
    {body}
  </main>
</body>
</html>
"""


def write_learning_curve(
    run_dir: Path,
    *,
    refresh_seconds: int = 10,
    curves: Sequence[Curve] = DEFAULT_CURVES,
) -> Path:
    """
    Recompute `learning_curve.html` from `metrics.csv`.

    Safe to call after every metrics flush; cheap for typical PPO cadence.
    """
    run_dir = Path(run_dir)
    rows = load_metrics_rows(run_dir / "metrics.csv")
    out = run_dir / "learning_curve.html"
    out.write_text(
        render_learning_curve_html(
            run_dir.name,
            rows,
            curves=curves,
            refresh_seconds=refresh_seconds,
        ),
        encoding="utf-8",
    )
    return out
