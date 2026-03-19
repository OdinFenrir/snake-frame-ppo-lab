from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.reporting.common import (
        prune_stamped_outputs,
        read_json,
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_PHASE3_COMPARE
except ModuleNotFoundError:
    import sys

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from scripts.reporting.common import (
        prune_stamped_outputs,
        read_json,
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_PHASE3_COMPARE


def _build_html(report: dict[str, Any]) -> str:
    compare = dict(report.get("compare", {})) if isinstance(report.get("compare"), dict) else {}
    left = dict(report.get("left", {})) if isinstance(report.get("left"), dict) else {}
    right = dict(report.get("right", {})) if isinstance(report.get("right"), dict) else {}
    summary = dict(report.get("summary", {})) if isinstance(report.get("summary"), dict) else {}
    checks = list(report.get("checks", [])) if isinstance(report.get("checks"), list) else []
    metric_rows = list(report.get("metric_rows", [])) if isinstance(report.get("metric_rows"), list) else []

    left_curve = list(left.get("eval_curve", [])) if isinstance(left.get("eval_curve"), list) else []
    right_curve = list(right.get("eval_curve", [])) if isinstance(right.get("eval_curve"), list) else []
    l_steps = [safe_int(r.get("step")) for r in left_curve]
    l_reward = [safe_float(r.get("mean_reward")) for r in left_curve]
    l_score = [safe_float(r.get("mean_score")) for r in left_curve]
    r_steps = [safe_int(r.get("step")) for r in right_curve]
    r_reward = [safe_float(r.get("mean_reward")) for r in right_curve]
    r_score = [safe_float(r.get("mean_score")) for r in right_curve]

    left_deaths = dict(left.get("agent", {}).get("deaths_pct", {}))
    right_deaths = dict(right.get("agent", {}).get("deaths_pct", {}))
    death_keys = sorted(set(list(left_deaths.keys()) + list(right_deaths.keys())))
    left_death_pct = [safe_float(left_deaths.get(k)) for k in death_keys]
    right_death_pct = [safe_float(right_deaths.get(k)) for k in death_keys]

    metrics = [str(m.get("metric", "")) for m in metric_rows]
    left_vals = [safe_float(m.get("left_value")) for m in metric_rows]
    right_vals = [safe_float(m.get("right_value")) for m in metric_rows]
    deltas = [safe_float(m.get("delta_right_minus_left")) for m in metric_rows]
    delta_colors = ["#63d18d" if v >= 0 else "#f38b8b" for v in deltas]

    left_art = dict(left.get("artifacts", {}))
    right_art = dict(right.get("artifacts", {}))
    art_keys = ["last_model_bytes", "vecnormalize_bytes", "arbiter_bytes", "tactic_memory_bytes"]
    art_labels = ["last_model", "vecnormalize", "arbiter", "tactic_memory"]
    left_art_mb = [safe_float(left_art.get(k)) / (1024.0 * 1024.0) for k in art_keys]
    right_art_mb = [safe_float(right_art.get(k)) / (1024.0 * 1024.0) for k in art_keys]

    ok_count = sum(1 for c in checks if isinstance(c, dict) and bool(c.get("ok")))
    fail_count = sum(1 for c in checks if isinstance(c, dict) and not bool(c.get("ok")))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "left_exp": str(compare.get("left_experiment", "")),
        "right_exp": str(compare.get("right_experiment", "")),
        "verdict": str(compare.get("verdict", "UNKNOWN")),
        "wins_left": safe_int(summary.get("wins_left")),
        "wins_right": safe_int(summary.get("wins_right")),
        "ties": safe_int(summary.get("ties")),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "metrics": metrics,
        "left_vals": left_vals,
        "right_vals": right_vals,
        "deltas": deltas,
        "delta_colors": delta_colors,
        "l_steps": l_steps,
        "l_reward": l_reward,
        "l_score": l_score,
        "r_steps": r_steps,
        "r_reward": r_reward,
        "r_score": r_score,
        "death_keys": death_keys,
        "left_death_pct": left_death_pct,
        "right_death_pct": right_death_pct,
        "art_labels": art_labels,
        "left_art_mb": left_art_mb,
        "right_art_mb": right_art_mb,
        "checks": checks,
    }

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Model + Agent Compare Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #0f1220;
      color: #e7edf5;
      font-family: Segoe UI, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .card {{
      background: #182236;
      border: 1px solid #31445f;
      border-radius: 10px;
      padding: 10px;
      min-height: 56px;
    }}
    .k {{ color: #98abc4; font-size: 12px; }}
    .v {{
      color: #f6f9fc;
      font-size: 21px;
      font-weight: 700;
      overflow-wrap: anywhere;
      word-break: break-word;
      line-height: 1.1;
    }}
    .plots {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .plot {{
      border: 1px solid #31445f;
      border-radius: 10px;
      min-height: 320px;
      background: #182236;
    }}
    .checks {{
      margin-top: 14px;
      border: 1px solid #31445f;
      border-radius: 10px;
      padding: 10px;
      background: #182236;
    }}
    .ok {{ color: #63d18d; }}
    .bad {{ color: #f38b8b; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Model + Agent Compare Dashboard</h2>
    <div id="cards" class="cards"></div>
    <div class="plots">
      <div id="plot_metrics" class="plot"></div>
      <div id="plot_deltas" class="plot"></div>
      <div id="plot_eval_reward" class="plot"></div>
      <div id="plot_eval_score" class="plot"></div>
      <div id="plot_deaths" class="plot"></div>
      <div id="plot_artifacts" class="plot"></div>
    </div>
    <div class="checks">
      <h3>Checks</h3>
      <div id="checks"></div>
    </div>
  </div>
  <script>
    const d = {json.dumps(payload)};
    const cardRows = [
      ['Left experiment', d.left_exp || 'n/a'],
      ['Right experiment', d.right_exp || 'n/a'],
      ['Verdict', d.verdict],
      ['Wins (L/R/T)', `${{d.wins_left}} / ${{d.wins_right}} / ${{d.ties}}`],
      ['Checks', `${{d.ok_count}} OK / ${{d.fail_count}} FAIL`],
    ];
    const cards = document.getElementById('cards');
    for (const [k, v] of cardRows) {{
      const el = document.createElement('div');
      el.className = 'card';
      el.innerHTML = `<div class="k">${{k}}</div><div class="v">${{v}}</div>`;
      cards.appendChild(el);
    }}

    const baseLayout = {{
      paper_bgcolor: '#182236',
      plot_bgcolor: '#182236',
      font: {{ color: '#e7edf5' }},
      margin: {{ l: 55, r: 20, t: 40, b: 75 }},
      xaxis: {{ gridcolor: '#2b3a54' }},
      yaxis: {{ gridcolor: '#2b3a54' }},
      showlegend: true,
    }};

    Plotly.newPlot('plot_metrics', [
      {{ x: d.metrics, y: d.left_vals, type: 'bar', name: d.left_exp, marker: {{ color: '#6bb8ff' }} }},
      {{ x: d.metrics, y: d.right_vals, type: 'bar', name: d.right_exp, marker: {{ color: '#ffd46b' }} }},
    ], {{ ...baseLayout, barmode: 'group', title: 'Metric Comparison' }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_deltas', [
      {{ x: d.metrics, y: d.deltas, type: 'bar', name: 'right-left', marker: {{ color: d.delta_colors }} }},
    ], {{
      ...baseLayout,
      title: 'Metric Delta (right - left)',
      yaxis: {{ ...baseLayout.yaxis, zeroline: true, zerolinecolor: '#7f8fa3' }},
    }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_eval_reward', [
      {{ x: d.l_steps, y: d.l_reward, mode: 'lines+markers', name: `${{d.left_exp}} mean_reward`, line: {{ color: '#6bb8ff' }} }},
      {{ x: d.r_steps, y: d.r_reward, mode: 'lines+markers', name: `${{d.right_exp}} mean_reward`, line: {{ color: '#ffd46b' }} }},
    ], {{ ...baseLayout, title: 'Eval Mean Reward by Step', xaxis: {{...baseLayout.xaxis, title: 'step'}}, yaxis: {{...baseLayout.yaxis, title: 'mean_reward'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_eval_score', [
      {{ x: d.l_steps, y: d.l_score, mode: 'lines+markers', name: `${{d.left_exp}} mean_score`, line: {{ color: '#63d18d' }} }},
      {{ x: d.r_steps, y: d.r_score, mode: 'lines+markers', name: `${{d.right_exp}} mean_score`, line: {{ color: '#ff8fc6' }} }},
    ], {{ ...baseLayout, title: 'Eval Mean Score by Step', xaxis: {{...baseLayout.xaxis, title: 'step'}}, yaxis: {{...baseLayout.yaxis, title: 'mean_score'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_deaths', [
      {{ x: d.death_keys, y: d.left_death_pct, type: 'bar', name: `${{d.left_exp}} death %`, marker: {{ color: '#4ea0df' }} }},
      {{ x: d.death_keys, y: d.right_death_pct, type: 'bar', name: `${{d.right_exp}} death %`, marker: {{ color: '#df8a4e' }} }},
    ], {{ ...baseLayout, barmode: 'group', title: 'Death Mix (%)', yaxis: {{...baseLayout.yaxis, title: 'percent'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_artifacts', [
      {{ x: d.art_labels, y: d.left_art_mb, type: 'bar', name: `${{d.left_exp}} MB`, marker: {{ color: '#8db2ff' }} }},
      {{ x: d.art_labels, y: d.right_art_mb, type: 'bar', name: `${{d.right_exp}} MB`, marker: {{ color: '#f2b56a' }} }},
    ], {{ ...baseLayout, barmode: 'group', title: 'Artifact Sizes (MB)', yaxis: {{...baseLayout.yaxis, title: 'MB'}} }}, {{responsive: true, displayModeBar: false}});

    const checks = document.getElementById('checks');
    for (const c of d.checks) {{
      const row = document.createElement('div');
      row.className = c.ok ? 'ok' : 'bad';
      row.textContent = `[${{c.ok ? 'OK' : 'FAIL'}}] ${{c.name}} - ${{c.detail}}`;
      checks.appendChild(row);
    }}
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual dashboard for model+agent compare report.")
    parser.add_argument("--in-dir", type=str, default="artifacts/phase3_compare")
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_compare")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()
    try:
        validate_retain_stamped(int(args.retain_stamped))
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc

    root = Path(__file__).resolve().parents[2]
    in_dir = (root / args.in_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    try:
        out_dir = validate_canonical_out_dir(root, REPORT_FAMILY_PHASE3_COMPARE, out_dir)
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(in_dir / "model_agent_compare_latest.json")
    html = _build_html(report)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"
    stamped = out_dir / f"model_agent_compare_dashboard_{ts}.html"
    latest = out_dir / f"model_agent_compare_dashboard_{tag}.html"
    stamped.write_text(html, encoding="utf-8")
    latest.write_text(html, encoding="utf-8")
    prune_stamped_outputs(
        out_dir,
        stem_prefix="model_agent_compare_dashboard",
        suffix=".html",
        retain=int(args.retain_stamped),
    )
    print(f"Wrote: {stamped}")
    print(f"Wrote: {latest}")


if __name__ == "__main__":
    main()
