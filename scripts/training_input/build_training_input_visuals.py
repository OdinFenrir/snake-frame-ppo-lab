from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

_STAMP_TOKEN_RE = re.compile(r"^\d{8}_\d{6}$")


def _prune_stamped_outputs(out_dir: Path, *, stem_prefix: str, suffix: str, retain: int) -> None:
    keep = max(0, int(retain))
    prefix = f"{stem_prefix}_"
    candidates: list[Path] = []
    for path in out_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : len(name) - len(suffix)]
        if middle == "latest":
            continue
        if not _STAMP_TOKEN_RE.fullmatch(middle):
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in candidates[keep:]:
        stale.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _build_deltas(labels: list[str], values: list[float]) -> tuple[list[str], list[float]]:
    if len(labels) < 2 or len(values) < 2:
        return [], []
    out_labels: list[str] = []
    out_vals: list[float] = []
    for i in range(1, min(len(labels), len(values))):
        out_labels.append(f"{labels[i - 1]}->{labels[i]}")
        out_vals.append(float(values[i] - values[i - 1]))
    return out_labels, out_vals


def _stability_status(max_drift_pct: float) -> str:
    if max_drift_pct <= 15.0:
        return "GREEN"
    if max_drift_pct <= 30.0:
        return "AMBER"
    return "RED"


def _performance_trend(reward_deltas: list[float], score_deltas: list[float]) -> str:
    if not reward_deltas and not score_deltas:
        return "UNKNOWN"
    reward_sum = float(sum(reward_deltas))
    score_sum = float(sum(score_deltas))
    if reward_sum > 0.0 and score_sum > 0.0:
        return "UP"
    if reward_sum < 0.0 and score_sum < 0.0:
        return "DOWN"
    return "MIXED"


def _recommended_action(perf: str, stability: str) -> str:
    if perf == "UP" and stability == "GREEN":
        return "KEEP"
    if perf == "DOWN" and stability in ("AMBER", "RED"):
        return "RETRAIN"
    if perf == "DOWN" and stability == "GREEN":
        return "ROLLBACK_BASELINE"
    if perf == "MIXED" and stability == "RED":
        return "RETRAIN"
    return "KEEP_WATCH"


def _build_html(report: dict[str, Any], timeline: dict[str, Any]) -> str:
    run = dict(report.get("run", {})) if isinstance(report.get("run"), dict) else {}
    contract = dict(report.get("input_contract", {})) if isinstance(report.get("input_contract"), dict) else {}
    vec = dict(report.get("vecnormalize_latest", {})) if isinstance(report.get("vecnormalize_latest"), dict) else {}
    checks = list(report.get("checks", [])) if isinstance(report.get("checks"), list) else []
    tl_rows = list(timeline.get("timeline", [])) if isinstance(timeline.get("timeline"), list) else []
    tl_summary = dict(timeline.get("summary", {})) if isinstance(timeline.get("summary"), dict) else {}

    steps = [_safe_int(r.get("step")) for r in tl_rows]
    step_labels = [f"{int(s/1000)}k" if s >= 1000 else str(s) for s in steps]
    vec_var_avg = [_safe_float(r.get("vec_obs_var_avg")) for r in tl_rows]
    vec_var_max = [_safe_float(r.get("vec_obs_var_max")) for r in tl_rows]
    vec_mean_abs_avg = [_safe_float(r.get("vec_obs_mean_abs_avg")) for r in tl_rows]
    vec_obs_count = [_safe_float(r.get("vec_obs_count")) for r in tl_rows]

    eval_points = [
        (
            _safe_int(r.get("step")),
            _float_or_none(r.get("eval_mean_reward")),
            _float_or_none(r.get("eval_mean_score")),
        )
        for r in tl_rows
    ]
    eval_points = [p for p in eval_points if p[1] is not None]
    eval_steps = [p[0] for p in eval_points]
    eval_step_labels = [f"{int(s/1000)}k" if s >= 1000 else str(s) for s in eval_steps]
    eval_mean_reward = [float(p[1]) for p in eval_points if p[1] is not None]
    eval_mean_score = [float(p[2] or 0.0) for p in eval_points]
    eval_delta_labels, eval_reward_deltas = _build_deltas(eval_step_labels, eval_mean_reward)
    _, eval_score_deltas = _build_deltas(eval_step_labels, eval_mean_score)

    baseline_var = float(vec_var_avg[0]) if vec_var_avg else 0.0
    drift_pct = [
        (100.0 * (v - baseline_var) / baseline_var) if baseline_var > 1e-12 else 0.0
        for v in vec_var_avg
    ]
    max_drift_pct = max(drift_pct) if drift_pct else 0.0
    stability_state = _stability_status(max_drift_pct)
    perf_trend = _performance_trend(eval_reward_deltas, eval_score_deltas)
    recommendation = _recommended_action(perf_trend, stability_state)
    green_upper = [baseline_var * 1.15 for _ in vec_var_avg]
    amber_upper = [baseline_var * 1.30 for _ in vec_var_avg]
    red_upper = [max(max(vec_var_avg) if vec_var_avg else baseline_var, baseline_var * 1.55) for _ in vec_var_avg]

    ok_count = sum(1 for c in checks if isinstance(c, dict) and bool(c.get("ok")))
    fail_count = sum(1 for c in checks if isinstance(c, dict) and not bool(c.get("ok")))
    checks_pretty = [
        {"name": str(c.get("name", "")), "ok": bool(c.get("ok", False)), "detail": str(c.get("detail", ""))}
        for c in checks
        if isinstance(c, dict)
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run.get("run_id", "")),
        "requested_total_timesteps": _safe_int(run.get("requested_total_timesteps")),
        "actual_total_timesteps": _safe_int(run.get("actual_total_timesteps")),
        "env_count": _safe_int(contract.get("env_count")),
        "n_steps": _safe_int(contract.get("n_steps")),
        "batch_size": _safe_int(contract.get("batch_size")),
        "n_epochs": _safe_int(contract.get("n_epochs")),
        "obs_dim": _safe_int(vec.get("obs_dim")),
        "obs_count": _safe_float(vec.get("obs_count")),
        "obs_mean_abs_avg": _safe_float(vec.get("obs_mean_abs_avg")),
        "obs_var_avg": _safe_float(vec.get("obs_var_avg")),
        "timeline_summary": tl_summary,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "checks": checks_pretty,
        "stability": {
            "baseline_var": baseline_var,
            "max_drift_pct": max_drift_pct,
            "status": stability_state,
        },
        "verdict": {
            "performance_trend": perf_trend,
            "stability": stability_state,
            "recommendation": recommendation,
        },
        "series": {
            "steps": steps,
            "step_labels": step_labels,
            "vec_var_avg": vec_var_avg,
            "vec_var_max": vec_var_max,
            "vec_mean_abs_avg": vec_mean_abs_avg,
            "vec_obs_count": vec_obs_count,
            "drift_pct": drift_pct,
            "green_upper": green_upper,
            "amber_upper": amber_upper,
            "red_upper": red_upper,
            "eval_steps": eval_steps,
            "eval_step_labels": eval_step_labels,
            "eval_mean_reward": eval_mean_reward,
            "eval_mean_score": eval_mean_score,
            "eval_delta_labels": eval_delta_labels,
            "eval_reward_deltas": eval_reward_deltas,
            "eval_score_deltas": eval_score_deltas,
        },
    }

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Training Input Dashboard</title>
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
    .v-mono {{
      font-family: Consolas, "Courier New", monospace;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
      font-size: 18px;
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
    <h2>Training Input Dashboard</h2>
    <div id="cards" class="cards"></div>
    <div class="plots">
      <div id="plot_eval" class="plot"></div>
      <div id="plot_eval_delta" class="plot"></div>
      <div id="plot_vec" class="plot"></div>
      <div id="plot_stability" class="plot"></div>
      <div id="plot_count" class="plot"></div>
      <div id="plot_score" class="plot"></div>
    </div>
    <div class="checks">
      <h3>Checks</h3>
      <div id="checks"></div>
    </div>
  </div>
  <script>
    const d = {json.dumps(payload)};
    const cardRows = [
      ['Run ID', d.run_id ? (d.run_id.length > 18 ? d.run_id.slice(0, 10) + '...' + d.run_id.slice(-6) : d.run_id) : 'n/a'],
      ['Timesteps', `${{d.actual_total_timesteps}} / ${{d.requested_total_timesteps}}`],
      ['Rollout', `${{d.env_count}} x ${{d.n_steps}} = ${{d.env_count * d.n_steps}}`],
      ['Batch / Epochs', `${{d.batch_size}} / ${{d.n_epochs}}`],
      ['Obs Dim', `${{d.obs_dim}}`],
      ['Vec Obs Count', `${{d.obs_count.toFixed(2)}}`],
      ['Vec Mean|abs| avg', `${{d.obs_mean_abs_avg.toFixed(6)}}`],
      ['Vec Var avg', `${{d.obs_var_avg.toFixed(6)}}`],
      ['Checks', `${{d.ok_count}} OK / ${{d.fail_count}} FAIL`],
      ['Eval Points', `${{d.timeline_summary.eval_trace_points || 0}}`],
      ['Vec Checkpoints', `${{d.timeline_summary.vec_checkpoint_count || 0}}`],
      ['Stability', `${{d.stability.status}} (${{d.stability.max_drift_pct.toFixed(1)}}%)`],
      ['Perf Trend', d.verdict.performance_trend],
      ['Action', d.verdict.recommendation],
    ];

    const cards = document.getElementById('cards');
    for (const [k, v] of cardRows) {{
      const el = document.createElement('div');
      el.className = 'card';
      const valueClass = (k === 'Run ID') ? 'v v-mono' : 'v';
      el.innerHTML = `<div class="k">${{k}}</div><div class="${{valueClass}}" title="${{String(v)}}">${{v}}</div>`;
      cards.appendChild(el);
    }}

    const baseLayout = {{
      paper_bgcolor: '#182236',
      plot_bgcolor: '#182236',
      font: {{ color: '#e7edf5' }},
      margin: {{ l: 55, r: 20, t: 40, b: 45 }},
      xaxis: {{ gridcolor: '#2b3a54', title: 'checkpoint' }},
      yaxis: {{ gridcolor: '#2b3a54' }},
      showlegend: true,
    }};

    Plotly.newPlot('plot_eval', [
      {{ x: d.series.eval_step_labels, y: d.series.eval_mean_reward, mode: 'lines+markers', name: 'eval_mean_reward', line: {{ color: '#f2d36b' }} }},
    ], {{ ...baseLayout, title: 'Eval Mean Reward by Eval Checkpoint', yaxis: {{ ...baseLayout.yaxis, title: 'mean_reward' }} }}, {{ responsive: true, displayModeBar: false }});

    const rewardDeltaPct = d.series.eval_reward_deltas.map((v, i) => {{
      const prev = d.series.eval_mean_reward[i] || 0.0;
      if (Math.abs(prev) < 1e-9) return 0.0;
      return (100.0 * v) / prev;
    }});
    const scoreDeltaPct = d.series.eval_score_deltas.map((v, i) => {{
      const prev = d.series.eval_mean_score[i] || 0.0;
      if (Math.abs(prev) < 1e-9) return 0.0;
      return (100.0 * v) / prev;
    }});
    const deltaColors = rewardDeltaPct.map(v => v >= 0 ? '#63d18d' : '#f38b8b');
    Plotly.newPlot('plot_eval_delta', [
      {{
        x: d.series.eval_delta_labels,
        y: rewardDeltaPct,
        type: 'bar',
        name: 'delta_reward_%',
        marker: {{ color: deltaColors }},
        customdata: d.series.eval_reward_deltas,
        hovertemplate: 'reward delta: %{{customdata:.3f}}<br>reward delta %: %{{y:.3f}}%<extra></extra>',
      }},
      {{
        x: d.series.eval_delta_labels,
        y: scoreDeltaPct,
        type: 'bar',
        name: 'delta_score_%',
        marker: {{ color: '#5db7e8' }},
        opacity: 0.75,
        customdata: d.series.eval_score_deltas,
        hovertemplate: 'score delta: %{{customdata:.3f}}<br>score delta %: %{{y:.3f}}%<extra></extra>',
      }},
    ], {{
      ...baseLayout,
      barmode: 'group',
      title: 'Checkpoint-to-Checkpoint Delta (Improvement/Regression)',
      yaxis: {{ ...baseLayout.yaxis, title: 'delta % (current - previous)', ticksuffix: '%', zeroline: true, zerolinecolor: '#7f8fa3' }},
    }}, {{ responsive: true, displayModeBar: false }});

    Plotly.newPlot('plot_vec', [
      {{ x: d.series.step_labels, y: d.series.vec_var_avg, mode: 'lines+markers', name: 'vec_var_avg', line: {{ color: '#59c5e8' }} }},
      {{ x: d.series.step_labels, y: d.series.vec_var_max, mode: 'lines+markers', name: 'vec_var_max', line: {{ color: '#9be15d' }} }},
      {{ x: d.series.step_labels, y: d.series.vec_mean_abs_avg, mode: 'lines+markers', name: 'vec_mean_abs_avg', line: {{ color: '#f08bb4' }} }},
    ], {{ ...baseLayout, title: 'VecNormalize Drift by Step', yaxis: {{ ...baseLayout.yaxis, title: 'value' }} }}, {{ responsive: true, displayModeBar: false }});

    Plotly.newPlot('plot_stability', [
      {{ x: d.series.step_labels, y: d.series.green_upper, mode: 'lines', name: 'green upper (+15%)', line: {{ color: 'rgba(91,209,133,0.8)', width: 1 }} }},
      {{ x: d.series.step_labels, y: d.series.amber_upper, mode: 'lines', name: 'amber upper (+30%)', line: {{ color: 'rgba(245,201,95,0.8)', width: 1 }} }},
      {{ x: d.series.step_labels, y: d.series.red_upper, mode: 'lines', name: 'red cap', line: {{ color: 'rgba(243,139,139,0.6)', width: 1 }} }},
      {{ x: d.series.step_labels, y: d.series.vec_var_avg, mode: 'lines+markers', name: 'vec_var_avg', line: {{ color: '#59c5e8', width: 3 }} }},
      {{ x: d.series.step_labels, y: d.series.drift_pct, mode: 'lines+markers', name: 'drift_pct', yaxis: 'y2', line: {{ color: '#d7e6f0', dash: 'dot' }} }},
    ], {{
      ...baseLayout,
      title: 'Stability Zone (Vec Var Avg with Drift Risk)',
      yaxis: {{ ...baseLayout.yaxis, title: 'vec_var_avg' }},
      yaxis2: {{
        title: 'drift % vs first checkpoint',
        overlaying: 'y',
        side: 'right',
        gridcolor: '#1f3850',
      }},
      annotations: [{{
        text: `status=${{d.stability.status}} max_drift=${{d.stability.max_drift_pct.toFixed(1)}}%`,
        xref: 'paper', yref: 'paper', x: 0.01, y: 0.98, showarrow: false, font: {{color: '#d9e6ee'}}
      }}],
    }}, {{ responsive: true, displayModeBar: false }});

    Plotly.newPlot('plot_count', [
      {{ x: d.series.step_labels, y: d.series.vec_obs_count, mode: 'lines+markers', name: 'vec_obs_count', line: {{ color: '#63d18d' }} }},
    ], {{ ...baseLayout, title: 'VecNormalize Obs Count by Step', yaxis: {{ ...baseLayout.yaxis, title: 'obs_count' }} }}, {{ responsive: true, displayModeBar: false }});

    Plotly.newPlot('plot_score', [
      {{ x: d.series.eval_step_labels, y: d.series.eval_mean_score, mode: 'lines+markers', name: 'eval_mean_score', line: {{ color: '#ffb067' }} }},
    ], {{ ...baseLayout, title: 'Eval Mean Score by Eval Checkpoint', yaxis: {{ ...baseLayout.yaxis, title: 'mean_score' }} }}, {{ responsive: true, displayModeBar: false }});

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
    parser = argparse.ArgumentParser(description="Build visual dashboard HTML for training-input reports.")
    parser.add_argument("--in-dir", type=str, default="artifacts/training_input")
    parser.add_argument("--out-dir", type=str, default="artifacts/training_input")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_dir = (root / args.in_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _read_json(in_dir / "training_input_latest.json")
    timeline = _read_json(in_dir / "training_input_timeline_latest.json")
    html = _build_html(report=report, timeline=timeline)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"
    stamped = out_dir / f"training_input_dashboard_{ts}.html"
    latest = out_dir / f"training_input_dashboard_{tag}.html"
    stamped.write_text(html, encoding="utf-8")
    latest.write_text(html, encoding="utf-8")
    _prune_stamped_outputs(
        out_dir,
        stem_prefix="training_input_dashboard",
        suffix=".html",
        retain=int(args.retain_stamped),
    )
    print(f"Wrote: {stamped}")
    print(f"Wrote: {latest}")


if __name__ == "__main__":
    main()
