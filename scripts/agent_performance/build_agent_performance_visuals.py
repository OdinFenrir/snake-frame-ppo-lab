from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / float(len(chunk)))
    return out


def _moving_std(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return [0.0 for _ in values]
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        mean = sum(chunk) / float(len(chunk))
        var = sum((v - mean) ** 2 for v in chunk) / float(len(chunk))
        out.append(var ** 0.5)
    return out


def _delta(values: list[int]) -> list[int]:
    if not values:
        return []
    return [0] + [int(values[i] - values[i - 1]) for i in range(1, len(values))]


def _status_interventions(mean_pct: float) -> str:
    if mean_pct <= 5.0:
        return "GOOD"
    if mean_pct <= 12.0:
        return "CAUTION"
    return "HIGH"


def _status_risk_drift(risk_delta_avg: float) -> str:
    if risk_delta_avg <= 400.0:
        return "STABLE"
    if risk_delta_avg <= 900.0:
        return "AMBER"
    return "DRIFT"


def _build_payload(report: dict[str, Any]) -> dict[str, Any]:
    ep = dict(report.get("episodes", {})) if isinstance(report.get("episodes"), dict) else {}
    ctl = dict(report.get("agent_control", {})) if isinstance(report.get("agent_control"), dict) else {}
    checks = list(report.get("checks", [])) if isinstance(report.get("checks"), list) else []
    rows = list(report.get("rows", [])) if isinstance(report.get("rows"), list) else []

    episodes = [_safe_int(r.get("episode_index")) for r in rows]
    scores = [_safe_int(r.get("score")) for r in rows]
    interventions_pct = [_safe_float(r.get("interventions_pct")) for r in rows]
    decisions_delta = [_safe_int(r.get("decisions_delta")) for r in rows]
    interventions_delta = [_safe_int(r.get("interventions_delta")) for r in rows]
    risk_total = [_safe_int(r.get("risk_total")) for r in rows]
    death_reason = [str(r.get("death_reason", "other")) for r in rows]
    mode = [str(r.get("mode", "unknown")) for r in rows]

    score_delta = _delta(scores)
    risk_delta = _delta(risk_total)
    score_delta_colors = ["#63d18d" if v >= 0 else "#f38b8b" for v in score_delta]
    score_roll = _rolling_mean([float(s) for s in scores], 5)
    score_std = _moving_std([float(s) for s in scores], 5)
    score_roll_up = [score_roll[i] + score_std[i] for i in range(len(score_roll))]
    score_roll_dn = [score_roll[i] - score_std[i] for i in range(len(score_roll))]
    int_roll = _rolling_mean(interventions_pct, 5)

    score_per_1k_decisions: list[float] = []
    interventions_per_1k_decisions: list[float] = []
    for i in range(len(scores)):
        d = max(1, decisions_delta[i])
        score_per_1k_decisions.append((float(scores[i]) / float(d)) * 1000.0)
        interventions_per_1k_decisions.append((float(interventions_delta[i]) / float(d)) * 1000.0)

    mean_interventions = (sum(interventions_pct) / float(len(interventions_pct))) if interventions_pct else 0.0
    hi_deltas = [score_delta[i] for i in range(len(score_delta)) if interventions_pct[i] >= mean_interventions]
    lo_deltas = [score_delta[i] for i in range(len(score_delta)) if interventions_pct[i] < mean_interventions]
    hi_delta_mean = (sum(hi_deltas) / float(len(hi_deltas))) if hi_deltas else 0.0
    lo_delta_mean = (sum(lo_deltas) / float(len(lo_deltas))) if lo_deltas else 0.0

    risk_delta_avg = (sum(risk_delta[1:]) / float(max(1, len(risk_delta) - 1))) if risk_delta else 0.0
    intervention_status = _status_interventions(_safe_float(ctl.get("interventions_pct_mean")))
    risk_status = _status_risk_drift(risk_delta_avg)
    quality_status = "HELPING" if hi_delta_mean >= lo_delta_mean else "NOISY"

    ok_count = sum(1 for c in checks if isinstance(c, dict) and bool(c.get("ok")))
    fail_count = sum(1 for c in checks if isinstance(c, dict) and not bool(c.get("ok")))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(report.get("run_id", "")),
        "artifact_dir": str(report.get("artifact_dir", "")),
        "episode_count": _safe_int(ep.get("count")),
        "score_mean": _safe_float(ep.get("score_mean")),
        "score_best": _safe_int(ep.get("score_best")),
        "score_last": _safe_int(ep.get("score_last")),
        "score_trend": str(ep.get("score_trend", "unknown")),
        "interventions_pct_mean": _safe_float(ctl.get("interventions_pct_mean")),
        "interventions_total": _safe_int(ctl.get("interventions_delta_total")),
        "decisions_total": _safe_int(ctl.get("decisions_delta_total")),
        "risk_total_last": _safe_int(ctl.get("risk_total_last")),
        "intervention_status": intervention_status,
        "risk_status": risk_status,
        "quality_status": quality_status,
        "quality_hi_delta_mean": _safe_float(hi_delta_mean),
        "quality_lo_delta_mean": _safe_float(lo_delta_mean),
        "risk_delta_avg": _safe_float(risk_delta_avg),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "checks": checks,
        "series": {
            "episodes": episodes,
            "scores": scores,
            "score_delta": score_delta,
            "score_delta_colors": score_delta_colors,
            "interventions_pct": interventions_pct,
            "decisions_delta": decisions_delta,
            "interventions_delta": interventions_delta,
            "risk_total": risk_total,
            "risk_delta": risk_delta,
            "death_reason": death_reason,
            "mode": mode,
            "score_roll": score_roll,
            "score_roll_up": score_roll_up,
            "score_roll_dn": score_roll_dn,
            "int_roll": int_roll,
            "score_per_1k_decisions": score_per_1k_decisions,
            "interventions_per_1k_decisions": interventions_per_1k_decisions,
        },
    }


def _build_html(report: dict[str, Any]) -> str:
    payload = _build_payload(report)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Agent Performance Dashboard</title>
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
    <h2>Agent Performance Dashboard</h2>
    <div id="cards" class="cards"></div>
    <div class="plots">
      <div id="plot_score" class="plot"></div>
      <div id="plot_score_delta" class="plot"></div>
      <div id="plot_interventions" class="plot"></div>
      <div id="plot_control_load" class="plot"></div>
      <div id="plot_risk" class="plot"></div>
      <div id="plot_efficiency" class="plot"></div>
      <div id="plot_pies" class="plot"></div>
    </div>
    <div class="checks">
      <h3>Checks</h3>
      <div id="checks"></div>
    </div>
  </div>
  <script>
    const d = {json.dumps(payload)};
    const cardRows = [
      ['Run ID', d.run_id || 'n/a'],
      ['Episodes', `${{d.episode_count}}`],
      ['Score mean / best / last', `${{d.score_mean.toFixed(2)}} / ${{d.score_best}} / ${{d.score_last}}`],
      ['Score trend', d.score_trend],
      ['Interventions mean %', `${{d.interventions_pct_mean.toFixed(2)}}%`],
      ['Intervention status', d.intervention_status],
      ['Risk drift status', `${{d.risk_status}} (${{d.risk_delta_avg.toFixed(1)}}/ep)`],
      ['Control quality', `${{d.quality_status}} (hi ${{d.quality_hi_delta_mean.toFixed(1)}} vs low ${{d.quality_lo_delta_mean.toFixed(1)}})`],
      ['Interventions total', `${{d.interventions_total}}`],
      ['Decisions total', `${{d.decisions_total}}`],
      ['Risk total last', `${{d.risk_total_last}}`],
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
      margin: {{ l: 55, r: 20, t: 40, b: 45 }},
      xaxis: {{ gridcolor: '#2b3a54' }},
      yaxis: {{ gridcolor: '#2b3a54' }},
      showlegend: true,
    }};

    Plotly.newPlot('plot_score', [
      {{ x: d.series.episodes, y: d.series.scores, mode: 'lines+markers', name: 'score', line: {{ color: '#f6cb5a' }} }},
      {{ x: d.series.episodes, y: d.series.score_roll, mode: 'lines', name: 'score_roll(5)', line: {{ color: '#5fd5ff', width: 2 }} }},
      {{ x: d.series.episodes, y: d.series.score_roll_up, mode: 'lines', name: 'roll +1std', line: {{ color: '#2e7f98', dash: 'dot' }} }},
      {{ x: d.series.episodes, y: d.series.score_roll_dn, mode: 'lines', name: 'roll -1std', line: {{ color: '#2e7f98', dash: 'dot' }} }},
    ], {{ ...baseLayout, title: 'Score by Episode', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'score'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_score_delta', [
      {{ x: d.series.episodes, y: d.series.score_delta, type: 'bar', name: 'score_delta', marker: {{ color: d.series.score_delta_colors }} }},
    ], {{ ...baseLayout, title: 'Score Delta by Episode', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'delta (current - previous)', zeroline: true, zerolinecolor: '#7f8fa3'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_interventions', [
      {{ x: d.series.episodes, y: d.series.interventions_pct, mode: 'lines+markers', name: 'interventions_pct', line: {{ color: '#65d5ff' }} }},
      {{ x: d.series.episodes, y: d.series.int_roll, mode: 'lines', name: 'interventions_roll(5)', line: {{ color: '#7effd5' }} }},
      {{ x: d.series.episodes, y: d.series.episodes.map(_ => 5), mode: 'lines', name: 'good <=5%', line: {{ color: '#4fcf84', dash: 'dot' }} }},
      {{ x: d.series.episodes, y: d.series.episodes.map(_ => 12), mode: 'lines', name: 'caution <=12%', line: {{ color: '#e0b44d', dash: 'dot' }} }},
    ], {{ ...baseLayout, title: 'Interventions % by Episode', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'interventions %'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_control_load', [
      {{ x: d.series.episodes, y: d.series.decisions_delta, mode: 'lines+markers', name: 'decisions_delta', line: {{ color: '#9be15d' }} }},
      {{ x: d.series.episodes, y: d.series.interventions_delta, mode: 'lines+markers', name: 'interventions_delta', line: {{ color: '#ff8fc6' }} }},
    ], {{ ...baseLayout, title: 'Decision vs Intervention Load', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'count'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_risk', [
      {{ x: d.series.episodes, y: d.series.risk_total, mode: 'lines+markers', name: 'risk_total', line: {{ color: '#ffa66b' }} }},
      {{ x: d.series.episodes, y: d.series.risk_delta, mode: 'lines+markers', name: 'risk_delta_per_episode', line: {{ color: '#ff6565' }} }},
      {{ x: d.series.episodes, y: d.series.episodes.map(_ => 400), mode: 'lines', name: 'stable <=400', line: {{ color: '#4fcf84', dash: 'dot' }} }},
      {{ x: d.series.episodes, y: d.series.episodes.map(_ => 900), mode: 'lines', name: 'amber <=900', line: {{ color: '#e0b44d', dash: 'dot' }} }},
    ], {{ ...baseLayout, title: 'Risk Total + Delta by Episode', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'risk'}} }}, {{responsive: true, displayModeBar: false}});

    Plotly.newPlot('plot_efficiency', [
      {{ x: d.series.episodes, y: d.series.score_per_1k_decisions, mode: 'lines+markers', name: 'score_per_1k_decisions', line: {{ color: '#ffd089' }} }},
      {{ x: d.series.episodes, y: d.series.interventions_per_1k_decisions, mode: 'lines+markers', name: 'interventions_per_1k_decisions', line: {{ color: '#96ffc9' }} }},
    ], {{ ...baseLayout, title: 'Efficiency by Episode', xaxis: {{...baseLayout.xaxis, title: 'episode'}}, yaxis: {{...baseLayout.yaxis, title: 'efficiency'}} }}, {{responsive: true, displayModeBar: false}});

    const reasonCounts = {{}};
    for (const r of d.series.death_reason) {{
      reasonCounts[r] = (reasonCounts[r] || 0) + 1;
    }}
    const modeCounts = {{}};
    for (const m of d.series.mode) {{
      modeCounts[m] = (modeCounts[m] || 0) + 1;
    }}
    Plotly.newPlot('plot_pies', [
      {{ type: 'pie', labels: Object.keys(reasonCounts), values: Object.values(reasonCounts), name: 'death_reason', domain: {{x:[0,0.48], y:[0,1]}} }},
      {{ type: 'pie', labels: Object.keys(modeCounts), values: Object.values(modeCounts), name: 'mode', domain: {{x:[0.52,1], y:[0,1]}} }},
    ], {{ ...baseLayout, title: 'Death Reasons and Control Modes', showlegend: true }}, {{responsive: true, displayModeBar: false}});

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
    parser = argparse.ArgumentParser(description="Build visual dashboard for agent performance report.")
    parser.add_argument("--in-dir", type=str, default="artifacts/agent_performance")
    parser.add_argument("--out-dir", type=str, default="artifacts/agent_performance")
    parser.add_argument("--tag", type=str, default="latest")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_dir = (root / args.in_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _read_json(in_dir / "agent_performance_latest.json")
    html = _build_html(report)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"
    stamped = out_dir / f"agent_performance_dashboard_{ts}.html"
    latest = out_dir / f"agent_performance_dashboard_{tag}.html"
    stamped.write_text(html, encoding="utf-8")
    latest.write_text(html, encoding="utf-8")
    print(f"Wrote: {stamped}")
    print(f"Wrote: {latest}")


if __name__ == "__main__":
    main()

