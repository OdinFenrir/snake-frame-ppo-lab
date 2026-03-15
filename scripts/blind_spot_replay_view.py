from __future__ import annotations

import argparse
import json
from pathlib import Path
import webbrowser
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _norm_points(value: Any) -> list[list[int]]:
    out: list[list[int]] = []
    for item in list(value or []):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        out.append([_safe_int(item[0]), _safe_int(item[1])])
    return out


def _infer_board_cells(window_rows: list[dict[str, Any]], default: int = 20) -> int:
    max_coord = -1
    for row in window_rows:
        for key in ("snake_before", "snake_after"):
            for x, y in _norm_points(row.get(key)):
                max_coord = max(max_coord, int(x), int(y))
        for key in ("food_before", "food_after", "head_before", "head_after"):
            p = row.get(key)
            if isinstance(p, (list, tuple)) and len(p) == 2:
                max_coord = max(max_coord, _safe_int(p[0]), _safe_int(p[1]))
    if max_coord < 0:
        return int(default)
    return max(int(default), int(max_coord + 1))


def normalize_window_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(window_rows or []):
        if not isinstance(row, dict):
            continue
        snake_before = _norm_points(row.get("snake_before"))
        snake_after = _norm_points(row.get("snake_after"))
        head_before = row.get("head_before")
        head_after = row.get("head_after")
        if not snake_before and isinstance(head_before, (list, tuple)) and len(head_before) == 2:
            snake_before = [[_safe_int(head_before[0]), _safe_int(head_before[1])]]
        if not snake_after and isinstance(head_after, (list, tuple)) and len(head_after) == 2:
            snake_after = [[_safe_int(head_after[0]), _safe_int(head_after[1])]]
        out.append(
            {
                "step": _safe_int(row.get("step"), 0),
                "score_before": _safe_int(row.get("score_before"), 0),
                "score_after": _safe_int(row.get("score_after"), 0),
                "predicted_confidence": float(row.get("predicted_confidence") or 0.0),
                "mode": str(row.get("mode", "")),
                "switch_reason": str(row.get("switch_reason", "")),
                "override_used": bool(row.get("override_used", False)),
                "steps_until_death": _safe_int(row.get("steps_until_death"), 0),
                "game_over": bool(row.get("game_over", False)),
                "death_reason": str(row.get("death_reason", "")),
                "snake_before": snake_before,
                "snake_after": snake_after,
                "food_before": _norm_points([row.get("food_before")])[:1],
                "food_after": _norm_points([row.get("food_after")])[:1],
                "head_before": _norm_points([head_before])[:1],
                "head_after": _norm_points([head_after])[:1],
            }
        )
    return out


def build_payload(report: dict[str, Any]) -> dict[str, Any]:
    spots: list[dict[str, Any]] = []
    for spot in list(report.get("blind_spots") or []):
        if not isinstance(spot, dict):
            continue
        window_rows = normalize_window_rows(list(spot.get("window_rows") or []))
        spots.append(
            {
                "seed": _safe_int(spot.get("seed"), -1),
                "trace_file": str(spot.get("trace_file", "")),
                "index": _safe_int(spot.get("index"), 0),
                "step": _safe_int(spot.get("step"), 0),
                "predicted_confidence": float(spot.get("predicted_confidence") or 0.0),
                "steps_until_death": _safe_int(spot.get("steps_until_death"), 0),
                "mode": str(spot.get("mode", "")),
                "switch_reason": str(spot.get("switch_reason", "")),
                "override_used": bool(spot.get("override_used", False)),
                "score_before": _safe_int(spot.get("score_before"), 0),
                "score_after": _safe_int(spot.get("score_after"), 0),
                "board_cells": _infer_board_cells(window_rows),
                "window_rows": window_rows,
            }
        )
    summary = dict(report.get("summary") or {})
    return {"summary": summary, "spots": spots}


def build_html(payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Blind Spot Replay</title>
  <style>
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #0b1220; color: #e5eef9; font-family: Segoe UI, Arial, sans-serif; }}
    #wrap {{ display: grid; grid-template-columns: 360px 1fr; width: 100%; height: 100%; }}
    #side {{ border-right: 1px solid #233247; padding: 12px; overflow: auto; background: #0a1525; }}
    #main {{ display: grid; grid-template-rows: auto 1fr auto; height: 100%; }}
    #toolbar {{ padding: 10px 12px; border-bottom: 1px solid #233247; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    #canvasWrap {{ display: grid; place-items: center; padding: 10px; }}
    #boardCanvas {{ background: #07101c; border: 1px solid #2b3e55; }}
    #status {{ border-top: 1px solid #233247; padding: 10px 12px; font-size: 13px; color: #b8cbe2; }}
    .lbl {{ color: #8fb3d9; margin-right: 6px; }}
    .row {{ margin-bottom: 8px; font-size: 13px; }}
    select, input, button {{ background: #0f2238; color: #e5eef9; border: 1px solid #2b3e55; border-radius: 6px; padding: 6px; }}
    button {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="side">
      <div class="row"><span class="lbl">Blind spots:</span><span id="spotCount"></span></div>
      <div class="row"><span class="lbl">Rows scanned:</span><span id="rowsScanned"></span></div>
      <div class="row"><span class="lbl">Min confidence:</span><span id="minConf"></span></div>
      <div class="row"><span class="lbl">Max steps-to-death:</span><span id="maxSteps"></span></div>
      <div class="row"><span class="lbl">Replay window:</span><span id="replayWindow"></span></div>
      <hr style="border-color:#233247; border-width:1px 0 0 0;">
      <div class="row"><span class="lbl">Spot:</span><select id="spotSelect"></select></div>
      <div class="row"><span class="lbl">Seed:</span><span id="seed"></span></div>
      <div class="row"><span class="lbl">Conf:</span><span id="conf"></span></div>
      <div class="row"><span class="lbl">Steps-until-death:</span><span id="sud"></span></div>
      <div class="row"><span class="lbl">Mode:</span><span id="mode"></span></div>
      <div class="row"><span class="lbl">Switch:</span><span id="switchReason"></span></div>
      <div class="row"><span class="lbl">Override:</span><span id="override"></span></div>
      <div class="row"><span class="lbl">Trace:</span><span id="traceFile"></span></div>
    </div>
    <div id="main">
      <div id="toolbar">
        <button id="prevSpot">Prev Spot</button>
        <button id="nextSpot">Next Spot</button>
        <button id="prevFrame">Prev Frame</button>
        <button id="nextFrame">Next Frame</button>
        <button id="playPause">Play</button>
        <span class="lbl">Speed</span>
        <select id="speedSel">
          <option value="900">x0.5</option>
          <option value="450" selected>x1</option>
          <option value="220">x2</option>
          <option value="120">x4</option>
        </select>
        <span class="lbl">Frame</span>
        <input type="range" id="frameSlider" min="0" max="0" value="0" style="width: 300px;">
      </div>
      <div id="canvasWrap"><canvas id="boardCanvas" width="760" height="760"></canvas></div>
      <div id="status"></div>
    </div>
  </div>
  <script>
    const data = {json.dumps(payload)};
    const summary = data.summary || {{}};
    const spots = data.spots || [];
    const byId = (id) => document.getElementById(id);
    byId("spotCount").textContent = String(spots.length);
    byId("rowsScanned").textContent = String(summary.rows_scanned || 0);
    byId("minConf").textContent = String(summary.min_confidence || 0);
    byId("maxSteps").textContent = String(summary.max_steps_to_death || 0);
    byId("replayWindow").textContent = String(summary.replay_window || 0);

    const spotSelect = byId("spotSelect");
    const frameSlider = byId("frameSlider");
    const status = byId("status");
    const canvas = byId("boardCanvas");
    const ctx = canvas.getContext("2d");

    let currentSpot = 0;
    let currentFrame = 0;
    let timer = null;

    function fillSpotSelect() {{
      spots.forEach((s, i) => {{
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `#${{i+1}} seed=${{s.seed}} conf=${{Number(s.predicted_confidence).toFixed(3)}} sud=${{s.steps_until_death}}`;
        spotSelect.appendChild(opt);
      }});
    }}

    function drawCell(x, y, cell, color) {{
      ctx.fillStyle = color;
      ctx.fillRect(x * cell, y * cell, cell - 1, cell - 1);
    }}

    function renderFrame() {{
      if (!spots.length) {{
        ctx.fillStyle = "#0b1220";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        status.textContent = "No blind spots in dataset.";
        return;
      }}
      const s = spots[currentSpot];
      const rows = s.window_rows || [];
      const row = rows[currentFrame] || {{}};
      const n = Math.max(8, Number(s.board_cells || 20));
      const cell = Math.floor(Math.min(canvas.width, canvas.height) / n);
      const drawW = cell * n;
      const ox = Math.floor((canvas.width - drawW) / 2);
      const oy = Math.floor((canvas.height - drawW) / 2);

      ctx.fillStyle = "#07101c";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(ox, oy);

      ctx.strokeStyle = "#132335";
      ctx.lineWidth = 1;
      for (let i = 0; i <= n; i++) {{
        const p = i * cell;
        ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, drawW); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(drawW, p); ctx.stroke();
      }}

      const snakeB = row.snake_before || [];
      const snakeA = row.snake_after || [];
      const foodB = (row.food_before || [])[0] || null;
      const foodA = (row.food_after || [])[0] || null;

      snakeB.forEach(([x,y]) => drawCell(x, y, cell, "#2563eb"));
      snakeA.forEach(([x,y]) => drawCell(x, y, cell, "#60a5fa"));
      if (snakeA.length) {{
        const [hx, hy] = snakeA[0];
        drawCell(hx, hy, cell, "#6ee7b7");
      }}
      if (foodB) drawCell(foodB[0], foodB[1], cell, "#f59e0b");
      if (foodA) drawCell(foodA[0], foodA[1], cell, "#ef4444");

      ctx.restore();

      status.textContent =
        `spot=${{currentSpot+1}}/${{spots.length}} frame=${{currentFrame+1}}/${{rows.length || 1}} `
        + `step=${{row.step ?? "?"}} score=${{row.score_before ?? "?"}}->${{row.score_after ?? "?"}} `
        + `conf=${{Number(row.predicted_confidence || 0).toFixed(3)}} sud=${{row.steps_until_death ?? "?"}} `
        + `mode=${{row.mode || ""}} switch=${{row.switch_reason || ""}} `
        + `override=${{row.override_used ? "yes":"no"}} game_over=${{row.game_over ? "yes":"no"}} death=${{row.death_reason || ""}}`;
    }}

    function renderSpotMeta() {{
      if (!spots.length) return;
      const s = spots[currentSpot];
      byId("seed").textContent = String(s.seed);
      byId("conf").textContent = Number(s.predicted_confidence || 0).toFixed(3);
      byId("sud").textContent = String(s.steps_until_death);
      byId("mode").textContent = String(s.mode || "");
      byId("switchReason").textContent = String(s.switch_reason || "");
      byId("override").textContent = s.override_used ? "yes" : "no";
      byId("traceFile").textContent = String(s.trace_file || "");
      const maxFrame = Math.max(0, (s.window_rows || []).length - 1);
      frameSlider.max = String(maxFrame);
      frameSlider.value = String(Math.min(currentFrame, maxFrame));
    }}

    function setSpot(i) {{
      if (!spots.length) return;
      currentSpot = (i + spots.length) % spots.length;
      currentFrame = 0;
      spotSelect.value = String(currentSpot);
      renderSpotMeta();
      renderFrame();
    }}

    function setFrame(i) {{
      if (!spots.length) return;
      const maxFrame = Math.max(0, (spots[currentSpot].window_rows || []).length - 1);
      currentFrame = Math.max(0, Math.min(maxFrame, i));
      frameSlider.value = String(currentFrame);
      renderFrame();
    }}

    function togglePlay() {{
      const btn = byId("playPause");
      if (timer) {{
        clearInterval(timer);
        timer = null;
        btn.textContent = "Play";
        return;
      }}
      const speed = Number(byId("speedSel").value || 450);
      timer = setInterval(() => {{
        const maxFrame = Math.max(0, (spots[currentSpot].window_rows || []).length - 1);
        if (currentFrame >= maxFrame) {{
          clearInterval(timer);
          timer = null;
          btn.textContent = "Play";
          return;
        }}
        setFrame(currentFrame + 1);
      }}, speed);
      btn.textContent = "Pause";
    }}

    byId("prevSpot").addEventListener("click", () => setSpot(currentSpot - 1));
    byId("nextSpot").addEventListener("click", () => setSpot(currentSpot + 1));
    byId("prevFrame").addEventListener("click", () => setFrame(currentFrame - 1));
    byId("nextFrame").addEventListener("click", () => setFrame(currentFrame + 1));
    byId("playPause").addEventListener("click", togglePlay);
    spotSelect.addEventListener("change", () => setSpot(Number(spotSelect.value || 0)));
    frameSlider.addEventListener("input", () => setFrame(Number(frameSlider.value || 0)));

    fillSpotSelect();
    setSpot(0);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render blind-spot replay windows into an interactive HTML viewer.")
    parser.add_argument("--input", type=str, default="artifacts/live_eval/blind_spot_replay_latest.json")
    parser.add_argument("--out", type=str, default="artifacts/live_eval/blind_spot_replay_latest.html")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input JSON not found: {in_path}")
    report = json.loads(in_path.read_text(encoding="utf-8"))
    payload = build_payload(report if isinstance(report, dict) else {})
    html = build_html(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path}")
    if bool(args.open):
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
