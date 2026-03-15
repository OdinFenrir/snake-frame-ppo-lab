# Self-Learning Snake RL

An interactive Snake RL lab built with `pygame` and PPO.  
The goal is simple: train an agent, watch it play live, measure failures, and iteratively improve behavior with reproducible data.

## Demo

### Live Training UI

![Snake Frame live training UI](docs/assets/live_training_ui.png)

## Project Resume

This project is a full training + evaluation environment for reinforcement learning experiments on Snake:

- Learning core:
  - PPO with action masking (`stable-baselines3` + `sb3-contrib`)
  - Gym-style environment (`snake_frame/ppo_env.py`)
- Runtime intelligence:
  - policy inference from PPO
  - dynamic safety/controller arbitration (`snake_frame/gameplay_controller.py`)
  - learned controller memory:
    - `arbiter_model.json` (online learned arbitration)
    - `tactic_memory.json` (clustered tactic memory)
- Experiment loop:
  - train in app UI
  - run holdout suites (`ppo_only` vs `controller_on`)
  - isolate worst seeds
  - capture per-step traces
  - patch and re-validate

## How It Works (For Enthusiasts)

At each game decision:
1. The PPO model predicts an action and confidence.
2. The controller scores local risk (danger, space viability, food pressure, loop signals).
3. The system either trusts PPO or applies controller logic (`escape` / `space_fill` behavior).
4. Outcomes are logged into telemetry and artifacts.

During training:
1. PPO runs in vectorized environments.
2. Evaluation/checkpoints are saved under `state/ppo/v2/`.
3. You can watch live behavior in the same app while training runs.
4. Post-run, focused seed tools identify exactly where controller behavior underperforms.

## Core Features

- Live dashboard with training KPIs, run KPIs, and risk/intervention counters
- Save/load/delete model lifecycle controls in UI
- Determinism and smoke-performance validation tools
- Worst-seed gate + focused per-step trace pipeline for factual debugging
- Clean local artifact model for iterative tuning

## Quick Start (Windows)

1. `setup_env.bat`
2. `run.bat`

Environment defaults:
- Python `3.12`
- Virtual environment at `.venv`
- Locked dependencies from `requirements-lock.txt`

## Reproducibility & Data

After cloning:
1. `setup_env.bat`
2. `run_dashboard.bat` for CI-equivalent validation

Not versioned by design (local experiment data):
- `state/` (local models/checkpoints/UI state)
- `artifacts/` (generated diagnostics/evals/reports)

## Main Controls (In App)

- `Start Train` / `Stop Train`
- `Save` / `Load` / `Delete`
- `Start Game` / `Stop Game` / `Restart`
- Options: adaptive reward, space strategy, themes/backgrounds, debug overlays, diagnostics export

## Persistence

Saved artifacts:
- `state/ui_state.json`
- `state/ppo/v2/*`
- `state/ppo/v2/metadata.json`
- `state/ppo/v2/arbiter_model.json`
- `state/ppo/v2/tactic_memory.json`

Metadata captures run IDs, timesteps, configs, provenance, and eval summaries for future tuning.

## Evaluation & Diagnostics

Core artifacts:
- Holdout suite: `artifacts/live_eval/suites/latest_suite.json`
- Focused worst-seed report: `artifacts/live_eval/worst10_latest.json`
- Focused per-step traces: `artifacts/live_eval/focused_traces/<timestamp>_<tag>/seed_<seed>.jsonl`

Useful scripts:
- `scripts/worst_seed_gate.py`
- `scripts/focused_controller_trace.py`
- `scripts/post_run_suite.py`

## Validation Commands (Local)

- Lint:
  - `.venv\Scripts\python.exe -m ruff check snake_frame tests main.py`
- Full tests:
  - `.venv\Scripts\python.exe -m pytest -q`
- Determinism:
  - `.venv\Scripts\python.exe scripts\validate_determinism.py --baseline tests\baselines\deterministic_windows.json`
- Smoke median gate:
  - `.venv\Scripts\python.exe scripts\smoke_gate_median.py --runs 3 --train-steps 2048 --game-steps 300 --ppo-profile fast --max-frame-p95-ms 40 --max-frame-avg-ms 34 --max-frame-jitter-ms 8 --max-inference-p95-ms 12 --min-training-steps-per-sec 250`
- Worst-seed gate:
  - `.venv\Scripts\python.exe scripts\worst_seed_gate.py --suite artifacts\live_eval\suites\latest_suite.json --top-n 10 --enforce --max-worse-count 8 --min-mean-delta -25`

## CI / Automation

Workflow files exist in `.github/workflows/`.

Note:
- GitHub Actions jobs are currently disabled in workflow config for this repository.
- Full validation is run locally with the commands above.

## Project Structure

- `main.py`
- `snake_frame/` core app + training + controller modules
- `scripts/` diagnostics/eval/research tooling
- `tests/` automated suite
- `ARCHITECTURE.md` deep architecture notes
