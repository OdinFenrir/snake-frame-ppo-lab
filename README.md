# Snake Frame PPO Lab

A research-oriented `pygame` Snake project with:
- PPO training + live inference
- Safety/controller arbitration on top of policy decisions
- Persistent model/runtime artifacts for retuning
- Deterministic + smoke + CI quality gates

## Demo

### Live Training UI

![Snake Frame live training UI](docs/assets/live_training_ui.png)

## Highlights

- Runtime app with training controls, model lifecycle controls, run/watch modes, and KPI dashboards
- PPO stack (`stable-baselines3` + `sb3-contrib` action masking)
- Dynamic controller with:
  - risk-aware arbitration (`ppo_conf_trust`, `food_pressure`, `risk`, etc.)
  - learned arbiter memory (`arbiter_model.json`)
  - tactic memory bank (`tactic_memory.json`)
- Holdout evaluation for `ppo_only` vs `controller_on`
- Focused worst-seed diagnostics + per-step trace tooling

## Quick Start (Windows)

1. `setup_env.bat`
2. `run.bat`

Environment defaults:
- Python `3.12`
- Virtual environment at `.venv`
- Locked dependencies from `requirements-lock.txt`

## Reproducibility

After cloning:
1. `setup_env.bat`
2. `run_dashboard.bat` for CI-equivalent validation

Not versioned by design:
- `state/` (local models/checkpoints/UI state)
- `artifacts/` (generated diagnostics/evals/reports)

## Main Controls

- `Start Train` / `Stop Train`
- `Save` / `Load` / `Delete`
- `Start Game` / `Stop Game` / `Restart`
- Options: adaptive reward, space strategy, themes/backgrounds, debug overlays, diagnostics export

## Model + Data Persistence

Saved artifacts:
- `state/ui_state.json`
- `state/ppo/v2/*`
- `state/ppo/v2/metadata.json`
- `state/ppo/v2/arbiter_model.json`
- `state/ppo/v2/tactic_memory.json`

Metadata captures run IDs, timesteps, configs, provenance, and eval summaries for future tuning.

## Evaluation and Diagnostics

Core artifacts:
- Holdout suite: `artifacts/live_eval/suites/latest_suite.json`
- Focused worst-seed report: `artifacts/live_eval/worst10_latest.json`
- Focused per-step traces: `artifacts/live_eval/focused_traces/<timestamp>_<tag>/seed_<seed>.jsonl`

Useful scripts:
- `scripts/worst_seed_gate.py`
- `scripts/focused_controller_trace.py`
- `scripts/post_run_suite.py`

## Validation Commands

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

## CI

Workflow: `.github/workflows/ci.yml`

Runs on Windows:
- dependency install
- lint (`ruff`)
- tests (`pytest`, including render marker)
- deterministic drift validation
- smoke performance gate (median-of-3)

## Project Structure

- `main.py`
- `snake_frame/` core app + training + controller modules
- `scripts/` diagnostics/eval/research tooling
- `tests/` automated suite
- `ARCHITECTURE.md` deep architecture notes
