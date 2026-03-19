# Architecture

## 1. System Overview

`Snake Frame PPO Lab` is a single-desktop Python application that combines:
- interactive gameplay and rendering (`pygame`)
- PPO training/inference (`stable-baselines3` + `sb3-contrib`)
- a safety/controller layer that can arbitrate policy actions
- persistence + diagnostics pipelines for iterative tuning

The architecture is optimized for iterative experimentation:
- run training in-app
- evaluate holdout seeds
- inspect artifacts and traces
- apply targeted controller/policy updates

## 2. Runtime Composition

Entry points:
- `snake_frame/app.py`: Application entry and runtime bootstrap
- `snake_frame/app_factory.py`: Composition/wiring layer for runtime dependencies

Core runtime modules:
- `snake_frame/game.py`: game state, mechanics, board updates, rendering primitives
- `snake_frame/gameplay_controller.py`: policy+controller action selection and telemetry
- `snake_frame/ppo_agent.py`: PPO model lifecycle (train/load/save/eval/inference sync)
- `snake_frame/training.py`: threaded training controller
- `snake_frame/holdout_eval.py`: holdout eval orchestration (`ppo_only`/`controller_on`)
- `snake_frame/state_io.py`: UI state schema and recovery behavior
- `snake_frame/app_rendering.py`, `panel_ui.py`, `graph_renderer.py`: dashboard rendering

## 3. Decision Stack (Agent Intelligence)

Per decision tick (`gameplay_controller`):
1. Build observation + action mask
2. Query PPO action and probabilities
3. Evaluate action viability/risk
4. Optionally arbitrate via controller modes:
   - `ppo` (trust policy)
   - `escape`
   - `space_fill`
5. Optionally apply learned memory influence:
   - learned arbiter model
   - tactic memory bank
6. Queue final direction into game loop

Important telemetry fields include:
- interventions, risk counters, mode switches
- loop/cycle indicators
- death reason distributions

## 4. Learning Memory Extensions

Persisted under `state/ppo/<experiment_name>/`:
- `arbiter_model.json`
- `tactic_memory.json`

Default baseline uses `state/ppo/baseline/`.

Purpose:
- retain controller-side adaptation signals between sessions
- bias arbitration away from repeated harmful interventions

## 5. Artifact Contracts

### Model Artifacts

Persisted under `state/ppo/<experiment_name>/`:
- `last_model.zip`
- `best_model.zip`
- `best_score_model.zip`
- `vecnormalize.pkl`
- `metadata.json`

Default baseline uses `state/ppo/baseline/`.

### Evaluation Artifacts

For trustworthy comparisons, use suite artifacts (NOT single-run summaries):
- `artifacts/live_eval/suites/suite_<timestamp>.json` (full paired evaluation)
- `artifacts/live_eval/suites/latest_suite.json` (most recent suite)

Single-run summaries (less authoritative):
- `artifacts/live_eval/latest_summary.json`
- `artifacts/live_eval/summary_*.json`

See `TRUSTED_BASELINES.md` for authoritative baseline references.

### Focused Trace Artifacts

- `artifacts/live_eval/focused_traces/<timestamp>_<tag>/seed_<seed>.jsonl`

Per-step rows include:
- predicted vs chosen action
- override flags
- switch reason/mode
- action probabilities
- risk/viability features
- step outcome (`score_before/after`, `death_reason`, `game_over`)

## 6. Validation and Quality Gates

### Local/CI Gates

- static lint: `ruff`
- full test suite: `pytest`
- deterministic drift check: `scripts/validate_determinism.py`
- smoke performance gate (median-of-3): `scripts/smoke_gate_median.py`

### Focused Regression Gates

- worst-seed extraction/enforcement: `scripts/worst_seed_gate.py`
- focused trace capture for root-cause analysis: `scripts/focused_controller_trace.py`

## 7. State and Schema Guarantees

- UI persistence uses a versioned schema (`uiStateVersion`)
- unsupported future schemas are rejected explicitly (safe failure)
- model/load errors propagate as structured operation codes

## 8. Concurrency Model

- UI/render loop on main thread
- PPO training on worker thread (`training.py`)
- holdout evaluation on dedicated worker thread (`holdout_eval.py`)

Guardrails:
- training-active checks prevent unsafe eval artifact mutation
- model selector/load flows explicitly validated before eval runs

## 9. Repository Structure

- `snake_frame/`: application and ML/control core
- `scripts/`: evaluation, diagnostics, deterministic/smoke tools
- `tests/`: unit/integration regression coverage
- `.github/workflows/`: CI/release workflows

## 10. Current Engineering Priorities

1. Reduce controller-on underperformance on hard holdout seeds.
2. Keep smoke and determinism gates green under strict thresholds.
3. Preserve artifact compatibility and reproducibility while iterating.
