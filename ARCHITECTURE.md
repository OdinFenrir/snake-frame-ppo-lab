# Architecture

## 1. System Overview

`Snake RL Research Lab` is a single-desktop Python application that combines:
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
- `snake_frame/welcome.py`: workspace router (`Live Training`, `Analysis Tools`, `Model Manager`, `Settings`)
- `snake_frame/gate_runner.py`: quick reliability gate wrapper used to block unsafe model-manager actions
- `snake_frame/app_rendering.py`, `panel_ui.py`, `graph_renderer.py`: dashboard rendering

Startup semantics:
- app starts in detached mode (`_detached_session`)
- no experiment is auto-loaded at startup
- binding to a real experiment happens only through explicit `Load`/`Save`

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

### Analysis Report Artifacts (Canonical)

Canonical report roots:
- `artifacts/training_input`
- `artifacts/agent_performance`
- `artifacts/phase3_compare`
- `artifacts/reports`

Contract guarantees:
- deterministic latest aliases (`*_latest.*`)
- stamped retention policy (`latest` + last `N`, default `N=5`)
- no writes outside canonical family roots
- compare tooling requires explicit Model 1 + Model 2 selection (no silent defaults)
- in-app report cleanup supports:
  - retention prune (`report_artifacts`)
  - hard purge (`report_artifacts_purge`)

Authoritative contract: `docs/REPORT_TOOLING_CONTRACT.md`.

## 6. Validation and Quality Gates

### Local/CI Gates

- static lint: `ruff`
- full test suite: `pytest`
- deterministic drift check: `scripts/validate_determinism.py`
- smoke performance gate (median-of-3): `scripts/smoke_gate_median.py`

### Focused Regression Gates

- worst-seed extraction/enforcement: `scripts/worst_seed_gate.py`
- focused trace capture for root-cause analysis: `scripts/focused_controller_trace.py`

## 6.5 Training Safety (PPO Health Monitoring)

The PPO training loop includes a custom `TrainingHealthMonitor` callback (`snake_frame/ppo_agent.py`) that runs **before all other callbacks** to detect and stop training failures early.

### Purpose

Prevent saving corrupted or catastrophically diverged models by detecting common PPO failure modes at rollout boundaries.

### Checked Signals

| Signal | Threshold | Action |
|--------|-----------|--------|
| NaN/Inf in any loss | Any | **FAIL** |
| Persistent high KL | > 0.2 for ≥ 2 consecutive rollouts | **FAIL** |
| Missing metrics | After train() completed + warmup | **FAIL** |
| Explained variance | < -5.0 (after warmup) | **FAIL** |
| KL divergence | > 0.1 | WARN |
| Clip fraction | > 0.85 (after warmup) | WARN |
| Entropy collapse | < 10% of recent average | WARN |
| Value loss spike | > 5× moving average | WARN |
| Reward stagnation | Flat slope + small range | WARN |
| Logger structure invalid | Non-dict `name_to_value` | WARN |

### Behavior on Failure

1. Sets internal failure flag
2. **Skips model checkpoint save** at end of `learn()` call
3. Emits failure event (logged + hookable by UI)
4. Writes `status: "failed"` to metadata

### Warmup

First 3 rollouts are ignored for most checks to allow initial stabilization.

### SB3 Timing Note

In SB3's `learn()` loop, `_on_rollout_end()` fires **before** `train()` runs. This means `train/*` metrics (policy_loss, value_loss, etc.) don't exist on the first rollout. The monitor tracks `_train_has_completed` to distinguish between "train() hasn't run yet" (expected, warn) vs "train() ran but no metrics" (failure).

### Configuration

All thresholds are configurable via constructor arguments:

```python
monitor = TrainingHealthMonitor(
    kl_fail_threshold=0.2,
    kl_persistence_required=2,
    warmup_rollouts=3,
    stagnation_window=5,
    stagnation_rel_threshold=0.01,  # relative slope
    stagnation_abs_threshold=0.05,  # absolute range
    # ...
)
```

Default thresholds are documented in `TrainingHealthMonitor.DEFAULT_THRESHOLDS`.

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

## 11. Legacy Path Timeline (UTC)

- Legacy write support ends: `2026-06-30 23:59 UTC`
- Legacy read-fallback support removed: `2026-07-31 23:59 UTC`
- After `2026-07-31 23:59 UTC`, fallback paths are hard errors with migration hints.

`.bat` wrappers remain compatibility helpers; app orchestration uses Python entrypoints directly.
