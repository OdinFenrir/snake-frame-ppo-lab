# Report Tooling Contract

This document is the source of truth for report outputs, retention, CLI validation, and failure semantics.

## Canonical Output Roots

- Training input: `artifacts/training_input`
- Agent performance: `artifacts/agent_performance`
- Model compare: `artifacts/phase3_compare`
- Reports hub: `artifacts/reports`

Report builders and visuals must not write outside these roots.

## Required Latest Files

### Training Input

- `training_input_latest.json`
- `training_input_latest.md`
- `training_input_checkpoint_vecnorm_latest.csv`
- `training_input_timeline_latest.json`
- `training_input_timeline_latest.md`
- `training_input_timeline_latest.csv`
- `training_input_dashboard_latest.html`

### Agent Performance

- `agent_performance_latest.json`
- `agent_performance_latest.md`
- `agent_performance_rows_latest.csv`
- `agent_performance_dashboard_latest.html`

### Model Compare

- `model_agent_compare_latest.json`
- `model_agent_compare_latest.md`
- `model_agent_compare_rows_latest.csv`
- `model_agent_compare_dashboard_latest.html`

### Reports Hub

- `reports_hub_latest.md`
- `reports_hub_latest.txt`

## Latest Pointer Semantics

- Canonical latest files are deterministic fixed names (`*_latest.*`).
- Consumers should read latest files directly and not infer “latest” by scanning stamped files.

## Retention Semantics

- Keep `latest` for each family.
- Keep last `N` stamped outputs per file family (`N` from `--retain-stamped`, default `5`).
- Never delete non-matching files.
- Stamped token format is `YYYYMMDD_HHMMSS`.

## CLI Contract

Common arguments:

- `--out-dir` must be the canonical output root for that report family.
- `--tag` defaults to `latest`.
- `--retain-stamped` must be `>= 0`.

Failure semantics:

- Invalid inputs exit non-zero.
- Error text names the invalid argument/path/dependency.
- Compare mode must not silently fallback to defaults for model selection.

## Wrapper Status

`.bat` wrappers are compatibility helpers. The app invokes Python tooling directly.

## Legacy Timeline (UTC)

- Legacy write support ends: `2026-06-30 23:59 UTC`
- Legacy read-fallback support removed: `2026-07-31 23:59 UTC`
- After `2026-07-31 23:59 UTC`, old fallback paths are hard errors with migration hints.

## Troubleshooting

- If a tool runs but viewer has no output, verify required latest files exist in canonical root.
- If command fails on `--out-dir`, use canonical family root listed above.
- If compare fails, verify both selected experiments exist under `state/ppo/`.
