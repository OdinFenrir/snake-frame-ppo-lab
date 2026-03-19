# Operating Rules for This Repo

## Experiment Isolation

- **Baseline runs**: use `baseline` (the default)
- **New experiments**: must use a distinct experiment name (for example `baseline_experiment_a`)
- **Never train into baseline without preserving baseline first**
- **Use explicit experiment targeting for model actions** (`Save` / `Load` / `Delete`)
- **Do not assume current slot**: confirm the selected experiment before any save/delete

## Canonical Baseline

For the current verified baseline, see `TRUSTED_BASELINES.md`.

Summary:
- **Baseline experiment:** `baseline` (default)
- **Observation schema:** 31-dim (all ObsConfig flags = true)
- **Baseline metrics:** pending fresh suite generation in this new cycle
- **Trust cutoff:** commit `cc22fee` onward for artifact-discipline rules

## Before Any Code Change

1. **Freeze baseline** (Windows):
   ```powershell
   xcopy /E /I state\ppo\baseline state\ppo\baseline_BASELINE
   copy artifacts\live_eval\suites\latest_suite.json artifacts\live_eval\suites\suite_BASELINE_%date:~0,4%%date:~5,2%%date:~8,2%.json
   ```

2. **Document what you're preserving**:
   - commit hash
   - suite artifact path
   - metadata.json snapshot

## Making Comparisons

**For any claim about results, cite**:

1. **Commit** - git hash
2. **Suite artifact** - `artifacts/live_eval/suites/suite_*.json` (NOT `latest_summary.json`)
3. **Metadata** - `state/ppo/<experiment>/metadata.json`
4. **Experiment name** - the experiment_name used

**Never trust**:
- `latest_summary.json` - single mode, easily overwritten
- Memory - always cite artifacts
- "It looked better" - always cite numbers from suite files

## Do Not Mix

- **Do not compare suites from different experiment_name values** unless the matching `metadata.json` files are cited.
- **Do not compare models with different observation schemas** as if they were the same experiment lineage.
- **Do not compare a new run's results to a preserved baseline** without explicitly citing both suite artifacts.

## Report Tooling Rules

- Canonical report contract: `docs/REPORT_TOOLING_CONTRACT.md`.
- Reports must write only to canonical family roots:
  - `artifacts/training_input`
  - `artifacts/agent_performance`
  - `artifacts/phase3_compare`
  - `artifacts/reports`
- `*_latest.*` files are deterministic aliases and must be used as primary readers.
- Retention semantics are strict:
  - keep `latest`
  - keep last `N` stamped (default `N=5`)
  - never delete non-matching files
- Compare tools must not silently default model selectors.

## Legacy Policy (UTC)

- Legacy write support ends: `2026-06-30 23:59 UTC`
- Legacy read-fallback support removed: `2026-07-31 23:59 UTC`
- After `2026-07-31 23:59 UTC`, fallback paths are hard errors with migration hints.

`.bat` wrappers are compatibility-only helpers; app workflow uses Python tools directly.

## When Changing experiment_name

The code now supports:
```python
runtime = build_runtime(settings, font, small_font, on_score, experiment_name="my_experiment")
```

This creates artifacts in `state/ppo/my_experiment/`.

## Test Before Proposing Changes

Run tests with:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

All tests in the current suite must pass.
