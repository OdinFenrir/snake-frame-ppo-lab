# Operating Rules for This Repo

## Experiment Isolation

- **Baseline runs**: use `v2` (the default)
- **New experiments**: must use a distinct experiment_name (e.g., `v2_my_experiment`)
- **Never train into v2 without preserving baseline first**

## Canonical Baseline

For the current verified baseline, see `TRUSTED_BASELINES.md`.

Summary:
- **Baseline experiment:** `v2` (default)
- **Observation schema:** 31-dim (all ObsConfig flags = true)
- **Suite artifact:** `artifacts/live_eval/suites/suite_20260317_092448.json`
- **Trust cutoff:** commit `cc22fee` onward

## Before Any Code Change

1. **Freeze baseline** (Windows):
   ```powershell
   xcopy /E /I state\ppo\v2 state\ppo\v2_BASELINE
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

All 232 tests must pass.
