# Trusted Baselines

This file declares the trust boundary between pre-discipline and disciplined artifact eras.

---

## Pre-Discipline Era (Historical)

**Artifacts before commit `cc22fee` (March 17, 2026) are considered historical.**

These artifacts may be useful for:
- Debugging
- Forensic analysis
- Understanding past behavior

But they are **NOT suitable for benchmark claims** because:
- Experiment isolation did not exist
- Artifact authority rules were not established
- Some runs used unstable or incorrect assumptions
- `latest_summary.json` was easy to misread as authoritative
- No explicit operating rules existed

---

## Disciplined Era (Trustworthy)

**Artifacts from commit `cc22fee` onward are benchmark-eligible IF they follow these rules:**

### Requirements for Trustworthy Claims

1. **Git commit** - must cite the exact commit hash
2. **Experiment name** - must be explicit (`baseline` for the default baseline, or custom name for experiments)
3. **Matching metadata.json** - must be cited alongside any comparison
4. **Suite artifact** - must use `artifacts/live_eval/suites/suite_*.json` (NOT `latest_summary.json`)
5. **Known comparison context** - must specify: ppo_only vs controller_on, seed set, model selector

### Current Baseline

| Artifact | Value |
|----------|-------|
| Experiment name | `baseline` |
| Observation schema | 31-dim (all ObsConfig flags = true) |
| Model | `state/ppo/baseline/last_model.zip` |
| Metadata | `state/ppo/baseline/metadata.json` |
| Latest suite | `pending (generate fresh suite in new cycle)` |
| Suite mean (ppo_only) | `pending` |
| Suite mean (controller_on) | `pending` |
| **Baseline-producing code** | `cc22fee` to `549ab63` era (trust cutoff through tail-trend) |
| **Baseline documented in repo at** | `7c05191` or later |
| **Trust cutoff begins at** | `cc22fee` |

---

## Artifact Classification Guide

| Classification | Use For | Don't Use For |
|---------------|---------|---------------|
| **Canonical** | Current baseline claims | N/A |
| **Comparable** | Trend analysis, non-baseline comparisons | Definitive benchmark claims |
| **Historical** | Debugging, understanding past behavior | Performance claims |
| **Not Comparable** | Nothing | Any comparison to baseline |
| **Unknown** | Nothing | Any claims |

---

## How to Preserve a Baseline

Before any code change that could affect model behavior:

```bash
# 1. Freeze model artifacts
cp -r state/ppo/baseline state/ppo/baseline_BASELINE

# 2. Freeze suite artifact  
cp artifacts/live_eval/suites/latest_suite.json artifacts/live_eval/suites/suite_BASELINE_$(date +%Y%m%d).json

# 3. Document commit
git log -1 --format="%H %s" > state/ppo/baseline_BASELINE/commit.txt
```

---

## Key Cutoff Commit

**`cc22fee`** - First commit with experiment isolation support (baseline-producing code era)

This is the formal boundary. Artifacts created before this commit are from the pre-discipline era.

- **Baseline-producing code**: `cc22fee` to `549ab63` - the actual code that trained the model and ran the suite
- **Documentation commits**: `7c05191`, `7b22942` - later commits that recorded the baseline
