# Evidence Bundle

This folder contains example artifacts that demonstrate the evaluation system.

## Files

### suite_example.json
A complete paired holdout evaluation suite showing:
- PPO-only mode: mean score 150.63
- Controller-on mode: mean score 150.20
- Per-seed scores for both modes (30 seeds, 17001-17030)
- Controller intervention telemetry
- Paired comparison statistics

**Proves**: Systematic evaluation with both controller modes, per-seed tracking, and quantitative comparison.

### trace_example_seed_17001.jsonl
Per-step decision trace for seed 17001 showing:
- Each step's snake state (head position, tail, food)
- PPO action prediction and confidence
- Controller decision (trust PPO / escape / space_fill)
- Risk metrics at each decision point

**Proves**: Granular debugging capability, ability to trace exactly why the controller made each decision.

## Source

Produced from: commit `549ab63` (tail-trend feature era)  
Experiment: historical `v2` baseline (superseded by current `baseline` experiment)  
Date: March 17, 2026

These files demonstrate the evaluation discipline documented in TRUSTED_BASELINES.md.
