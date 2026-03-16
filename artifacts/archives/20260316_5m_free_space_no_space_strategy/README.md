# Free Space Feature + No Space Strategy - 2026-03-16

## Training Config
- Steps: 5,005,312
- free_space_features: ON
- space_strategy: OFF
- adaptive_reward: ON

## Eval Results (30 seeds)

| Metric | Baseline (space_strat ON) | This Run | Delta |
|--------|---------------------------|----------|-------|
| Mean | 142.07 | 134.93 | -7.14 |
| Median | 145.5 | 138.5 | -7 |
| Best | 168 | 165 | -3 |
| Min | 102 | 70 | -32 |
| Interventions% | 3.93% | 3.91% | -0.02% |

## Seed Changes

| Seed | Baseline | This Run | Delta |
|------|----------|----------|-------|
| 17010 | 165 | 94 | -71 |
| 17015 | 125 | 70 | -55 |
| 17004 | 160 | 116 | -44 |
| 17005 | 167 | 123 | -44 |
| 17012 | 154 | 116 | -38 |
| 17021 | 102 | 165 | +63 |
| 17002 | 105 | 155 | +50 |
| 17007 | 135 | 165 | +30 |
| 17008 | 105 | 136 | +31 |

## Blind Spots

- 50 blind spots across 19 seeds
- Dominated by food_pressure trigger in space_fill/escape modes
- 0 overrides used
- Worst: 17018 (11), 17019/17030 (10)

## Changes from Baseline

- Added use_free_space_features observation input
- Added low-confidence fallback in SPACE_FILL/ESCAPE modes
- space_strategy was OFF (caused instability)

## Files

- `summary_20260316_151059.json` - Full 30-seed eval results
- `blind_spot_replay_latest.html` - Blind spot visualization
- `blind_spot_replay_latest.json` - Blind spot data
