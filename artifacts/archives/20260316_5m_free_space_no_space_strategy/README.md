# Free Space Feature + No Space Strategy - 2026-03-16

## Training Config
- Steps: 5,005,312
- free_space_features: ON
- space_strategy: OFF
- adaptive_reward: ON

## Eval Results (30 seeds)

| Metric | Value |
|--------|-------|
| Mean | 134.93 |
| Median | 138.5 |
| Best | 165 |
| Min | 70 |
| Interventions | 3.91% |

## Blind Spots
- 50 blind spots across 19 seeds
- Dominated by food_pressure trigger in space_fill/escape modes
- 0 overrides used

## Notes
- This run showed instability compared to baseline (space_strategy ON)
- The free_space observation feature was added
- Controller low-confidence fallback was also added
- Space strategy OFF caused some seeds to crash (17010: 94, 17015: 70)
