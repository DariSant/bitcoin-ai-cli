# Activity gate study - 2026-10-04

- Data: BTC/USDT:USDT on binanceusdm, decisions from 2025-09-29 to 2026-10-03 (17,712 decision points, one every 30 minutes).
- Simulated rules: stop floor = max(1 x 15m ATR, 0.5 x 4H ATR), target 1.5R, 24 h time limit, fee 0.050% + slippage 0.020% per side.
- Trades are taken in a random direction, so results before costs are about 0R; the differences come from costs and missed movement.
- Verdict KEEP = blocked moments were at least 0.05R worse than allowed ones in BOTH halves. TOO FEW CASES = fewer than 50 blocked moments in one half.

## Current thresholds
```
                                             rule  blocked_%  blocked_avg_R  allowed_avg_R  blocked_move_24h_%  allowed_move_24h_%  blocked_cost_R  allowed_cost_R  gap_R_first  gap_R_second verdict
                          LOW_VOLUME (RVOL < 0.5)     17.276         -0.252         -0.166               2.265               3.217           0.238           0.161        0.112         0.060    KEEP
LOW_VOLATILITY (15m ATR < 0.15% or 4H ATR < 0.6%)     12.133         -0.277         -0.167               1.859               3.235           0.286           0.163        0.166         0.069    KEEP
                          COSTS_TOO_HIGH (> 0.3R)      5.956         -0.347         -0.170               2.083               3.134           0.335           0.168        0.330         0.111    KEEP
                    WHOLE GATE (any of the above)     22.047         -0.257         -0.159               2.207               3.298           0.254           0.157        0.123         0.072    KEEP
```
Whole gate blocks 7.5% of weekday decisions and 58.8% of weekend decisions.

## Threshold sweeps (to spot a better value)
### Low volume: RVOL below
```
 threshold     current  blocked_%  blocked_avg_R  allowed_avg_R  blocked_move_24h_%  allowed_move_24h_%  blocked_cost_R  allowed_cost_R  gap_R_first  gap_R_second verdict
       0.3                  6.668         -0.259         -0.175               2.044               3.125           0.256           0.169        0.103         0.064    KEEP
       0.4                 11.885         -0.247         -0.172               2.205               3.174           0.247           0.165        0.092         0.059    KEEP
       0.5 <-- current     17.276         -0.252         -0.166               2.265               3.217           0.238           0.161        0.112         0.060    KEEP
       0.6                 23.092         -0.237         -0.164               2.451               3.245           0.229           0.158        0.095         0.051    KEEP
       0.7                 30.211         -0.234         -0.158               2.586               3.273           0.221           0.154        0.089         0.062    KEEP
```

### Low volatility: 15m ATR % below
```
 threshold     current  blocked_%  blocked_avg_R  allowed_avg_R  blocked_move_24h_%  allowed_move_24h_%  blocked_cost_R  allowed_cost_R  gap_R_first  gap_R_second verdict
     0.100                  3.760         -0.359         -0.174               1.302               3.129           0.309           0.170        0.256         0.137    KEEP
     0.125                  7.148         -0.322         -0.170               1.620               3.170           0.299           0.167        0.223         0.105    KEEP
     0.150 <-- current     11.811         -0.288         -0.166               1.831               3.228           0.286           0.163        0.166         0.087    KEEP
     0.175                 17.090         -0.285         -0.159               1.968               3.288           0.274           0.158        0.181         0.084    KEEP
     0.200                 23.380         -0.277         -0.151               2.122               3.353           0.262           0.153        0.165         0.092    KEEP
```

### Low volatility: 4H ATR % below
```
 threshold     current  blocked_%  blocked_avg_R  allowed_avg_R  blocked_move_24h_%  allowed_move_24h_%  blocked_cost_R  allowed_cost_R  gap_R_first  gap_R_second       verdict
       0.5                  0.226         -0.373         -0.180               2.401               3.065           0.459           0.173          NaN         0.172 TOO FEW CASES
       0.6 <-- current      0.932         -0.256         -0.180               1.918               3.091           0.389           0.173          NaN         0.055 TOO FEW CASES
       0.7                  2.433         -0.272         -0.178               1.867               3.117           0.372           0.171          NaN         0.074 TOO FEW CASES
       0.8                  5.087         -0.326         -0.173               2.013               3.124           0.337           0.169        0.267         0.111          KEEP
```

### Costs too high: fees + slippage above (R)
```
 threshold     current  blocked_%  blocked_avg_R  allowed_avg_R  blocked_move_24h_%  allowed_move_24h_%  blocked_cost_R  allowed_cost_R  gap_R_first  gap_R_second       verdict
      0.20                 35.716         -0.251         -0.141               2.438               3.463           0.245           0.142        0.124         0.092          KEEP
      0.25                 15.961         -0.284         -0.161               2.261               3.237           0.286           0.159        0.198         0.078          KEEP
      0.30 <-- current      5.956         -0.347         -0.170               2.083               3.134           0.335           0.168        0.330         0.111          KEEP
      0.35                  2.343         -0.416         -0.175               1.874               3.109           0.384           0.172        0.650         0.171 TOO FEW CASES
```

## How to read this
- `blocked_avg_R` should be clearly lower (worse) than `allowed_avg_R`.
- `gap_R_first` / `gap_R_second` = allowed minus blocked, in each half of the period.
- Prefer the threshold where the gap stays clear in both halves without blocking much more time.
- Change a threshold only if the current one shows REVIEW in two monthly checks in a row.
- TOO FEW CASES is normal for the 4H ATR safety net: it rarely fires. Keep it unless it starts blocking a lot (more than about 5% of the time).
