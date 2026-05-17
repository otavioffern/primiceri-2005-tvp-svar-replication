# Benchmark Results

This report records the benchmark run using Fabian Krueger's `bvarsv` R/C++ package. The package implements the Primiceri (2005) TVP-VAR-SV model and incorporates the Del Negro and Primiceri (2015) corrigendum.

## Run

Command:

```powershell
.\scripts\run_bvarsv_benchmark.ps1 -NRep 10000 -NBurn 2000 -ThinFac 10 -ItPrint 2000
```

Environment:

- R: local CRAN R 4.6.0
- `bvarsv`: 1.1.1
- Data: `bvarsv::usmacro`
- Estimation output window: 1963Q3 to 2001Q3
- Runtime on this Windows machine: 142.1 seconds

## Posterior-Mean Shock Volatilities

The table reports period averages of posterior-mean structural shock standard deviations.

| Period | Inflation shock | Unemployment shock | T-bill / policy shock |
|---|---:|---:|---:|
| 1963Q3-1979Q2 | 0.3350 | 0.2388 | 0.6687 |
| 1979Q3-1983Q4 | 0.4953 | 0.3627 | 1.3791 |
| 1984Q1-2001Q3 | 0.2287 | 0.1713 | 0.3697 |

The policy-shock volatility is about 3.73 times larger in 1979Q3-1983Q4 than in 1984Q1-2001Q3. This reproduces the main qualitative Primiceri result that non-systematic monetary policy shocks were especially volatile around the Volcker transition and lower in the post-1984 period.

## Interpretation

This benchmark should be treated as the repository's reference implementation. The Python implementation remains useful for transparency and learning, but `bvarsv` is compiled, mature, and already aligned with the corrigendum.

Generated files from this run are intentionally ignored by Git:

- `outputs/r_bvarsv_volatility_summary.csv`
- `outputs/r_bvarsv_volatility_paths.csv`
- `outputs/figures/r_bvarsv_volatilities.png`
- `outputs/r_bvarsv_benchmark_summary.rds`
