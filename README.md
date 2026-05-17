# Primiceri (2005) TVP-SVAR replication

This repository contains a reproducible replication attempt for Giorgio Primiceri's "Time Varying Structural Vector Autoregressions and Monetary Policy".

It is written as a public-facing research repository: source code, reproducible data construction, generated figures, citation metadata, and Colab instructions are included. The article PDF is not vendored; see [REFERENCES.md](REFERENCES.md) for links and copyright notes.

The code implements the model described in Sections 2 and 3.1 of the paper:

- a three-variable quarterly VAR with time-varying lag coefficients;
- a lower-triangular contemporaneous matrix with time-varying free elements;
- stochastic volatility for the structural shocks;
- Gibbs sampling with Carter-Kohn forward-filtering backward-sampling and the Kim, Shephard, and Chib seven-component mixture approximation for log chi-square volatility errors;
- the corrected MCMC step ordering from Del Negro and Primiceri's 2015 corrigendum.

The model is a TVP-SVAR-SV: time-varying parameters, structural VAR identification, and stochastic volatility. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for naming and algorithm details.

## Data

The paper uses S&P DRI data. To keep this repository reproducible without proprietary data, the scripts use public FRED proxies:

- `GDPCTPI`: chain-type GDP price index, transformed into annualized quarterly inflation;
- `UNRATE`: civilian unemployment rate, quarterly average;
- `TB3MS`: 3-month Treasury bill rate, quarterly average.

The sample matches the paper's timing as closely as possible: 1953Q1 to 2001Q3, with the first 40 quarters used to calibrate priors and the remaining observations used for the TVP-SVAR estimation.

## Run

```powershell
python scripts\run_replication.py --iterations 800 --burn 300 --seed 123
```

For a closer paper-style run, increase the sampler length:

```powershell
python scripts\run_replication.py --iterations 10000 --burn 2000 --seed 123
```

## R Benchmark

The fastest benchmark path uses Fabian Krueger's R/C++ package `bvarsv`, which implements the Primiceri model and includes the `usmacro` dataset used in common replications.

```powershell
Rscript scripts\run_bvarsv_benchmark.R --install --nrep=10000 --nburn=2000 --thinfac=10
```

For a smoke test:

```powershell
Rscript scripts\run_bvarsv_benchmark.R --install --nrep=200 --nburn=100 --thinfac=10
```

The script writes `outputs/r_bvarsv_volatility_summary.csv`, `outputs/r_bvarsv_volatility_paths.csv`, and `outputs/figures/r_bvarsv_volatilities.png`.

## Outputs

Generated outputs are written to:

- `data/processed/primiceri_fred_proxy.csv`
- `outputs/figures/01_fred_proxy_data.png`
- `outputs/figures/02_stochastic_volatilities.png`
- `outputs/figures/03_policy_shock_irfs.png`
- `outputs/posterior_volatility_summary.csv`
- `outputs/replication_summary.md`
- `outputs/r_bvarsv_volatility_summary.csv`
- `outputs/r_bvarsv_volatility_paths.csv`

The generated figures are meant to be transparent replication artifacts, not exact scans of the paper figures. Exact numerical equality is not expected because the original data source is proprietary and the default run is shorter than the paper's 10,000-iteration benchmark.

## Colab

The notebook [notebooks/primiceri_colab.ipynb](notebooks/primiceri_colab.ipynb) can be opened in Google Colab after this repository is pushed to GitHub. Replace the placeholder `REPO_URL` in the first code cell with the final GitHub URL.

The current Python sampler is CPU-bound and does not use a GPU. Colab can still help with reproducibility and with running longer jobs away from your local machine, but GPU acceleration would require a separate JAX, Numba, C++, or parallel-chain implementation.

## Related Public Replications

For a mature R/C++ implementation, see Fabian Krueger's [`bvarsv`](https://cran.r-project.org/package=bvarsv) package. It implements the Primiceri model, incorporates the 2015 corrigendum, and includes the `usmacro` dataset for the 1953Q1-2001Q3 sample.

## Notes

The PDF used to derive the model is not included in this repository. Cite the published paper and the corrigendum separately; citation metadata is provided in [CITATION.cff](CITATION.cff).
