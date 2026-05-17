# Reproducibility Notes

## Model Label

The model in this repository is best described as a TVP-SVAR-SV:

- TVP: time-varying VAR coefficients.
- SVAR: a structural VAR with a lower-triangular contemporaneous impact normalization.
- SV: stochastic volatility in the structural shocks.

It is also common to call this the Primiceri (2005) TVP-VAR with stochastic volatility. If using the term "Cholesky stochastic volatility", the label is reasonable because the covariance matrix is decomposed through a triangular matrix and diagonal stochastic volatilities. It is not a common-factor stochastic-volatility model.

## Algorithm

The sampler follows the corrected step ordering from Del Negro and Primiceri (2015), Algorithm 2:

1. Draw the time-varying VAR coefficients conditional on the volatility history.
2. Draw the time-varying contemporaneous coefficients conditional on the volatility history.
3. Draw the hyperparameters conditional on the current state histories.
4. Draw the Kim-Shephard-Chib mixture indicators.
5. Draw the log-volatility paths conditional on those indicators.

The Metropolis-Hastings correction from Algorithm 3 is not implemented. Del Negro and Primiceri report that Algorithm 2 was empirically indistinguishable from Algorithm 3 in their application, but this remains an approximation.

## Data

The original paper uses S&P DRI data. This repository defaults to public FRED proxies for full reproducibility from Python:

- `GDPCTPI` transformed into annualized quarterly inflation.
- `UNRATE` averaged to quarterly frequency.
- `TB3MS` averaged to quarterly frequency.

For closer comparability with common public replications, compare against the `usmacro` dataset in the R package `bvarsv`.

