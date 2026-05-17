from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import invwishart


KSC_PROB = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_MEAN = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
KSC_VAR = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
KSC_CENTERING = 1.2704


@dataclass
class PriorSpec:
    beta_mean: np.ndarray
    beta_cov: np.ndarray
    alpha_mean: np.ndarray
    alpha_cov_blocks: list[np.ndarray]
    h_mean: np.ndarray
    h_cov: np.ndarray
    q_scale: np.ndarray
    q_df: int
    w_scale: np.ndarray
    w_df: int
    s_scales: list[np.ndarray]
    s_dfs: list[int]


@dataclass
class TvpSvarResult:
    dates: pd.DatetimeIndex
    variables: list[str]
    lags: int
    beta_draws: np.ndarray
    alpha_draws: np.ndarray
    h_draws: np.ndarray
    sigma_draws: np.ndarray
    iterations: int
    burn: int
    seed: int

    @property
    def beta_mean(self) -> np.ndarray:
        return self.beta_draws.mean(axis=0)

    @property
    def alpha_mean(self) -> np.ndarray:
        return self.alpha_draws.mean(axis=0)

    @property
    def h_mean(self) -> np.ndarray:
        return self.h_draws.mean(axis=0)


def _sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _with_jitter(matrix: np.ndarray, min_jitter: float = 1e-10) -> np.ndarray:
    matrix = _sym(np.asarray(matrix, dtype=float))
    jitter = min_jitter
    eye = np.eye(matrix.shape[0])
    for _ in range(8):
        try:
            np.linalg.cholesky(matrix + jitter * eye)
            return matrix + jitter * eye
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_jitter)
    return _sym((eigvecs * eigvals) @ eigvecs.T)


def _sample_mvn(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cov = _with_jitter(cov)
    return rng.multivariate_normal(mean, cov)


def make_lagged_design(values: np.ndarray, dates: pd.DatetimeIndex, lags: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    rows_y: list[np.ndarray] = []
    rows_x: list[np.ndarray] = []
    rows_dates: list[pd.Timestamp] = []
    for t in range(lags, len(values)):
        lag_stack = [values[t - lag] for lag in range(1, lags + 1)]
        rows_x.append(np.r_[1.0, np.concatenate(lag_stack)])
        rows_y.append(values[t])
        rows_dates.append(dates[t])
    return np.asarray(rows_y), np.asarray(rows_x), pd.DatetimeIndex(rows_dates)


def fit_ols_var(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta_hat = xtx_inv @ x.T @ y
    resid = y - x @ beta_hat
    dof = max(1, y.shape[0] - x.shape[1])
    omega = resid.T @ resid / dof
    beta_cov = np.kron(_with_jitter(omega), _with_jitter(xtx_inv))
    return beta_hat, resid, _with_jitter(omega), _with_jitter(beta_cov)


def fit_triangular_a(resid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    n = resid.shape[1]
    alpha_parts: list[np.ndarray] = []
    cov_blocks: list[np.ndarray] = []
    sigmas = np.empty(n)

    sigmas[0] = np.sqrt(np.mean(resid[:, 0] ** 2))
    for i in range(1, n):
        x_i = -resid[:, :i]
        y_i = resid[:, i]
        xtx_inv = np.linalg.pinv(x_i.T @ x_i)
        alpha_i = xtx_inv @ x_i.T @ y_i
        err_i = y_i - x_i @ alpha_i
        dof = max(1, resid.shape[0] - i)
        sigma2_i = float(err_i @ err_i / dof)
        sigmas[i] = np.sqrt(max(sigma2_i, 1e-10))
        alpha_parts.append(alpha_i)
        cov_blocks.append(_with_jitter(sigma2_i * xtx_inv))

    return np.concatenate(alpha_parts), cov_blocks, sigmas


def calibrate_priors(
    data: pd.DataFrame,
    lags: int = 2,
    prior_obs: int = 40,
    k_q: float = 0.01,
    k_s: float = 0.1,
    k_w: float = 0.01,
) -> PriorSpec:
    values = data.to_numpy(dtype=float)
    y0, x0, _ = make_lagged_design(values[:prior_obs], data.index[:prior_obs], lags)
    beta_ols, resid_ols, _, beta_cov = fit_ols_var(y0, x0)
    alpha_ols, alpha_cov_blocks, sigmas = fit_triangular_a(resid_ols)

    n = values.shape[1]
    beta_mean = beta_ols.T.reshape(-1)
    beta_cov = _with_jitter(beta_cov)
    alpha_cov = block_diag(*alpha_cov_blocks)

    q_df = prior_obs
    w_df = n + 1
    s_dfs = [block.shape[0] + 1 for block in alpha_cov_blocks]

    return PriorSpec(
        beta_mean=beta_mean,
        beta_cov=beta_cov,
        alpha_mean=alpha_ols,
        alpha_cov_blocks=[4.0 * block for block in alpha_cov_blocks],
        h_mean=np.log(np.maximum(sigmas, 1e-8)),
        h_cov=np.eye(n),
        q_scale=_with_jitter((k_q**2) * q_df * beta_cov),
        q_df=q_df,
        w_scale=_with_jitter((k_w**2) * w_df * np.eye(n)),
        w_df=w_df,
        s_scales=[_with_jitter((k_s**2) * df * block) for df, block in zip(s_dfs, alpha_cov_blocks)],
        s_dfs=s_dfs,
    )


def estimation_sample(
    data: pd.DataFrame,
    lags: int = 2,
    prior_obs: int = 40,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = data.to_numpy(dtype=float)
    y_all, x_all, dates_all = make_lagged_design(values, data.index, lags)
    start = prior_obs - lags
    return y_all[start:], x_all[start:], dates_all[start:]


def alpha_to_a(alpha_t: np.ndarray, n: int) -> np.ndarray:
    a = np.eye(n)
    cursor = 0
    for row in range(1, n):
        width = row
        a[row, :row] = alpha_t[cursor : cursor + width]
        cursor += width
    return a


def beta_to_coefficients(beta_t: np.ndarray, n: int, lags: int) -> np.ndarray:
    m = 1 + n * lags
    return beta_t.reshape(n, m).T


def beta_to_var_matrices(beta_t: np.ndarray, n: int, lags: int) -> list[np.ndarray]:
    coeff = beta_to_coefficients(beta_t, n, lags)
    mats = []
    for lag in range(lags):
        rows = coeff[1 + lag * n : 1 + (lag + 1) * n, :]
        mats.append(rows.T)
    return mats


def observation_matrices(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray([np.kron(np.eye(n), x_t) for x_t in x])


def omega_from_alpha_h(alpha: np.ndarray, h: np.ndarray, n: int) -> np.ndarray:
    omega = np.empty((alpha.shape[0], n, n))
    for t in range(alpha.shape[0]):
        a_t = alpha_to_a(alpha[t], n)
        sigma2 = np.diag(np.exp(2.0 * h[t]))
        a_inv = np.linalg.inv(a_t)
        omega[t] = _with_jitter(a_inv @ sigma2 @ a_inv.T)
    return omega


def structural_residuals(y: np.ndarray, x: np.ndarray, beta: np.ndarray, alpha: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    h_mats = observation_matrices(x, n)
    u = np.asarray([y[t] - h_mats[t] @ beta[t] for t in range(len(y))])
    y_star = np.empty_like(u)
    for t in range(len(y)):
        y_star[t] = alpha_to_a(alpha[t], n) @ u[t]
    return u, y_star


def ffbs_random_walk(
    y: np.ndarray,
    h_mat: np.ndarray,
    r_mat: np.ndarray,
    q_mat: np.ndarray,
    m0: np.ndarray,
    p0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    t_count, state_dim = h_mat.shape[0], m0.shape[0]
    filt_mean = np.empty((t_count, state_dim))
    filt_cov = np.empty((t_count, state_dim, state_dim))
    pred_cov = np.empty((t_count, state_dim, state_dim))

    mean_prev = m0
    cov_prev = _with_jitter(p0)
    q_mat = _with_jitter(q_mat)
    eye_state = np.eye(state_dim)

    for t in range(t_count):
        mean_pred = mean_prev
        cov_pred = _with_jitter(cov_prev + q_mat)
        pred_cov[t] = cov_pred

        h_t = h_mat[t]
        r_t = _with_jitter(r_mat[t])
        forecast_cov = _with_jitter(h_t @ cov_pred @ h_t.T + r_t)
        gain = cov_pred @ h_t.T @ np.linalg.pinv(forecast_cov)
        innovation = y[t] - h_t @ mean_pred
        mean_filt = mean_pred + gain @ innovation
        update = eye_state - gain @ h_t
        cov_filt = _with_jitter(update @ cov_pred @ update.T + gain @ r_t @ gain.T)

        filt_mean[t] = mean_filt
        filt_cov[t] = cov_filt
        mean_prev = mean_filt
        cov_prev = cov_filt

    states = np.empty((t_count, state_dim))
    states[-1] = _sample_mvn(filt_mean[-1], filt_cov[-1], rng)
    for t in range(t_count - 2, -1, -1):
        next_pred_cov = _with_jitter(filt_cov[t] + q_mat)
        smoother_gain = filt_cov[t] @ np.linalg.pinv(next_pred_cov)
        mean = filt_mean[t] + smoother_gain @ (states[t + 1] - filt_mean[t])
        cov = _with_jitter(filt_cov[t] - smoother_gain @ next_pred_cov @ smoother_gain.T)
        states[t] = _sample_mvn(mean, cov, rng)

    return states


def draw_beta(
    y: np.ndarray,
    x: np.ndarray,
    alpha: np.ndarray,
    h: np.ndarray,
    q_mat: np.ndarray,
    priors: PriorSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    n = y.shape[1]
    h_mats = observation_matrices(x, n)
    r_mats = omega_from_alpha_h(alpha, h, n)
    return ffbs_random_walk(y, h_mats, r_mats, q_mat, priors.beta_mean, 4.0 * priors.beta_cov, rng)


def draw_alpha(
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    h: np.ndarray,
    s_blocks: list[np.ndarray],
    priors: PriorSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    n = y.shape[1]
    u, _ = structural_residuals(y, x, beta, np.tile(priors.alpha_mean, (len(y), 1)), n)
    draws: list[np.ndarray] = []
    cursor = 0
    for row in range(1, n):
        width = row
        obs_y = u[:, row : row + 1]
        obs_h = -u[:, :row][:, np.newaxis, :]
        obs_r = np.exp(2.0 * h[:, row])[:, np.newaxis, np.newaxis]
        mean0 = priors.alpha_mean[cursor : cursor + width]
        cov0 = priors.alpha_cov_blocks[row - 1]
        draws.append(ffbs_random_walk(obs_y, obs_h, obs_r, s_blocks[row - 1], mean0, cov0, rng))
        cursor += width
    return np.concatenate(draws, axis=1)


def draw_h_given_indicators(
    y_star: np.ndarray,
    indicators: np.ndarray,
    w_mat: np.ndarray,
    priors: PriorSpec,
    rng: np.random.Generator,
    offset: float = 0.001,
) -> np.ndarray:
    t_count, n = y_star.shape
    y_log = np.log(y_star**2 + offset)

    obs_y = np.empty_like(y_log)
    obs_r = np.empty((t_count, n, n))
    obs_h = np.tile(2.0 * np.eye(n), (t_count, 1, 1))
    for t in range(t_count):
        means = KSC_MEAN[indicators[t]] - KSC_CENTERING
        vars_t = KSC_VAR[indicators[t]]
        obs_y[t] = y_log[t] - means
        obs_r[t] = np.diag(vars_t)

    return ffbs_random_walk(obs_y, obs_h, obs_r, w_mat, priors.h_mean, priors.h_cov, rng)


def draw_indicators(
    y_star: np.ndarray,
    h: np.ndarray,
    rng: np.random.Generator,
    offset: float = 0.001,
) -> np.ndarray:
    t_count, n = y_star.shape
    y_log = np.log(y_star**2 + offset)
    new_indicators = np.empty((t_count, n), dtype=int)
    for t in range(t_count):
        for i in range(n):
            means = 2.0 * h[t, i] + KSC_MEAN - KSC_CENTERING
            log_probs = (
                np.log(KSC_PROB)
                - 0.5 * np.log(2.0 * np.pi * KSC_VAR)
                - 0.5 * ((y_log[t, i] - means) ** 2) / KSC_VAR
            )
            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / probs.sum()
            new_indicators[t, i] = rng.choice(len(KSC_PROB), p=probs)

    return new_indicators


def _draw_iw(df: int, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draw = invwishart.rvs(df=df, scale=_with_jitter(scale), random_state=rng)
    return _with_jitter(np.atleast_2d(draw))


def draw_hyperparameters(
    beta: np.ndarray,
    alpha: np.ndarray,
    h: np.ndarray,
    priors: PriorSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    beta_diff = np.vstack([beta[0] - priors.beta_mean, np.diff(beta, axis=0)])
    q_scale = priors.q_scale + beta_diff.T @ beta_diff
    q_draw = _draw_iw(priors.q_df + beta_diff.shape[0], q_scale, rng)

    h_diff = np.vstack([h[0] - priors.h_mean, np.diff(h, axis=0)])
    w_scale = priors.w_scale + h_diff.T @ h_diff
    w_draw = _draw_iw(priors.w_df + h_diff.shape[0], w_scale, rng)

    s_draws: list[np.ndarray] = []
    cursor = 0
    for block_index, scale_prior in enumerate(priors.s_scales):
        width = block_index + 1
        block = alpha[:, cursor : cursor + width]
        mean0 = priors.alpha_mean[cursor : cursor + width]
        diff = np.vstack([block[0] - mean0, np.diff(block, axis=0)])
        scale = scale_prior + diff.T @ diff
        s_draws.append(_draw_iw(priors.s_dfs[block_index] + diff.shape[0], scale, rng))
        cursor += width

    return q_draw, s_draws, w_draw


def run_mcmc(
    data: pd.DataFrame,
    lags: int = 2,
    prior_obs: int = 40,
    iterations: int = 800,
    burn: int = 300,
    seed: int = 123,
    k_q: float = 0.01,
    k_s: float = 0.1,
    k_w: float = 0.01,
    progress: bool = True,
) -> TvpSvarResult:
    rng = np.random.default_rng(seed)
    priors = calibrate_priors(data, lags=lags, prior_obs=prior_obs, k_q=k_q, k_s=k_s, k_w=k_w)
    y, x, dates = estimation_sample(data, lags=lags, prior_obs=prior_obs)
    t_count, n = y.shape

    beta = np.tile(priors.beta_mean, (t_count, 1))
    alpha = np.tile(priors.alpha_mean, (t_count, 1))
    h = np.tile(priors.h_mean, (t_count, 1))
    indicators = rng.choice(len(KSC_PROB), size=(t_count, n), p=KSC_PROB)

    q_mat = _with_jitter(priors.q_scale / max(priors.q_df, 1))
    w_mat = _with_jitter(priors.w_scale / max(priors.w_df, 1))
    s_blocks = [_with_jitter(scale / max(df, 1)) for scale, df in zip(priors.s_scales, priors.s_dfs)]

    kept_beta: list[np.ndarray] = []
    kept_alpha: list[np.ndarray] = []
    kept_h: list[np.ndarray] = []

    for iteration in range(1, iterations + 1):
        # Del Negro and Primiceri's corrigendum samples the mixture indicators
        # immediately before the volatility history.
        beta = draw_beta(y, x, alpha, h, q_mat, priors, rng)
        alpha = draw_alpha(y, x, beta, h, s_blocks, priors, rng)
        q_mat, s_blocks, w_mat = draw_hyperparameters(beta, alpha, h, priors, rng)
        _, y_star = structural_residuals(y, x, beta, alpha, n)
        indicators = draw_indicators(y_star, h, rng)
        h = draw_h_given_indicators(y_star, indicators, w_mat, priors, rng)

        if iteration > burn:
            kept_beta.append(beta.copy())
            kept_alpha.append(alpha.copy())
            kept_h.append(h.copy())

        if progress and (iteration == 1 or iteration % max(1, iterations // 10) == 0):
            print(f"iteration {iteration}/{iterations}", flush=True)

    h_draws = np.asarray(kept_h)
    return TvpSvarResult(
        dates=dates,
        variables=list(data.columns),
        lags=lags,
        beta_draws=np.asarray(kept_beta),
        alpha_draws=np.asarray(kept_alpha),
        h_draws=h_draws,
        sigma_draws=np.exp(h_draws),
        iterations=iterations,
        burn=burn,
        seed=seed,
    )


def compute_policy_irf(
    beta_t: np.ndarray,
    alpha_t: np.ndarray,
    h_t: np.ndarray,
    lags: int,
    horizon: int = 20,
    policy_index: int = 2,
) -> np.ndarray:
    n = h_t.shape[0]
    a_t = alpha_to_a(alpha_t, n)
    sigma_t = np.diag(np.exp(h_t))
    shock = np.zeros(n)
    shock[policy_index] = 1.0
    impact = np.linalg.inv(a_t) @ sigma_t @ shock

    mats = beta_to_var_matrices(beta_t, n, lags)
    responses = np.zeros((horizon + 1, n))
    responses[0] = impact
    for h_step in range(1, horizon + 1):
        response = np.zeros(n)
        for lag in range(1, lags + 1):
            if h_step - lag >= 0:
                response += mats[lag - 1] @ responses[h_step - lag]
        responses[h_step] = response
    return responses
