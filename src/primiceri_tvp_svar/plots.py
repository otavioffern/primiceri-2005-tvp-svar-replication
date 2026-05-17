from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import TvpSvarResult, compute_policy_irf


LABELS = {
    "inflation": "Inflation",
    "unemployment": "Unemployment",
    "tbill_3m": "3-month T-bill",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_data(data: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col in zip(axes, data.columns):
        ax.plot(data.index, data[col], color="#1f4e79", linewidth=1.3)
        ax.set_title(LABELS.get(col, col), loc="left", fontsize=10)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("FRED proxy data for the Primiceri (2005) three-variable system", fontsize=12)
    _save(fig, path)


def plot_volatilities(result: TvpSvarResult, path: Path) -> None:
    sigma = result.sigma_draws
    mean = sigma.mean(axis=0)
    p16 = np.quantile(sigma, 0.16, axis=0)
    p84 = np.quantile(sigma, 0.84, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.fill_between(result.dates, p16[:, i], p84[:, i], color="#9bb7d4", alpha=0.45, label="16th-84th")
        ax.plot(result.dates, mean[:, i], color="#1f4e79", linewidth=1.4, label="posterior mean")
        ax.set_title(f"Structural shock volatility: {LABELS.get(result.variables[i], result.variables[i])}", loc="left", fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("TVP-SVAR stochastic volatilities", fontsize=12)
    _save(fig, path)


def _nearest_date_index(dates: pd.DatetimeIndex, quarter: str) -> int:
    target = pd.Period(quarter, freq="Q").to_timestamp()
    distances = np.abs((dates - target).days)
    return int(np.argmin(distances))


def plot_policy_irfs(result: TvpSvarResult, path: Path, horizon: int = 20) -> None:
    selected = ["1975Q1", "1981Q3", "1996Q1"]
    colors = ["#1f4e79", "#b04a3a", "#3b7d3a"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x_axis = np.arange(horizon + 1)
    for quarter, color in zip(selected, colors):
        idx = _nearest_date_index(result.dates, quarter)
        irf = compute_policy_irf(
            result.beta_mean[idx],
            result.alpha_mean[idx],
            result.h_mean[idx],
            lags=result.lags,
            horizon=horizon,
            policy_index=2,
        )
        axes[0].plot(x_axis, irf[:, 0], color=color, linewidth=1.5, label=quarter)
        axes[1].plot(x_axis, irf[:, 1], color=color, linewidth=1.5, label=quarter)

    axes[0].set_title("Inflation response to a monetary policy shock", loc="left", fontsize=10)
    axes[1].set_title("Unemployment response to a monetary policy shock", loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Quarters after shock")
    fig.suptitle("Posterior-mean impulse responses at selected dates", fontsize=12)
    _save(fig, path)


def build_summary_table(result: TvpSvarResult) -> pd.DataFrame:
    sigma_mean = result.sigma_draws.mean(axis=0)
    dates = result.dates
    periods = {
        "pre_volcker_1963Q1_1979Q2": ("1963-01-01", "1979-04-01"),
        "volcker_transition_1979Q3_1983Q4": ("1979-07-01", "1983-10-01"),
        "post_1984Q1_2001Q3": ("1984-01-01", "2001-07-01"),
    }
    rows = []
    for period, (start, end) in periods.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        values = sigma_mean[mask]
        row = {"period": period}
        for i, var in enumerate(result.variables):
            row[f"{var}_shock_std_mean"] = float(values[:, i].mean())
        rows.append(row)
    return pd.DataFrame(rows)

