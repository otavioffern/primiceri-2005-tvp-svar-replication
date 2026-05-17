from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from primiceri_tvp_svar.data import load_or_fetch
from primiceri_tvp_svar.model import run_mcmc
from primiceri_tvp_svar.plots import build_summary_table, plot_data, plot_policy_irfs, plot_volatilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Primiceri (2005) TVP-SVAR replication.")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--burn", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def write_markdown_summary(path: Path, result, summary_table) -> None:
    sigma_policy = result.sigma_draws[:, :, 2]
    final_policy_mean = float(sigma_policy.mean(axis=0)[-1])
    volcker_row = summary_table.loc[summary_table["period"].str.contains("volcker")].iloc[0]
    post_row = summary_table.loc[summary_table["period"].str.contains("post")].iloc[0]
    ratio = volcker_row["tbill_3m_shock_std_mean"] / post_row["tbill_3m_shock_std_mean"]

    text = f"""# Replication run summary

This run estimates a three-variable Primiceri-style TVP-SVAR with FRED proxies for the data used in the paper.

- MCMC iterations: {result.iterations}
- Burn-in: {result.burn}
- Retained draws: {result.sigma_draws.shape[0]}
- Seed: {result.seed}
- Sample used for estimation after prior calibration: {result.dates[0].date()} to {result.dates[-1].date()}
- Variables: {", ".join(result.variables)}

## Main diagnostic numbers

The posterior-mean standard deviation of the monetary policy shock is {ratio:.2f} times larger in 1979Q3-1983Q4 than in 1984Q1-2001Q3 in this FRED-proxy run.

The final posterior-mean standard deviation of the monetary policy shock is {final_policy_mean:.3f}.

These numbers should be read as a replication attempt, not as an exact reproduction of the published tables. The original paper uses S&P DRI data; this repository uses public FRED proxies so the workflow is reproducible without proprietary data.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.burn >= args.iterations:
        raise ValueError("--burn must be smaller than --iterations")

    data_path = ROOT / "data" / "processed" / "primiceri_fred_proxy.csv"
    figures_dir = ROOT / "outputs" / "figures"
    data = load_or_fetch(data_path, refresh=args.refresh_data)

    plot_data(data, figures_dir / "01_fred_proxy_data.png")
    result = run_mcmc(
        data,
        iterations=args.iterations,
        burn=args.burn,
        seed=args.seed,
        progress=not args.no_progress,
    )

    plot_volatilities(result, figures_dir / "02_stochastic_volatilities.png")
    plot_policy_irfs(result, figures_dir / "03_policy_shock_irfs.png")

    summary_table = build_summary_table(result)
    summary_table.to_csv(ROOT / "outputs" / "posterior_volatility_summary.csv", index=False)
    write_markdown_summary(ROOT / "outputs" / "replication_summary.md", result, summary_table)

    np.savez_compressed(
        ROOT / "outputs" / "posterior_draws_quick.npz",
        dates=result.dates.astype(str).to_numpy(),
        variables=np.asarray(result.variables),
        beta_mean=result.beta_mean,
        alpha_mean=result.alpha_mean,
        h_mean=result.h_mean,
        sigma_mean=result.sigma_draws.mean(axis=0),
    )

    print("Wrote:")
    print(f"  {data_path}")
    print(f"  {figures_dir / '01_fred_proxy_data.png'}")
    print(f"  {figures_dir / '02_stochastic_volatilities.png'}")
    print(f"  {figures_dir / '03_policy_shock_irfs.png'}")
    print(f"  {ROOT / 'outputs' / 'posterior_volatility_summary.csv'}")
    print(f"  {ROOT / 'outputs' / 'replication_summary.md'}")


if __name__ == "__main__":
    main()

