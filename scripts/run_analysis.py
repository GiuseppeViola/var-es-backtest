"""Run the analysis on the local price data (data/prices.csv).

Run from the repository root, after scripts/download_data.py:

    python scripts/run_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from varbacktest.backtest import (
    christoffersen_independence,
    conditional_coverage,
    exceptions,
    kupiec_pof,
    traffic_light,
)
from varbacktest.data import (
    ASML_2012_CASH_FRACTION,
    ASML_2012_EX_DATE,
    adjust_cash_distribution,
    align_dates,
    load_prices,
)
from varbacktest.returns import (
    autocorrelation,
    buy_and_hold_weights,
    excess_kurtosis,
    portfolio_returns,
    simple_returns,
)
from varbacktest.var_models import rolling_forecasts

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "prices.csv"
LAGS = [1, 5, 10]
METHODS = ["hs", "gauss", "fhs"]
LEVELS = {"var99": 0.99, "var975": 0.975}  # forecast column suffix -> confidence level


def describe(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tables, one row per series: (excess kurtosis + ACF of r, ACF of r^2)."""
    tails, clustering = {}, {}
    for name, r in returns.items():
        tails[name] = {"excess_kurtosis": excess_kurtosis(r)}
        tails[name].update({f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r, LAGS))})
        clustering[name] = {f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r**2, LAGS))}
    return pd.DataFrame(tails).T, pd.DataFrame(clustering).T


def coverage_tests(hits: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, VaR level): exception count and the three LR tests."""
    rows = {}
    for column, h in hits.items():
        level = LEVELS[column.split("_")[1]]  # column "{method}_{suffix}", e.g. "hs_var99" -> 0.99
        lr_pof, p_pof = kupiec_pof(h, level)
        lr_ind, p_ind = christoffersen_independence(h)
        lr_cc, p_cc = conditional_coverage(h, level)
        rows[column] = {"T": len(h), "x": h.sum(), "expected": len(h) * (1 - level),
                        "x/T %": 100 * h.mean(), "LR_pof": lr_pof, "p_pof": p_pof,
                        "LR_ind": lr_ind, "p_ind": p_ind, "LR_cc": lr_cc, "p_cc": p_cc,
                        "reject_pof": p_pof < 0.05, "reject_cc": p_cc < 0.05}
    # Transposing mixed rows gives object columns: infer_objects restores int, float and bool.
    return pd.DataFrame(rows).T.infer_objects()


def main() -> None:
    raw = load_prices(DATA_FILE)
    fixed = adjust_cash_distribution(raw, "ASML.AS", ASML_2012_EX_DATE, ASML_2012_CASH_FRACTION)
    prices, n_dropped = align_dates(fixed)
    print(f"Aligned sample: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices)} dates ({n_dropped} dropped: not all exchanges open)")

    returns = simple_returns(prices)
    weights = buy_and_hold_weights(returns)
    r_p = portfolio_returns(returns, weights)

    # Final weights: buy-and-hold drifts towards the stocks that grew most.
    final = weights.iloc[-1] * (1 + returns.iloc[-1])
    print("\nBuy-and-hold weights at the last date:")
    print((final / final.sum()).round(3).to_string())

    tails, clustering = describe(pd.concat([returns, r_p], axis=1))
    band = 1.96 / np.sqrt(len(r_p))
    print(f"\n{len(r_p)} daily returns; i.i.d. band for the ACF: +-{band:.3f}")
    print("\nFat tails and linear predictability: excess kurtosis and ACF of r")
    print(tails.round(3).to_string())
    print("\nVolatility clustering: ACF of r^2")
    print(clustering.round(3).to_string())

    # Rolling one-day forecasts: row t uses only data up to t-1.
    forecasts = rolling_forecasts(returns, weights, window=250, burn_in=250, lam=0.94)
    pct = 100 * forecasts
    print(f"\nEvaluation period: {forecasts.index[0].date()} to {forecasts.index[-1].date()}, "
          f"{len(forecasts)} days. Forecasts, % of portfolio value:")
    summary = pd.DataFrame({"mean": pct.mean(), "min": pct.min(), "max": pct.max(),
                            "date of max": pct.idxmax().dt.date})
    print(summary.round(2).to_string())

    t = forecasts.index[-1]
    print(f"\nForecasts for {t.date()} (%):")
    print(pct.iloc[-1].round(2).to_string())
    print(f"Realised loss on {t.date()}: {-100 * r_p.iloc[-1]:.2f}%")

    # Backtesting: compare each forecast with the realised loss L_t = -r_{p,t} of the same day.
    losses = -r_p.loc[forecasts.index]
    # Hit sequences, one column per "{method}_{suffix}" (same names as the forecast columns).
    hits = pd.DataFrame({f"{m}_{suffix}": exceptions(losses, forecasts[f"{m}_{suffix}"])
                         for m in METHODS for suffix in LEVELS})

    print("\nCoverage tests (reject at 5%: p < 0.05; critical values chi2(1) 3.84, chi2(2) 5.99):")
    print(coverage_tests(hits).round(3).to_string())

    year = hits.index.year.rename("year")
    counts = hits.groupby(year).sum()
    days = hits.groupby(year).size()
    yearly = pd.concat([days.rename("days"),
                        (days * 0.01).round(1).rename("exp99"),
                        counts[[f"{m}_var99" for m in METHODS]],
                        (days * 0.025).round(1).rename("exp975"),
                        counts[[f"{m}_var975" for m in METHODS]]], axis=1)
    print("\nExceptions per calendar year (exp = expected):")
    print(yearly.to_string())

    print("\nBasel traffic light, VaR 99%, rolling 250 days: % of days in each zone")
    zones, last = {}, {}
    for m in METHODS:
        tl = traffic_light(hits[f"{m}_var99"])
        zones[m] = 100 * tl["zone"].value_counts(normalize=True)
        last[m] = f"{tl['zone'].iloc[-1]} ({tl['count'].iloc[-1]} exceptions)"
    shares = pd.DataFrame(zones).T
    # A zone never reached is missing from value_counts: show it as 0%.
    shares = shares.reindex(columns=["green", "yellow", "red"]).fillna(0)
    print(shares.round(1).to_string())
    print(f"Zone on the last 250 days (to {hits.index[-1].date()}):")
    for m in METHODS:
        print(f"  {m}: {last[m]}")


if __name__ == "__main__":
    main()
