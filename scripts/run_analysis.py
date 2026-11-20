"""Run the analysis on the local price data (data/prices.csv).

Run from the repository root, after scripts/download_data.py:

    python scripts/run_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

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

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "prices.csv"
LAGS = [1, 5, 10]


def describe(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tables, one row per series: (excess kurtosis + ACF of r, ACF of r^2)."""
    tails, clustering = {}, {}
    for name, r in returns.items():
        tails[name] = {"excess_kurtosis": excess_kurtosis(r)}
        tails[name].update({f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r, LAGS))})
        clustering[name] = {f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r**2, LAGS))}
    return pd.DataFrame(tails).T, pd.DataFrame(clustering).T


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


if __name__ == "__main__":
    main()
