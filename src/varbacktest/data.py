"""Price data: loading and date alignment.

The input is data/prices.csv written by scripts/download_data.py: adjusted close
prices (dividends reinvested in the same stock), one column per stock, one row
per date on which at least one exchange was open.
"""

from pathlib import Path

import pandas as pd


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read the adjusted close prices from the CSV file.

    Output: one column per stock, indexed by date (increasing, unique). Missing
    prices (an exchange closed) are NaN. Units: EUR per share. No sign convention,
    no confidence level. Data used: the whole file.
    """
    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    if not (prices.index.is_monotonic_increasing and prices.index.is_unique):
        raise ValueError("dates must be increasing and unique")
    return prices


def align_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only the dates on which every stock has a price.

    Returns (aligned prices, number of dates dropped). A date is dropped if at least
    one exchange was closed: a portfolio return needs the return of every stock.
    Units: EUR per share. Data used: the whole price table.
    """
    aligned = prices.dropna(how="any")
    n_dropped = len(prices) - len(aligned)
    return aligned, n_dropped
