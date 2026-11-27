"""Price data: loading, one cash-distribution correction, date alignment.

The input is data/prices.csv written by scripts/download_data.py: adjusted close
prices (dividends reinvested in the same stock), one column per stock, one row
per date on which at least one exchange was open.
"""

from pathlib import Path

import pandas as pd

# ASML, November 2012: synthetic share buyback = cash repayment of 9.18 EUR per old
# share plus a reverse split of 77 new shares for 100 old ones (ex-date 26 Nov 2012).
# Yahoo adjusted the split but not the repayment, so the adjusted close falls by
# about 21% on the ex-date: a loss that no shareholder actually suffered.
ASML_2012_EX_DATE = "2012-11-26"
ASML_2012_CASH_PER_OLD_SHARE = 9.18           # EUR
ASML_2012_NEW_SHARES_PER_OLD_SHARE = 77 / 100
ASML_2012_CLOSE_BEFORE_EX_DATE = 57.922077    # EUR, Yahoo close of 23 Nov 2012 (split-adjusted)
# Written here, not read from prices.csv: the CSV holds adjusted closes, which are also
# scaled down by every later dividend, while the cash fraction must be computed on the
# close actually quoted that day (adjusted for the split only, not for dividends).
# Cash per new share, as a fraction of that close (about 0.206).
ASML_2012_CASH_FRACTION = (
    ASML_2012_CASH_PER_OLD_SHARE / ASML_2012_NEW_SHARES_PER_OLD_SHARE
) / ASML_2012_CLOSE_BEFORE_EX_DATE


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


def adjust_cash_distribution(
    prices: pd.DataFrame, stock: str, ex_date: str, cash_fraction: float
) -> pd.DataFrame:
    """Treat a cash distribution as reinvested in the same stock, as Yahoo does for dividends.

    Every price of `stock` strictly before `ex_date` is multiplied by
        1 - cash_fraction,
        cash_fraction = cash per share / last unadjusted close before the ex-date,
    so the return on the ex-date no longer contains the fall due to the cash paid out.
    Returns before the ex-date are unchanged (all those prices are scaled by the same
    factor). Other stocks and other dates are unchanged; a new DataFrame (float) is returned.
    Apply it before align_dates: the reference is the last close of `stock` itself,
    whatever the other exchanges did that day.

    Guard (a heuristic): the return on the ex-date, before the correction, must show a
    fall of at least half the distribution; otherwise the prices look already adjusted
    and a ValueError is raised (adjusting twice would create a fictitious gain). It can
    be fooled only by a market move larger than half the distribution on that very day.
    Units: prices in EUR per share, cash_fraction as a fraction of the price.
    Data used: the price series of `stock` only.
    """
    if not 0 < cash_fraction < 1:
        raise ValueError("cash_fraction must be strictly between 0 and 1")
    if stock not in prices.columns:
        raise ValueError(f"unknown stock {stock}")
    ex_date = pd.Timestamp(ex_date)
    series = prices[stock]
    if ex_date not in series.index or pd.isna(series[ex_date]):
        raise ValueError(f"no price of {stock} on the ex-date {ex_date.date()}")
    before = series.index < ex_date
    prices_before = series[before].dropna()  # dropna: skip days when its exchange was closed
    if prices_before.empty:
        raise ValueError(f"no price of {stock} before the ex-date {ex_date.date()}")
    raw_return = series[ex_date] / prices_before.iloc[-1] - 1
    if raw_return > -cash_fraction / 2:
        raise ValueError(f"{stock} does not fall on {ex_date.date()}: already adjusted?")

    adjusted = prices.astype(float)  # a new DataFrame; float, so scaled prices fit
    adjusted.loc[before, stock] = series[before] * (1 - cash_fraction)
    return adjusted


def align_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only the dates on which every stock has a price.

    Returns (aligned prices, number of dates dropped). A date is dropped if at least
    one exchange was closed: a portfolio return needs the return of every stock.
    Units: EUR per share. Data used: the whole price table.
    """
    aligned = prices.dropna(how="any")
    n_dropped = len(prices) - len(aligned)
    return aligned, n_dropped
