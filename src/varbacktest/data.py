"""Loading and date alignment of the price data."""

from pathlib import Path

import pandas as pd


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read adjusted close prices from the CSV written by scripts/download_data.py.

    Units: prices in EUR (not returns, so no sign convention and no confidence level).
    Data used: the whole file, as downloaded; missing values are kept.

    Returns a DataFrame indexed by date (sorted), one column per ticker.
    """
    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    return prices.sort_index()


# ASML synthetic share buyback, ex-date 2012-11-26: cash capital repayment of EUR 9.18
# per old share plus a 77:100 reverse split. Yahoo adjusts the split but not the cash,
# so the ex-date shows a fake -20.8% return. Cash per new share = 9.18 * 100/77 EUR;
# previous close (split-adjusted, new shares) = 57.922077 EUR.
# Source: https://www.sec.gov/Archives/edgar/data/0000937966/000119312512470856/d439738dex991.htm
ASML_2012_EX_DATE = "2012-11-26"
ASML_2012_CASH_FRACTION = (9.18 * 100 / 77) / 57.922077


def adjust_cash_distribution(
    prices: pd.DataFrame, ticker: str, ex_date: str, cash_fraction: float
) -> pd.DataFrame:
    """Adjust for a cash distribution that the data provider did not adjust.

    Same rule Yahoo uses for dividends: every price of `ticker` before `ex_date`
    is multiplied by (1 - cash_fraction), where cash_fraction = cash / previous close.
    The return on the ex-date then includes the cash received; all other returns
    are unchanged (a constant factor cancels in P_t / P_{t-1}).

    Units: prices in EUR (no sign convention, no confidence level).
    Data used: the whole price table.
    """
    # Guard against adjusting twice: if the provider has already adjusted the cash,
    # the raw ex-date return no longer shows a drop of about cash_fraction.
    series = prices[ticker].dropna()
    ex = pd.Timestamp(ex_date)
    raw_return = series.loc[ex] / series[series.index < ex].iloc[-1] - 1
    if raw_return > -cash_fraction / 2:
        raise ValueError(f"{ticker} on {ex_date} looks already adjusted (return {raw_return:.1%})")

    adjusted = prices.copy()
    before = adjusted.index < ex
    adjusted.loc[before, ticker] *= 1 - cash_fraction
    return adjusted


def align_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only the dates on which every stock has a price (inner join).

    The exchanges have different holidays. Filling the gaps with the previous
    price would create fake zero returns and understate volatility, so those
    dates are dropped instead. Returns must be computed AFTER this step: the
    return on the next common date then covers the whole period since the
    previous common date, and no price move is lost.

    Units: prices in EUR (no sign convention, no confidence level).
    Data used: the whole price table.

    Returns the aligned prices and the number of dropped dates.
    """
    aligned = prices.dropna(how="any")
    n_dropped = len(prices) - len(aligned)
    return aligned, n_dropped
