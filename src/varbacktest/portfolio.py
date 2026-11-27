"""Buy-and-hold equity portfolio: stock returns, weights, portfolio returns and losses.

Time convention. Prices of N stocks are observed at the close of dates 0, 1, ..., T.
    r_{i,t} = P_{i,t} / P_{i,t-1} - 1                    return of stock i on day t (t >= 1)
    w_{i,0} = 1 / N                                       equal weights at the first date
    w_{i,t} = w_{i,t-1} (1 + r_{i,t}) / sum_j w_{j,t-1} (1 + r_{j,t})
                                                          weights at the close of day t
    r_{p,t} = sum_i w_{i,t-1} r_{i,t}                     portfolio return on day t
    L_t     = -r_{p,t}                                    loss on day t
The return of day t is earned with the weights of the close of day t-1 (positions
fixed at t-1: hypothetical P&L). No rebalancing: each position grows with its own
stock, and the weights drift towards the stocks that grew most.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd


# eq=False: comparing DataFrames with == gives a DataFrame, not True/False.
@dataclass(frozen=True, eq=False)
class Portfolio:
    """Buy-and-hold portfolio with equal initial weights, built from aligned prices.

    prices: adjusted close prices, one column per stock, indexed by date (increasing,
            unique), no missing values (see data.align_dates), all positive.

    Two weight tables, both summing to 1 on each row:
        weights       T+1 rows, dates 0..T     row t = w_t, weights at the close of day t
        weights_held  T rows,   dates 1..T     row t = w_{t-1}, weights held during day t
    weights_held has the same rows as returns; the VaR models use only weights_held.

    Frozen: the portfolio is defined by its prices; returns, weights and losses are
    derived from them. They are cached properties: computed at the first access, then
    stored, so there is a single source of truth (the prices) and no state to keep in
    sync. cached_property works on a frozen dataclass because it writes the value
    directly into the instance __dict__, without the __setattr__ that frozen blocks
    (it would not work with slots=True, which removes __dict__). The constructor keeps
    its own float copy of the prices, so later changes to the caller's DataFrame cannot
    make the cache stale. As in any Python object, the DataFrames returned can still be
    modified in place by the caller: frozen is not deep immutability.

    Units: prices in EUR per share; returns, weights and losses as fractions.
    Sign: losses are positive for a fall in portfolio value. No confidence level.
    Data used: returns, weights and losses of day t use prices up to day t only.
    """

    prices: pd.DataFrame

    def __post_init__(self) -> None:
        prices = self.prices
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame, one column per stock")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must be indexed by date")
        if not (prices.index.is_monotonic_increasing and prices.index.is_unique):
            raise ValueError("dates must be increasing and unique")
        if len(prices) < 2 or prices.shape[1] < 1:
            raise ValueError("need at least 2 dates and 1 stock")
        if not prices.columns.is_unique:
            raise ValueError("each stock must appear in one column only")
        values = prices.to_numpy(dtype=float)  # raises if a column is not numeric
        if np.isnan(values).any():
            raise ValueError("prices contain missing values: align the dates first")
        if not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError("prices must be finite and positive")
        # Own float copy: the cache below cannot go stale if the caller changes its DataFrame.
        object.__setattr__(self, "prices", prices.astype(float))

    @property
    def stocks(self) -> list[str]:
        """Names of the stocks, in column order."""
        return list(self.prices.columns)

    @cached_property
    def returns(self) -> pd.DataFrame:
        """r_{i,t} = P_{i,t} / P_{i,t-1} - 1, one row per day t = 1, ..., T.

        Simple returns, fractions of value. Data used: prices of days t-1 and t.
        """
        previous_close = self.prices.shift(1)
        return (self.prices / previous_close - 1).iloc[1:]

    @cached_property
    def weights(self) -> pd.DataFrame:
        """w_{i,t}: buy-and-hold weights at the close of day t, one row per date 0, ..., T.

        Row 0 holds the equal initial weights 1/N; each row sums to 1.
        The loop follows the recursion of the specification. Without rebalancing the
        number of shares is fixed, so the same weights have a closed form:
            w_{i,t} = (P_{i,t} / P_{i,0}) / sum_j (P_{j,t} / P_{j,0})
        each stock weighs in proportion to how much it has grown since date 0.
        Fractions of portfolio value. Data used: returns up to day t.
        """
        r = self.returns.to_numpy()
        n_days, n_stocks = r.shape
        w = np.empty((n_days + 1, n_stocks))
        w[0] = 1.0 / n_stocks
        for t in range(1, n_days + 1):
            grown = w[t - 1] * (1 + r[t - 1])  # r[t - 1] is the return of day t (row 0 = day 1)
            w[t] = grown / grown.sum()
        return pd.DataFrame(w, index=self.prices.index, columns=self.prices.columns)

    @cached_property
    def weights_held(self) -> pd.DataFrame:
        """w_{i,t-1}: the weights held during day t, one row per day t = 1, ..., T.

        Same rows as `returns`, so the two can be multiplied row by row: row t is the
        position at the close of day t-1, the one exposed to the return of day t.
        The VaR models use this row as the current weights for the forecast of day t.
        Fractions of portfolio value. Data used: returns up to day t-1.
        """
        held = self.weights.iloc[:-1]  # w_0, ..., w_{T-1}
        return held.set_axis(self.returns.index)

    @cached_property
    def portfolio_returns(self) -> pd.Series:
        """r_{p,t} = sum_i w_{i,t-1} r_{i,t}, one value per day t = 1, ..., T.

        Fraction of portfolio value; positive for a gain. Data used: prices up to day t.
        """
        return (self.weights_held * self.returns).sum(axis=1).rename("r_p")

    @cached_property
    def losses(self) -> pd.Series:
        """L_t = -r_{p,t}, one value per day t = 1, ..., T.

        Fraction of portfolio value; positive for a loss. Data used: prices up to day t.
        """
        return (-self.portfolio_returns).rename("loss")
