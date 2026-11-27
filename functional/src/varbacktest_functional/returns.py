"""Stock returns, buy-and-hold weights, portfolio returns and descriptive statistics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns r_t = P_t / P_{t-1} - 1, one column per stock.

    Units: fraction of value (0.01 = +1%). Sign: positive = gain; the loss is L_t = -r_t.
    No confidence level. Data used: prices at t and t-1.

    The first date has no previous price, so the output starts on the second date.
    Simple (not log) returns are used because the portfolio return is then exactly
    the weighted sum of the stock returns.
    """
    return (prices / prices.shift(1) - 1).iloc[1:]


def buy_and_hold_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Portfolio weights held at the START of each day t, i.e. w_{t-1}.

    Buy-and-hold: equal weights (1/n) on the first date of the sample, then the
    number of shares never changes and the weights drift with prices:
        w_{i,t} = w_{i,t-1} (1 + r_{i,t}) / sum_j w_{j,t-1} (1 + r_{j,t}).

    Row t contains the weights known at the close of t-1, so a forecast for day t
    can use row t without look-ahead. The first row is 1/n (weights set on the
    first price date, which has no return).

    Units: fractions of portfolio value, summing to 1. No sign convention, no
    confidence level. Data used: returns up to t-1.
    """
    r = returns.to_numpy()
    n_assets = r.shape[1]
    w = np.full(n_assets, 1.0 / n_assets)  # weights on the first price date
    held = np.empty_like(r)
    for t in range(len(r)):
        held[t] = w                # weights held during day t (= w_{t-1})
        grown = w * (1 + r[t])     # value of each position at the close of t
        w = grown / grown.sum()    # w_t, used on day t+1
    return pd.DataFrame(held, index=returns.index, columns=returns.columns)


def portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio return r_{p,t} = sum_i w_{i,t-1} r_{i,t} (hypothetical P&L).

    `weights` must come from buy_and_hold_weights: row t already holds w_{t-1},
    so no shift is needed here. Positions are frozen at t-1 for the whole day.

    Units: fraction of portfolio value. Sign: positive = gain; L_t = -r_{p,t}.
    No confidence level. Data used: weights at t-1, returns at t.
    """
    if not returns.index.equals(weights.index):
        raise ValueError("returns and weights must have the same dates")
    if not returns.columns.equals(weights.columns):
        raise ValueError("returns and weights must have the same stocks")
    # skipna=False: a missing return gives a missing portfolio return, not a silent 0.
    return (weights * returns).sum(axis=1, skipna=False).rename("portfolio")


def excess_kurtosis(x: Sequence[float]) -> float:
    """Excess kurtosis m4 / m2^2 - 3, with population moments (divide by n).

    Equal to 0 for a normal distribution; positive values mean fat tails.
    No small-sample correction (unlike pandas .kurt()).

    Units: dimensionless. Sign convention and confidence level: not applicable.
    Data used: the whole sample passed in.
    """
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    m2 = np.mean(d**2)
    m4 = np.mean(d**4)
    return m4 / m2**2 - 3


def autocorrelation(x: Sequence[float], lags: Sequence[int]) -> np.ndarray:
    """Sample autocorrelation at the given lags (standard ACF estimator).

        rho_k = sum_t (x_t - m)(x_{t-k} - m) / sum_t (x_t - m)^2,

    with m the mean of the whole sample. Applied to r it measures linear
    predictability; applied to r^2 it measures volatility clustering.

    Units: dimensionless, between -1 and 1. Sign convention and confidence level:
    not applicable. Data used: the whole sample passed in.
    """
    x = np.asarray(x, dtype=float)
    if any(k < 1 or k >= len(x) for k in lags):
        raise ValueError("each lag must satisfy 1 <= lag < len(x)")
    d = x - x.mean()
    denom = np.sum(d**2)
    return np.array([np.sum(d[k:] * d[:-k]) / denom for k in lags])
