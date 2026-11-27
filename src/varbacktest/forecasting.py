"""Rolling one-day forecasts: EWMA volatility and the engine that feeds the models.

The engine is the only place that decides which data a model sees. For each forecast
day t it builds a Snapshot with the returns of days t-n, ..., t-1, the weights w_{t-1}
and the EWMA volatilities, all known at the close of t-1, and asks every model for
VaR and ES. Changing any data from day t on cannot change the forecast for day t.
"""

import numpy as np
import pandas as pd

from .models import Snapshot, VaRModel
from .portfolio import Portfolio


def ewma_volatility(returns: pd.DataFrame, lam: float = 0.94, burn_in: int = 250) -> pd.DataFrame:
    """EWMA volatility forecasts per stock (RiskMetrics, zero mean).

    Row t is sigma_{i,t}, the forecast for day t built with returns up to t-1:
        sigma^2_{i,burn_in} = (1/burn_in) sum_{s < burn_in} r_{i,s}^2    (initial condition)
        sigma^2_{i,t}       = lam sigma^2_{i,t-1} + (1 - lam) r_{i,t-1}^2,   t > burn_in
    The initial condition is the zero-mean variance of the burn-in returns: a deliberate
    choice, the same zero-mean assumption as the recursion (the sample mean of 250 daily
    returns is estimated with more noise than its size).
    Rows before burn_in are NaN: no forecast without enough past data. With lam = 0.94
    the weight of r^2_{t-k} decays as lam^(k-1): half-life ln(0.5)/ln(lam), about 11 days.
    Rows are counted on the returns index (row 0 = first return).
    Units: volatility per day, fraction of value. No sign, no confidence level.
    Data used: for row t, returns up to t-1.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be strictly between 0 and 1")
    r = returns.to_numpy(dtype=float)
    if not 0 < burn_in < len(r):
        raise ValueError("burn_in must be between 1 and the number of days - 1")

    variance = np.full_like(r, np.nan)
    variance[burn_in] = np.mean(r[:burn_in] ** 2, axis=0)  # returns 0, ..., burn_in-1
    for t in range(burn_in + 1, len(r)):
        variance[t] = lam * variance[t - 1] + (1 - lam) * r[t - 1] ** 2  # r[t-1], never r[t]
    return pd.DataFrame(np.sqrt(variance), index=returns.index, columns=returns.columns)


def rolling_forecasts(
    portfolio: Portfolio,
    models: list[VaRModel],
    window: int = 250,
    burn_in: int = 250,
    lam: float = 0.94,
) -> dict[str, pd.DataFrame]:
    """Daily one-day VaR 99%, VaR 97.5% and ES 97.5% forecasts for every model.

    window: number of past days n in each snapshot. For each forecast day t (counted on
    the returns index) the Snapshot holds:
        returns r_{i,s} and EWMA sigma_{i,s} for s = t-window, ..., t-1,
        weights w_{t-1} (portfolio.weights_held, row t) and sigma_{i,t}.
    Evaluation period: t >= burn_in + window, the same for every model. FHS needs
    sigma_s on every day of its window, and the first EWMA forecast is sigma_{burn_in};
    HS and Gaussian could start earlier, but use the same dates so that all models
    are backtested on the same sample.

    Output: {model.name: DataFrame}, one row per forecast day t, columns var_99,
    var_975, es_975. Units: fraction of portfolio value. Sign: positive for a loss.
    Confidence levels: 99% and 97.5%. Data used: for row t, data up to t-1 only.
    """
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    returns = portfolio.returns.to_numpy()
    weights_held = portfolio.weights_held.to_numpy()  # row t = w_{t-1}
    sigma = ewma_volatility(portfolio.returns, lam, burn_in).to_numpy()
    first = burn_in + window
    if window < 2 or first >= len(returns):
        raise ValueError("need window >= 2 and more than burn_in + window days")

    rows: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    for t in range(first, len(returns)):
        past = slice(t - window, t)  # days t-window, ..., t-1: never day t
        snapshot = Snapshot(
            returns=returns[past],
            weights=weights_held[t],
            sigma_past=sigma[past],
            sigma_now=sigma[t],
        )
        for model in models:
            # The measures of the specification: VaR at 99%, VaR and ES at 97.5%.
            at_99 = model.estimate(snapshot, 0.99)
            at_975 = model.estimate(snapshot, 0.975)
            rows[model.name].append(
                {"var_99": at_99.var, "var_975": at_975.var, "es_975": at_975.es}
            )

    dates = portfolio.returns.index[first:]
    return {name: pd.DataFrame(rows[name], index=dates) for name in names}
