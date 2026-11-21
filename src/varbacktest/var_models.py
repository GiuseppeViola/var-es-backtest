"""One-day VaR and ES on a single window: historical, Gaussian and filtered historical simulation.

Every function receives the stock returns of the estimation window (days s = t-n, ..., t-1)
and the current weights w_{t-1}, and returns the forecast (VaR_t, ES_t) for day t.
All methods apply the scenarios to the current weights (static portfolio over one day).
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")


def _as_window_and_weights(window_returns: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert to arrays and check shapes: window (n days x k stocks), weights (k,).

    Arrays are matched by position: column i of the window goes with weights[i].
    If pandas objects are passed, the stock labels must match in the same order,
    so that a weight can never be applied to the wrong stock.
    """
    if isinstance(window_returns, pd.DataFrame) and isinstance(weights, pd.Series):
        if not window_returns.columns.equals(weights.index):
            raise ValueError("window_returns columns and weights index must be the same stocks, same order")
    r = np.asarray(window_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 2:
        raise ValueError("window_returns must be a 2D array (days x stocks)")
    if w.shape != (r.shape[1],):
        raise ValueError("weights must have one entry per stock (column of window_returns)")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(w))):
        raise ValueError("window_returns and weights must not contain NaN or inf")
    return r, w


def scenario_var_es(scenario_returns: ArrayLike, level: float) -> tuple[float, float]:
    """Empirical VaR and ES from a set of portfolio return scenarios.

    Scenario s gives a portfolio return X_s and a loss L_s = -X_s. VaR is their alpha-quantile with np.quantile (default
    'linear' method: position (n-1)*alpha in the sorted losses, linear interpolation
    between the two neighbours). ES is the mean of the losses >= VaR, the empirical
    version of E[L | L >= VaR]. With n = 250 and alpha = 0.975 this is the mean of
    the 7 largest losses (if there are no ties).

    Units: fraction of portfolio value. Sign: VaR and ES are positive for a loss.
    Confidence level: `level` (e.g. 0.99 or 0.975). Data used: the scenarios passed
    in, which must be built from data up to t-1 only.
    """
    _check_level(level)
    losses = -np.asarray(scenario_returns, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("scenario_returns must be a non-empty 1D array")
    if not np.all(np.isfinite(losses)):
        raise ValueError("scenario_returns must not contain NaN or inf")
    var = float(np.quantile(losses, level))
    es = float(losses[losses >= var].mean())
    return var, es


def historical_var_es(window_returns: ArrayLike, weights: ArrayLike, level: float) -> tuple[float, float]:
    """One-day VaR and ES by historical simulation (HS).

    Scenario s replays the stock returns of day s on today's portfolio:
        X_s = sum_i w_{i,t-1} r_{i,s},   s in the window,
    with i = stock (risk factor) and r_{i,s} its observed return on day s.
    VaR and ES are then read from the empirical distribution of L_s = -X_s
    (see scenario_var_es). No distributional assumption, but the n days are
    treated as equally likely (i.i.d.) scenarios for day t.

    Units: fraction of portfolio value. Sign: VaR and ES are positive for a loss.
    Confidence level: `level`. Data used: stock returns of the window (up to t-1)
    and the weights w_{t-1}.
    """
    r, w = _as_window_and_weights(window_returns, weights)
    return scenario_var_es(r @ w, level)


def gaussian_var_es(window_returns: ArrayLike, weights: ArrayLike, level: float) -> tuple[float, float]:
    """One-day VaR and ES assuming normal portfolio returns (variance-covariance method).

    From the window: mean vector mu and covariance matrix Sigma of the stock returns
    (sample covariance, ddof=1). Portfolio: mu_p = w'mu, sigma_p = sqrt(w' Sigma w),
    with w = w_{t-1}. Since L ~ N(-mu_p, sigma_p^2), with z = Phi^{-1}(level):
        VaR = -mu_p + sigma_p * z
        ES  = -mu_p + sigma_p * phi(z) / (1 - level)
    (phi = standard normal density; E[Z | Z > z] = phi(z) / (1 - level)).

    Units: fraction of portfolio value. Sign: VaR and ES are positive for a loss.
    Confidence level: `level`. Data used: stock returns of the window (up to t-1)
    and the weights w_{t-1}.
    """
    _check_level(level)
    r, w = _as_window_and_weights(window_returns, weights)
    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False, ddof=1)  # k x k; rowvar=False: columns are stocks
    mu_p = w @ mu
    sigma_p = np.sqrt(w @ cov @ w)
    z = norm.ppf(level)
    var = -mu_p + sigma_p * z
    es = -mu_p + sigma_p * norm.pdf(z) / (1 - level)
    return float(var), float(es)


def fhs_var_es(
    window_returns: ArrayLike,
    window_sigma: ArrayLike,
    current_sigma: ArrayLike,
    weights: ArrayLike,
    level: float,
) -> tuple[float, float]:
    """One-day VaR and ES by filtered historical simulation (FHS).

    Each stock return is standardised by the volatility forecast for its own day
    and rescaled by the volatility forecast for day t:
        z_{i,s} = r_{i,s} / sigma_{i,s}     (devolatilize: shock of day s in units of its volatility)
        x_{i,s} = z_{i,s} * sigma_{i,t}     (revolatilize: same shock at today's volatility)
        X_s     = sum_i w_{i,t-1} x_{i,s}   (portfolio return in scenario s)
    i = stock (risk factor), s = day of the window, t = forecast day. Every scenario s
    is rescaled with the same sigma_{i,t}, so x_{i,s} = r_{i,s} * sigma_{i,t} / sigma_{i,s}.
    The residuals of the same day s are kept together (one row per scenario), so
    the cross-sectional dependence of that day is preserved. VaR and ES are then
    read as in historical simulation (see scenario_var_es).

    The volatilities are inputs: window_sigma[s, i] = sigma_{i,s} (forecast for day s
    using data up to s-1), current_sigma[i] = sigma_{i,t} (forecast for day t using
    data up to t-1). Zero mean is assumed, as in the EWMA model that produces them.

    Units: returns, volatilities, VaR and ES as fractions of value. Sign: VaR and ES
    are positive for a loss. Confidence level: `level`. Data used: stock returns and
    volatility forecasts of the window, sigma_t and w_{t-1}: all known at t-1.
    """
    r, w = _as_window_and_weights(window_returns, weights)
    sigma_s = np.asarray(window_sigma, dtype=float)
    sigma_t = np.asarray(current_sigma, dtype=float)
    if sigma_s.shape != r.shape:
        raise ValueError("window_sigma must have the same shape as window_returns")
    if sigma_t.shape != w.shape:
        raise ValueError("current_sigma must have one entry per stock")
    # Written as "all > 0" (not "any <= 0") so that NaN volatilities are rejected too.
    if not (np.all(sigma_s > 0) and np.all(sigma_t > 0)):
        raise ValueError("volatilities must be strictly positive (and not NaN)")
    z = r / sigma_s                # n x k standardised residuals
    stock_scenarios = z * sigma_t  # broadcast: column i rescaled by sigma_{i,t}
    return scenario_var_es(stock_scenarios @ w, level)
