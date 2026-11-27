"""One-day VaR and ES: historical, Gaussian and filtered historical simulation.

Single window: each *_var_es function receives the stock returns of the estimation
window (days s = t-n, ..., t-1) and the current weights w_{t-1}, and returns the
forecast (VaR_t, ES_t) for day t. All methods apply the scenarios to the current
weights (static portfolio over one day).

Rolling: ewma_volatility gives the per-stock volatility forecasts needed by FHS, and
rolling_forecasts moves the window through the sample, one forecast per day.
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


def ewma_volatility(returns: pd.DataFrame, lam: float = 0.94, burn_in: int = 250) -> pd.DataFrame:
    """EWMA volatility forecasts, one column per stock (RiskMetrics, zero mean).

    Initial condition, from the burn-in returns 0, ..., burn_in-1 only:
        sigma^2_{i,burn_in} = (1/burn_in) * sum_{s < burn_in} r_{i,s}^2
    (zero-mean variance, the same assumption as the recursion). Then, for t > burn_in:
        sigma^2_{i,t} = lam * sigma^2_{i,t-1} + (1 - lam) * r_{i,t-1}^2
    lam = 0.94 is the RiskMetrics value for daily data: the weight of r^2_{t-k} decays
    as lam^(k-1), with half-life ln(0.5)/ln(lam) ~ 11 days, so sigma reacts to a shock
    within about two weeks.
    Row t is the forecast for day t, built with returns up to t-1 (same row
    convention as buy_and_hold_weights). Rows before burn_in are NaN: there is
    not enough past data for a forecast without look-ahead.

    Units: volatility per day, as a fraction of value. No sign convention (r enters
    squared), no confidence level. Data used: returns up to t-1.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be strictly between 0 and 1")
    r = returns.to_numpy(dtype=float)
    if not 0 < burn_in < len(r):
        raise ValueError("burn_in must be between 1 and the number of days - 1")
    sigma2 = np.full_like(r, np.nan)
    sigma2[burn_in] = np.mean(r[:burn_in] ** 2, axis=0)  # returns 0..burn_in-1 only
    for t in range(burn_in + 1, len(r)):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2  # r[t-1], never r[t]
    return pd.DataFrame(np.sqrt(sigma2), index=returns.index, columns=returns.columns)


def rolling_forecasts(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    window: int = 250,
    burn_in: int = 250,
    lam: float = 0.94,
) -> pd.DataFrame:
    """Daily one-day VaR 99%, VaR 97.5% and ES 97.5% forecasts for HS, Gaussian and FHS.

    For each forecast day t the inputs are: the stock returns of days t-window, ..., t-1,
    the weights w_{t-1} (row t of `weights`, see buy_and_hold_weights) and, for FHS,
    the EWMA volatilities of the same days plus sigma_t (see ewma_volatility).
    Evaluation period: t >= burn_in + window. FHS needs sigma_s for every day
    s = t-window, ..., t-1 of its window, and the first EWMA forecast is
    sigma_{burn_in}; hence t - window >= burn_in. HS and Gaussian could start
    `burn_in` days earlier: those days are dropped on purpose, so that all methods
    are backtested on the same dates.

    Output: one row per forecast day t, columns '<method>_var99', '<method>_var975',
    '<method>_es975' with method in {hs, gauss, fhs}.
    Units: fraction of portfolio value. Sign: VaR and ES are positive for a loss.
    Confidence levels: 99% and 97.5%. Data used: for row t, data up to t-1 only.
    """
    if not (returns.columns.equals(weights.columns) and returns.index.equals(weights.index)):
        raise ValueError("returns and weights must have the same dates and stocks, in the same order")
    first = burn_in + window
    if window < 2 or first >= len(returns):
        raise ValueError("need window >= 2 and more than burn_in + window days")
    sigma = ewma_volatility(returns, lam, burn_in).to_numpy()
    r = returns.to_numpy(dtype=float)
    w_all = weights.to_numpy(dtype=float)

    rows = []
    for t in range(first, len(r)):
        past = slice(t - window, t)  # days t-window, ..., t-1: never day t
        past_returns, w = r[past], w_all[t]  # w_all[t] = w_{t-1}
        past_sigma, sigma_t = sigma[past], sigma[t]
        row = {}
        row["hs_var99"], _ = historical_var_es(past_returns, w, 0.99)
        row["hs_var975"], row["hs_es975"] = historical_var_es(past_returns, w, 0.975)
        row["gauss_var99"], _ = gaussian_var_es(past_returns, w, 0.99)
        row["gauss_var975"], row["gauss_es975"] = gaussian_var_es(past_returns, w, 0.975)
        row["fhs_var99"], _ = fhs_var_es(past_returns, past_sigma, sigma_t, w, 0.99)
        row["fhs_var975"], row["fhs_es975"] = fhs_var_es(past_returns, past_sigma, sigma_t, w, 0.975)
        rows.append(row)
    return pd.DataFrame(rows, index=returns.index[first:])
