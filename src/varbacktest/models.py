"""One-day VaR and ES models: historical simulation, Gaussian, filtered historical simulation.

Every model answers the same question: given what is known at the close of day t-1,
what are VaR_t and ES_t of the portfolio loss L_t = -r_{p,t}? What is known is packed
in a Snapshot, built by the forecasting engine with data up to t-1 only; a model sees
nothing else, so it cannot look ahead.

Notation: s = t-n, ..., t-1 are the n days of the window, i = stock,
w = w_{t-1} = current weights, alpha = confidence level (`level` in the code).
All models are static-portfolio models: the scenarios of the past days s are applied
to the current weights w, not to the weights held on day s.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def _read_only_copy(values: ArrayLike) -> np.ndarray:
    """Float copy that cannot be modified.

    The engine passes slices of the full data arrays, and a slice of a numpy array is a
    view: without a copy, a model writing into its input would silently change the data
    of every later forecast. frozen alone does not prevent this (it only blocks
    reassigning a field); writeable = False does.
    """
    array = np.array(values, dtype=float)  # np.array copies
    array.flags.writeable = False
    return array


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")


# ---------------------------------------------------------------- inputs and outputs


@dataclass(frozen=True, eq=False)  # eq=False: == on numpy arrays is element-wise
class Snapshot:
    """Everything a model may use for the forecast of day t: what is known at the close of t-1.

    returns:    r_{i,s} for the n days s = t-n, ..., t-1          (n x N)
    weights:    w_{i,t-1}, the current weights, summing to 1      (N)
    sigma_past: EWMA volatility sigma_{i,s} of the same days       (n x N)
    sigma_now:  EWMA volatility sigma_{i,t}                        (N)
    Every sigma_{i,u} is the forecast for day u built with returns up to u-1. So in a
    residual z_{i,s} = r_{i,s} / sigma_{i,s} the return r_{i,s} is not inside its own
    volatility (out-of-sample standardisation), and sigma_{i,t} uses returns up to t-1.
    The volatilities are used by FHS only. Arrays are stored as read-only copies.
    Units: fractions (returns, weights), volatility per day as a fraction of value.
    """

    returns: np.ndarray
    weights: np.ndarray
    sigma_past: np.ndarray
    sigma_now: np.ndarray

    def __post_init__(self) -> None:
        for name in ("returns", "weights", "sigma_past", "sigma_now"):
            object.__setattr__(self, name, _read_only_copy(getattr(self, name)))
        if self.returns.ndim != 2:
            raise ValueError("returns must be a (days x stocks) array")
        n_days, n_stocks = self.returns.shape
        if n_days < 2 or n_stocks < 1:
            raise ValueError("need at least 2 days and 1 stock")
        if self.weights.shape != (n_stocks,):
            raise ValueError("weights must have one value per stock")
        if self.sigma_past.shape != self.returns.shape or self.sigma_now.shape != (n_stocks,):
            raise ValueError("sigma_past must match returns, sigma_now must match weights")
        if not (np.isfinite(self.returns).all() and np.isfinite(self.weights).all()):
            raise ValueError("returns and weights must be finite")
        if abs(self.weights.sum() - 1) > 1e-10:
            raise ValueError("weights must sum to 1")

    @property
    def n_days(self) -> int:
        """n: number of days in the window."""
        return self.returns.shape[0]


@dataclass(frozen=True)
class RiskEstimate:
    """VaR and ES at one confidence level.

    Both positive for a loss, as a fraction of portfolio value; ES >= VaR.
    """

    var: float
    es: float


# ---------------------------------------------------------------- shared tool


def empirical_var_es(scenario_returns: ArrayLike, level: float) -> RiskEstimate:
    """VaR and ES from equally weighted portfolio scenarios (used by HS and FHS).

    Losses L_s = -X_s. VaR = empirical alpha-quantile of the losses, with np.quantile
    (default method: linear interpolation at position h = (n-1) alpha of the losses
    sorted in increasing order, positions counted from 0).
    ES = mean of the losses >= VaR. If there are no ties: when h is not an integer, VaR
    lies strictly between two losses and the ES averages the losses above it; when h is
    an integer, VaR equals one loss, which is included. With n = 250: at 99%, h = 246.51,
    VaR lies between the 4th and the 3rd largest loss and ES averages the 3 largest;
    at 97.5%, h = 242.775 and ES averages the 7 largest.
    Units: fraction of portfolio value; positive for a loss. Confidence level: `level`.
    Data used: the scenarios given (built from the snapshot, data up to t-1).
    """
    _check_level(level)
    losses = -np.asarray(scenario_returns, dtype=float)
    if losses.ndim != 1 or losses.size < 2:
        raise ValueError("need a 1D array of at least 2 scenarios")
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    return RiskEstimate(var=var, es=float(tail.mean()))


# ---------------------------------------------------------------- the interface


class VaRModel(ABC):
    """Interface of a one-day VaR/ES model.

    Why an abstract base class: the engine depends only on this interface, so a new
    model is a new subclass and the engine does not change; and a subclass that does
    not implement estimate() cannot be instantiated.
    A model only turns a Snapshot into a RiskEstimate: it does not choose its data.
    Which days it sees is decided by the forecasting engine (no look-ahead by design).
    Subclasses must also set the class attribute `name` (key of the forecast tables);
    the ABC does not enforce it: a missing name fails when the engine reads it.
    """

    name: str

    @abstractmethod
    def estimate(self, snapshot: Snapshot, level: float) -> RiskEstimate:
        """VaR_t and ES_t at confidence level `level`, from the data in `snapshot`.

        Units: fraction of portfolio value. Sign: positive for a loss.
        Data used: the snapshot only (data up to t-1).
        """


# ---------------------------------------------------------------- the three models


class HistoricalSimulation(VaRModel):
    """Historical simulation: the past days of the window, replayed on today's portfolio.

    Scenarios X_s = sum_i w_{i,t-1} r_{i,s}, then empirical VaR and ES (empirical_var_es)
    at the confidence level requested. No distributional assumption; reacts to a crisis
    only as fast as the window moves.
    Units: fraction of portfolio value, positive for a loss. Data used: snapshot (up to t-1).
    """

    name = "hs"

    def estimate(self, snapshot: Snapshot, level: float) -> RiskEstimate:
        scenarios = snapshot.returns @ snapshot.weights  # X_s, one per day of the window
        return empirical_var_es(scenarios, level)


class GaussianModel(VaRModel):
    """Gaussian (variance-covariance) model with the mean and covariance of the window.

        mu = sample mean of r_s,  Sigma = sample covariance (ddof = 1)
        mu_p = w' mu,  sigma_p = sqrt(w' Sigma w),  z = Phi^{-1}(alpha)
        VaR = -mu_p + sigma_p z
        ES  = -mu_p + sigma_p phi(z) / (1 - alpha)
    Phi and phi: standard normal CDF and density; alpha: the confidence level requested.
    Thin tails: it underestimates the risk of fat-tailed returns, more at 99% than at 97.5%.
    Units: fraction of portfolio value, positive for a loss. Data used: snapshot (up to t-1).
    """

    name = "gauss"

    def estimate(self, snapshot: Snapshot, level: float) -> RiskEstimate:
        _check_level(level)
        w = snapshot.weights
        mu = snapshot.returns.mean(axis=0)
        # atleast_2d: with a single stock np.cov returns a number, not a 1 x 1 matrix.
        sigma = np.atleast_2d(np.cov(snapshot.returns, rowvar=False, ddof=1))
        mu_p = float(w @ mu)
        sigma_p = float(np.sqrt(w @ sigma @ w))
        z = norm.ppf(level)
        var = -mu_p + sigma_p * z
        es = -mu_p + sigma_p * norm.pdf(z) / (1 - level)
        return RiskEstimate(var=float(var), es=float(es))


class FilteredHistoricalSimulation(VaRModel):
    """Filtered historical simulation with EWMA volatility per stock.

        z_{i,s} = r_{i,s} / sigma_{i,s}          standardised residual of day s
        x_{i,s} = z_{i,s} * sigma_{i,t}          the same shock at today's volatility
        X_s     = sum_i w_{i,t-1} x_{i,s}        portfolio scenario
    then empirical VaR and ES at the confidence level requested. The residuals of the
    same day s stay together, so the cross-correlation of the stocks is kept. The same
    sigma_{i,t} rescales every day s: calm-period shocks are scaled up in a crisis,
    crisis shocks scaled down when calm.
    Units: fraction of portfolio value, positive for a loss. Data used: snapshot (up to
    t-1), including sigma_{i,t}, which is built from returns up to t-1.
    """

    name = "fhs"

    def estimate(self, snapshot: Snapshot, level: float) -> RiskEstimate:
        for sigma in (snapshot.sigma_past, snapshot.sigma_now):
            # NaN > 0 is False, so a missing volatility is rejected here too.
            if not (np.isfinite(sigma).all() and (sigma > 0).all()):
                raise ValueError("FHS needs finite, positive volatilities on every day")
        residuals = snapshot.returns / snapshot.sigma_past   # z_{i,s}
        rescaled = residuals * snapshot.sigma_now             # x_{i,s}; broadcast over days s
        scenarios = rescaled @ snapshot.weights               # X_s
        return empirical_var_es(scenarios, level)
