"""Descriptive statistics of returns: fat tails and autocorrelation.

They are diagnostics, not model inputs: they show why a Gaussian i.i.d. model is
a poor description of daily equity returns (fat tails, volatility clustering)
and motivate filtered historical simulation.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def _as_1d_array(x: ArrayLike) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("need a 1D sequence of at least 2 values")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    # A constant series has zero variance: kurtosis and autocorrelation are undefined.
    # Checking m2 == 0 would not be enough: the floating-point mean of [0.1] * 7 is not
    # exactly 0.1, so m2 comes out tiny but positive and the result is a meaningless number.
    if np.ptp(values) == 0:
        raise ValueError("values are all equal: zero variance")
    return values


def excess_kurtosis(x: ArrayLike) -> float:
    """Excess kurtosis m4 / m2^2 - 3, with central sample moments m_k = mean((x - mean)^k).

    0 for a normal distribution; > 0 for fat tails (extreme values more frequent than
    under a normal with the same variance). Moment estimator, without small-sample
    correction. No units. Data used: the whole sample given.
    """
    values = _as_1d_array(x)
    deviations = values - values.mean()
    m2 = np.mean(deviations**2)
    m4 = np.mean(deviations**4)
    return float(m4 / m2**2 - 3)


def autocorrelation(x: ArrayLike, lags: Sequence[int]) -> np.ndarray:
    """Sample autocorrelation at the given lags:

        rho_k = sum_{t >= k} (x_t - mean)(x_{t-k} - mean) / sum_t (x_t - mean)^2

    Applied to r it measures linear predictability; applied to r^2, volatility
    clustering. Under i.i.d. data rho_k is about N(0, 1/n), hence the band +-1.96/sqrt(n);
    with volatility clustering the band is too narrow for the ACF of r, so a value just
    outside it is weak evidence of linear predictability.
    No units. Data used: the whole sample given.
    """
    values = _as_1d_array(x)
    n = values.size
    deviations = values - values.mean()
    denominator = np.sum(deviations**2)
    result = []
    for k in lags:
        if not 1 <= k < n:
            raise ValueError("each lag must satisfy 1 <= k < number of values")
        numerator = np.sum(deviations[k:] * deviations[:-k])
        result.append(numerator / denominator)
    return np.array(result)
