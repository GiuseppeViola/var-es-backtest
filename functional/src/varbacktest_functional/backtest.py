"""VaR backtesting: exceptions, Kupiec POF, Christoffersen tests, Basel traffic light.

A VaR forecast at level alpha is correct if the hit sequence
    I_t = 1 if L_t > VaR_t else 0
is i.i.d. Bernoulli(p) with p = 1 - alpha. The likelihood ratio (LR) tests check this:
    unconditional coverage (Kupiec POF): the share of exceptions is p;
    independence (Christoffersen): exceptions do not cluster;
    conditional coverage: both at once.
Backtesting is ex post: it uses the realised losses of the whole evaluation period,
each compared with a forecast that was built with data up to t-1 only.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import chi2


def exceptions(losses: pd.Series, var: pd.Series) -> pd.Series:
    """Hit sequence I_t = 1 if L_t > VaR_t (strictly), else 0.

    Units: losses and VaR as fractions of portfolio value. Sign: L_t = -r_{p,t},
    positive for a loss; VaR positive. Confidence level: that of `var`. Data used:
    realised loss on day t and the forecast for day t (built with data up to t-1).
    """
    if not losses.index.equals(var.index):
        raise ValueError("losses and var must have the same dates")
    # NaN > x is False: without this check a missing value would silently count as "no exception".
    if losses.isna().any() or var.isna().any():
        raise ValueError("losses and var must not contain NaN")
    return (losses > var).astype(int)


def _as_hits(hits: ArrayLike) -> np.ndarray:
    """Convert to an integer array and check that it is a non-empty 0/1 sequence."""
    h = np.asarray(hits)
    if h.ndim != 1 or h.size == 0 or not np.all((h == 0) | (h == 1)):
        raise ValueError("hits must be a non-empty 1D sequence of 0 and 1")
    return h.astype(int)


def kupiec_pof(hits: ArrayLike, level: float) -> tuple[float, float]:
    """Kupiec proportion of failures (POF) test of unconditional coverage.

    With x exceptions in T days and p = 1 - level:
        log L0 = (T-x) ln(1-p) + x ln(p)            (model: exception probability p)
        log L1 = (T-x) ln(1-x/T) + x ln(x/T)        (alternative: x/T, the maximum-
                                                     likelihood estimate of p, so L1 >= L0)
        LR_pof = -2 (log L0 - log L1) ~ chi2(1)
    One degree of freedom: H0 fixes p, H1 estimates it (one free parameter). The chi2
    distribution is asymptotic, so it is only approximate when few exceptions are expected.
    Two-sided: it rejects both too many and too few exceptions (an over-conservative VaR
    wastes capital). With T = 250 at 99% only ~2.5 exceptions are expected, so the
    test has low power: it rarely rejects a wrong model.
    xlogy(a, b) = a ln(b) with 0 ln(0) = 0, so x = 0 and x = T are handled.

    Returns (LR_pof, p-value); a small p-value rejects correct coverage.
    No units (counts). Confidence level: `level` of the VaR being tested.
    Data used: the hit sequence of the evaluation period.
    """
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")
    h = _as_hits(hits)
    p = 1 - level
    T, x = h.size, h.sum()
    log_l0 = xlogy(T - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(T - x, 1 - x / T) + xlogy(x, x / T)
    lr = max(0.0, -2 * (log_l0 - log_l1))  # >= 0 in theory; clip rounding noise
    return float(lr), float(chi2.sf(lr, df=1))


def transition_counts(hits: ArrayLike) -> tuple[int, int, int, int]:
    """Counts (n00, n01, n10, n11) of consecutive pairs (I_{t-1}, I_t).

    n_ij = number of days with state j preceded by a day with state i
    (0 = no exception, 1 = exception). They sum to T - 1.
    No units, no sign convention, no confidence level. Data used: the hit sequence.
    """
    h = _as_hits(hits)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen_independence(hits: ArrayLike) -> tuple[float, float]:
    """Christoffersen independence test (first-order Markov alternative).

    pi0 = n01/(n00+n01): P(exception | no exception yesterday)
    pi1 = n11/(n10+n11): P(exception | exception yesterday)
    pi  = (n01+n11)/(T-1): P(exception), the same after any day under independence
        log L0 = (n00+n10) ln(1-pi) + (n01+n11) ln(pi)
        log L1 = n00 ln(1-pi0) + n01 ln(pi0) + n10 ln(1-pi1) + n11 ln(pi1)
        LR_ind = -2 (log L0 - log L1) ~ chi2(1)
    One degree of freedom: H1 has two parameters (pi0, pi1), H0 only one (pi).
    pi1 > pi0 means clustering: after an exception another one is more likely.

    Edge cases: n11 = 0 is handled by xlogy (0 ln 0 = 0). If a row has no days
    (e.g. no exception before the last day, so n10 + n11 = 0), its pi is set to 0;
    it only multiplies zero counts, and then L1 = L0, so LR_ind = 0: no pairs, no
    evidence of clustering.

    Returns (LR_ind, p-value). No units. No confidence level (it tests only the
    timing of the exceptions). Data used: the hit sequence of the evaluation period.
    """
    n00, n01, n10, n11 = transition_counts(hits)
    if n00 + n01 + n10 + n11 == 0:
        raise ValueError("need at least 2 days")
    pi0 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = max(0.0, -2 * (log_l0 - log_l1))
    return float(lr), float(chi2.sf(lr, df=1))


def conditional_coverage(hits: ArrayLike, level: float) -> tuple[float, float]:
    """Christoffersen conditional coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2).

    Rejects if the number of exceptions is wrong, or if they cluster, or both.
    Two degrees of freedom: H0 fixes pi0 = pi1 = p, H1 estimates pi0 and pi1.
    Note: LR_pof uses all T days and LR_ind the T-1 transitions. The exact likelihood
    decomposition would compute the POF part on hits[1:]; the difference is one
    observation, negligible for T in the thousands.
    Returns (LR_cc, p-value). No units. Confidence level: `level` of the VaR
    being tested. Data used: the hit sequence of the evaluation period.
    """
    lr = kupiec_pof(hits, level)[0] + christoffersen_independence(hits)[0]
    return float(lr), float(chi2.sf(lr, df=2))


BASEL_WINDOW = 250  # the zone thresholds below are only valid for 250 days at 99%


def traffic_light(hits: pd.Series) -> pd.DataFrame:
    """Basel traffic light on a rolling 250-day window of the 99% VaR hit sequence.

    count_t = sum of I_u for u = t-249, ..., t (exceptions in the last 250 days,
    day t included); zone: green 0-4, yellow 5-9 ("amber" in Basel MAR99), red >= 10.
    MAR99.17 places the amber zone where P(X <= x) reaches 95% and the red zone where
    it reaches 99.99%, with X ~ Binomial(250, 0.01): P(X <= 4) = 89.22%,
    P(X <= 5) = 95.88%, P(X <= 9) = 99.97%, P(X <= 10) = 99.99%.
    Rows start on the first day with a full window. Basel counts the exceptions of
    the last 250 trading days (MAR32.3); computing the zone every day is a monitoring choice.

    Returns a DataFrame with columns 'count' (int) and 'zone' (str). No units.
    Confidence level: 99% (the thresholds are only valid for it). Data used: hit
    sequence up to and including t (ex post, as a supervisor would do on day t).
    """
    _as_hits(hits)  # validation only: rolling() below needs the pandas Series and its dates
    if len(hits) < BASEL_WINDOW:
        raise ValueError(f"need at least {BASEL_WINDOW} days")
    count = hits.rolling(BASEL_WINDOW).sum().dropna().astype(int)
    zone = np.select([count <= 4, count <= 9], ["green", "yellow"], default="red")
    return pd.DataFrame({"count": count, "zone": zone}, index=count.index)
