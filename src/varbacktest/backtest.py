"""VaR backtesting: hit sequence, likelihood-ratio (LR) coverage tests, Basel traffic light.

A VaR forecast at confidence level alpha (`level` in the code) is correct if the hit sequence
    I_t = 1 if L_t > VaR_t, else 0
is i.i.d. Bernoulli(p) with p = 1 - alpha. Three LR tests check this:
    Kupiec POF (unconditional coverage): the share of exceptions is p;
    Christoffersen independence: an exception today does not make one tomorrow more likely;
    conditional coverage: both at once.
Each test compares the maximised likelihood under H0 (L0) with the one under the
alternative (L1):
    LR = -2 (ln L0 - ln L1), asymptotically chi2(df) under H0,
with df = number of free parameters under the alternative minus under H0.
xlogy(a, b) = a ln(b) with the convention 0 ln(0) = 0: it handles zero counts
(no exceptions, all exceptions, n11 = 0).

Backtesting is ex post: it uses the realised losses of the whole evaluation period,
each compared with a forecast built with data up to t-1 only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import chi2

SIGNIFICANCE = 0.05  # reject H0 when p-value < 5%

# Basel traffic light (MAR99.17): exceptions of the 99% VaR in the last 250 days.
# With X ~ Binomial(250, 0.01): P(X <= 4) = 89.22%, P(X <= 5) = 95.88%,
# P(X <= 9) = 99.97%, P(X <= 10) = 99.99%; the yellow ("amber") zone starts where
# P(X <= x) reaches 95%, the red zone where it reaches 99.99%.
BASEL_WINDOW = 250
BASEL_LEVEL = 0.99
BASEL_MAX_GREEN = 4    # green: 0-4 exceptions
BASEL_MAX_YELLOW = 9   # yellow: 5-9; red: 10 or more


# ---------------------------------------------------------------- result of one test


@dataclass(frozen=True)
class LRTest:
    """Result of one likelihood-ratio test.

    name:      which test, e.g. "Kupiec POF".
    statistic: LR = -2 (ln L0 - ln L1) >= 0; large values are evidence against H0.
    df:        degrees of freedom of the chi2 limit of LR under H0.
    Frozen: a test result is a fact about observed data, it must not change after
    it is computed. The p-value and the decision are derived from these three
    fields (properties), not stored: they cannot get out of sync with the statistic.
    No units: the statistic is built from counts of days.
    """

    name: str
    statistic: float
    df: int

    @property
    def p_value(self) -> float:
        """P(chi2_df > LR): probability, under H0, of a statistic at least this large."""
        return float(chi2.sf(self.statistic, self.df))

    def rejects(self, significance: float = SIGNIFICANCE) -> bool:
        """True if H0 (the VaR model is correct) is rejected at this significance level."""
        return self.p_value < significance


# ---------------------------------------------------------------- input checks


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")


def _as_hits(hits: ArrayLike) -> np.ndarray:
    """Return the hits as an integer array, after checking they are a non-empty 0/1 sequence."""
    raw = np.asarray(hits)
    # Only booleans and numbers: otherwise strings such as "0" and "1" would be converted silently.
    if raw.dtype.kind not in "biuf":
        raise ValueError("hits must be numeric or boolean")
    h = raw.astype(float)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("hits must be a non-empty 1D sequence")
    # NaN is neither 0 nor 1, so a missing value is rejected here too.
    if not np.all((h == 0) | (h == 1)):
        raise ValueError("hits must contain only 0 and 1")
    return h.astype(int)


def _likelihood_ratio(log_l0: float, log_l1: float) -> float:
    """LR = -2 (ln L0 - ln L1). It is >= 0 in theory (L1 is maximised over a larger
    set); max() removes a negative value of the order of rounding errors."""
    return float(max(0.0, -2.0 * (log_l0 - log_l1)))  # float, not np.float64: cleaner repr


# ---------------------------------------------------------------- transitions


@dataclass(frozen=True)
class TransitionCounts:
    """Counts of consecutive pairs (I_{t-1}, I_t) of a hit sequence.

    n_ij = number of days in state j preceded by a day in state i
    (0 = no exception, 1 = exception). They sum to T - 1.
    No units, no sign convention, no confidence level.
    """

    n00: int
    n01: int
    n10: int
    n11: int

    @classmethod
    def from_hits(cls, hits: ArrayLike) -> "TransitionCounts":
        """Count the transitions of a 0/1 hit sequence.

        A classmethod constructor: an alternative way to build the object, starting
        from the raw data; the default constructor takes the four counts directly.
        """
        h = _as_hits(hits)
        yesterday, today = h[:-1], h[1:]
        return cls(
            n00=int(np.sum((yesterday == 0) & (today == 0))),
            n01=int(np.sum((yesterday == 0) & (today == 1))),
            n10=int(np.sum((yesterday == 1) & (today == 0))),
            n11=int(np.sum((yesterday == 1) & (today == 1))),
        )

    @property
    def n_transitions(self) -> int:
        """T - 1: number of consecutive pairs."""
        return self.n00 + self.n01 + self.n10 + self.n11

    @property
    def pi0(self) -> float:
        """P(exception today | no exception yesterday) = n01 / (n00 + n01).
        Set to 0 if no day follows a day without exception (empty row)."""
        days_after_no_exception = self.n00 + self.n01
        if days_after_no_exception == 0:
            return 0.0
        return self.n01 / days_after_no_exception

    @property
    def pi1(self) -> float:
        """P(exception today | exception yesterday) = n11 / (n10 + n11).
        Set to 0 if no day follows an exception (empty row)."""
        days_after_exception = self.n10 + self.n11
        if days_after_exception == 0:
            return 0.0
        return self.n11 / days_after_exception

    @property
    def pi(self) -> float:
        """P(exception) over all transitions = (n01 + n11) / (T - 1)."""
        if self.n_transitions == 0:
            raise ValueError("need at least 2 days (one transition)")
        return (self.n01 + self.n11) / self.n_transitions


# ---------------------------------------------------------------- the three tests


def kupiec_pof_test(hits: ArrayLike, level: float) -> LRTest:
    """Kupiec proportion of failures (POF) test of unconditional coverage.

    With x exceptions in T days and p = 1 - level:
        ln L0 = (T - x) ln(1 - p)     + x ln(p)          (H0: probability p)
        ln L1 = (T - x) ln(1 - x/T)   + x ln(x/T)        (alternative: x/T, the
                                                          maximum-likelihood estimate)
        LR_pof ~ chi2(1): H0 fixes p, the alternative estimates one parameter.
    Two-sided: it rejects too many and too few exceptions. Low power on short samples
    (T = 250 at 99% expects only 2.5 exceptions); the chi2 limit is asymptotic.

    Input: hit sequence of the evaluation period (0/1). No units.
    Confidence level: `level` of the VaR being tested.
    """
    _check_level(level)
    h = _as_hits(hits)
    p = 1 - level                     # exception probability under H0
    n_days = h.size                   # T
    n_exceptions = int(h.sum())       # x
    n_no_exceptions = n_days - n_exceptions
    p_hat = n_exceptions / n_days     # x/T

    log_l0 = xlogy(n_no_exceptions, 1 - p) + xlogy(n_exceptions, p)
    log_l1 = xlogy(n_no_exceptions, 1 - p_hat) + xlogy(n_exceptions, p_hat)
    return LRTest("Kupiec POF", _likelihood_ratio(log_l0, log_l1), df=1)


def independence_test(hits: ArrayLike) -> LRTest:
    """Christoffersen independence test, against a first-order Markov alternative.

        ln L0 = (n00 + n10) ln(1 - pi)  + (n01 + n11) ln(pi)       (H0: same pi after any day)
        ln L1 = n00 ln(1 - pi0) + n01 ln(pi0) + n10 ln(1 - pi1) + n11 ln(pi1)
        LR_ind ~ chi2(1): the alternative has two parameters (pi0, pi1), H0 one (pi).
    pi1 > pi0 means clustering: after an exception another one is more likely.
    With an empty row (e.g. no exception before the last day) its pi is 0, it only
    multiplies zero counts, so L1 = L0 and LR_ind = 0: no pairs, no evidence of clustering.

    Input: hit sequence of the evaluation period (0/1), at least 2 days. No units.
    No confidence level: it tests only the timing of the exceptions.
    """
    counts = TransitionCounts.from_hits(hits)
    if counts.n_transitions == 0:
        raise ValueError("need at least 2 days")
    pi, pi0, pi1 = counts.pi, counts.pi0, counts.pi1

    days_without_exception = counts.n00 + counts.n10  # I_t = 0, whatever happened at t-1
    days_with_exception = counts.n01 + counts.n11     # I_t = 1, whatever happened at t-1
    log_l0 = xlogy(days_without_exception, 1 - pi) + xlogy(days_with_exception, pi)
    log_l1 = (xlogy(counts.n00, 1 - pi0) + xlogy(counts.n01, pi0)
              + xlogy(counts.n10, 1 - pi1) + xlogy(counts.n11, pi1))
    return LRTest("Christoffersen independence", _likelihood_ratio(log_l0, log_l1), df=1)


def conditional_coverage_test(hits: ArrayLike, level: float) -> LRTest:
    """Christoffersen conditional coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2).

    Rejects if the number of exceptions is wrong, or if they cluster, or both.
    Two degrees of freedom: H0 fixes pi0 = pi1 = p, the alternative estimates pi0 and pi1.
    LR_pof uses all T days and LR_ind the T - 1 transitions; the exact decomposition
    would compute LR_pof on days 2..T, a difference of one day, negligible for large T.

    Input: hit sequence of the evaluation period (0/1). No units.
    Confidence level: `level` of the VaR being tested.
    """
    pof = kupiec_pof_test(hits, level)
    independence = independence_test(hits)
    statistic = float(pof.statistic + independence.statistic)
    return LRTest("Conditional coverage", statistic, df=2)


# ---------------------------------------------------------------- backtest of one VaR series


# eq=False: comparing two pandas Series with == gives a Series, not True/False,
# so the comparison generated by dataclass would fail.
@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Backtest of one VaR series at one confidence level.

    hits:  I_t = 1 if L_t > VaR_t, else 0, one value per day of the evaluation
           period, indexed by date.
    level: confidence level of the VaR being tested (e.g. 0.99).
    Two ways to build it: BacktestResult.from_forecasts(losses, var, level), from
    realised losses and VaR forecasts, or BacktestResult(hits, level), from a hit
    sequence computed elsewhere.
    Frozen: the backtest of a given series must not change after it is built.
    The hits are the only data stored; counts and tests are properties derived from
    them (recomputed at each access, O(T)), so there is no state that can get out of sync.
    Inputs of from_forecasts: fractions of portfolio value. Results: counts,
    probabilities and LR statistics, without units. Data used: realised loss of
    day t against the forecast for day t (built with data up to t-1).
    """

    hits: pd.Series
    level: float

    def __post_init__(self) -> None:
        _check_level(self.level)
        if not isinstance(self.hits, pd.Series):
            raise TypeError("hits must be a pandas Series indexed by date")
        _as_hits(self.hits)
        # astype(int) returns a new Series: later changes to the caller's Series do not
        # reach the result. frozen only blocks reassigning the field (result.hits = ...),
        # not changes inside the Series (result.hits.iloc[0] = 1): Python has no deep
        # immutability. object.__setattr__ is the documented way to set a field of a
        # frozen dataclass during initialisation.
        object.__setattr__(self, "hits", self.hits.astype(int))

    @classmethod
    def from_forecasts(cls, losses: pd.Series, var: pd.Series, level: float) -> "BacktestResult":
        """Compare each realised loss with the VaR forecast for the same day.

        A classmethod constructor: it builds the hits from the raw data, then calls
        the default constructor BacktestResult(hits, level).
        losses: L_t = -r_{p,t}, positive for a loss, fraction of portfolio value.
        var:    VaR_t, positive, fraction of portfolio value, built with data up to t-1.
        An exception is strict: L_t = VaR_t is not an exception.
        """
        if not losses.index.equals(var.index):
            raise ValueError("losses and var must have the same dates")
        if not losses.index.is_unique:
            raise ValueError("dates must be unique")
        # NaN > x is False: without this check a missing value would count as "no exception".
        if losses.isna().any() or var.isna().any():
            raise ValueError("losses and var must not contain NaN")
        hits = (losses > var).astype(int).rename("hit")
        return cls(hits, level)

    @property
    def n_days(self) -> int:
        """T: number of days in the evaluation period."""
        return len(self.hits)

    @property
    def n_exceptions(self) -> int:
        """x: number of days with L_t > VaR_t."""
        return int(self.hits.sum())

    @property
    def expected_exceptions(self) -> float:
        """T (1 - level): exceptions expected if the model is correct."""
        return self.n_days * (1 - self.level)

    @property
    def exception_rate(self) -> float:
        """x / T, to compare with 1 - level."""
        return self.n_exceptions / self.n_days

    @property
    def pof(self) -> LRTest:
        """Kupiec POF test on the hits at `level`; recomputed at each access."""
        return kupiec_pof_test(self.hits, self.level)

    @property
    def independence(self) -> LRTest:
        """Christoffersen independence test on the hits; recomputed at each access."""
        return independence_test(self.hits)

    @property
    def conditional_coverage(self) -> LRTest:
        """Christoffersen conditional coverage test at `level`; recomputed at each access."""
        return conditional_coverage_test(self.hits, self.level)

    # ------------------------------------------------------------ views by date

    def _check_dates(self) -> None:
        if not isinstance(self.hits.index, pd.DatetimeIndex):
            raise ValueError("this view needs hits indexed by date")

    def exceptions_per_year(self) -> pd.DataFrame:
        """Exceptions per calendar year, next to the number expected if the model is correct.

        Columns: days (days in the evaluation period that year), exceptions,
        expected = days (1 - level). No units (counts). Confidence level: `level`.
        Data used: the hit sequence.
        """
        self._check_dates()
        year = self.hits.index.year.rename("year")
        days = self.hits.groupby(year).size()
        exceptions = self.hits.groupby(year).sum()
        return pd.DataFrame({
            "days": days,
            "exceptions": exceptions,
            "expected": days * (1 - self.level),
        })

    def traffic_light(self) -> pd.DataFrame:
        """Basel traffic light on a rolling 250-day window.

        count_t = number of exceptions on days t-249, ..., t (day t included);
        zone: green 0-4, yellow 5-9 ("amber" in Basel MAR99), red >= 10.
        Rows start on the first day with a full window. Basel counts the exceptions
        of the last 250 trading days (MAR32.3); computing the zone every day is a
        monitoring choice.
        Columns: count (int), zone (str). Only for the 99% VaR: the thresholds are
        derived from Binomial(250, 0.01). Data used: hits up to and including day t
        (ex post, as a supervisor would do on day t).
        """
        self._check_dates()
        if not np.isclose(self.level, BASEL_LEVEL):
            raise ValueError("the Basel zones are defined for the 99% VaR only")
        if self.n_days < BASEL_WINDOW:
            raise ValueError(f"need at least {BASEL_WINDOW} days")
        count = self.hits.rolling(BASEL_WINDOW).sum().dropna().astype(int)
        zone = np.select(
            [count <= BASEL_MAX_GREEN, count <= BASEL_MAX_YELLOW],
            ["green", "yellow"],
            default="red",
        )
        return pd.DataFrame({"count": count, "zone": zone}, index=count.index)
