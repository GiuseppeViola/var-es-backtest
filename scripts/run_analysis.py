"""Run the analysis on the local price data (data/prices.csv) and write the figures.

Run from the repository root, after scripts/download_data.py:

    python scripts/run_analysis.py

Steps: data -> buy-and-hold portfolio -> descriptive statistics -> rolling VaR/ES
forecasts of every model -> backtests -> summary table and figures (figures/).
To add a model: add an instance to MODELS and its label and colour to
plots.MODEL_STYLE; tables and figures follow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from varbacktest.backtest import BacktestResult
from varbacktest.data import (
    ASML_2012_CASH_FRACTION,
    ASML_2012_EX_DATE,
    adjust_cash_distribution,
    align_dates,
    load_prices,
)
from varbacktest.forecasting import rolling_forecasts
from varbacktest.models import (
    FilteredHistoricalSimulation,
    GaussianModel,
    HistoricalSimulation,
)
from varbacktest.plots import plot_losses_and_var, plot_traffic_light, plot_zoom
from varbacktest.portfolio import Portfolio
from varbacktest.statistics import autocorrelation, excess_kurtosis

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "prices.csv"
FIGURES = ROOT / "figures"

MODELS = [HistoricalSimulation(), GaussianModel(), FilteredHistoricalSimulation()]
WINDOW = 250   # days in each estimation window
BURN_IN = 250  # days used only to initialise the EWMA volatility
LAM = 0.94     # EWMA decay factor (RiskMetrics, daily data)
LAGS = [1, 5, 10]
VAR_COLUMNS = {"var_99": 0.99, "var_975": 0.975}  # forecast column -> confidence level

# Backtests, nested as results[var_column][model_name], e.g. results["var_99"]["fhs"].
Backtests = dict[str, dict[str, BacktestResult]]


def build_portfolio() -> Portfolio:
    """Load the prices, correct the ASML 2012 cash distribution, keep the common dates."""
    raw = load_prices(DATA_FILE)
    corrected = adjust_cash_distribution(raw, "ASML.AS", ASML_2012_EX_DATE, ASML_2012_CASH_FRACTION)
    prices, n_dropped = align_dates(corrected)
    print(f"Aligned sample: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices)} dates ({n_dropped} dropped: not all exchanges open)")
    return Portfolio(prices)


def print_descriptive_statistics(portfolio: Portfolio) -> None:
    """Final weights, fat tails (excess kurtosis), ACF of r and of r^2."""
    print("\nBuy-and-hold weights at the last date:")
    print(portfolio.weights.iloc[-1].round(3).to_string())

    series = pd.concat([portfolio.returns, portfolio.portfolio_returns.rename("portfolio")], axis=1)
    kurtosis_and_acf_r, acf_r2 = {}, {}
    for name, r in series.items():
        row = {"excess_kurtosis": excess_kurtosis(r)}
        row.update({f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r, LAGS))})
        kurtosis_and_acf_r[name] = row
        acf_r2[name] = {f"lag{k}": rho for k, rho in zip(LAGS, autocorrelation(r**2, LAGS))}

    band = 1.96 / np.sqrt(len(series))
    print(f"\n{len(series)} daily returns; i.i.d. band for the ACF: +-{band:.3f}")
    print("\nFat tails and linear predictability: excess kurtosis and ACF of r")
    print(pd.DataFrame(kurtosis_and_acf_r).T.round(3).to_string())
    print("\nVolatility clustering: ACF of r^2")
    print(pd.DataFrame(acf_r2).T.round(3).to_string())


def print_forecast_summary(forecasts: dict[str, pd.DataFrame], losses: pd.Series) -> None:
    """Mean, min and max of every forecast series, and the forecasts of the last day."""
    dates = losses.index
    print(f"\nEvaluation period: {dates[0].date()} to {dates[-1].date()}, "
          f"{len(dates)} days. Forecasts, % of portfolio value:")
    rows = {}
    for name, table in forecasts.items():
        for column, values in table.items():
            pct = 100 * values
            rows[f"{name} {column}"] = {"mean": pct.mean(), "min": pct.min(), "max": pct.max(),
                                        "date of max": pct.idxmax().date()}
    # The date column makes every column object after .T: infer_objects restores the
    # float columns, otherwise round() would leave them untouched.
    print(pd.DataFrame(rows).T.infer_objects().round(2).to_string())

    last = dates[-1]
    print(f"\nForecasts for {last.date()} (%):")
    for name, table in forecasts.items():
        print(f"  {name}: " + ", ".join(f"{c} {100 * v:.2f}" for c, v in table.loc[last].items()))
    print(f"Realised loss on {last.date()}: {100 * losses.loc[last]:.2f}%")


def backtest_all(forecasts: dict[str, pd.DataFrame], losses: pd.Series) -> Backtests:
    """One backtest per VaR column and model: realised loss of day t vs VaR forecast for t."""
    results: Backtests = {}
    for column, level in VAR_COLUMNS.items():
        results[column] = {
            name: BacktestResult.from_forecasts(losses.loc[table.index], table[column], level)
            for name, table in forecasts.items()
        }
    return results


def coverage_table(results: Backtests) -> pd.DataFrame:
    """One row per (model, VaR column): exceptions and the three LR tests."""
    rows = {}
    for column, by_model in results.items():
        for name, result in by_model.items():
            pof, ind, cc = result.pof, result.independence, result.conditional_coverage
            rows[f"{name} {column}"] = {
                "T": result.n_days, "x": result.n_exceptions,
                "expected": result.expected_exceptions, "x/T %": 100 * result.exception_rate,
                "LR_pof": pof.statistic, "p_pof": pof.p_value,
                "LR_ind": ind.statistic, "p_ind": ind.p_value,
                "LR_cc": cc.statistic, "p_cc": cc.p_value,
                "reject_pof": pof.rejects(), "reject_cc": cc.rejects(),
            }
    # Transposing mixed rows gives object columns: infer_objects restores int, float and bool.
    return pd.DataFrame(rows).T.infer_objects()


def yearly_table(results: Backtests) -> pd.DataFrame:
    """Exceptions per calendar year for every model, with the expected numbers."""
    columns = {}
    for column, by_model in results.items():
        per_year = {name: result.exceptions_per_year() for name, result in by_model.items()}
        any_model = next(iter(per_year.values()))  # days and expected are the same for all
        columns["days"] = any_model["days"]
        columns[f"exp {column}"] = any_model["expected"].round(1)
        for name, table in per_year.items():
            columns[f"{name} {column}"] = table["exceptions"]
    return pd.DataFrame(columns)


def zone_shares(results_99: dict[str, BacktestResult]) -> pd.DataFrame:
    """% of days in each Basel zone (VaR 99%, rolling 250 days), one row per model."""
    shares = {}
    for name, result in results_99.items():
        shares[name] = 100 * result.traffic_light()["zone"].value_counts(normalize=True)
    # A zone never reached is missing from value_counts: show it as 0%.
    return pd.DataFrame(shares).T.reindex(columns=["green", "yellow", "red"]).fillna(0)


def summary_table(results: Backtests, forecasts: dict[str, pd.DataFrame],
                  shares: pd.DataFrame) -> pd.DataFrame:
    """Summary for the README: accuracy (exceptions, tests) next to cost (mean VaR and ES)."""
    rows = []  # a list of rows keeps the type of each column (int counts stay int)
    for name in forecasts:
        at_99, at_975 = results["var_99"][name], results["var_975"][name]
        rows.append({
            "x 99%": at_99.n_exceptions,
            "p_cc 99%": at_99.conditional_coverage.p_value,
            "x 97.5%": at_975.n_exceptions,
            "p_cc 97.5%": at_975.conditional_coverage.p_value,
            "mean VaR 99% (%)": 100 * forecasts[name]["var_99"].mean(),
            "mean ES 97.5% (%)": 100 * forecasts[name]["es_975"].mean(),
            "yellow days (%)": shares.loc[name, "yellow"],
            "red days (%)": shares.loc[name, "red"],
        })
    return pd.DataFrame(rows, index=list(forecasts))


def main() -> None:
    portfolio = build_portfolio()
    print_descriptive_statistics(portfolio)

    # Rolling one-day forecasts: row t uses only data up to t-1.
    forecasts = rolling_forecasts(portfolio, MODELS, window=WINDOW, burn_in=BURN_IN, lam=LAM)
    dates = next(iter(forecasts.values())).index  # the same evaluation dates for every model
    losses = portfolio.losses.loc[dates]
    print_forecast_summary(forecasts, losses)

    # Backtesting: compare each forecast with the realised loss L_t = -r_{p,t} of the same day.
    results = backtest_all(forecasts, losses)
    results_99 = results["var_99"]

    print("\nCoverage tests (reject at 5%: p < 0.05; critical values chi2(1) 3.84, chi2(2) 5.99):")
    print(coverage_table(results).round(3).to_string())

    print("\nExceptions per calendar year (exp = expected):")
    print(yearly_table(results).to_string())

    print("\nBasel traffic light, VaR 99%, rolling 250 days: % of days in each zone")
    shares = zone_shares(results_99)
    print(shares.round(1).to_string())
    print(f"Zone on the last 250 days (to {dates[-1].date()}):")
    for name, result in results_99.items():
        last = result.traffic_light().iloc[-1]
        print(f"  {name}: {last['zone']} ({last['count']} exceptions)")

    any_99, any_975 = next(iter(results_99.values())), next(iter(results["var_975"].values()))
    print(f"\nSummary ({len(dates)} days; expected exceptions "
          f"{any_99.expected_exceptions:.1f} at 99%, {any_975.expected_exceptions:.1f} at 97.5%):")
    print(summary_table(results, forecasts, shares).round(3).to_string())

    FIGURES.mkdir(exist_ok=True)
    plot_losses_and_var(losses, forecasts, results_99, FIGURES / "losses_var.png")
    plot_zoom(losses, forecasts, "2008-06-01", "2009-12-31", FIGURES / "zoom_2008.png")
    plot_zoom(losses, forecasts, "2020-01-01", "2020-12-31", FIGURES / "zoom_2020.png")
    plot_traffic_light(results_99, FIGURES / "traffic_light.png")
    print(f"\nFigures saved in {FIGURES}")


if __name__ == "__main__":
    main()
