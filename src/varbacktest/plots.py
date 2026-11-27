"""Figures for the validation report: losses vs VaR, crisis zooms, rolling traffic light.

The functions only draw results that have already been computed (forecasts and
backtests of the 99% VaR) and save a PNG. Losses and VaR are plotted in % of
portfolio value. Each model keeps the same colour in every figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .backtest import BASEL_MAX_GREEN, BASEL_MAX_YELLOW, BacktestResult

MODEL_STYLE = {  # model name -> (label, colour)
    "hs": ("Historical simulation", "#2a78d6"),
    "gauss": ("Gaussian", "#eb6834"),
    "fhs": ("Filtered HS (EWMA)", "#1baf7a"),
}
LOSS_COLOUR = "#a3a29c"  # neutral grey: the data, not a model
TEXT_COLOUR = "#52514e"
# Basel zones for the 250-day count at 99%: (label, lower, upper, colour). The band
# edges sit half-way between integer counts (e.g. 4.5 between green 4 and yellow 5).
ZONES = [
    ("green", 0, BASEL_MAX_GREEN + 0.5, "#0ca30c"),
    ("yellow", BASEL_MAX_GREEN + 0.5, BASEL_MAX_YELLOW + 0.5, "#fab219"),
    ("red", BASEL_MAX_YELLOW + 0.5, None, "#d03b3b"),
]


def _style_of(name: str) -> tuple[str, str]:
    if name not in MODEL_STYLE:
        raise ValueError(f"no label and colour defined for model {name!r} in MODEL_STYLE")
    return MODEL_STYLE[name]


def _check_same_models(forecasts: dict, results: dict) -> None:
    if set(forecasts) != set(results):
        raise ValueError("forecasts and results must cover the same models")


def _panels(n_panels: int, height: float) -> tuple[plt.Figure, list[plt.Axes]]:
    """One panel per model, stacked, sharing both axes.

    squeeze=False: with a single model plt.subplots would return one Axes instead of
    an array, and the loops over the panels would fail.
    """
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, height), sharex=True, sharey=True,
                             squeeze=False)
    return fig, list(axes[:, 0])


def _style(ax: plt.Axes) -> None:
    """Recessive axes: light horizontal grid, no top/right frame."""
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_COLOUR, labelsize=8)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _save(fig: plt.Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_losses_and_var(losses: pd.Series, forecasts: dict[str, pd.DataFrame],
                        results: dict[str, BacktestResult], path: str | Path) -> None:
    """One panel per model: daily loss, VaR 99% forecast and exceptions, whole period.

    forecasts: model name -> forecast table (column var_99 is used);
    results:   model name -> backtest of that model's VaR 99%.
    Units: % of portfolio value. Sign: loss L_t = -r_{p,t} (positive = loss), VaR
    positive. Confidence level: 99%. Data used: realised losses and the forecasts
    (built with data up to t-1); exceptions are the days with L_t > VaR_t.
    """
    _check_same_models(forecasts, results)
    for name in forecasts:
        # The exception markers are selected by position: misaligned dates would put
        # them on the wrong days without any error.
        if not (forecasts[name].index.equals(losses.index)
                and results[name].hits.index.equals(losses.index)):
            raise ValueError("losses, forecasts and hits must have the same dates")
    fig, axes = _panels(len(forecasts), height=8)
    for ax, name in zip(axes, forecasts):
        label, colour = _style_of(name)
        result = results[name]
        hit = result.hits.astype(bool)
        ax.plot(losses.index, 100 * losses, color=LOSS_COLOUR, linewidth=0.5,
                label="Daily loss (negative = gain)")
        ax.plot(losses.index, 100 * forecasts[name]["var_99"], color=colour, linewidth=1.2,
                label="VaR 99%")
        ax.scatter(losses.index[hit], 100 * losses[hit], s=14, color=colour,
                   edgecolors="white", linewidths=0.6, zorder=3, label="Exception")
        ax.set_title(f"{label}: {result.n_exceptions} exceptions "
                     f"(expected {result.expected_exceptions:.1f})",
                     loc="left", fontsize=10, color=TEXT_COLOUR)
        ax.set_ylabel("% of value", fontsize=8, color=TEXT_COLOUR)
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncols=3)
    fig.suptitle("One-day loss vs 99% VaR forecast", x=0.125, ha="left", fontsize=12)
    _save(fig, path)


def plot_zoom(losses: pd.Series, forecasts: dict[str, pd.DataFrame], start: str, end: str,
              path: str | Path) -> None:
    """The VaR 99% forecasts of every model and the daily loss over a crisis period [start, end].

    Units: % of portfolio value. Sign: loss positive, VaR positive. Confidence
    level: 99%. Data used: realised losses and forecasts between start and end.
    """
    loss = losses.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(loss.index, 100 * loss, color=LOSS_COLOUR, linewidth=0.8,
            label="Daily loss (negative = gain)")
    for name, table in forecasts.items():
        label, colour = _style_of(name)
        var = table.loc[start:end, "var_99"]
        ax.plot(var.index, 100 * var, color=colour, linewidth=1.6, label=f"VaR 99%, {label}")
    ax.axhline(0, color="#c3c2b7", linewidth=0.6)
    ax.set_title(f"VaR 99% forecasts, {loss.index[0]:%b %Y} to {loss.index[-1]:%b %Y}",
                 loc="left", fontsize=11, color=TEXT_COLOUR)
    ax.set_ylabel("% of value", fontsize=8, color=TEXT_COLOUR)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax)
    _save(fig, path)


def plot_traffic_light(results: dict[str, BacktestResult], path: str | Path) -> None:
    """One panel per model: exceptions of the VaR 99% in the last 250 days, with Basel zones.

    results: model name -> backtest of that model's VaR 99%.
    Units: number of exceptions. No sign convention. Confidence level: 99% (the
    zone thresholds are only valid for it). Data used: hit sequence up to and
    including day t (ex post count, see BacktestResult.traffic_light).
    """
    lights = {name: result.traffic_light() for name, result in results.items()}
    top = max(light["count"].max() for light in lights.values()) + 2
    fig, axes = _panels(len(results), height=7)
    for ax, name in zip(axes, results):
        label, colour = _style_of(name)
        for zone, low, high, zone_colour in ZONES:
            upper = high if high is not None else top
            ax.axhspan(low, upper, color=zone_colour, alpha=0.12, linewidth=0)
            ax.text(1.005, (low + upper) / 2, zone, transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8, color=TEXT_COLOUR)
        count = lights[name]["count"]
        ax.step(count.index, count, where="post", color=colour, linewidth=1.2)
        share_red = 100 * (lights[name]["zone"] == "red").mean()
        ax.set_title(f"{label}: {share_red:.1f}% of days in the red zone", loc="left",
                     fontsize=10, color=TEXT_COLOUR)
        ax.set_ylabel("exceptions", fontsize=8, color=TEXT_COLOUR)
        _style(ax)
    axes[0].set_ylim(0, top)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are integers
    fig.suptitle("Basel traffic light: VaR 99% exceptions in the last 250 days",
                 x=0.125, ha="left", fontsize=12)
    _save(fig, path)
