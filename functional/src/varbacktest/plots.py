"""Figures for the validation report: losses vs VaR, crisis zooms, rolling traffic light.

The functions only draw data that has already been computed (forecasts, hit
sequences) and save a PNG. Losses and VaR are plotted in % of portfolio value.
Each method keeps the same colour in every figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from varbacktest.backtest import traffic_light

METHODS = {  # column prefix -> (label, colour)
    "hs": ("Historical simulation", "#2a78d6"),
    "gauss": ("Gaussian", "#eb6834"),
    "fhs": ("Filtered HS (EWMA)", "#1baf7a"),
}
LOSS_COLOUR = "#a3a29c"  # neutral grey: the data, not a model
TEXT_COLOUR = "#52514e"
ZONES = [  # (label, lower, upper, colour): Basel zones for the 250-day count at 99%
    ("green", 0, 4.5, "#0ca30c"),
    ("yellow", 4.5, 9.5, "#fab219"),
    ("red", 9.5, None, "#d03b3b"),
]


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


def plot_losses_and_var(losses: pd.Series, forecasts: pd.DataFrame, hits: pd.DataFrame,
                        path: str | Path) -> None:
    """One panel per method: daily loss, VaR 99% forecast and exceptions, whole period.

    Units: % of portfolio value. Sign: loss L_t = -r_{p,t} (positive = loss), VaR
    positive. Confidence level: 99%. Data used: realised losses and the forecasts
    (built with data up to t-1); exceptions are the days with L_t > VaR_t.
    """
    # The exception markers are selected by position: misaligned dates would put them
    # on the wrong days without any error.
    if not (hits.index.equals(losses.index) and forecasts.index.equals(losses.index)):
        raise ValueError("losses, forecasts and hits must have the same dates")
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (m, (label, colour)) in zip(axes, METHODS.items()):
        hit = hits[f"{m}_var99"].astype(bool)
        ax.plot(losses.index, 100 * losses, color=LOSS_COLOUR, linewidth=0.5,
                label="Daily loss (negative = gain)")
        ax.plot(forecasts.index, 100 * forecasts[f"{m}_var99"], color=colour, linewidth=1.2,
                label="VaR 99%")
        ax.scatter(losses.index[hit], 100 * losses[hit], s=14, color=colour,
                   edgecolors="white", linewidths=0.6, zorder=3, label="Exception")
        expected = 0.01 * len(hit)
        ax.set_title(f"{label}: {hit.sum()} exceptions (expected {expected:.1f})",
                     loc="left", fontsize=10, color=TEXT_COLOUR)
        ax.set_ylabel("% of value", fontsize=8, color=TEXT_COLOUR)
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncols=3)
    fig.suptitle("One-day loss vs 99% VaR forecast", x=0.125, ha="left", fontsize=12)
    _save(fig, path)


def plot_zoom(losses: pd.Series, forecasts: pd.DataFrame, start: str, end: str,
              path: str | Path) -> None:
    """The three VaR 99% forecasts and the daily loss over a crisis period [start, end].

    Units: % of portfolio value. Sign: loss positive, VaR positive. Confidence
    level: 99%. Data used: realised losses and forecasts between start and end.
    """
    loss = losses.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(loss.index, 100 * loss, color=LOSS_COLOUR, linewidth=0.8,
            label="Daily loss (negative = gain)")
    for m, (label, colour) in METHODS.items():
        var = forecasts.loc[start:end, f"{m}_var99"]
        ax.plot(var.index, 100 * var, color=colour, linewidth=1.6, label=f"VaR 99%, {label}")
    ax.axhline(0, color="#c3c2b7", linewidth=0.6)
    ax.set_title(f"VaR 99% forecasts, {loss.index[0]:%b %Y} to {loss.index[-1]:%b %Y}",
                 loc="left", fontsize=11, color=TEXT_COLOUR)
    ax.set_ylabel("% of value", fontsize=8, color=TEXT_COLOUR)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax)
    _save(fig, path)


def plot_traffic_light(hits: pd.DataFrame, path: str | Path) -> None:
    """One panel per method: exceptions of the VaR 99% in the last 250 days, with Basel zones.

    Units: number of exceptions. No sign convention. Confidence level: 99% (the
    zone thresholds are only valid for it). Data used: hit sequence up to and
    including day t (ex post count, see backtest.traffic_light).
    """
    counts = {m: traffic_light(hits[f"{m}_var99"])["count"] for m in METHODS}
    top = max(c.max() for c in counts.values()) + 2
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (m, (label, colour)) in zip(axes, METHODS.items()):
        for zone, low, high, zone_colour in ZONES:
            ax.axhspan(low, high if high is not None else top, color=zone_colour,
                       alpha=0.12, linewidth=0)
            ax.text(1.005, (low + (high if high is not None else top)) / 2, zone,
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                    color=TEXT_COLOUR)
        ax.step(counts[m].index, counts[m], where="post", color=colour, linewidth=1.2)
        share_red = 100 * (counts[m] >= 10).mean()
        ax.set_title(f"{label}: {share_red:.1f}% of days in the red zone", loc="left",
                     fontsize=10, color=TEXT_COLOUR)
        ax.set_ylabel("exceptions", fontsize=8, color=TEXT_COLOUR)
        _style(ax)
    axes[0].set_ylim(0, top)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are integers
    fig.suptitle("Basel traffic light: VaR 99% exceptions in the last 250 days",
                 x=0.125, ha="left", fontsize=12)
    _save(fig, path)
