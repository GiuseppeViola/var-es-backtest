"""Download daily adjusted close prices for the portfolio from Yahoo Finance.

Run manually, once, from the repository root:

    python scripts/download_data.py

Outputs (the data/ folder is not published):
    data/prices.csv          adjusted close prices in EUR, one column per ticker.
                             Missing values (different exchange holidays) are
                             kept here: date alignment is done by the analysis code.
    data/download_info.txt   download metadata, for reproducibility.

This is the only file that touches the internet: the analysis reads only these files.
"""

from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

TICKERS = ["TTE.PA", "DTE.DE", "SAN.MC", "NOKIA.HE", "ENEL.MI"]
START = "2005-01-01"
# Fixed end date, exclusive in yfinance, so the last possible row is 2026-11-19
# (the actual last date is recorded in download_info.txt).
# It freezes the sample dates: re-running the script later gives the same date range,
# and a download during today's trading hours cannot add an intraday price.
# It does not freeze the values: Yahoo may revise historical prices, which is why
# the download metadata are saved alongside the prices.
END = "2026-11-20"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    # auto_adjust=True: the "Close" column is the adjusted close. Past prices are
    # rescaled for splits and dividends, so returns computed from it are
    # (approximately) total returns, with dividends reinvested in the same stock,
    # and no fake jumps on split or ex-dividend dates.
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"][TICKERS]  # fixed column order
    prices.index.name = "Date"

    # Fail loudly instead of saving an empty or incomplete file.
    empty = [t for t in TICKERS if prices[t].isna().all()]
    if empty:
        raise RuntimeError(f"No data downloaded for: {empty}")

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(DATA_DIR / "prices.csv")

    # Yahoo revises its historical data, so record when and how the file was made.
    lines = [
        f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"yfinance_version: {yf.__version__}",
        f"tickers: {', '.join(TICKERS)}",
        f"start: {START}",
        f"end_exclusive: {END}",
        f"rows: {len(prices)}",
        f"first_date: {prices.index[0].date()}",
        f"last_date: {prices.index[-1].date()}",
        f"rows_with_missing_values: {int(prices.isna().any(axis=1).sum())}",
    ]
    for t in TICKERS:
        lines.append(f"first_valid_date {t}: {prices[t].first_valid_index().date()}")
    (DATA_DIR / "download_info.txt").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
