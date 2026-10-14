"""Download, snapshot and audit SPY daily OHLCV data."""

from datetime import date
from pathlib import Path

import pandas as pd

from src import config

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_snapshot(ticker: str = config.TICKER) -> Path:
    """Download adjusted daily OHLCV with yfinance and save a dated CSV snapshot.

    auto_adjust=True adjusts all prices for splits and dividends, so returns are
    total-return consistent (unadjusted prices would show ex-dividend drops).
    """
    import yfinance as yf

    df = yf.download(ticker, start=config.START, end=config.END,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[OHLCV]
    df.index.name = "Date"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{ticker.lower()}_{date.today():%Y%m%d}.csv"
    df.to_csv(path)
    return path


def latest_snapshot_path(ticker: str = config.TICKER):
    files = sorted(config.RAW_DIR.glob(f"{ticker.lower()}_*.csv"))
    if not files:
        raise FileNotFoundError("No snapshot found - run download_snapshot() first.")
    return files[-1]


def load_snapshot(path=None) -> pd.DataFrame:
    """Load the (latest) snapshot; all notebooks read this instead of re-downloading."""
    path = path or latest_snapshot_path()
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[OHLCV].sort_index()


def audit(df: pd.DataFrame) -> dict:
    """Basic integrity checks on OHLCV data. Returns counts of each issue."""
    ret = df["Close"].pct_change()
    return {
        "rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "unsorted_index": bool(not df.index.is_monotonic_increasing),
        "high_below_open_close": int((df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-8).sum()),
        "low_above_open_close": int((df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-8).sum()),
        "non_positive_volume": int((df["Volume"] <= 0).sum()),
        "gaps_over_4_calendar_days": int((df.index.to_series().diff().dt.days > 4).sum()),
        "abs_daily_return_over_7pct": int((ret.abs() > 0.07).sum()),
    }
