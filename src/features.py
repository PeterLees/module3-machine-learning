"""Feature engineering. Every feature at row t uses only data up to the close of
day t (causal rolling windows, recursive EWMs, positive shifts only). Signals
are executed at the next day's open, so including day t's close is legitimate.

Centered windows and negative shifts are forbidden here (enforced by
tests/test_no_lookahead.py)."""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "close_sma20", "close_sma50", "close_sma200", "sma20_sma50",
    "rsi_14", "macd_hist",
    "bb_pctb", "bb_width",
    "vol_20", "atr_14_norm",
    "vol_z20",
]

# Lower-redundancy subset used as an ablation (one representative per family).
COMPACT_COLS = ["ret_1", "ret_20", "close_sma200", "rsi_14", "macd_hist",
                "bb_width", "vol_20", "vol_z20"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI: 100 - 100 / (1 + avg_gain / avg_loss), Wilder smoothing alpha=1/n."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram normalised by price: (EMA12 - EMA26 - EMA9(MACD)) / Close."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return (macd - sig) / close


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Bollinger %B = (C - Lower)/(Upper - Lower); width = (Upper - Lower)/SMA."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    return (close - lower) / (upper - lower), (upper - lower) / mid


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from an OHLCV frame. Rows in the warm-up period are NaN."""
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    for k in (1, 5, 10, 20):
        f[f"ret_{k}"] = c.pct_change(k)
    sma = {k: c.rolling(k).mean() for k in (20, 50, 200)}
    for k, s in sma.items():
        f[f"close_sma{k}"] = c / s - 1
    f["sma20_sma50"] = sma[20] / sma[50] - 1
    f["rsi_14"] = rsi(c)
    f["macd_hist"] = macd_hist(c)
    f["bb_pctb"], f["bb_width"] = bollinger(c)
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["atr_14_norm"] = atr(df) / c
    logv = np.log(df["Volume"].astype(float))
    f["vol_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    return f[FEATURE_COLS]
