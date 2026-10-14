"""Target construction. This is the ONLY module allowed to look into the future
(negative shifts): the label defines the answer, it is never a model input."""

import numpy as np
import pandas as pd

from src import config


def forward_return(close: pd.Series, horizon: int = config.HORIZON) -> pd.Series:
    """r_fwd(t, h) = Close(t+h) / Close(t) - 1. The last h rows are NaN."""
    return close.shift(-horizon) / close - 1


def forward_return_open(open_: pd.Series, horizon: int = config.HORIZON) -> pd.Series:
    """Execution-aligned alternative: Open(t+1+h) / Open(t+1) - 1."""
    return open_.shift(-(horizon + 1)) / open_.shift(-1) - 1


def forward_realized_vol(close: pd.Series, horizon: int = 20) -> pd.Series:
    """Annualised std of daily returns over days t+1..t+h (target for volatility models)."""
    r = close.pct_change()
    # std over the window ending at t+h (= returns t+1..t+h), moved back to row t
    return r.rolling(horizon).std().shift(-horizon) * np.sqrt(252)


def forward_drop_event(close: pd.Series, horizon: int = 10, drop: float = -0.03) -> pd.Series:
    """1 if Close(t+h)/Close(t) - 1 < drop, else 0 (NaN where unknown)."""
    r = forward_return(close, horizon)
    return (r < drop).astype(float).where(r.notna())


def to_class(r: pd.Series, buy_thr=config.BUY_THR, sell_thr=config.SELL_THR) -> pd.Series:
    """Map returns to Buy=1 / Hold=0 / Sell=-1. NaN returns stay NaN.

    buy_thr / sell_thr may be scalars or Series aligned with r (volatility-scaled)."""
    out = pd.Series(np.where(r > buy_thr, 1, np.where(r < sell_thr, -1, 0)),
                    index=r.index, dtype="float")
    out[r.isna()] = np.nan
    return out


def quantile_thresholds(r_train: pd.Series, lo: float = 1 / 3, hi: float = 2 / 3):
    """Thresholds giving roughly balanced classes, estimated on TRAINING returns only."""
    return float(r_train.quantile(hi)), float(r_train.quantile(lo))


def vol_scaled_thresholds(vol_20: pd.Series, k: float = 0.5, horizon: int = config.HORIZON):
    """+/- k * sigma_20(t) * sqrt(h); sigma uses data up to t only."""
    band = k * vol_20 * np.sqrt(horizon)
    return band, -band
