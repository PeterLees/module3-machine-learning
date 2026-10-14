"""Look-ahead audit: features at day t must not change when future data is removed."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.backtest import run_backtest
from src.features import FEATURE_COLS, build_features

SRC = Path(__file__).resolve().parents[1] / "src"


def _check_truncation(raw: pd.DataFrame, n_dates: int = 25):
    full = build_features(raw)
    valid = full.dropna().index
    rng = np.random.default_rng(1)
    dates = sorted(rng.choice(valid, size=n_dates, replace=False))
    for t in dates:
        trunc = build_features(raw.loc[:t])
        pd.testing.assert_series_equal(trunc.loc[t], full.loc[t], check_names=False,
                                       rtol=1e-10, atol=1e-12)


def test_truncation_synthetic(synthetic_ohlcv):
    _check_truncation(synthetic_ohlcv)


def test_truncation_real_snapshot():
    from src.data import load_snapshot
    try:
        raw = load_snapshot()
    except FileNotFoundError:
        pytest.skip("no snapshot downloaded")
    _check_truncation(raw)


def test_future_shock_does_not_move_past_features(synthetic_ohlcv):
    """Multiply all prices after a cut-off by 10: features up to the cut-off must be identical."""
    raw = synthetic_ohlcv.copy()
    cut = raw.index[3000]
    shocked = raw.copy()
    shocked.loc[shocked.index > cut, ["Open", "High", "Low", "Close"]] *= 10
    a, b = build_features(raw).loc[:cut], build_features(shocked).loc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_no_negative_shift_or_centered_window_outside_labels():
    pattern = re.compile(r"shift\(\s*-|center\s*=\s*True")
    offenders = [p.name for p in SRC.glob("*.py")
                 if p.name != "labels.py" and pattern.search(p.read_text())]
    assert offenders == [], f"future-looking operations found in {offenders}"


def test_feature_columns_have_no_raw_prices(synthetic_ohlcv):
    cols = build_features(synthetic_ohlcv).columns
    assert list(cols) == FEATURE_COLS
    assert not {"Open", "High", "Low", "Close", "Volume"} & set(cols)


def test_backtest_executes_on_next_bar(synthetic_ohlcv):
    prices = synthetic_ohlcv.loc["2023-01-01":"2023-03-31"]
    pos = pd.Series(0, index=prices.index)
    signal_day = prices.index[10]
    pos.loc[signal_day:] = 1
    pf = run_backtest(pos, prices)
    orders = pf.orders.records_readable
    assert len(orders) == 1
    fill_day = orders["Timestamp"].iloc[0]
    assert fill_day == prices.index[11], "signal at t must fill at t+1"
    expected = prices.loc[fill_day, "Open"] * (1 + config.SLIPPAGE)
    assert abs(orders["Price"].iloc[0] - expected) < 1e-8, "fill must use the open price"
