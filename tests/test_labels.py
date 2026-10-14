import numpy as np
import pandas as pd

from src.labels import (
    forward_return,
    forward_return_open,
    quantile_thresholds,
    to_class,
)


def test_forward_return_and_classes():
    close = pd.Series([100, 101, 102, 103, 104, 102.0, 99, 98, 110, 111, 112],
                      index=pd.bdate_range("2020-01-01", periods=11))
    r = forward_return(close, 5)
    assert np.isclose(r.iloc[0], 102 / 100 - 1)  # +2.0% -> Buy
    assert np.isclose(r.iloc[1], 99 / 101 - 1)  # -1.98% -> Sell
    assert np.isclose(r.iloc[2], 98 / 102 - 1)  # -3.9% -> Sell
    assert r.iloc[-5:].isna().all()
    y = to_class(r)
    assert y.iloc[0] == 1 and y.iloc[1] == -1 and y.iloc[2] == -1
    assert y.iloc[-5:].isna().all()


def test_hold_band_is_inclusive():
    r = pd.Series([0.01, -0.01, 0.0100001, -0.0100001, 0.0])
    assert to_class(r).tolist() == [0, 0, 1, -1, 0]


def test_open_label_alignment():
    o = pd.Series(np.arange(1, 13, dtype=float))
    r = forward_return_open(o, 5)
    assert np.isclose(r.iloc[0], o.iloc[6] / o.iloc[1] - 1)
    assert r.iloc[-6:].isna().all()


def test_quantile_thresholds_balance():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 3000))
    hi, lo = quantile_thresholds(r)
    counts = to_class(r, hi, lo).value_counts(normalize=True)
    assert (counts - 1 / 3).abs().max() < 0.01
