import numpy as np
import pandas as pd

from src.dataset import build_dataset
from src.labels import forward_realized_vol
from src.research import (
    apply_band,
    backtest_weights_vbt,
    simulate_weights,
    walk_forward,
)


def test_simulator_matches_vectorbt_unlevered(synthetic_ohlcv):
    prices = synthetic_ohlcv.loc["2015":"2016"]
    rng = np.random.default_rng(3)
    raw_w = pd.Series(rng.choice([0, 0.3, 0.6, 1.0], len(prices)), index=prices.index)
    w = apply_band(raw_w.rolling(10).mean().fillna(0).round(1))
    v_vbt = backtest_weights_vbt(w, prices).value()
    v_sim = simulate_weights(w, prices, borrow_rate=0.0)["value"]
    assert np.max(np.abs(v_vbt.to_numpy() / v_sim.to_numpy() - 1)) < 1e-3


def test_leverage_charges_interest(synthetic_ohlcv):
    prices = synthetic_ohlcv.loc["2015"].copy()
    prices[["Open", "Close"]] = 100.0  # flat market: only fees and interest matter
    w = pd.Series(1.5, index=prices.index)
    v_free = simulate_weights(w, prices, borrow_rate=0.0, fees=0, slippage=0)["value"]
    v_cost = simulate_weights(w, prices, borrow_rate=0.05, fees=0, slippage=0)["value"]
    assert abs(v_free.iloc[-1] - 100_000) < 1e-6
    # ~50k borrowed at 5% for ~1 year -> about 2.5k interest
    assert 2000 < 100_000 - v_cost.iloc[-1] < 3000


def test_forward_realized_vol_window():
    close = pd.Series(100 * np.cumprod(1 + np.r_[0, np.tile([0.01, -0.01], 20)]))
    v = forward_realized_vol(close, horizon=4)
    expected = close.pct_change().iloc[1:5].std() * np.sqrt(252)
    assert np.isclose(v.iloc[0], expected)
    assert v.iloc[-4:].isna().all()


def test_walk_forward_trains_only_on_past_labels(synthetic_ohlcv):
    ds = build_dataset(synthetic_ohlcv)
    seen = {}

    def spy(X, y, X_new):
        seen[X_new.index.min().year] = X.index.max()
        return np.zeros(len(X_new))

    walk_forward(ds, "label", spy, years=[2015, 2016], horizon=5)
    for year, last_train_day in seen.items():
        start = ds.index.get_indexer([ds.loc[f"{year}"].index.min()])[0]
        assert ds.index.get_loc(last_train_day) + 5 < start
