"""Strategy research: walk-forward model refits, weight-based strategies and backtests.

Each strategy produces a target portfolio weight w(t) in [0, w_max] after the close
of day t; the weight is executed at the OPEN of t+1. Unlevered strategies (w <= 1)
are backtested with vectorbt; levered ones (w > 1) with `simulate_weights`, a small
transparent simulator that charges borrowing interest and is cross-checked against
vectorbt on unlevered weights (tests/test_research.py)."""

import numpy as np
import pandas as pd
import vectorbt as vbt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURE_COLS

BORROW_RATE = 0.05  # annual interest on borrowed cash for levered strategies
REBALANCE_BAND = 0.10  # trade only when |target - current| >= band (or to/from zero)
TARGET_VOL = 0.15  # annualised volatility target
W_MAX_LEVERED = 1.5


# --- walk-forward engine --------------------------------------------------------


def walk_forward(ds: pd.DataFrame, target: str, fit_predict, years, horizon: int,
                 train_start: str = "2006-01-01") -> pd.Series:
    """For each year Y: train on rows from train_start whose label window ends before
    Y-01-01 (the last `horizon` rows before the year are dropped), then predict every
    day of year Y. Returns the stitched out-of-sample predictions."""
    out = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        hist = ds.loc[train_start:start - pd.Timedelta(days=1)].dropna(subset=[target])
        hist = hist.iloc[:-horizon]
        test = ds.loc[f"{y}-01-01":f"{y}-12-31"]
        if test.empty:
            continue
        out.append(pd.Series(fit_predict(hist[FEATURE_COLS], hist[target], test[FEATURE_COLS]),
                             index=test.index))
    return pd.concat(out)


# --- model recipes (fixed in advance; no tuning on out-of-sample years) ---------


def rf3_proba_diff(X, y, X_new):
    """Frozen E06 random forest; returns P(Sell) - P(Buy)."""
    m = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=68,
                               class_weight="balanced", n_jobs=-1, random_state=config.SEED)
    m.fit(X, y.astype(int))
    p = pd.DataFrame(m.predict_proba(X_new), columns=m.classes_)
    return (p[-1] - p[1]).to_numpy()


def drop_event_signal(X, y, X_new, pct: float = 85):
    """LR classifier for 'next 10 days fall more than 3%'. Returns 1 when the
    predicted probability exceeds the `pct` percentile of training-period
    probabilities (i.e. the model's top-15% risk days), else 0."""
    m = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))])
    m.fit(X, y.astype(int))
    thr = np.percentile(m.predict_proba(X)[:, 1], pct)
    return (m.predict_proba(X_new)[:, 1] > thr).astype(int)


def vol_forecast(X, y, X_new):
    """Gradient-boosted regression of log forward 20-day realised volatility."""
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                      min_samples_leaf=50, random_state=config.SEED)
    m.fit(X, np.log(y))
    return np.exp(m.predict(X_new))


# --- weights -> trades ------------------------------------------------------------


def apply_band(weights: pd.Series, band: float = REBALANCE_BAND) -> pd.Series:
    """Hold the last executed target until it moves by >= band (moves to/from 0 always trade)."""
    out, cur = [], 0.0
    for w in weights.to_numpy():
        if (abs(w - cur) >= band) or (w == 0 and cur != 0) or (cur == 0 and w > 0):
            cur = float(w)
        out.append(cur)
    return pd.Series(out, index=weights.index)


def backtest_weights_vbt(weights: pd.Series, prices: pd.DataFrame, fees=config.FEES,
                         slippage=config.SLIPPAGE, init_cash=config.INIT_CASH):
    """Unlevered (w <= 1) weight strategy in vectorbt: target set at close t, filled at open t+1."""
    assert weights.max() <= 1 + 1e-9, "vectorbt path is for unlevered weights"
    prices = prices.loc[weights.index.min():weights.index.max()]
    w = weights.reindex(prices.index).shift(1)
    size = w.where(w.ne(w.shift(1)))  # orders only when the target changes
    size.iloc[0] = np.nan
    return vbt.Portfolio.from_orders(close=prices["Close"], price=prices["Open"], size=size,
                                     size_type="targetpercent", fees=fees, slippage=slippage,
                                     init_cash=init_cash, freq="1D")


def simulate_weights(weights: pd.Series, prices: pd.DataFrame, fees=config.FEES,
                     slippage=config.SLIPPAGE, init_cash=config.INIT_CASH,
                     borrow_rate: float = BORROW_RATE) -> pd.DataFrame:
    """Daily simulator supporting leverage. Target w(t) is executed at Open(t+1): shares
    are set to w * equity(at open) / fill price; fees + slippage are charged on the traded
    notional; negative cash pays borrow_rate/252 per day. Returns value / trades / fees."""
    prices = prices.loc[weights.index.min():weights.index.max()]
    target = weights.reindex(prices.index).shift(1)
    o, c = prices["Open"].to_numpy(), prices["Close"].to_numpy()
    cash, shares, fees_paid, n_trades = float(init_cash), 0.0, 0.0, 0
    prev_w, values = np.nan, []
    for i, w in enumerate(target.to_numpy()):
        if i > 0 and cash < 0:
            cash *= 1 + borrow_rate / 252  # interest accrued overnight
        if not np.isnan(w) and w != prev_w:
            equity_open = cash + shares * o[i]
            new_shares = w * equity_open / o[i]
            delta = new_shares - shares
            if abs(delta) > 1e-12:
                fill = o[i] * (1 + slippage * np.sign(delta))
                fee = abs(delta) * fill * fees
                cash -= delta * fill + fee
                shares, fees_paid, n_trades = new_shares, fees_paid + fee, n_trades + 1
            prev_w = w
        values.append(cash + shares * c[i])
    return pd.DataFrame({"value": values}, index=prices.index).assign(
        n_orders=n_trades, fees=fees_paid)


# --- metrics ----------------------------------------------------------------------


def metrics_from_values(v: pd.Series, init_cash=config.INIT_CASH, name="", n_orders=0,
                        fees=0.0, exposure=np.nan) -> dict:
    r = pd.concat([pd.Series([init_cash]), pd.Series(v.to_numpy())]).pct_change().dropna()
    years = len(r) / 252
    total = v.iloc[-1] / init_cash - 1
    dd = (v / v.cummax()).min() - 1
    dd = min(dd, v.min() / init_cash - 1)
    cagr = (1 + total) ** (1 / years) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return {"strategy": name, "total_return": round(float(total), 4), "cagr": round(float(cagr), 4),
            "ann_vol": round(float(r.std() * np.sqrt(252)), 4), "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(dd), 4),
            "calmar": round(float(cagr / abs(dd)), 3) if dd < 0 else np.nan,
            "n_orders": int(n_orders), "avg_exposure": round(float(exposure), 3),
            "total_fees": round(float(fees), 2), "final_value": round(float(v.iloc[-1]), 2)}


def evaluate(weights: pd.Series, prices: pd.DataFrame, name: str, fees=config.FEES,
             slippage=config.SLIPPAGE):
    """Backtest one weight series; vectorbt if unlevered, simulator otherwise.
    Returns (metrics dict, value series)."""
    exposure = weights.shift(1).fillna(0).mean()
    if weights.max() <= 1 + 1e-9:
        pf = backtest_weights_vbt(weights, prices, fees, slippage)
        orders = pf.orders.records_readable
        v = pf.value()
        return metrics_from_values(v, name=name, n_orders=len(orders),
                                   fees=orders["Fees"].sum() if len(orders) else 0.0,
                                   exposure=exposure), v
    sim = simulate_weights(weights, prices, fees, slippage)
    return metrics_from_values(sim["value"], name=name, n_orders=sim["n_orders"].iloc[0],
                               fees=sim["fees"].iloc[0], exposure=exposure), sim["value"]


# --- the pre-registered candidates (outputs/research/preregistration.md) -----------


def _hold_rule(exit_flag: pd.Series, min_hold: int = 5) -> pd.Series:
    """Default long; go to cash when exit_flag == 1, respecting a minimum holding period."""
    from src.backtest import to_positions
    actions = pd.Series(np.where(exit_flag.to_numpy() == 1, -1, 1), index=exit_flag.index)
    return to_positions(actions, min_hold=min_hold).astype(float)


def _vol_weight(sigma: pd.Series, w_max: float) -> pd.Series:
    w = (TARGET_VOL / sigma).clip(0, w_max).round(1)
    return apply_band(w)


def build_candidates(ds: pd.DataFrame, years) -> tuple[dict, pd.DataFrame]:
    """Weights of S0-S6L for every trading day in `years`, using yearly walk-forward
    refits. `ds` must contain FEATURE_COLS, 'label' (3-class), 'drop10' and 'fvol20'.
    Also returns the out-of-sample volatility forecasts for diagnostics."""
    days = ds.loc[f"{min(years)}-01-01":f"{max(years)}-12-31"].index
    trend = (ds.loc[days, "close_sma200"] > 0).astype(float)
    sigma20 = ds.loc[days, "vol_20"] * np.sqrt(252)

    diff = walk_forward(ds, "label", rf3_proba_diff, years, horizon=config.HORIZON)
    drop_sig = walk_forward(ds, "drop10", drop_event_signal, years, horizon=10)
    sigma_ml = walk_forward(ds, "fvol20", vol_forecast, years, horizon=20)

    w = {
        "S0 Buy & Hold": pd.Series(1.0, index=days),
        "S1 Trend SMA200": trend,
        "S2 RF default-long": _hold_rule((diff > 0.10).astype(int)),
        "S3 Drop-risk LR": _hold_rule(drop_sig),
        "S4 Vol-target σ20": _vol_weight(sigma20, 1.0),
        "S5 Vol-target ML": _vol_weight(sigma_ml, 1.0),
        "S4L Vol-target σ20 ×1.5": _vol_weight(sigma20, W_MAX_LEVERED),
        "S5L Vol-target ML ×1.5": _vol_weight(sigma_ml, W_MAX_LEVERED),
        "S6L Trend + Vol-target ML ×1.5": apply_band(
            ((TARGET_VOL / sigma_ml).clip(0, W_MAX_LEVERED).round(1) * trend)),
    }
    diag = pd.DataFrame({"sigma20": sigma20, "sigma_ml": sigma_ml,
                         "fvol20": ds.loc[days, "fvol20"], "drop_sig": drop_sig, "rf_diff": diff})
    return w, diag


def research_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Modelling table with the three research targets."""
    from src.dataset import build_dataset
    from src.labels import forward_drop_event, forward_realized_vol
    ds = build_dataset(raw)
    ds["drop10"] = forward_drop_event(raw["Close"]).reindex(ds.index)
    ds["fvol20"] = forward_realized_vol(raw["Close"]).reindex(ds.index)
    return ds
