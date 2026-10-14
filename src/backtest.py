"""Signal rules and vectorbt backtests.

Timing: a prediction made with data up to the close of day t is shifted by one
bar and filled at the OPEN of day t+1 (never at the close of t)."""

import numpy as np
import pandas as pd
import vectorbt as vbt

from src import config

# --- turning class probabilities into actions --------------------------------


def apply_rule(proba: pd.DataFrame, margin: float = None) -> pd.Series:
    """proba columns = [-1, 0, 1].
    margin=None -> argmax class.
    margin=m    -> Buy if P(Buy)-P(Sell) > m, Sell if P(Sell)-P(Buy) > m, else Hold."""
    if margin is None:
        return proba.idxmax(axis=1).astype(int)
    d = proba[1] - proba[-1]
    return pd.Series(np.where(d > margin, 1, np.where(d < -margin, -1, 0)), index=proba.index)


def to_positions(actions: pd.Series, long_short: bool = False, min_hold: int = 0) -> pd.Series:
    """State machine. Buy -> long; Sell -> flat (long-only) or short; Hold -> keep.
    min_hold: after a position change, ignore opposite signals for this many bars."""
    pos, held, out = 0, 0, []
    for a in actions.to_numpy():
        target = pos
        if a == 1:
            target = 1
        elif a == -1:
            target = -1 if long_short else 0
        if target != pos and (pos == 0 or held >= min_hold):
            pos, held = target, 0
        held += 1
        out.append(pos)
    return pd.Series(out, index=actions.index)


# --- backtest -----------------------------------------------------------------


def run_backtest(positions: pd.Series, prices: pd.DataFrame, fees: float = config.FEES,
                 slippage: float = config.SLIPPAGE, init_cash: float = config.INIT_CASH):
    """positions indexed by signal date t (1 long, 0 flat, -1 short). Executed at Open(t+1)."""
    prices = prices.loc[positions.index.min():positions.index.max()]
    target = positions.reindex(prices.index).shift(1).fillna(0)  # act on next bar
    prev = target.shift(1).fillna(0)
    entries = (target == 1) & (prev != 1)
    exits = (target != 1) & (prev == 1)
    short_entries = (target == -1) & (prev != -1)
    short_exits = (target != -1) & (prev == -1)
    return vbt.Portfolio.from_signals(
        close=prices["Close"], price=prices["Open"],
        entries=entries, exits=exits, short_entries=short_entries, short_exits=short_exits,
        fees=fees, slippage=slippage, init_cash=init_cash, freq="1D")


def buy_and_hold(prices: pd.DataFrame, fees: float = config.FEES,
                 slippage: float = config.SLIPPAGE, init_cash: float = config.INIT_CASH):
    """Benchmark: buy at the first open of the period, hold to the end."""
    entries = pd.Series(False, index=prices.index)
    entries.iloc[0] = True
    return vbt.Portfolio.from_signals(
        close=prices["Close"], price=prices["Open"], entries=entries,
        exits=pd.Series(False, index=prices.index), fees=fees, slippage=slippage,
        init_cash=init_cash, freq="1D")


def summarize(pf, name: str = "") -> dict:
    """Metrics computed from the equity curve (252 trading days per year, rf = 0)."""
    v = pf.value()
    # measure from initial cash, so the first day's fill costs and move are included
    init = float(np.asarray(pf.init_cash).ravel()[0])
    r = pd.concat([pd.Series([init]), pd.Series(v.to_numpy())]).pct_change().dropna()
    years = len(r) / 252
    total = v.iloc[-1] / init - 1
    dd = (v / v.cummax() - 1).min()
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    orders = pf.orders.records_readable
    pos_value = pf.asset_value().abs()
    return {
        "strategy": name,
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(r.std() * np.sqrt(252)), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd), 4),
        "calmar": round(float(cagr / abs(dd)), 3) if dd < 0 else np.nan,
        "n_orders": int(len(orders)),
        "n_trades": int(pf.trades.count()),
        "exposure": round(float((pos_value > 0).mean()), 3),
        "total_fees": round(float(orders["Fees"].sum()) if len(orders) else 0.0, 2),
        "final_value": round(float(v.iloc[-1]), 2),
    }
