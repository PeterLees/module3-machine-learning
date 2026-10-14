"""Assemble the modelling table: features (<= t) + forward return + label."""

import pandas as pd

from src import config, labels
from src.features import FEATURE_COLS, build_features


def build_dataset(raw: pd.DataFrame, label_kind: str = "close",
                  buy_thr=config.BUY_THR, sell_thr=config.SELL_THR) -> pd.DataFrame:
    """Return one row per trading day with FEATURE_COLS, r_fwd and label.

    label_kind: "close" -> Close(t+5)/Close(t)-1 (project brief, primary)
                "open"  -> Open(t+6)/Open(t+1)-1 (execution-aligned sensitivity)
    Warm-up rows (any feature NaN) are dropped; the unlabelled tail is kept
    (label NaN) so the model can still emit signals on the last days.
    """
    feats = build_features(raw)
    if label_kind == "close":
        r = labels.forward_return(raw["Close"])
    elif label_kind == "open":
        r = labels.forward_return_open(raw["Open"])
    else:
        raise ValueError(label_kind)
    out = feats.copy()
    out["r_fwd"] = r
    out["label"] = labels.to_class(r, buy_thr, sell_thr)
    return out.dropna(subset=FEATURE_COLS)
