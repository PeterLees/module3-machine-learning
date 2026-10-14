"""Chronological splitting with a gap at each boundary.

The test segment is only returned when explicitly requested
(include_test=True); this is used in exactly one place, after the
configuration has been frozen (Gate G4)."""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src import config


def _segment(df: pd.DataFrame, start: str, end: str, gap: int) -> pd.DataFrame:
    """Rows within [start, end], minus the last `gap` rows whose label windows
    would reach into the next segment."""
    seg = df.loc[start:end]
    return seg.iloc[:-gap] if gap else seg


def get_segments(df: pd.DataFrame, gap: int = config.GAP, include_test: bool = False) -> dict:
    """Return labelled train / val (/ test) segments, in chronological order."""
    labelled = df.dropna(subset=["label"])
    segs = {
        "train": _segment(labelled, *config.TRAIN, gap),
        "val": _segment(labelled, *config.VAL, gap),
    }
    if include_test:
        # test is the final segment: its tail is already unlabelled (NaN dropped)
        segs["test"] = labelled.loc[config.TEST[0]:config.TEST[1]]
    assert_chronological(segs, df.index, gap)
    return segs


def train_val_combined(df: pd.DataFrame, gap: int = config.GAP) -> pd.DataFrame:
    """Train+Val (2010-2022) for the final refit; drops only the last `gap` rows."""
    labelled = df.dropna(subset=["label"])
    return _segment(labelled, config.TRAIN[0], config.VAL[1], gap)


def assert_chronological(segs: dict, calendar: pd.Index, gap: int = config.GAP) -> None:
    """Each segment's last sample sits at least `gap` trading days before the next
    segment starts, so its h-day label window ends before that segment begins."""
    names = list(segs)
    for a, b in zip(names, names[1:]):
        end_pos = calendar.get_loc(segs[a].index.max())
        start_pos = calendar.get_loc(segs[b].index.min())
        assert start_pos - end_pos > gap, f"{a} -> {b}: gap {start_pos - end_pos} <= {gap}"
    for s in segs.values():
        assert s.index.is_monotonic_increasing and not s.index.duplicated().any()


def make_cv(n_splits: int = config.CV_SPLITS, gap: int = config.GAP) -> TimeSeriesSplit:
    """Expanding-window CV inside the training segment, with a label-overlap gap."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)
