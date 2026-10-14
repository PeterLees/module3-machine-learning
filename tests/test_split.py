import pytest

from src import config
from src.dataset import build_dataset
from src.split import get_segments, make_cv, train_val_combined


@pytest.fixture(scope="module")
def ds(synthetic_ohlcv):
    return build_dataset(synthetic_ohlcv)


def test_segments_are_chronological_with_gap(ds):
    segs = get_segments(ds, include_test=True)
    cal = ds.index
    for a, b in [("train", "val"), ("val", "test")]:
        end = cal.get_loc(segs[a].index.max())
        start = cal.get_loc(segs[b].index.min())
        # label of the last sample uses Close(end + h): it must be before segment b
        assert end + config.HORIZON < start


def test_segment_date_ranges(ds):
    segs = get_segments(ds, include_test=True)
    assert segs["train"].index.min().year == 2010 and segs["train"].index.max().year == 2020
    assert segs["val"].index.min().year == 2021 and segs["val"].index.max().year == 2022
    assert segs["test"].index.min().year == 2023 and segs["test"].index.max().year == 2024
    assert not segs["test"]["label"].isna().any()


def test_test_hidden_by_default(ds):
    assert "test" not in get_segments(ds)


def test_train_val_combined_ends_before_test(ds):
    tv = train_val_combined(ds)
    test_start = ds.loc[config.TEST[0]:].index.min()
    assert ds.index.get_loc(tv.index.max()) + config.HORIZON < ds.index.get_loc(test_start)


def test_cv_folds_are_forward_with_gap(ds):
    train = get_segments(ds)["train"]
    for tr, va in make_cv().split(train):
        assert tr.max() + config.GAP < va.min()
