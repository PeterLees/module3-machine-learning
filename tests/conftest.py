import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """Deterministic random-walk OHLCV covering the full project calendar."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2005-01-03", "2024-12-31")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
    open_ = close * (1 + rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(idx)))
    vol = rng.integers(5e7, 2e8, len(idx)).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=idx)
