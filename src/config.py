"""Project-wide constants. Every tunable design decision lives here so that
notebooks and modules share one source of truth (see docs/ plan, Section 3)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# --- Data -------------------------------------------------------------------
TICKER = "SPY"
START = "2005-01-01"
END = "2025-01-01"  # yfinance end is exclusive -> last bar 2024-12-31

# --- Target -----------------------------------------------------------------
HORIZON = 5  # trading days
BUY_THR = 0.01  # r_fwd > +1%  -> Buy (1)
SELL_THR = -0.01  # r_fwd < -1%  -> Sell (-1)
CLASSES = [-1, 0, 1]
CLASS_NAMES = {-1: "Sell", 0: "Hold", 1: "Buy"}

# --- Chronological split ----------------------------------------------------
TRAIN = ("2010-01-01", "2020-12-31")
VAL = ("2021-01-01", "2022-12-31")
TEST = ("2023-01-01", "2024-12-31")
GAP = HORIZON  # rows dropped at the end of each segment (overlapping labels)
CV_SPLITS = 5

# --- Backtest ---------------------------------------------------------------
INIT_CASH = 100_000
FEES = 0.001  # per side
SLIPPAGE = 0.0005  # per side
FEE_GRID = [0.0, 0.0005, 0.001, 0.002]

SEED = 42

# --- Frozen configuration (filled at the end of P6, see 02_modeling.ipynb) --
# Selection rules stated in advance:
#   model       -> highest validation macro-F1 among tuned candidates
#   signal rule -> highest after-cost Sharpe on the validation-period backtest;
#                  ties broken by fewer trades.
# Frozen at the end of P6 (mirrors outputs/frozen_config.json, written by 02_modeling).
FINAL_MODEL = {"experiment": "E06", "model": "RandomForestClassifier",
               "params": {"class_weight": "balanced", "n_estimators": 300, "max_depth": None,
                          "max_features": "sqrt", "min_samples_leaf": 68},
               "features": "all 15 (FEATURE_COLS)"}
FINAL_SIGNAL_RULE = {"margin": 0.05, "min_hold": 5}
