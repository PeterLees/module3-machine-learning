# Model Card — SPY 5-day Buy / Hold / Sell Classifier (E06)

| Item | Value |
|---|---|
| Model | `RandomForestClassifier(n_estimators=300, max_depth=None, max_features="sqrt", min_samples_leaf=68, class_weight="balanced", random_state=42)` inside an sklearn Pipeline |
| Artifact | `models/final_pipeline.joblib` (not committed; regenerate with `02_modeling.ipynb`) |
| Task | 3-class classification of SPY's next-5-trading-day close→close return: Buy > +1%, Sell < −1%, Hold otherwise |
| Prediction time | After the close of day *t*; intended execution at the open of *t+1* |
| Inputs | 15 features computed from SPY daily OHLCV up to the close of *t*: `ret_1/5/10/20`, `close_sma20/50/200`, `sma20_sma50`, `rsi_14`, `macd_hist`, `bb_pctb`, `bb_width`, `vol_20`, `atr_14_norm`, `vol_z20` (definitions in `src/features.py`) |
| Data | yfinance SPY, auto-adjusted, 2005-01-03 → 2024-12-31 (snapshot `data/raw/spy_20261010.csv`); 2005–2009 used only for indicator warm-up |
| Training | Hyper-parameters: `RandomizedSearchCV` (20 draws), `TimeSeriesSplit(n_splits=5, gap=5)` on Train 2010–2020, scored by macro-F1. Final fit: Train + Val, 2010-01-04 → 2022-12-22 (3,267 rows) |
| Selection | Highest validation (2021–2022) macro-F1 among 7 tuned candidates (LR ×4 variants, RF, HGB ×2) |
| Signal rule | Buy if P(Buy) − P(Sell) > 0.05; Sell if P(Sell) − P(Buy) > 0.05; else Hold; minimum holding period 5 bars (chosen on the validation backtest) |
| Validation | macro-F1 0.404, accuracy 0.402 |
| **Test 2023–2024 (once)** | **macro-F1 0.408**, accuracy 0.449, balanced accuracy 0.413, log loss 1.058 (uniform = 1.099). Best baseline macro-F1 0.305 |
| Backtest (test, after 0.1% fee + 0.05% slippage per side, long-only) | +23.7% vs Buy & Hold +56.5%; Sharpe 1.14 vs 1.82; max DD −9.6% vs −10.0% |

## Intended use
Teaching and methodology demonstration: a leakage-free time-series classification and backtest workflow. **Not** for real trading or investment decisions.

## Known limitations
- Essentially no skill at separating Buy from Sell; the edge over baselines comes from recognising Hold (quiet weeks).
- Probabilities are close to uniform and are not calibrated.
- Trained on one asset and 2010–2022 regimes; relationships (e.g. `close_sma200`) changed sign in 2022.
- Label (close→close) is not aligned with the execution price (next open).
- Performance is a single 2-year realisation; no significance testing.
- yfinance data may be revised; results are tied to the stored snapshot.

## Ethical / risk notes
Using the strategy with real money would, based on this evidence, most likely underperform passive investing after costs.
