# S&P 500 (SPY) Trading Signal Model — Module 3 Group 7

Time-series 3-class classification (Buy / Hold / Sell) of SPY's next-5-day return. The emphasis is on **leakage-free validation** and a **realistic, cost-aware backtest**, not on profit.

> 中文摘要：本项目用严格滞后的技术指标预测 SPY 未来 5 日收益的 Buy/Hold/Sell 类别。按时间切分并设 gap，用 vectorbt 做含费用的回测，并与 Buy & Hold 比较。结论：模型的分类效果（macro-F1 0.41）优于所有朴素基线，但扣费后回测收益 +23.7%，远低于 Buy & Hold 的 +56.5%。详见 [outputs/report_CN.md](outputs/report_CN.md)。后续研究预先登记了 9 种策略（趋势、ML 择时、ML 波动率目标、杠杆版本），先在 2013–2022 滚动回测，再在全新的 2025–2026 数据上检验，扣费后没有一个跑赢 Buy & Hold，只是能明显降低回撤。详见 [outputs/research/research_report_CN.md](outputs/research/research_report_CN.md)。计划文档见 [docs/](docs/)。

## Results at a glance (sealed test set 2023–2024)

| | Macro-F1 | Accuracy | Backtest return (after costs) | Sharpe |
|---|---|---|---|---|
| **Random Forest (final)** | **0.408** | 0.449 | +23.7% | 1.14 |
| Best naive baseline | 0.305 | — | — | — |
| Buy & Hold SPY | — | — | **+56.5%** | **1.82** |

The model beats every naive classifier, but it cannot separate up-moves from down-moves, so as a trading strategy it underperforms Buy & Hold. Why is explained in [outputs/report.md](outputs/report.md).

**Follow-up research: can any strategy beat Buy & Hold?** Nine strategies were pre-registered (trend filter, ML direction/crash signals, ML and rule-based volatility targeting, levered versions). They were tested on a 2013–2022 walk-forward and then on fresh 2025-01 → 2026-10 data. **None beat Buy & Hold's return after costs.** The trend and volatility-targeting strategies reduce drawdowns substantially at similar Sharpe, but they do not earn more. See [outputs/research/research_report.md](outputs/research/research_report.md) ([中文](outputs/research/research_report_CN.md)).

![equity](outputs/figures/equity_vs_bh.png)

## Repository layout

```text
docs/                 requirements analysis & development plan (EN / 中文)
knowledge&docs/       course materials and project brief (read-only)
data/raw/             yfinance snapshot (spy_20261010.csv) – all notebooks read this file
notebooks/
  01_eda.ipynb        EDA: distributions, indicator correlations, target balance
  02_modeling.ipynb   pipeline, baselines, tuning, validation selection, one-time test
  03_backtest.ipynb   vectorbt backtest with fees/slippage vs Buy & Hold
  04_strategy_research.ipynb  follow-up: 9 pre-registered strategies, walk-forward 2013–2022
  05_holdout_2025.ipynb       follow-up: fresh holdout 2025-01 → 2026-10 (downloads data/holdout/)
src/
  config.py           every design constant (dates, thresholds, fees, frozen config)
  data.py             download / load snapshot, data audit
  features.py         15 strictly-lagged features
  labels.py           forward return & 3-class labels (only module allowed to look ahead)
  dataset.py          features + labels table
  split.py            chronological segments with gap, TimeSeriesSplit CV
  models.py           baselines, Pipelines, search spaces
  evaluate.py         metrics, confusion matrices, experiment log
  backtest.py         signal rules, position state machine, vectorbt runner, metrics
  research.py         walk-forward refits, candidate strategies, levered simulator
tests/                look-ahead audit, label and split tests (pytest)
outputs/
  report.md / report_CN.md   Lessons Learned (EN / 中文)
  model_card.md
  experiment_log.csv         all validation experiments E00–E13
  frozen_config.json         configuration frozen before the test set was opened
  metrics.json, backtest_metrics.json, *_predictions.csv, figures/
  research/                  pre-registration (+ sha256), dev/holdout results, research report
```

## Reproduce

```bash
conda create -y -n spy-m3 python=3.11
```

```bash
conda run -n spy-m3 pip install -r requirements.txt
```

```bash
conda run -n spy-m3 python -m ipykernel install --user --name spy-m3
```

```bash
conda run -n spy-m3 python -m pytest -q
```

```bash
cd notebooks && for n in 01_eda 02_modeling 03_backtest 04_strategy_research 05_holdout_2025; do conda run -n spy-m3 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spy-m3 $n.ipynb; done
```

The snapshot in `data/raw/` is committed so results are reproducible offline. To refresh it, run `python -c "from src.data import download_snapshot; download_snapshot()"` from the project root. Notebooks always load the latest snapshot, and yfinance history can change, so a refreshed snapshot may give slightly different numbers. A full run takes about 1–2 minutes.

## Methodology guardrails

- **Prediction time:** close of *t*; **execution:** open of *t+1*.
- **Split:** Train 2010–2020 / Val 2021–2022 / Test 2023–2024. The last 5 rows before each boundary are dropped so no label window crosses a boundary. Tuning uses `TimeSeriesSplit(n_splits=5, gap=5)`.
- **Preprocessing:** all scaling and resampling sit inside sklearn/imblearn Pipelines and are fit per training fold.
- **Imbalance:** `class_weight="balanced"`. SMOTE (in-fold) and no-weight variants were tested as ablations.
- **Selection rules were stated in advance:** model = highest validation macro-F1; signal rule = highest validation after-cost Sharpe. The test set is read in one place only, after freezing.
- **Costs:** 0.1% fee + 0.05% slippage per side, with a fee-sensitivity table (0–0.2%).
