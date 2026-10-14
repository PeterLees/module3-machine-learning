# Can any model beat Buy & Hold SPY? — Strategy research

Module 3 Group 7 · 2026-10-13 · 中文版：[research_report_CN.md](research_report_CN.md) · Notebooks: `04_strategy_research.ipynb`, `05_holdout_2025.ipynb`

## Answer

**No.** Nine pre-registered strategies were tested: rules, ML signals and levered volatility targeting. None of them earned more than Buy & Hold after costs, either in the 10-year walk-forward development period (2013–2022) or on genuinely new data (2025-01 → 2026-10). What the better strategies do achieve is **lower risk**: similar or better Sharpe ratios and much smaller drawdowns at the cost of some return. Even that edge was not stable across periods.

## How the question was tested (and why this way)

The original test period (2023–2024) had already been used, so selecting a strategy there would be test-set snooping. A new protocol was written and hashed **before** any backtest was run or any 2025+ data was downloaded ([preregistration.md](preregistration.md)):

- **Development:** yearly walk-forward 2013–2022. Each year is predicted by models refit on all earlier data whose labels were already known. Selection rule: eligible = higher CAGR than Buy & Hold; pick = highest Sharpe among eligible.
- **Secondary:** 2023–2024 (already seen; reported only).
- **Fresh holdout:** 2025-01-02 → 2026-10-12, downloaded after pre-registration. **All** candidates are reported.
- **Costs and execution:** same as the main project (fill at next open, 0.1% fee + 0.05% slippage per side). Levered strategies pay 5% a year on borrowed cash.
- **Code checks:** four new tests: walk-forward training ends > h days before each predicted year; the leverage simulator matches vectorbt for unlevered weights; borrowing interest is charged correctly; the forward-volatility label covers exactly t+1..t+h. The last test caught a real bug, fixed before any results.

## Results

| Strategy | Dev 2013–22 CAGR | Dev Sharpe | Dev max DD | Holdout 2025–26 return | Holdout Sharpe | Holdout max DD |
|---|---|---|---|---|---|---|
| **S0 Buy & Hold** | **12.2%** | 0.75 | −33.7% | **+34.8%** | **1.09** | −18.8% |
| S1 Trend (Close > SMA200) | 8.3% | 0.74 | −20.0% | +22.8% | 1.08 | **−11.2%** |
| S2 RF 3-class, default-long | 8.2% | 0.56 | −31.3% | +32.1% | 1.06 | −16.6% |
| S3 Drop-risk classifier (LR) | 7.8% | 0.58 | −25.5% | +14.6% | 0.65 | −16.2% |
| S4 Vol-target, trailing σ20 | 10.9% | **0.89** | **−17.8%** | +24.8% | 0.98 | −15.7% |
| S5 Vol-target, ML σ̂ | 10.1% | 0.80 | −22.3% | +24.5% | 1.00 | −16.8% |
| S4L Vol-target σ20, ≤1.5× | 11.0% | 0.76 | −18.2% | +25.1% | 0.86 | −17.7% |
| S5L Vol-target ML, ≤1.5× | 8.7% | 0.63 | −24.4% | +21.6% | 0.79 | −18.4% |
| S6L Trend + Vol-target ML, ≤1.5× | 6.2% | 0.53 | −23.7% | +20.2% | 0.83 | −12.5% |
| *Original E06 model (reference)* | — | — | — | +24.0% | 0.96 | −12.9% |

Pre-registered pick from development: **none** (no candidate eligible).

![holdout](../figures/holdout_equity.png)

## What we learned

1. **Timing strategies lose in V-shaped rebounds.** In the development years the candidates beat Buy & Hold mainly in the slow 2022 bear market. They lost most in 2019 (S1 −17 pp) and 2020 (S3 −30 pp, S2 −20 pp, S1 −12 pp). The April-2025 crash and recovery repeated this on fresh data. The market's best days come right after its worst, and a strategy that reduces risk after a fall misses them.
2. **Volatility is predictable; that alone does not beat the market.** The ML volatility forecast explained 34% of out-of-sample variance in log 20-day volatility, against 8% for trailing volatility. Yet ML-based sizing (S5) did not beat simple trailing-vol sizing (S4). Better statistics did not mean a better strategy.
3. **Leverage did not close the gap.** Even with free borrowing, the best levered strategy made +27.6% on the holdout vs +34.8% for Buy & Hold. Leverage is added in calm periods and removed after shocks, just before the rebounds.
4. **A short window can crown the wrong model.** On the already-used 2023–2024 period, S2 *beat* Buy & Hold (+63.7% vs +57.0%). Over 2013–2022 it was among the worst (8.2% vs 12.2% a year), and on the holdout it trailed (+32.1% vs +34.8%). Had the strategy been chosen from 2023–2024, the result would have been a false discovery.
5. **The real achievable improvement is risk control.** S1 and S4 reduce maximum drawdown substantially (holdout: −11.2% / −15.7% vs −18.8%; development: −20.0% / −17.8% vs −33.7%) with similar Sharpe ratios. For an investor who values smaller losses this is a legitimate trade-off. It is not higher return, and its Sharpe advantage was not stable (S4: 0.89 vs 0.75 in development, 0.98 vs 1.09 on the holdout).

## Limits of this research

- Only nine candidates and a single asset. Other information (rates, credit spreads, VIX futures, breadth) was not used.
- The holdout is fresh data but not a perfectly blind researcher: the AI researcher has general knowledge of 2025 events (stated in the pre-registration).
- The 2025–26 holdout is 445 days, so a single period decides the ranking. No significance testing.
- Borrowing at 5% is an assumption; a post-hoc check at 0% and 2% did not change the conclusion.

## If you want to keep looking

These are hypotheses for a new, separately pre-registered study, not results:
- **Equal-risk comparison:** lever the best risk-adjusted strategy to Buy & Hold's volatility and test whether its return exceeds Buy & Hold. This only works if the Sharpe edge is real, and here it was unstable.
- **Exogenous data** with careful availability timing (e.g. term structure of VIX, credit spreads) for drawdown prediction.
- **Longer and more diverse test history** (e.g. include 2000–2002 and 2008, or other indices) before trusting any timing rule.
