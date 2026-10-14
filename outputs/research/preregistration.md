# Pre-registration — "Can any strategy beat Buy & Hold SPY?"

Written 2026-10-13, **before** any of the candidates below was backtested and **before** any data after 2024-12-31 was downloaded. It may not be edited after results exist; any deviation must be reported as a deviation.

## Why a new protocol
The original test period (2023–2024) has already been opened and examined. Choosing a strategy because it looks good on 2023–2024 would be test-set snooping. Therefore:

| Phase | Data | Role |
|---|---|---|
| Development | Walk-forward, yearly refit, out-of-sample years **2013–2022** (first training window 2006–2012) | The only data used to choose a strategy |
| Secondary (already seen) | 2023–2024, same walk-forward | Reported, but **not clean** evidence |
| **Fresh holdout** | **2025-01-01 → latest available**, downloaded only after this file is written; models refit yearly on all earlier data | Final, one-time evaluation of **all** candidates |

Caveat on the holdout: the researcher (an AI model) has general knowledge of market events up to its training cutoff, which overlaps 2025. The candidate list and parameters were fixed on research grounds (well-known strategy families plus the project's own finding that volatility is predictable while direction is not), but the holdout cannot be called perfectly blind.

## Common assumptions
Target weight set after the close of t, executed at the open of t+1; 0.1% fee + 0.05% slippage per side on traded notional; cash earns 0%; borrowed cash (weights > 1) pays 5% per year; fractional weights rounded to 0.1 and rebalanced only when the target moves by ≥ 0.10 (moves to/from 0 always trade). Unlevered strategies use vectorbt; levered ones use `src/research.py::simulate_weights` (cross-checked against vectorbt).

## Candidates (all parameters fixed here)
| ID | Weight rule | Model |
|---|---|---|
| S0 | 1 (Buy & Hold) | — |
| S1 | 1 if Close > SMA200 else 0 | rule |
| S2 | 0 if P(Sell) − P(Buy) > 0.10 else 1; minimum hold 5 days | frozen E06 RF 3-class, refit yearly |
| S3 | 0 if P(next-10-day return < −3%) exceeds the 85th percentile of training-period probabilities, else 1; minimum hold 5 days | LR (C=0.01, balanced), refit yearly |
| S4 | min(1, 15% / σ20) with σ20 = trailing 20-day realised vol (annualised) | rule |
| S5 | min(1, 15% / σ̂) with σ̂ = predicted next-20-day realised vol | HGB regressor on log vol (200 iters, lr 0.05, depth 3, leaf 50), refit yearly |
| S4L | min(1.5, 15% / σ20) | rule, **levered** |
| S5L | min(1.5, 15% / σ̂) | ML, **levered** |
| S6L | S5L weight × (1 if Close > SMA200 else 0) | ML + rule, **levered** |

## Selection rule (development period 2013–2022 only)
1. **Eligible** = after-cost CAGR above Buy & Hold's over 2013–2022.
2. Among eligible strategies, the **pick** is the one with the highest Sharpe ratio.
3. If none is eligible, the answer to the question is "no" for the development period, and the holdout is still reported for all candidates.
4. Levered and unlevered results are always shown separately: beating Buy & Hold with leverage is a statement about risk taken, so Sharpe and max drawdown are reported next to returns.

## Holdout reporting
Every candidate is reported on the holdout, not only the pick. Success for the pick = higher after-cost total return than Buy & Hold over the holdout. Its Sharpe and drawdown are reported alongside.
