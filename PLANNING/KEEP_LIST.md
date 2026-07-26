# KEEP-LIST — files to copy from CryptoTrader (pending user approval)

Target repo: **crypto-trading-bot-pipeline** (MIT, English docs, import-clean reference
pipeline — NOT a turnkey product). Source repo stays private; `.git` history is NEVER
copied. Every copied file passes LEAK_CHECKLIST.md before commit.

Legend: ✅ copy+clean · 🔧 copy, heavy strip (formulas/features → template) · ✂ structure
only (toy content) · ❌ excluded

## Stage 1 — Data layer (`stages/01_data/`)
| source | action | notes |
|---|---|---|
| backtest/common/download_klines.py | ✅ | public Binance klines, keyless |
| backtest/common/download_verify.py | ✅ | integrity checks |
| backtest/common/cache_builder_common.py | ✅ | parquet cache engine, parallel build |
| backtest/common/cache_builder_monthly.py | ✅ | |
| backtest/common/registry.py | 🔧 | coin universe → 3-coin example |
| backtest/totalmix_1/pipeline/fetch_listing_dates.py | ✅ | |

## Stage 2 — Features & gate sweep (`stages/02_gates/`)
| source | action | notes |
|---|---|---|
| backtest/common/features_common.py | 🔧 | real feature set OUT → 2-3 toy features (RSI, ATR-pct, vol-z) |
| backtest/common/gate_sweep.py | 🔧 | engine stays (asym SL/TP, EV ranking, JSON out); production gate values → example |
| backtest/common/oos_report.py | ✅ | |

## Stage 3 — Phase-2 / blocks (`stages/03_blocks/`)
| source | action | notes |
|---|---|---|
| backtest/common/phase2.py | 🔧 | exhaustive parallel + independence blocks; thresholds → config |
| backtest/common/run_all.py | ✅ | orchestration |
| backtest/common/scan_shards.py | ✅ | |
| backtest/common/accept_monthly.py, age_filter.py | optional | include if cheap to clean |

## Stage 4 — Strategy research layer (`strategies/`)
| source | action | notes |
|---|---|---|
| backtest/momentum_strategy/ (ONE substrategy dir) | ✂ | structure template with TOY signal (SMA-cross); real 16 strategies ❌ |
| backtest/unified_cell_comparison.py | 🔧 | aggregator engine stays |

## Stage 5 — QS pipeline (`stages/04_qs/`)
| source | action | notes |
|---|---|---|
| backtest/common/qs_common.py | 🔧 | |
| backtest/common/qs_features.py | 🔧 | feature names stripped |
| backtest/totalmix_1/pipeline/qs_to_cellcomp.py | 🔧 | |
| backtest/totalmix_1/pipeline/bootstrap_qs.py | ✅ | |
| bots/totalmix_1/qs_core.py, qs_daily.py, qs_replay.py | 🔧 | live-side QS (deterministic within-bar order = parity story); account/exchange refs OUT |

## Stage 6 — Portfolio assembly (`stages/05_portfolio/`)
| source | action | notes |
|---|---|---|
| backtest/totalmix_1/pipeline/build_cell_files.py | 🔧 | **Sv5 → TEMPLATE_FORMULA in scoring.py** |
| backtest/totalmix_1/pipeline/save_cell_trades.py | ✅ | (incl. the _tmp-resume fix) |
| backtest/totalmix_1/pipeline/trade_combo.py | 🔧 | budget-merge + diversity/JK/crash flags = showcase; score fn → template |
| backtest/totalmix_1/pipeline/trade_regime_combo.py, trade_full_combo.py | 🔧 | |
| backtest/common/budget_merge.py, style_map.py, beam_diversity.py | ✅ | generic selection mechanisms |
| backtest/totalmix_1/pipeline/deploy_sweep.py | 🔧 | lev×caution×cyc grid + TM_FIX_* fixed mode; ladder values → example |
| backtest/totalmix_1/analysis/deploy_consensus.py | 🔧 | 8-variant gates, graded relaxation, abstain (BOS-GEC) |
| backtest/totalmix_1/pipeline/veto_eval.py | optional | |

## Stage 7 — Blind walk-forward validation (`stages/06_walkforward/`)
| source | action | notes |
|---|---|---|
| backtest/totalmix_1/pipeline/procedure_wf_quarter.sh | 🔧 | quarterly blind chain |
| backtest/totalmix_1/pipeline/pwf_arm_iso.sh | 🔧 | TM_WORK isolation (A/B arms) |
| backtest/totalmix_1/pipeline/pwf_judge.py | 🔧 | forward judge |
| backtest/totalmix_1/pipeline/pwf_sequential.sh | 🔧 | ONE runner as example |
| pwf_party*/redo/resume/eblitz/tournament/round2 | ❌ | session-specific runners, not template material |

## Stage 8 — Live↔backtest parity (`stages/07_parity/`)
Curated subset (~5 of 19; pattern showcase, account layer OUT):
| source | action |
|---|---|
| parity/daily_live_reconcile.py | 🔧 |
| parity/test_qs_reconcile.py | 🔧 |
| parity/verify_qs_math.py | ✅ |
| parity/qs_guard_test.py | ✅ |
| parity/replica_vs_backtest.py | 🔧 |

## Excluded entirely
Real strategies/signals/params · feature set · Sv5 constants · production gate/ladder
values · coin whitelist · results/, pickles, parquet, logs · bots/ execution layer
(brain, agent, position_*, exchange_sync) · server/Tailscale/panel · docs/gate_search_playbook
· `.git` history.

## New content to write
- README.md (EN): 7-month engineering story, stage diagram, "bring your own alpha"
- docs/: one page per stage + architecture overview
- docs/CASE_STUDY.md: anonymized findings (blind-DD inflation 1.5-2.2×, anti-signal
  leverage selection, Goodhart lesson, diversity-quota tail cut across two regimes)
- config/example.yaml: all thresholds/params in one place, clearly EXAMPLE
- illustrative outputs per stage (synthetic numbers, labeled as such)
- scoring.py: TEMPLATE_FORMULA (simple sharpe-like) with "plug your formula here"
