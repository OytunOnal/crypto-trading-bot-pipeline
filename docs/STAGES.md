# Stage-by-stage guide

Each stage is a package under `factory/`. Stages communicate through files
(parquet/pickle caches, txt/json result tables) — no shared in-memory state —
so any stage can be re-run, resumed, or A/B-forked in isolation.

## Stage 1 — Data (`factory/data`)

- `download_klines.py` — all USDT-perp 5m klines, keyless public API.
  Append-mode with a **trailing overlap re-fetch + content diff** (exchange
  revision protection), then a verify loop: freshness, missing, and
  *internal-gap* detection with targeted re-download. Exits non-zero if any
  symbol stays dirty — data problems surface, never silently persist.
- `cache_builder.py` — year-split feature/PnL caches per strategy: signals →
  features (3 timeframes + BTC-relative) → vectorized SL/TP simulation over a
  full SL×TP grid → per-year pickles, chunk-flushed to bound RAM.
- `cache_builder_monthly.py` — the production format: one shard per
  (month, regime-cell). Engines stream only their cell's 10–150MB shards
  instead of unpickling multi-GB year monoliths; daily incremental update =
  rebuild the current month. Coin age is counted in *bars since listing* to
  exactly match the live gate.
- `registry.py` — single source of truth mapping strategy codes to builders.
  Add a row = the whole factory picks the strategy up.

## Stage 2 — Gates (`factory/gates`)

- `features_common.py` — the feature battery (template ships 6 features ×
  3 TFs + 6 BTC-relative columns; production ran a much larger battery).
  Also: SL/TP simulation, rolling resample, BTC regime terciles, per-feature
  IC analysis with per-year sign stability.
- `gate_sweep.py` — per (cell, direction, SL, TP): IC-stable independent
  feature selection → per-feature threshold optimization → exhaustive AND
  combos (k 3–8) with failed-subset pruning → forward AND reverse
  walk-forward validation → only STRONG verdicts survive. Parallel mode uses
  **event-driven measured-RAM admission**: at most one worker in its load
  phase, admission decided on live `psutil` numbers, workers signal
  loaded/done through a queue.

## Stage 3 — Blocks (`factory/blocks`)

- `phase2.py` — for each STRONG gate config, search *block rules*: tail
  percentile cuts on other features that remove losing trades year-stably.
  Exhaustive k-combos with the same FWD+REV verdicts. Multi-strategy mode
  pipelines cell loads against config workers so the pool never starves.
- `run_all.py` — stage-major orchestrator (all gates → all phase2 → all QS);
  a failed stage stops the run rather than feeding garbage downstream.
- `scan_shards.py` — parallel try-unpickle integrity scan of every shard.

## Stage 4 — QS (`factory/qs`)

- `qs_features.py` — quality-score feature selection with the anti-overfit
  redesign documented in its docstring: selection never touches the OOS
  year, sign-stability ≥4/5 years, a FWD+REV top-quantile lift gate, and an
  absolute-positivity insurance clause. Weighting schemes compete under the
  same gates instead of being assumed.
- `qs_core.py` — the canonical QS math (exclude-self rolling rank, IC-weighted
  meta-rank) **shared verbatim by backtest and live**, plus
  `assert_chronological` — the guard born from a coin-major-ordering bug.
- `live_scorer.py` / `qs_daily.py` / `qs_replay.py` — the live side:
  incremental scoring, a vectorized pipeline replay, and a daily self-healing
  history refresh whose within-bar ordering is pinned coin-alphabetical to
  match the backtest stream exactly.
- `qs_to_cellcomp.py` / `bootstrap_qs.py` — bridges: stats tables for the
  portfolio stage (with a **parametric IS/OOS boundary** for blind replays)
  and the cold-start live seed (whitelist + age fail-closed).

## Stage 5 — Portfolio (`factory/portfolio`)

- `build_cell_files.py` — score every validated config per regime cell
  (`TEMPLATE FORMULA` slot) with a loud stale-menu sentinel.
- `trade_combo.py` — the combo engine: per-cell brute force over config
  combos with a real capital/cycle simulator, a **global cell queue** that
  keeps all cores busy across cells, a style concentration cap, and three
  optional menu mechanisms behind flags (jackknife median-rank, diversity
  quota, co-crash penalty) — the A/B arms of Stage 7.
- `budget_merge.py` — DD-diverse retention: interleave rankings under three
  leverage budgets so low/mid/high-DD combos all survive to the next layer.
- `trade_regime_combo.py` / `trade_full_combo.py` — assemble cells into
  regime portfolios into full portfolios, same retention discipline.
- `deploy_sweep.py` — the final sizing sweep (combo × lev × caution × cyc)
  with `TM_FIX_*` fixed-knob modes and a pre-registered lev-stratified tier
  rule.
- `deploy_consensus.py` — the decision layer: every candidate re-simulated
  under 8 data variants, pre-registered robustness gates, graded relaxation,
  and an explicit **ABSTAIN** outcome.

## Stage 6 — (part of Portfolio above: deploy sweep + consensus)

## Stage 7 — Blind walk-forward (`factory/walkforward`)

- `procedure_wf_quarter.sh` — replay the ENTIRE selection procedure blind to
  a cutoff, archive everything *before* judging, then measure the pick on the
  unseen quarter. Repeat over quarters → the procedure's expectation band.
- `pwf_arm_iso.sh` — run a variant ("arm") of the procedure in an isolated
  workspace (`TM_WORK`); the canonical tree is never touched, arms are
  parallel-safe, and the resume-cache poisoning guard is built in.
- `pwf_tournament_seq.sh` — race arms sequentially with full workers (the
  header documents why sequential beats parallel here — measured, not vibes).
- `pwf_judge.py` — the only code allowed to look at the forward window.

## Stage 8 — Parity (`factory/parity`)

- `verify_qs_math.py` — proves the live incremental scorer equals the batch
  scorer row-for-row, including the pass/fail decision.
- `qs_guard_test.py` — the chronological-stream guard's self-test.
- `daily_live_reconcile.py` / `test_qs_reconcile.py` — live trades ⊆ backtest
  trades, and the daily QS-history refresh reproduces the expected stream.
- `replica_vs_backtest.py` — full replica harness: re-run the live decision
  path over a window and diff it against the backtest.
