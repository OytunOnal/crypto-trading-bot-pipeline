# crypto-trading-bot-pipeline

**A production-grade strategy factory for crypto trading — shared as a reference
architecture.** This is the full engineering pipeline behind a live multi-strategy
Binance futures bot: data ingestion → feature/gate search → validation → portfolio
assembly → blind walk-forward testing → live/backtest parity. The proprietary parts
(strategy signals, feature battery, scoring constants, deployed configs) are replaced
with clearly-marked toy examples; **everything else is the real machinery.**

> **Bring your own alpha.** The pipeline never needs to know what your strategies
> are. Register a strategy, extend the feature battery, plug your scoring formula
> into the marked `TEMPLATE FORMULA` slots — the factory does the rest.

---

## Why this exists

Most public trading-bot repos show a strategy. Almost none show the part that
actually decides whether a system survives: **the validation infrastructure**.
This repo is that part — the product of ~8 months and 300+ commits of iterating
on one question: *how do you select trading configurations without fooling
yourself?*

The honest answers we converged on (measured, not assumed — see
[docs/CASE_STUDY.md](docs/CASE_STUDY.md)):

- **Blind-window drawdown understates real drawdown ~2×.** Selection metrics
  must never be trusted as forward estimates; measure forward.
- **Optimizers strip insurance.** Left free, the deploy optimizer picks
  maximum leverage and no caution ladder on blind data — then forward DD
  doubles. Risk knobs must be **fixed policy**, outside the optimizer.
- **Score-formula "improvements" can hide risk instead of reducing it**
  (a textbook Goodhart failure we caught in an A/B arm).
- **Menu diversity cuts tail drawdown** — confirmed by three independent
  mechanisms (jackknife ranking, composition quotas, co-crash penalties)
  converging on the same storm-quarter improvement in two different regimes.
- **The apparent edge can be an artifact of the search itself.** Two
  boundary-shift experiments proved a config layer's edge died *exactly* at
  the training-data cutoff — a selection artifact, not a market regime. The
  harness was built to catch precisely this before it reaches live capital,
  and it did.
- **The selection stack is the edge; signals are raw material.** Even
  *winning* redesigned signal bases were raw-negative on held-out data —
  every point of forward value was created by the gate/jury/floor layers.
  Raw signal EV, the number every public backtest advertises, predicted
  nothing.
- **One clean holdout year is not enough.** Five base redesigns passed a
  never-scanned holdout year *plus* hidden-coin and half-year robustness
  batteries; a second independent year kept one of five. Batteries are
  built from fitting data and cannot suppress selection-to-the-judge —
  two independent judge years is the minimum honest bar.
- **Challengers only beat weak incumbents.** Ranking all units by how much
  the validation stack already extracts predicted every redesign verdict in
  advance (5/5): rebuild effort only pays where the stack has nothing to
  work with.

All of it compresses to one sentence — the project's final takeaway:

> **Coarse filters work but don't clear the bar; fine filters clear the bar
> but don't survive it.**

## Pipeline at a glance

```
       ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ 1 DATA     │──▶│ 2 GATES    │──▶│ 3 BLOCKS   │──▶│ 4 QS       │
       │ klines,    │   │ feature    │   │ phase-2    │   │ anti-      │
       │ verified   │   │ battery +  │   │ block-rule │   │ overfit    │
       │ parquet    │   │ AND-gate   │   │ search     │   │ quality    │
       │ caches     │   │ SL/TP sweep│   │            │   │ ranking    │
       └────────────┘   └────────────┘   └────────────┘   └────────────┘
                                                                │
       ┌────────────┐   ┌────────────┐   ┌────────────┐        ▼
       │ 8 PARITY   │◀──│ 7 WALK-    │◀──│ 6 DEPLOY   │   ┌────────────┐
       │ live ==    │   │ FORWARD    │   │ sweep +    │◀──│ 5 PORTFOLIO│
       │ backtest,  │   │ blind      │   │ 8-variant  │   │ cell→regime│
       │ row-for-row│   │ replay +   │   │ consensus, │   │ →full combo│
       │            │   │ A/B arms   │   │ abstain    │   │ assembly   │
       └────────────┘   └────────────┘   └────────────┘   └────────────┘
```

| stage | package | what it does | highlights |
|---|---|---|---|
| 1 Data | `factory/data` | verified kline download, year-split + monthly cell-sharded parquet/pickle caches | append-mode with **content-diff re-fetch** (exchange revisions), freshness/gap verify loop, RAM-lean shard streaming |
| 2 Gates | `factory/gates` | feature battery (N × 3 timeframes + BTC-relative) + AND-gate / SL/TP sweep | IC-stable independent selection, threshold optimization, exhaustive combos with failed-subset pruning, **FWD+REV walk-forward verdicts**, event-driven measured-RAM admission |
| 3 Blocks | `factory/blocks` | phase-2 "block rule" search (cut the worst tail without touching the edge) | year-stable tail-percentile candidates, pipelined multi-strategy global queue |
| 4 QS | `factory/qs` | anti-overfit quality-score feature selection + the **canonical QS math** shared with live | no-OOS-in-selection, sign-stability ≥4/5, FWD+REV lift gates, absolute-positivity insurance; chronological-stream guard |
| 5 Portfolio | `factory/portfolio` | cell → regime → full combo assembly | **budget-merge** (DD-diverse round-robin retention), style concentration cap, diversity-quota / jackknife / co-crash A/B mechanisms |
| 6 Deploy | `factory/portfolio` | lev × caution × cyc sweep + deploy-level consensus | 8-variant vote (year-jackknife + truncation), pre-registered gates, graded relaxation, **abstain semantics** ("no package" is a first-class answer) |
| 7 Walk-forward | `factory/walkforward` | blind quarterly replay + isolated A/B arm tournament | cutoff truncation, uniform trailing-OOS boundary, workspace isolation (`TM_WORK`), forward judge, stale-resume sentinels |
| 8 Parity | `factory/parity` | live ⊆ backtest, proven | incremental live scorer == batch scorer **row-for-row**, daily self-healing reconcile, within-bar deterministic ordering |

## What is real vs. template

**Real (production machinery, lightly sanitized):** every engine, runner,
parallelism scheme, cache format, gate/consensus rule, isolation mechanism,
parity harness, and most inline comments — including the ones documenting
bugs we hit and the sentinels we added after them.

**Template (replace with yours):** the 3 toy strategies (`strategies/`), the
6-feature example battery (`factory/gates/features_common.py`), every function
marked `TEMPLATE FORMULA`, the example config table
(`factory/qs/live_config.py`), and all numeric thresholds collected in
[config/example.yaml](config/example.yaml).

**Deliberately absent:** live execution/exchange-account layer, real strategy
signals and parameters, the production feature battery, deployed
configurations, and any real performance artifacts.

## Status & honest scope

This is a **reference implementation, not a turnkey product**. All modules are
import-clean (`python -m compileall` passes; the QS/parity packages import and
the stream-guard self-test passes), but the repo ships no market data and no
end-to-end demo run. Read it as you would an architecture review: start at
[docs/STAGES.md](docs/STAGES.md), then dive into the stage that interests you
— the code is heavily commented with the *why*, not just the *what*.

**Project status (2026-08): paused, live bot stopped — deliberately.** After
the search-artifact finding and a final signal-base rebuild campaign (case
study, chapters 8–9), the honest tally was: one durable improvement, many
well-earned rejections, and no layer left where in-sample strength survived
an independent judge. We stopped pushing rather than lowering the bar. The
machinery is exactly as valuable as before — arguably more, since its main
product turned out to be *verified no's*.

## A short history

Eight months, 300+ commits, one live system:

- **Month 1–2** — multi-strategy bot skeleton, first Docker deploy, live QS
  history plumbing.
- **Month 3** — the 9-cell regime framework (BTC trend × momentum) and
  per-strategy research at scale; the shared gate-sweep engine replaces 20+
  copy-pasted scripts.
- **Month 4** — asymmetric SL/TP sweeps, phase-2 block search, execution
  hardening.
- **Month 5 (peak)** — monthly cell-sharded cache (multi-GB monoliths → 10-150MB
  streams), the parity push (look-ahead fixes, windowed indicators, deep
  reconcile), full rebuild, deploy, live.
- **Month 6–7** — 8-variant consensus, blind walk-forward harness, A/B arm
  tournament — and most of the findings in the case study.
- **Month 8** — the boundary-shift experiments (the edge was a search
  artifact), then a final strategy-by-strategy signal-base rebuild: mechanism
  inventories, time-scaled cross-scans, design races, Pareto-frontier
  candidate selection — judged by a clean year, then re-judged by a second
  independent year. One of five survivors carried; the bot was stopped and
  the project paused with its ledger clean and a reserve of unseen data
  intact.

## License

MIT — see [LICENSE](LICENSE). No warranty; nothing here is financial advice.
Trading crypto futures can lose money faster than any backtest suggests.
