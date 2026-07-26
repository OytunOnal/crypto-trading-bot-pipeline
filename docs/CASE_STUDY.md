# Case study: what blind walk-forward testing taught us

Findings from running this pipeline's validation harness against a live
production system. Numbers are anonymized/rounded; the *methods* are the
point. Every claim below was measured with the tooling in this repo — none
of it was assumed.

## 1. Blind-window DD understates real DD ~2×

We replayed the full selection procedure blind (chain sees nothing past a
cutoff) over 10 historical quarters and compared the DD the selector *saw*
with the DD the pick *realized* in the unseen next quarter.

- Selection-window DD inflated **1.5–2.2×** into the forward window across
  measurements; in storm quarters the ratio hit **2.3–2.5×**.
- Direction wasn't even stable: in a calm quarter the blind DD *overestimated*
  forward DD (ratio 0.6).

**Consequence:** any absolute gate on blind DD misprices risk. We moved DD
control out of selection entirely: fixed caution ladders + a leverage frontier
judged on *forward* windows. Honest deploy expectations come from the
band of realized forward results, not from the selector's numbers.

## 2. The procedure's own risk-knob choices are anti-signal

Given freedom, the deploy optimizer picked **maximum leverage in exactly the
quarters where it hurt most** (every arm chose max lev going into the storm
quarter), and once picked **no caution ladder at all** on a pick whose blind
DD looked pristine — forward DD then roughly doubled.

A counterfactual sweep (same picks, knobs varied at judge time) showed a
**fixed low-leverage policy retained ~90% of the procedure-chosen-lev return
while cutting worst-quarter DD by ~10pp**. We therefore removed lev/caution/
cyc from the optimizer's hands (`TM_FIX_*`), and — because the DD gate's real
mechanism had been leverage selection — retired the absolute blind-DD gate
with it.

## 3. Goodhart in the wild: a "risk-aware" formula that hid risk

One A/B arm extended the scoring formula with worst-month/worst-week penalty
terms — intended to prefer robust combos. Measured blind:

- Its picks' *selection* DD was the lowest of every arm (7–13%)…
- …while its *forward* DD inflation was the **worst** (up to **3.2×**), and
  the optimizer confidently stacked maximum leverage on top of the
  "safe-looking" picks.

The penalty terms didn't make combos safer — they made risk *invisible to the
selector*. The arm went to the graveyard, and the lesson became policy: score
changes are validated on **forward DD inflation**, never on how good they make
the blind numbers look.

## 4. Menu diversity genuinely cuts tail DD

Three independent mechanisms — year-jackknife median-rank ordering,
style×direction×DD×SLTP quota selection, and a measured co-crash penalty —
were each run as isolated arms over the same elimination quarters:

- All three cut storm-quarter forward DD by a similar **~6–7pp** vs the
  baseline menu, at no net cost.
- The quota arm repeated its improvement under a *second* regime (fixed
  risk knobs), cutting both raw forward DD and the DD-inflation ratio.

Three different roads reaching the same place is the kind of convergence a
single backtest can never give you. Diversity earned its place as a default.

## 5. Abstaining is an answer

The consensus layer's pre-registered gates include an explicit **ABSTAIN**
outcome: if no candidate passes at any relaxation grade, the procedure
proposes *no package* and the live meaning is "stay on the incumbent". Two
arms abstained in quarters where their candidate pools couldn't support a
robust pick — which is exactly what you want a selector to do instead of
shipping its least-bad option.

## 6. Infrastructure bugs are result bugs

Two incidents that shaped the harness's defensive design:

- **Stale-resume poisoning.** An arm workspace copied the production pickle
  tree *including its resume cache*; the builder silently skipped ~60% of its
  tasks and the arm's menu shrank ~100×. Every conclusion drawn from those
  runs was invalid. Fixes: clear resume caches in copies, add loud sentinels
  ("0 rows = FATAL", minimum-runtime checks), and treat *suspiciously fast
  runs* as a diagnostic signal.
- **Exchange kline revisions.** The REST API served deficient bars minutes
  after a market event and revised them later; incremental download made the
  bad bars permanent across most of the universe. Fix: always re-fetch a
  trailing overlap window and content-diff it, deepening until an error-free
  region is reached (`factory/data/download_klines.py`).

## 7. Determinism is a parity feature

Live and backtest only reconcile if *every* ordering decision is pinned:
within-bar candidates sort coin-alphabetically on both sides, QS streams are
asserted chronological (`assert_chronological`), and set-iteration
nondeterminism (hash randomization) is sorted away before any capital cap can
bind. With that discipline the incremental live scorer reproduces the batch
scorer **row-for-row** (`factory/parity/verify_qs_math.py`) and live trades
are a strict subset of backtest trades.
