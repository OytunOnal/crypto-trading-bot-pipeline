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

## 8. The sharpest finding: the "edge" was a search artifact, and the pipeline proved it

For months the tournament measured *menu mechanisms* (a composition-quota arm
won on tail-DD), while forward 2026 windows returned ~zero EV — explained away
as "the storm regime is eating it." One question broke that story open: **were
the configs simply overfit?** Four measurements, each run with the tooling in
this repo, answered it.

**a. The regime defense collapsed (regime-conditional test).** We split every
day into storm vs. calm (a pre-registered volatility tercile) and re-scored
thousands of frozen configs across 22 strategies:

| | in-sample (2021–25) | out-of-sample (2026) |
|---|---|---|
| calm days | **+0.40** | **+0.04** |
| storm days | **+0.67** | **−0.02** |

The configs carried no edge in 2026 even on *calm* days (retention ~0.10,
far under the pre-registered 0.5 bar). "The storm ate it" was dead — the
problem wasn't the regime.

**b. The entire edge lived in the gate layer, and that layer had died
(decomposition).** Scoring each config with and without its gate/block layer
(`gated = raw + gate_lift`) showed the raw signals were *never* the edge —
even in-sample they sat at ~+0.03–0.09. **~94% of the system's in-sample edge
was lift from gate/block selection** (+0.37–0.58). In 2026 that lift
evaporated ~91% and turned *negative* on storm days: the gates were now
*removing* value from the raw signal.

**c. The cliff wasn't the storm — it was the data boundary (monthly
timeline).** Lift month by month: the final in-sample months read healthy
(+0.24 / +0.46 / +0.30), then **+0.30 in the last training month → −0.04 the
month after**. The collapse landed *before* the February storm — exactly where
the gates' training data ended. The world didn't change on Jan 1; the dataset
boundary did.

**d. The decisive test: shift the boundary, and the cliff follows
(nested-boundary experiment).** To rule out "maybe 2026 really is a different
world," we re-ran the gate/block search **from scratch on ≤2024 data only** and
measured 2025 as a genuinely unseen period. In-sample 2024 lift climbed
+0.27 → **+0.88, peaking in the final month** — the textbook signature of
overfitting. Then: 2025-01 +0.23 → 2025-02 −0.02 → ~zero/negative after. We
pulled the fit boundary back a year, and the cliff moved back with it.

Two independent experiments, two different boundaries, two collapses — each
landing just past the training cutoff. This is not a market shift; it is a
**selection artifact manufactured by the search-and-validation procedure
itself**. The existing FWD/REV walk-forward folds don't protect against it,
because the *selection* still scans those years — validation-in-selection.

## What this all adds up to

The methodology in this repo is sound and the earlier findings hold — but
turned on its own output, the honest verdict is blunt:

- **This config/gate layer has no durable forward edge.** The strength seen
  in-sample is an artifact that decays within ~1–2 months past the training
  boundary.
- **Rolling-refit is not the fix.** Every fresh fit reproduces the same cliff
  at *its* own boundary.
- **The earlier findings remain valid but relative.** Diversity really does
  cut tail-DD, abstain really is the right default — but on top of a
  roughly-zero-edge base, not a real one.
- **The point of the harness was exactly this.** It was built to answer "how
  do you select trading configurations without fooling yourself?" — and its
  most valuable output turned out to be catching that we *had* fooled
  ourselves, before the illusion reached live capital. Sometimes the most
  valuable thing a validation system can tell you is *don't trust this.*
