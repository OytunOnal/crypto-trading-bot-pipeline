# Case study: what blind walk-forward testing taught us

> **Coarse filters work but don't clear the bar; fine filters clear the bar
> but don't survive it.** Eight months of measurement, one sentence. The
> data holds ~3–4 independent regime samples: the statistical resolution a
> fine filter needs is below what that sample can support, and the
> resolution a coarse filter can honestly support doesn't clear the
> cost-adjusted profitability bar. Everything below is the long version.

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

## 9. The last campaign: rebuilding the signal bases

After finding 8, one layer had never been seriously questioned: the **signal
bases themselves** — the ~22 trigger designs everything downstream selects
from, most of them years-old defaults with accidentally tiny windows (one
"channel" was 50 minutes long because a timeframe scaling had been forgotten;
a "divergence lookback" was 45 minutes). If any honest edge was left to find,
it should be here. So we ran one final, maximally disciplined campaign:
strategy by strategy, a full mechanism inventory (question every inherited
constant), time-scaled cross-scans, design races between hand-built trigger
variants, and an acceptance bar calibrated not to raw EV but to **what the
incumbent's full validation stack already extracts** — because that, not the
raw signal, is the thing a challenger must beat. (The rebuilt selection
stack these campaigns ran through — unified worst-year gate, feature-family
jury, flow-tiered floor, capital-sim arbiter — is described at design level
in [STAGES.md](STAGES.md), stage 9.)

What we measured, in order of increasing discomfort:

- **Challengers only ever beat weak incumbents.** A one-table triage —
  ranking all 44 direction-units by how much the validation stack extracts
  from each — predicted every verdict in advance (5/5). Where the stack
  already turned bland raw flow into a strong selected slice, no redesigned
  trigger could beat it; where the stack had nothing to work with, redesigns
  won easily.
- **The stack is the edge; bases are raw material.** A side measurement we
  almost didn't run: even the *winning* redesigned bases were **raw-negative**
  in the held-out year before selection. Every point of forward value was
  created by the gate/jury/floor layers. Raw signal EV — the number every
  public backtest advertises — told us nothing about anything.
- **One clean holdout year is not enough.** Five redesigns beat their
  incumbents on a clean, never-scanned year, after passing hidden-coin and
  half-year robustness batteries. Then a second, fully independent year
  judged them: **one of five carried.** The batteries were powerless to
  predict this — in one strategy family a candidate with a near-perfect
  battery (5/5 coin groups, 5/6 half-years) couldn't even sustain a gate
  out-of-sample, while its battery-passing sibling lost 1.25%/trade in the
  clean year — because batteries, however clever, are still built from the
  fitting data.
  Selection-to-the-judge is real, and it survives everything except a second
  judge the search has never met.

## What this all adds up to — and where it ended

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
- **The base-rebuild campaign confirmed the pattern from the other side.**
  Eight strategy passes, run with every anti-overfit discipline we had,
  produced exactly one durable improvement — and its durability claim rests
  on two independent years instead of one, which is now the minimum bar we
  would accept for anything.

So this is where the project landed, and we are saying it plainly rather than
quietly rebranding: **the live bot is stopped and the project is paused.**
Not because the machinery failed — because it worked. It was built to answer
"how do you select trading configurations without fooling yourself?", and
after eight months its most consistent, best-replicated answer was *you were
fooling yourself* — delivered before the illusion reached live capital, at
every layer we pointed it at. A final untouched reserve of data remains
unspent, banked for whichever future attempt earns the right to spend it.

If you take one thing from this repo, take the shape of that answer: build
the half that can tell you *no*. Ours paid for itself entirely in prevented
losses — which is a strange kind of profit, but it is the only kind this
system ever reliably produced.
