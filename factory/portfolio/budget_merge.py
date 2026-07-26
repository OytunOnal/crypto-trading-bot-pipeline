r"""Round-robin budget merge for combo selection (trade_combo / regime / full).

Replaces single-score top-K retention with a DD-DIVERSE top-K built by
interleaving three leverage-budget rankings, so low-/mid-/high-DD combos all
survive and propagate cell -> regime -> full. The deploy stage then levers each
candidate to a DD budget (with caution) and picks the real winner.

  Sv(B) = net x min(lev_cap, B/dd) x quality x stability x log2(oos_n)
          \____ leveraged net @ budget B ____/  \______ robustness ______/

- net/oos_n are RAW (deployment reality = recent large universe), NOT normalized.
  Normalization was tested and REJECTED: it collapsed the low-DD diversity
  (0-3% band 8->1 combos) because low-DD combos are recent-concentrated and norm
  penalizes exactly those. Raw preserves the DD spectrum the round-robin exists for.
  The B/dd term already internalizes the risk/return tradeoff per budget; deploy's
  real-sim re-checks recency. Temporal robustness still enforced via the q gate
  (is_wr AND oos_wr > 50) and s (neg weeks across all years).
- quality   = (oos_wr/100) x (is_wr/100)
- stability = 1 / (1 + nw/15 + nw26)            (nw = neg weeks, nw26 = 2026 neg weeks)

round_robin_merge(): tour r appends ranked[B10][r], ranked[B15][r], ranked[B20][r]
(skipping already-seen combos) until `target` unique combos are collected. B=10
surfaces cap-bound low-DD champions; B=20 the high-net/dd-optimal ones; list depth
fills the spectrum in between.
"""
import math

BUDGETS = [10.0, 15.0, 20.0]
LEV_CAP = 10.0



def lev_for(dd, B, cap=LEV_CAP):
    """Leverage to spend DD budget B given intrinsic dd (%), capped at `cap`."""
    return cap if dd <= 0.01 else min(cap, B / dd)


def sv_budget(r, B, cap=LEV_CAP):
    """Leveraged-net-under-budget score for record r at budget B. RAW net.
    Returns -1 if r fails the validity floors.

    EXAMPLE FORMULA — the mechanism (budget-interleaved DD-diverse ranking)
    is the point; plug your own quality/stability terms here."""
    net = r.get('net', 0)
    dd = r.get('max_dd', 0)
    oos_wr = r.get('oos_wr', 0)
    is_wr = r.get('is_wr', 0)
    oos_n = r.get('oos_n', 0)
    nw = r.get('nw', 0)
    nw26 = r.get('nw26', 0)
    if net <= 0 or oos_n < 50 or oos_wr <= 50 or is_wr <= 50:
        return -1.0
    quality = (oos_wr / 100) * (is_wr / 100)
    lev = lev_for(dd, B, cap)
    s = net * lev * quality * (1 / (1 + nw / 15 + nw26)) \
        * math.log2(max(oos_n, 50))
    return s


def round_robin_merge(results, key_fn, target=100, budgets=BUDGETS, cap=LEV_CAP):
    """Interleave per-budget rankings, dedup by key_fn(r), return top `target`
    records (ordered by round-robin position). `results` must contain at least
    each budget's true top-`target` for the merge to be faithful."""
    ranked = {B: sorted(results, key=lambda r: sv_budget(r, B, cap), reverse=True)
              for B in budgets}
    out = []
    seen = set()
    maxlen = max((len(v) for v in ranked.values()), default=0)
    rk = 0
    while len(out) < target and rk < maxlen:
        for B in budgets:
            lst = ranked[B]
            if rk >= len(lst):
                continue
            r = lst[rk]
            if sv_budget(r, B, cap) <= 0:
                continue
            k = key_fn(r)
            if k not in seen:
                seen.add(k)
                out.append(r)
                if len(out) >= target:
                    break
        rk += 1
    return out
