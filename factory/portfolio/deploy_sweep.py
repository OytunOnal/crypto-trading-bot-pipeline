"""Deploy-sizing sweep (pipeline step after trade_full_combo).

Takes the top-N full combos from trade_full_combo.txt and runs the leak-fixed
DEPLOY sim (real leverage + caution-grid + cycle-max) over the cross product
combo x lev x caution x cyc, ranked by the RAW score (sqrt(net)/(1+dd/3) x q x s x
log2(oos_n)) on the actual levered outcomes -- consistent with the raw round-robin
selection (combo/regime/full). No hard DD cap: the (1+dd/3) penalty handles risk
softly; real leverage is brute-swept (LEVS) so the score picks the risk/return
point. Normalization was rejected (collapses low-DD diversity). Picks final config.

  python factory/portfolio/deploy_sweep.py            # defaults
  python factory/portfolio/deploy_sweep.py 50 40      # TOPN=50, SHOW=40
"""
import sys, re, math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factory.portfolio import trade_full_combo as F
from factory.qs.age_filter import norm_factor   # per-month-universe normalization

# Rule B (close-on-opposite + skip-both-fire) is the production default and the ONLY
# mode now: reads trade_full_combo.txt, writes deploy_sweep.txt (canonical un-suffixed),
# builds a cache price map for the early exit. The old A / _B env-gated split is gone.
import os as _os
_RULE = 'B'   # Rule B (close-on-opposite + skip-both-fire) is the production default
_SFX = ''     # write to / read from the canonical (un-suffixed) result files
_CACHE_DIR = ROOT / 'data' / 'cache'
# Blind walk-forward: parametric OOS boundary (default = production, bit-identical)
_OOS_MS = _os.environ.get('TM_OOS_START', '2026')
_OOS_WS = _os.environ.get('TM_OOS_WSTART', '2026')
_W_PRICE = {}   # worker global: {(coin, et): close}


def _build_price_map(cells):
    """{(coin, entry_time): close} for all trade entries (rule B early exit)."""
    import pandas as pd
    coin_times = defaultdict(set)
    for trd in cells.values():
        for cfg_trades in trd.values():
            for t in cfg_trades:
                coin_times[t['coin']].add(t['entry_time'])
    pm = {}
    for coin, qset in coin_times.items():
        parts = []
        for f in sorted(_CACHE_DIR.glob(f'{coin}_5m_*.parquet')):
            d = pd.read_parquet(f, columns=['close'])
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            parts.append(d['close'])
        if not parts:
            continue
        s = pd.concat(parts); s = s[~s.index.duplicated(keep='last')].sort_index()
        qt = pd.DatetimeIndex(sorted(qset))
        vals = s.reindex(qt, method='ffill')
        for ts, v in zip(qt, vals.values):
            if v == v:
                pm[(coin, ts)] = float(v)
    print(f'  [price map] {len(pm):,} (coin,bar) priced (rule B)', flush=True)
    return pm


MAXC, FEE = 100, 0.0008
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SAVE_N = 100   # rows written to results/deploy_sweep.txt
_WORK = Path(_os.environ.get('TM_WORK', ROOT))  # isolated workdir (parallel arms)
OUT_PATH = _WORK / 'results' / f'deploy_sweep{_SFX}.txt'
CYCS = [15, 20, 25]
LEVS = [5, 6, 7, 8, 9, 10]
# (name, [(dd_thresh, pos_mult), ...] DESC). base pos_pct=0.01; mult reduces it.
CAUTION_GRID = [
    ('none',     []),
    ('8/15/22',  [(22, 0.0025), (15, 0.005), (8, 0.0075)]),   # current
    ('8/15/25',  [(25, 0.0025), (15, 0.005), (8, 0.0075)]),
    ('5/12/20',  [(20, 0.0025), (12, 0.005), (5, 0.0075)]),   # earlier trigger
    ('5/10/20',  [(20, 0.0025), (10, 0.005), (5, 0.0075)]),
    ('8/12/20',  [(20, 0.0025), (12, 0.005), (8, 0.0075)]),
    ('8/15/22D', [(22, 0.001), (15, 0.0025), (8, 0.005)]),    # deeper cuts
    ('8/15',     [(15, 0.005), (8, 0.0075)]),                 # two-level
]
# TM_FIX_CAUT="8/15/22D" -> collapse the caution grid to one FIXED ladder
# (lesson: left free, the optimizer strips the insurance — picks 'none' on
# blind data whose DD then doubles forward). No env -> full grid, bit-identical.
_FIX_CAUT = _os.environ.get('TM_FIX_CAUT')
if _FIX_CAUT:
    CAUTION_GRID = [cg for cg in CAUTION_GRID if cg[0] == _FIX_CAUT]
    assert CAUTION_GRID, f'TM_FIX_CAUT={_FIX_CAUT} not in grid'
# TM_FIX_CYC: cyc is an insurance/capacity knob too — a blind window that
# lacks clustered-day risk makes high cyc look free.
# TM_FIX_LEV: the procedure's own leverage choice measured ANTI-SIGNAL in
# blind replays (fixed low lev dominated). Selection runs at a FIXED lev; the
# winning pick is judged across a lev frontier; final lev = a human risk call.
_FIX_LEV = _os.environ.get('TM_FIX_LEV')
if _FIX_LEV:
    LEVS = [int(_FIX_LEV)]

# TM_FIX_CYC=25 -> collapse the cyc grid to one fixed policy value.
_FIX_CYC = _os.environ.get('TM_FIX_CYC')
if _FIX_CYC:
    CYCS = [int(_FIX_CYC)]

# parse top-N full combos (regime ranks) from trade_full_combo.txt
_full = (_WORK / 'results' / f'trade_full_combo{_SFX}.txt').read_text(encoding='utf-8')
combos = []
for _ln in _full.splitlines():
    _m = re.match(r'#\d+ .*BEAR#(\d+) \+ FLAT#(\d+) \+ BULL#(\d+)', _ln)
    if _m:
        combos.append((int(_m.group(1)), int(_m.group(2)), int(_m.group(3))))
    if len(combos) >= TOPN:
        break


def raw_score(r):
    """RAW score on levered sim outcomes (no normalization). Same shape as the
    pipeline's combo_score_fn but on raw net/oos_n. -999 if floors fail.
    TEMPLATE FORMULA — plug your own; constants are EXAMPLES."""
    net = r['net']
    if net <= 0 or r['net26'] <= 0:
        return -999.0
    dd = r['max_dd']; oos_wr = r['oos_wr']; is_wr = r['is_wr']; oos_n = r['oos_n']
    nw = r['nw']; nw26 = r['nw26']
    if oos_n < 50 or oos_wr <= 50 or is_wr <= 50:
        return -999.0
    q = (oos_wr / 100) * (is_wr / 100)
    return (math.sqrt(net) / (1 + dd / 3)) * q \
        * (1 / (1 + nw / 15 + nw26)) * math.log2(max(oos_n, 50))


def combo_cellrefs(b, fl, bu):
    refs = []
    for regime, rank in [('BEAR', b), ('FLAT', fl), ('BULL', bu)]:
        combo = [c for c in F.parse_regime_combo_file(regime, 30) if c['rank'] == rank][0]
        refs += combo['cell_refs']
    return refs


def build_tg(b, fl, bu, cells):
    tg = defaultdict(list)
    for cr in combo_cellrefs(b, fl, bu):
        cn, rk = cr.rsplit('#', 1)
        trd = cells[cn]
        for cfg in F.parse_cell_combo_file(cn, int(rk)):
            for t in trd.get(cfg, []):
                et = t['entry_time']; iso = et.isocalendar()
                tg[et].append((t['coin'], t['pnl_pct'], t['exit_time'],
                               et.strftime('%Y-%m'), '%d-W%02d' % (iso[0], iso[1]),
                               t.get('direction', '')))   # [5] direction (rule B)
    return tg


def sim(tg, max_new, caution, lev, detail=False):
    # detail=True: adds the mo/wk breakdown + end-of-day equity series to the
    # result dict (behavior identical; for analysis scripts).
    times = sorted(tg.keys())
    cash, wd, opos, peak, maxdd, liq = 10000.0, 0.0, {}, 10000.0, 0.0, 0
    eqd = {}
    mo, wk = defaultdict(float), defaultdict(float)
    mo_tr = defaultdict(int)  # trades per month (per-month-universe norm)
    ow = ot = iw = it = 0  # oos(2026) + is(pre-2026) wins/trades
    rule_b = (_RULE == 'B')
    for et in times:
        for c in [c for c, p in opos.items() if p[0] <= et]:
            p = opos.pop(c); m = p[1]; notl = p[2]; pnl = p[3]  # p[0]=xt; 4/6-tuple
            fin = m + notl * (pnl / 100) - notl * FEE
            if fin < 0: liq += 1; fin = 0
            cash += fin
        lk = sum(p[1] for p in opos.values()); eq = cash + lk
        if eq > 10000 and cash > 0:
            w = min(eq - 10000, cash); cash -= w; wd += w; eq = cash + lk
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > maxdd: maxdd = dd
        if detail: eqd[et.date()] = (eq, len(opos))
        pp = 0.01
        for th, p in caution:
            if dd >= th: pp = p; break
        nc = 0
        cands = tg[et] if rule_b else [t for t in tg[et] if t[0] not in opos]
        both = set()
        if rule_b:
            _ds = defaultdict(set)
            for t in cands:
                _ds[t[0]].add(t[5])
            both = {c for c, s in _ds.items() if len(s) >= 2}
        for tr in cands:
            c0 = tr[0]
            if c0 in opos:
                if not rule_b:
                    continue
                pos = opos[c0]                 # (xt, m, notl, pnl, dir, ep)
                if pos[4] == tr[5]:
                    continue                   # same dir -> skip
                ep = pos[5]; pxe = _W_PRICE.get((c0, et))
                if ep is None or pxe is None or ep <= 0:
                    continue
                early = (pxe / ep - 1) * 100 if pos[4] == 'LONG' else (1 - pxe / ep) * 100
                p = opos.pop(c0)
                early_real = p[2] * (early / 100) - p[2] * FEE   # p[2]=notl
                full_booked = p[2] * (p[3] / 100) - p[2] * FEE   # what open booked
                cash += max(p[1] + early_real, 0)
                # correct the ENTRY-bar booking: replace assumed-full with early-realized
                mo[p[6]] += early_real - full_booked   # p[6]=entry month
                wk[p[7]] += early_real - full_booked   # p[7]=entry week
                # fix the win flag if the realized sign flipped vs the assumed-full
                if (p[3] > 0) != (early_real > 0):
                    if p[6] >= _OOS_MS: ow += 1 if early_real > 0 else -1
                    else:              iw += 1 if early_real > 0 else -1
                continue
            if rule_b and c0 in both:
                continue                       # both-fire, no position -> skip
            if len(opos) >= MAXC: continue
            if nc >= max_new: continue
            m = eq * pp
            if m < 2.5: continue
            if m > cash: m = cash
            if m < 2.5: continue
            notl = m * lev; cash -= m
            if rule_b:
                opos[c0] = (tr[2], m, notl, tr[1], tr[5], _W_PRICE.get((c0, et)),
                            tr[3], tr[4])   # +entry month/week for early-close correction
            else:
                opos[c0] = (tr[2], m, notl, tr[1])
            mo[tr[3]] += notl * (tr[1] / 100) - notl * FEE
            wk[tr[4]] += notl * (tr[1] / 100) - notl * FEE
            mo_tr[tr[3]] += 1
            if tr[3] >= _OOS_MS: ot += 1; ow += 1 if tr[1] > 0 else 0
            else: it += 1; iw += 1 if tr[1] > 0 else 0
            nc += 1
    for c, p in opos.items():
        cash += max(p[1] + p[2] * (p[3] / 100) - p[2] * FEE, 0)
    net = wd + min(cash, 10000) - 10000
    net26 = sum(v for k, v in mo.items() if k >= _OOS_MS)
    nw = sum(1 for v in wk.values() if v < 0)
    nw26 = sum(1 for k, v in wk.items() if k >= _OOS_WS and v < 0)
    nm = sum(1 for v in mo.values() if v < 0)
    # per-month-universe normalized magnitudes (x350/univ) via shared norm_factor
    net_norm = sum(v * norm_factor(k) for k, v in mo.items())
    net26_norm = sum(v * norm_factor(k) for k, v in mo.items() if k >= _OOS_MS)
    oos_n_norm = sum(c * norm_factor(k) for k, c in mo_tr.items() if k >= _OOS_MS)
    r = dict(net=net, max_dd=maxdd, net26=net26, nw=nw, nw26=nw26, nm=nm, liq=liq,
             oos_n=ot, oos_wr=(ow / ot * 100 if ot else 0),
             is_wr=(iw / it * 100 if it else 0),
             net_norm=net_norm, net26_norm=net26_norm, oos_n_norm=oos_n_norm)
    if detail:
        r['mo'] = dict(mo); r['wk'] = dict(wk); r['eq_daily'] = eqd
        r['n_tr'] = it + ot
    return r


_W_CELLS = None
def _init(cells, price=None):
    global _W_CELLS, _W_PRICE
    _W_CELLS = cells
    _W_PRICE = price or {}


def process_combo(arg):
    i, b, fl, bu = arg
    tg = build_tg(b, fl, bu, _W_CELLS)
    label = 'B%d+F%d+BU%d' % (b, fl, bu)
    out = []
    for lev in LEVS:
        for cname, clev in CAUTION_GRID:
            for cyc in CYCS:
                r = sim(tg, cyc, clev, lev)
                # RAW score on levered outcomes (consistent with raw round-robin selection)
                r['score'] = raw_score(r)
                r['label'] = label; r['orig'] = i; r['cyc'] = cyc
                r['caut'] = cname; r['lev'] = lev
                out.append(r)
    return out


def main():
    import time
    t0 = time.time()
    needed = sorted({cr.rsplit('#', 1)[0] for (b, fl, bu) in combos
                     for cr in combo_cellrefs(b, fl, bu)})
    cells = {cn: F.load_cell_trades(cn) for cn in needed}
    print('loaded %d cells in %.0fs' % (len(cells), time.time() - t0), flush=True)
    price_map = _build_price_map(cells) if _RULE == 'B' else None
    nw = F.auto_workers(cells, 'cells')
    args = [(i, b, fl, bu) for i, (b, fl, bu) in enumerate(combos, 1)]
    with ProcessPoolExecutor(max_workers=nw, initializer=_init, initargs=(cells, price_map)) as ex:
        results = list(ex.map(process_combo, args))
    rows = [r for sub in results for r in sub]
    print('swept %d configs in %.0fs (%d workers)' % (len(rows), time.time() - t0, nw))

    # ranked by the RAW score (pipeline-canonical)
    norm_sorted = sorted(rows, key=lambda r: r['score'], reverse=True)
    for rk, r in enumerate(norm_sorted, 1):
        r['rk'] = rk

    hdr = ('%-3s %-14s %3s %-8s %3s %7s %9s %4s %9s %3s %3s %4s %5s %5s %3s' % (
        'rk', 'combo', 'lev', 'caut', 'cyc', 'Score', 'Net$', 'DD%', 'OOS26$',
        'NM', 'NW', 'NW26', 'isWR', 'oWR', 'Liq'))

    def prow(r):
        print('%-3d %-14s %3d %-8s %3d %7.1f %9s %4.1f %9s %3d %3d %4d %5.1f %5.1f %3d' % (
            r['rk'], r['label'], r['lev'], r['caut'], r['cyc'], r['score'],
            '{:,.0f}'.format(r['net']), r['max_dd'], '{:,.0f}'.format(r['net26']),
            r['nm'], r['nw'], r['nw26'], r['is_wr'], r['oos_wr'], r['liq']))

    print('%d combo x %d caution x cyc%s x lev%s = %d configs | ranked by RAW score'
          % (TOPN, len(CAUTION_GRID), CYCS, LEVS, len(rows)))
    print(hdr); print('-' * 100)
    for r in norm_sorted[:SHOW]:
        prow(r)

    print('\n=== FRONTIER: best config per leverage ===')
    print(hdr); print('-' * 100)
    for lev in LEVS:
        best = max((r for r in rows if r['lev'] == lev), key=lambda r: r['score'])
        prow(best)

    # LEV-STRATIFIED view: the score's DD penalty systematically favors low
    # lev, hiding the leverage decision. Stratified view = each lev's own
    # top-20 (round-robin) + a PRE-REGISTERED tier rule: the highest lev whose
    # tier leader satisfies Liq=0 AND NW26<=cap AND DD<=cap = LEV-RULE PICK.
    DD_CAP = 20.0
    NW26_MAX = 1
    per_lev = {lev: sorted((r for r in rows if r['lev'] == lev),
                           key=lambda r: r['score'], reverse=True)[:20]
               for lev in LEVS}
    strat_rows = []
    for i in range(20):
        for lev in LEVS:
            if i < len(per_lev[lev]):
                strat_rows.append(per_lev[lev][i])
    print('\n=== LEV-STRATIFIED TOP (each lev its own top-20, round-robin) ===')
    print(hdr); print('-' * 100)
    for r in strat_rows[:SHOW]:
        prow(r)

    rule_pick = None
    for lev in sorted(LEVS, reverse=True):
        cand = per_lev[lev][0] if per_lev[lev] else None
        if cand and cand['liq'] == 0 and cand['nw26'] <= NW26_MAX and cand['max_dd'] <= DD_CAP:
            rule_pick = cand
            break
    b = norm_sorted[0]
    pick = ('DEPLOY PICK (#1): {label} lev{lev} {caut} cyc{cyc} -> '
            'Score {score:.1f} Net ${net:,.0f} DD {max_dd:.1f}% OOS ${net26:,.0f} '
            'NM {nm} NW26 {nw26} Liq {liq}'.format(**b))
    print('\n' + pick)
    if rule_pick is not None:
        rp = (('LEV-KURAL PICK (Liq=0 & NW26<=%d & DD<=%.0f%% en-yuksek-lev): '
               % (NW26_MAX, DD_CAP))
              + '{label} lev{lev} {caut} cyc{cyc} -> Score {score:.1f} '
                'Net ${net:,.0f} DD {max_dd:.1f}% OOS ${net26:,.0f}'.format(**rule_pick))
    else:
        rp = 'LEV-RULE PICK: no tier satisfied the rule (DD_CAP=%.0f%%)' % DD_CAP
    print(rp)

    # write top-SAVE_N ranked table to results/deploy_sweep.txt
    def frow(r):
        return ('%-3d %-14s %3d %-8s %3d %7.1f %9s %4.1f %9s %3d %3d %4d %5.1f %5.1f %3d'
                % (r['rk'], r['label'], r['lev'], r['caut'], r['cyc'], r['score'],
                   '{:,.0f}'.format(r['net']), r['max_dd'], '{:,.0f}'.format(r['net26']),
                   r['nm'], r['nw'], r['nw26'], r['is_wr'], r['oos_wr'], r['liq']))
    fhdr = ('%-3s %-14s %3s %-8s %3s %7s %9s %4s %9s %3s %3s %4s %5s %5s %3s' % (
        'rk', 'combo', 'lev', 'caut', 'cyc', 'Score', 'Net$', 'DD%',
        'OOS26$', 'NM', 'NW', 'NW26', 'isWR', 'oWR', 'Liq'))
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('DEPLOY SWEEP | %d combo x %d caution x cyc%s x lev%s = %d configs '
                '| ranked by RAW score\n' % (
                    TOPN, len(CAUTION_GRID), CYCS, LEVS, len(rows)))
        f.write(pick + '\n')
        f.write(rp + '\n\n')
        f.write('=== LEV-STRATIFIED TOP (each lev its own top-20, round-robin) ===\n')
        f.write(fhdr + '\n' + '-' * 105 + '\n')
        for r in strat_rows:
            f.write(frow(r) + '\n')
        f.write('\n=== RAW score ranking ===\n')
        f.write(fhdr + '\n' + '-' * 105 + '\n')
        for r in norm_sorted[:SAVE_N]:
            f.write(frow(r) + '\n')
        f.write('\n=== FRONTIER: best config per leverage ===\n')
        f.write(fhdr + '\n' + '-' * 105 + '\n')
        for lev in LEVS:
            f.write(frow(max((r for r in rows if r['lev'] == lev),
                             key=lambda r: r['score'])) + '\n')
    print('wrote top-%d -> %s' % (SAVE_N, OUT_PATH))


if __name__ == '__main__':
    main()
