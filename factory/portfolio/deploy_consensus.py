"""Deploy-level consensus: vote on the FULL DEPLOYABLE CONFIG.

Consensus runs over (triple + lev + caution + cyc) — not just the combo
triple — which closes the divergence between a leverage tier-rule pick and a
combo-only consensus: one vote, one winner. Candidate pool = the blind
sweep's stratified top-20 x levs + raw top-100 rows (deduped). Every
candidate is re-simulated under 8 data variants (year-jackknife x5,
drop-two-years, trunc-last-10wk, full) with the production deploy sim
(close-on-opposite + caution + lev) and ranked in-variant by the raw score.

PRE-REGISTERED RULE (written before the run; rev1's "global best median"
rule was low-lev-biased — the raw top-100 was 100/100 minimum-lev):
  ROBUSTNESS GATE: in ALL 8 variants Liq=0 AND NW26<=cap AND DD<=cap.
  STRATUM LEADER: each lev's candidates are ranked AMONG THEMSELVES per
  variant; leader = best gate-passing median.
  WINNER (lev-rule x consensus): the leader of the HIGHEST lev whose leader
  passes the gate. The global best-median is also reported (to keep the
  bias visible).

  python factory/portfolio/deploy_consensus.py
Output: results/deploy_consensus.txt (+ _all.tsv raw dump)
"""
import sys, os, re, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.portfolio import deploy_sweep as ds   # sim, raw_score, CAUTION_GRID, price map
from factory.portfolio.trade_full_combo import load_cell_trades
from factory.portfolio import trade_full_combo as F

# The blind walk-forward wrapper redirects these via env; defaults = live tree.
V2 = ROOT / os.environ.get('DC_RESULTS_DIR', 'results')
_OUTDIR = Path(os.environ.get('TM_WORK', ROOT)) / 'results'  # isolated output (parallel)
# SAME DATA AS THE SWEEP = the TRUNCATED (blind) stream. Lesson: consensus on
# the full stream once leaked the judging window into selection — invalidated.
F.PICKLE_DIR = ROOT / os.environ.get('DC_TRUNC_DIR', 'trade_pickles')
CAUT = dict(ds.CAUTION_GRID)
# trunc10wk variant boundary (EXAMPLE — set to your blind stream's last ~10wk)
TRUNC_TS = pd.Timestamp(os.environ.get('DC_TRUNC_TS', '2026-02-23'))
VARIANTS = [('full', None), ('drop2021', {2021}), ('drop2022', {2022}),
            ('drop2023', {2023}), ('drop2024', {2024}), ('drop2025', {2025}),
            ('drop21+22', {2021, 2022}), ('trunc10wk', 'TRUNC')]
DD_CAP, NW26_MAX = 20.0, 1
# TM_NO_DD_GATE=1: in the FIXED-knob regime the absolute DD gate is removed.
# Its mechanism was leverage selection ("highest lev passing the gate"); with
# lev fixed, DD>cap became a dead-end abstain that killed variant-ROBUST
# candidates on a blind number known to misprice forward DD. DD is not
# unpoliced: the score's DD penalty + the fixed caution ladder + the forward
# lev-frontier + the wDD report remain. Liq=0 & graded NW26 gates STAY.
if os.environ.get('TM_NO_DD_GATE') == '1':
    DD_CAP = float('inf')


def parse_regime(regime, rank):
    for line in open(V2 / f'trade_regime_{regime}.txt', encoding='utf-8'):
        if line.startswith(f'#{rank} '):
            return [c.strip() for c in line.split('|')[2].strip().split('+')]
    return []


def parse_cell(cell, rank):
    for line in open(V2 / f'trade_combo_{cell}.txt', encoding='utf-8'):
        if line.startswith(f'#{rank} '):
            return [c.strip() for c in line.split('|')[2].strip().split(',')]
    return []


def build_tg(label, cells):
    b, fl, bu = (int(x) for x in re.match(r'B(\d+)\+F(\d+)\+BU(\d+)', label).groups())
    tg = defaultdict(list)
    for regime, rank in (('BEAR', b), ('FLAT', fl), ('BULL', bu)):
        for cr in parse_regime(regime, rank):
            cn, rk = cr.rsplit('#', 1)
            trd = cells.get(cn, {})
            for cfg in parse_cell(cn, int(rk)):
                for t in trd.get(cfg, []):
                    et = t['entry_time']; iso = et.isocalendar()
                    tg[et].append((t['coin'], t['pnl_pct'], t['exit_time'],
                                   et.strftime('%Y-%m'),
                                   '%d-W%02d' % (iso[0], iso[1]),
                                   t.get('direction', '')))
    return tg


def main():
    t0 = time.time()
    # candidate pool: every row of the stratified + raw tables (deduped)
    rxrow = re.compile(r'^(\d+)\s+(B\d+\+F\d+\+BU\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s')
    cands = []
    seen = set()
    for line in open(V2 / 'deploy_sweep.txt', encoding='utf-8'):
        m = rxrow.match(line.strip())
        if not m:
            continue
        key = (m.group(2), int(m.group(3)), m.group(4), int(m.group(5)))
        if key in seen or key[2] not in CAUT:
            continue
        seen.add(key)
        cands.append(key)
    labels = sorted({c[0] for c in cands})
    print(f'  {len(cands)} candidate configs, {len(labels)} triples', flush=True)

    cells_needed = set()
    for lb in labels:
        b, fl, bu = (int(x) for x in re.match(r'B(\d+)\+F(\d+)\+BU(\d+)', lb).groups())
        for regime, rank in (('BEAR', b), ('FLAT', fl), ('BULL', bu)):
            for cr in parse_regime(regime, rank):
                cells_needed.add(cr.rsplit('#', 1)[0])
    cells = {cn: load_cell_trades(cn) for cn in sorted(cells_needed)}
    print(f'  {len(cells)} cells loaded ({time.time()-t0:.0f}s)', flush=True)

    ds._W_PRICE = ds._build_price_map(cells)

    tg_full = {lb: build_tg(lb, cells) for lb in labels}
    print(f'  tg ready ({time.time()-t0:.0f}s)', flush=True)

    def keep(et, drop):
        if drop is None:
            return True
        if drop == 'TRUNC':
            return et < TRUNC_TS
        return et.year not in drop

    ranks = defaultdict(dict)      # cand -> {variant: rank}
    gates = defaultdict(dict)      # cand -> {variant: (liq, nw26, dd)}
    full_metrics = {}
    for vname, drop in VARIANTS:
        vtg = {lb: {et: v for et, v in tg_full[lb].items() if keep(et, drop)}
               for lb in labels}
        res = {}
        for (lb, lev, caut, cyc) in cands:
            r = ds.sim(vtg[lb], cyc, CAUT[caut], lev)
            r['score'] = ds.raw_score(r)
            res[(lb, lev, caut, cyc)] = r
            gates[(lb, lev, caut, cyc)][vname] = (r['liq'], r['nw26'], r['max_dd'])
            if vname == 'full':
                full_metrics[(lb, lev, caut, cyc)] = r
        order = sorted(res, key=lambda c: -res[c]['score'])
        for i, c in enumerate(order):
            ranks[c][vname] = i + 1
        print(f'  [{vname}] simulated+ranked ({(time.time()-t0)/60:.1f}min)',
              flush=True)

    vnames = [v for v, _ in VARIANTS]
    rows = []
    for c in cands:
        rr = [ranks[c][v] for v in vnames]
        gate_ok = all(g[0] == 0 and g[1] <= NW26_MAX and g[2] <= DD_CAP
                      for g in gates[c].values())
        fm = full_metrics[c]
        rows.append({'cand': c, 'med': float(np.median(rr)), 'worst': max(rr),
                     'rr': rr, 'gate': gate_ok, 'net': fm['net'],
                     'dd': fm['max_dd'], 'net26': fm['net26'],
                     'nw26': fm['nw26'], 'liq': fm['liq'],
                     'wdd': max(g[2] for g in gates[c].values())})
    rows.sort(key=lambda r: (r['med'], r['cand'][1], -r['net']))

    out = ['DEPLOY-LEVEL CONSENSUS (candidate = triple+lev+caut+cyc, 8 variants)',
           f'Pre-registered: gate = ALL variants Liq=0 & NW26<={NW26_MAX} & '
           f'DD<={DD_CAP:.0f}%; winner = best gate-passing median; '
           'tie -> lower lev.', '=' * 118, '',
           '%-16s %3s %-9s %3s | %-28s %6s %5s | %4s | %9s %5s %8s %6s' % (
               'triple', 'lev', 'caut', 'cyc', 'variant ranks', 'median',
               'worst', 'GATE', 'Net$full', 'DD%', 'OOS26$', 'wDD%')]
    for r in rows[:40]:
        lb, lev, caut, cyc = r['cand']
        out.append('%-16s %3d %-9s %3d | %-28s %6.1f %5d | %4s | %9s %5.1f %8s %6.1f' % (
            lb, lev, caut, cyc, ' '.join('%3d' % x for x in r['rr']),
            r['med'], r['worst'], 'OK' if r['gate'] else '-',
            '{:,.0f}'.format(r['net']), r['dd'],
            '{:,.0f}'.format(r['net26']), r['wdd']))
    # raw dump of every candidate (re-readable later without re-simulating)
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTDIR / 'deploy_consensus_all.tsv', 'w',
              encoding='utf-8') as f:
        f.write('label\tlev\tcaut\tcyc\tgate\tmed\tworst\tnet\tdd\twdd\t'
                'net26\tnw26\tliq\t' + '\t'.join(vnames) + '\n')
        for r in rows:
            lb, lev, caut, cyc = r['cand']
            f.write(f'{lb}\t{lev}\t{caut}\t{cyc}\t{int(r["gate"])}\t'
                    f'{r["med"]}\t{r["worst"]}\t{r["net"]:.0f}\t{r["dd"]:.2f}\t'
                    f'{r["wdd"]:.2f}\t{r["net26"]:.0f}\t{r["nw26"]}\t{r["liq"]}\t'
                    + '\t'.join(str(x) for x in r['rr']) + '\n')

    passing = [r for r in rows if r['gate']]
    out.append('')
    if passing:
        w = passing[0]
        lb, lev, caut, cyc = w['cand']
        out.append(f'GLOBAL best-median (low-lev-biased, report only): '
                   f'{lb} lev{lev} {caut} cyc{cyc} median {w["med"]:.1f}')
        out.append(f'gate-passing candidates: {len(passing)}/{len(rows)}')
    else:
        out.append('NO CANDIDATE PASSED THE ALL-VARIANT GATE')

    # IN-STRATUM consensus + GRADED-RELAXATION LEV RULE.
    # PRE-REGISTERED: the NW26 gate relaxes in grades (<=1 -> <=2 -> <=3); at
    # each grade the HIGHEST lev with a passing leader wins. If all three
    # grades are empty -> ABSTAIN (no package; the live meaning = stay on the
    # incumbent). Thresholds are NEVER tuned after seeing results.
    def stratum_ranks(lev):
        stratum = [r for r in rows if r['cand'][1] == lev]
        srows = []
        for r in stratum:
            rr = [sorted(x['rr'][vi] for x in stratum).index(r['rr'][vi]) + 1
                  for vi in range(len(vnames))]
            srows.append((float(np.median(rr)), max(rr), r))
        srows.sort(key=lambda x: (x[0], -x[2]['net']))
        return srows

    def passes(cand, nw26_cap):
        return all(g[0] == 0 and g[1] <= nw26_cap and g[2] <= DD_CAP
                   for g in gates[cand].values())

    out.append('')
    out.append('IN-STRATUM CONSENSUS (NW26 graded: <=1 -> <=2 -> <=3):')
    levs = sorted({c[1] for c in cands})
    strat = {lev: stratum_ranks(lev) for lev in levs}
    for lev in levs:
        top = strat[lev][0]
        lb, _, caut, cyc = top[2]['cand']
        g1 = next((s for s in strat[lev] if passes(s[2]['cand'], 1)), None)
        out.append(f'  lev{lev}: stratum #1 {lb} {caut} cyc{cyc} '
                   f'(median {top[0]:.1f}) | NW26<=1 passing: '
                   + (f"{g1[2]['cand'][0]} median {g1[0]:.1f} DD {g1[2]['dd']:.1f}%"
                      if g1 else 'NONE'))

    winner = None
    for cap in (1, 2, 3):
        lev_leader = {}
        for lev in levs:
            ldr = next((s for s in strat[lev] if passes(s[2]['cand'], cap)), None)
            if ldr:
                lev_leader[lev] = ldr
        if lev_leader:
            best_lev = max(lev_leader)
            winner = (cap, lev_leader[best_lev])
            break
    out.append('')
    if winner:
        cap, (m, wst, r) = winner
        lb, lev, caut, cyc = r['cand']
        tag = '' if cap == 1 else f' [RELAXED: NW26<={cap}]'
        out.append(f'WINNER (gate-passing leader at the HIGHEST lev){tag}: '
                   f'{lb} lev{lev} {caut} cyc{cyc} | stratum-median {m:.1f} '
                   f'| Net ${r["net"]:,.0f} DD {r["dd"]:.1f}% '
                   f'(variant-worst {r["wdd"]:.1f}%) OOS26 ${r["net26"]:,.0f}')
    else:
        out.append('WINNER: ABSTAIN (no grade passed incl. NW26<=3; '
                   'live = stay on incumbent). Lowest-lev stratum #1 for reference:')
        m, wst, r = strat[levs[0]][0]
        lb, lev, caut, cyc = r['cand']
        viol = []
        for vn, g in gates[r['cand']].items():
            pr = []
            if g[0] != 0: pr.append(f'Liq={g[0]}')
            if g[1] > 3: pr.append(f'NW26={g[1]}')
            if g[2] > DD_CAP: pr.append(f'DD={g[2]:.1f}')
            if pr: viol.append(f'{vn}:{"/".join(pr)}')
        out.append(f'  ref: {lb} lev{lev} {caut} cyc{cyc} '
                   f'DD {r["dd"]:.1f}% | violations: '
                   + ('; '.join(viol) if viol else 'none'))
    txt = '\n'.join(out)
    (_OUTDIR / 'deploy_consensus.txt').write_text(txt, encoding='utf-8')
    print(txt, flush=True)
    print(f'\nDONE ({(time.time()-t0)/60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
