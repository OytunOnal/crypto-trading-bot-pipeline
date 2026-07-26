"""Blind walk-forward JUDGE step: forward-window performance of the pick.

Called after the blind chain + deploy consensus finish:
  python pwf_judge.py <PICK "B3+F30+BU2 lev7 8/15/22D cyc15"> <JS> <JE> <OUTDIR>

Judge window [JS, JE): from the FULL (uncut) stream, the pick's config set is
measured as (a) trade-level total pnl% / EV / n, (b) the production sim's
in-window monthly booking net + in-window DD. Result -> OUTDIR/judge.txt.

The pick was selected BLIND (chain saw nothing past the cutoff); the judge is
the only step that touches the forward window — that separation is the whole
point of the harness.
"""
import sys, os, re, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    pick, js, je, outdir = sys.argv[1], pd.Timestamp(sys.argv[2]), \
        pd.Timestamp(sys.argv[3]), Path(sys.argv[4])
    m = re.match(r'B(\d+)\+F(\d+)\+BU(\d+) lev(\d+) (\S+) cyc(\d+)', pick)
    b, fl, bu, lev, caut, cyc = (int(m.group(1)), int(m.group(2)),
                                 int(m.group(3)), int(m.group(4)),
                                 m.group(5), int(m.group(6)))
    sys.argv = sys.argv[:1]
    from factory.portfolio import deploy_sweep as ds
    from factory.portfolio import trade_full_combo as F
    F.PICKLE_DIR = ROOT / os.environ.get('PWF_FULL_DIR', 'pickle_sets/FULL')
    # PWF_BASE_DIR: where the combo/regime definition files are read from.
    # Inside the chain the working results/ dir is correct; LATER (manual)
    # judging must point at the QUARTER ARCHIVE — the working dir may already
    # hold another quarter's files (a silent-mixup lesson).
    base = Path(os.environ.get('PWF_BASE_DIR', ROOT / 'results'))

    def parse_regime(regime, rank):
        for line in open(base / f'trade_regime_{regime}.txt', encoding='utf-8'):
            if line.startswith(f'#{rank} '):
                return [c.strip() for c in line.split('|')[2].strip().split('+')]
        return []

    def parse_cell(cell, rank):
        for line in open(base / f'trade_combo_{cell}.txt', encoding='utf-8'):
            if line.startswith(f'#{rank} '):
                return [c.strip() for c in line.split('|')[2].strip().split(',')]
        return []

    cfgs = set()
    for regime, rank in (('BEAR', b), ('FLAT', fl), ('BULL', bu)):
        for cr in parse_regime(regime, rank):
            cn, rk = cr.rsplit('#', 1)
            for cfg in parse_cell(cn, int(rk)):
                cfgs.add((cn, cfg))
    cells = {cn: F.load_cell_trades(cn) for cn in sorted({c for c, _ in cfgs})}
    ds._W_PRICE = ds._build_price_map(cells)

    tg = defaultdict(list)
    win = []
    # sorted: set-iteration nondeterminism (hash randomization) shuffled the
    # within-bar candidate order -> once cyc/capital caps bind, results jitter.
    for (cn, cfg) in sorted(cfgs):
        for t in cells[cn].get(cfg, []):
            et = t['entry_time']; iso = et.isocalendar()
            tg[et].append((t['coin'], t['pnl_pct'], t['exit_time'],
                           et.strftime('%Y-%m'), '%d-W%02d' % (iso[0], iso[1]),
                           t.get('direction', '')))
            if js <= et < je:
                win.append(t['pnl_pct'])
    # PWF_CAUT_JSON: pass a caution ladder as JSON [[thr,mult],...] (for
    # counterfactual sweeps). Default: the pick's named ladder. PWF_LEV_OVR:
    # override lev (judge every quarter at the same fixed lev in CFs).
    _cj = os.environ.get('PWF_CAUT_JSON')
    caut_ladder = json.loads(_cj) if _cj else dict(ds.CAUTION_GRID)[caut]
    lev = int(os.environ.get('PWF_LEV_OVR', lev))
    r = ds.sim(tg, cyc, caut_ladder, lev, detail=True)
    mo = r['mo']
    wnet = sum(v for k, v in mo.items()
               if js.strftime('%Y-%m') <= k < je.strftime('%Y-%m'))
    eqd = r['eq_daily']
    days = [d for d in sorted(eqd) if js.date() <= d < je.date()]
    if days:
        eq = np.array([eqd[d][0] for d in days])
        pk = np.maximum.accumulate(np.maximum(eq, 10000.0))
        wdd = float(((pk - eq) / pk * 100).max())
    else:
        wdd = float('nan')
    wv = np.array(win) if win else np.array([0.0])
    out = [f'PWF JUDGE: {pick}',
           f'window [{js.date()}, {je.date()})',
           f'trade-level: n={len(win):,} total {wv.sum():+.0f}% '
           f'EV/tr {wv.mean():+.3f} WR {(wv > 0).mean() * 100:.1f}%',
           f'sim booking: window-net ${wnet:,.0f}  window-DD {wdd:.1f}%',
           f'full-stream reference: Net ${r["net"]:,.0f} DD {r["max_dd"]:.1f}%']
    txt = '\n'.join(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'judge.txt').write_text(txt, encoding='utf-8')
    print(txt, flush=True)


if __name__ == '__main__':
    main()
