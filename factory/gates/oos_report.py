"""OOS (2026) report for frozen configs — evaluation ONLY, never selection.

Reads each strategy's phase2 results (gate + block rules frozen on IS
2021-2025), applies them to 2026 rows, and prints IS-vs-OOS EV/WR/N per
config. QS entries (unified_qs_features.json) are evaluated as top-q
lift on 2026 where present.

Run AFTER run_all; output is informational (a human look before deploy decisions).

Usage:
  python factory/gates/oos_report.py --strategies ALL
  python factory/gates/oos_report.py --strategies SMA_X --json out.json
"""
import sys, os, time, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.data.registry import STRATEGIES, load_builder, BACKTEST
from factory.gates.features_common import ALL_COL_NAMES
from factory.blocks.phase2 import apply_gate
from factory.qs.qs_core import rolling_rank_exclude_self

TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
VOL_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}
IS_YEARS = list(range(2021, 2026))   # in-sample years (example split)
OOS_YEAR = 2026                      # held-out evaluation year
N_ROLL = 100


def _stats(pnl, mask):
    n = int(mask.sum())
    if n == 0:
        return {'n': 0, 'ev': None, 'wr': None}
    p = pnl[mask]
    return {'n': n, 'ev': round(float(p.mean()), 4),
            'wr': round(float((p > 0).mean() * 100), 2)}


def report_strategy(code, qs_data):
    rel, _sig, _cat = STRATEGIES[code]
    strat_dir = BACKTEST / Path(rel).parent
    p2_path = strat_dir / f'{strat_dir.name}_phase2_results.json'
    if not p2_path.exists():
        print(f'  [{code}] no phase2 results, skipped', flush=True)
        return []
    p2 = json.load(open(p2_path))
    inst = load_builder(code)._instance
    rows = []

    for cell_key in sorted(p2.keys()):
        configs = p2[cell_key]
        if not configs:
            continue
        direction, cell = cell_key.split('_', 1)
        t_idx = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
        v_idx = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
        cache = inst.load_lite(direction, years=None, cell=(t_idx, v_idx))
        if cache is None:
            continue
        d = cache[direction]
        feat, years = d['features'], d['years']

        for ci, cfg in enumerate(configs):
            sl, tp = cfg['sl'], cfg['tp']
            pnl = d.get(f'pnl_{sl}_{tp}')
            if pnl is None:
                continue
            resolved = np.isfinite(pnl) & (pnl != 0)
            gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<',
                           g['value']) for g in cfg.get('gate', [])]
            gp = apply_gate(feat, resolved, gate_rules, ALL_COL_NAMES)
            for b in cfg.get('blocks', []):
                if b['name'] not in ALL_COL_NAMES:
                    continue
                col = feat[:, ALL_COL_NAMES.index(b['name'])].astype(np.float64)
                v = np.isfinite(col)
                if b['block_op'] == '<':
                    gp &= ~(v & (col < b['value']))
                else:
                    gp &= ~(v & (col > b['value']))

            is_m = gp & np.isin(years, IS_YEARS)
            oos_m = gp & (years == OOS_YEAR)
            row = {'strategy': code, 'cell_key': cell_key, 'ci': ci,
                   'sl': sl, 'tp': tp,
                   'is': _stats(pnl, is_m), 'oos': _stats(pnl, oos_m)}

            # QS top-q OOS lift (when present)
            for q in (2, 3, 4, 5):
                key = f'{direction}_{cell}_{code}_q{q}_c{ci}'
                e = qs_data.get(key)
                if not e:
                    continue
                wsum = np.zeros(len(pnl)); wt = 0.0
                valid = gp.copy()
                for fname, fic in e['ic'].items():
                    if fname not in ALL_COL_NAMES:
                        valid &= False
                        continue
                    col = feat[:, ALL_COL_NAMES.index(fname)].astype(np.float64)
                    if fic < 0:
                        col = -col
                    rr = rolling_rank_exclude_self(col, N_ROLL)
                    w = max(abs(fic), 0.01)
                    valid &= np.isfinite(rr)
                    wsum += np.where(np.isfinite(rr), rr * w, 0)
                    wt += w
                if wt <= 0:
                    continue
                pct = rolling_rank_exclude_self(wsum / wt, N_ROLL)
                ok = valid & np.isfinite(pct)
                ok[:N_ROLL] = False
                top = ok & (pct >= 100.0 * (1 - 1.0 / q))
                row[f'q{q}_oos_top'] = _stats(pnl, top & (years == OOS_YEAR))
                row[f'q{q}_oos_rest'] = _stats(pnl, ok & ~top & (years == OOS_YEAR))
            rows.append(row)
        del cache, d, feat, years

    # print table
    print(f'\n  [{code}] {len(rows)} config', flush=True)
    print(f'  {"cell":<22} {"sl/tp":<9} {"IS ev/wr/n":<24} {"OOS ev/wr/n":<24}',
          flush=True)
    for r in rows:
        i, o = r['is'], r['oos']
        istr = f"{i['ev']:+.3f}/{i['wr']:.1f}/{i['n']:,}" if i['n'] else '-'
        ostr = f"{o['ev']:+.3f}/{o['wr']:.1f}/{o['n']:,}" if o['n'] else '-'
        print(f"  {r['cell_key']:<22} {r['sl']}/{r['tp']:<6} {istr:<24} {ostr:<24}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategies', required=True)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    if args.strategies.upper() == 'ALL':
        codes = sorted(STRATEGIES)
    else:
        codes = [c.strip().upper() for c in args.strategies.split(',')]

    all_rows = []
    t0 = time.time()
    for code in codes:
        # per-category QS file
        rel, _sig, _cat = STRATEGIES[code]
        cat_dir = (BACKTEST / Path(rel).parent).parent
        qs_path = cat_dir / 'unified_qs_features.json'
        qs_data = json.load(open(qs_path)) if qs_path.exists() else {}
        try:
            all_rows.extend(report_strategy(code, qs_data))
        except Exception as e:
            print(f'  [{code}] ERROR: {type(e).__name__}: {e}', flush=True)

    if args.json:
        json.dump(all_rows, open(args.json, 'w'), indent=1)
        print(f'\n  JSON -> {args.json}', flush=True)
    print(f'\n  OOS report done ({(time.time()-t0)/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
