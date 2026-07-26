"""Common AND-gate + SL/TP sweep engine — works for ANY registry strategy.

One shared engine instead of a copy-pasted per-strategy script: add a
strategy to the registry and it sweeps with zero extra code.

Usage:
  python factory/gates/gate_sweep.py --strategy SMA_X
  python factory/gates/gate_sweep.py --strategy SMA_X --cells BEAR_MED,FLAT_LOW

Method (unchanged from the canonical diag):
  Regime: BTC trend x momentum (3x3 cells). Per (cell, direction, SL, TP):
  IC-stable independent features -> per-feature threshold optimization ->
  exhaustive AND combos (K 3-8) -> FWD+REV walk-forward validation ->
  keep STRONG only. Ranking: STRONG -> worst_yr_EV -> overall_EV.
"""
import sys, os, time, gc, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.gates.features_common import (
    ALL_COL_NAMES, _spearman, compute_all_feature_stats, dedup_1h_4h,
)
from factory.data.registry import STRATEGIES, load_builder, BACKTEST

EXCLUDED_FEATURES = set()   # feature names to hard-exclude from gate search
FEE_RT = 0.08
TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
VOL_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}

IC_MIN = 0.010
STAB_MIN = 2
CORR_THRESH = 0.60
K_RANGE = range(3, 9)
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MIN_YR_TRADES = 1000

SL_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TP_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

IS_YEARS = list(range(2021, 2026))
MID_YEARS = [2022, 2023, 2024]

ALL_CELLS = ['BEAR_LOW', 'BEAR_MED', 'BEAR_HIGH', 'FLAT_LOW', 'FLAT_MED',
             'FLAT_HIGH', 'BULL_LOW', 'BULL_MED', 'BULL_HIGH']

FWD_WF = [('FWF1', [2021, 2022], [2023]), ('FWF2', [2021, 2022, 2023], [2024]),
          ('FWF3', [2021, 2022, 2023, 2024], [2025])]
REV_WF = [('RWF1', [2024, 2025], [2023]), ('RWF2', [2023, 2024, 2025], [2022]),
          ('RWF3', [2022, 2023, 2024, 2025], [2021])]


def select_independent(feats, features, resolved):
    feats = sorted(feats, key=lambda f: f['abs_ic'], reverse=True)
    selected = []
    for f in feats:
        col_f = features[resolved, f['idx']].astype(np.float64)
        ok = True
        for s in selected:
            col_s = features[resolved, s['idx']].astype(np.float64)
            v = np.isfinite(col_f) & np.isfinite(col_s)
            if v.sum() < 100:
                continue
            if abs(_spearman(col_f[v], col_s[v])) > CORR_THRESH:
                ok = False
                break
        if ok:
            selected.append(f)
    return selected


def optimize_feature_threshold(features, pnl, resolved, feat):
    fi = feat['idx']
    col = features[:, fi].astype(np.float64)
    valid = resolved & np.isfinite(col)
    if valid.sum() < 500:
        return None
    col_valid = col[valid]
    pnl_valid = pnl[valid]
    pct_values = np.percentile(col_valid, PCTS)
    best = None
    for pi, pval in zip(PCTS, pct_values):
        if feat['ic'] > 0:
            mask_local = col_valid > pval
            direction = 'gt'
        else:
            mask_local = col_valid < pval
            direction = 'lt'
        n_pass = mask_local.sum()
        if n_pass < 200 or n_pass > len(col_valid) * 0.95:
            continue
        ev = pnl_valid[mask_local].mean()
        if best is None or ev > best['ev']:
            wr = (pnl_valid[mask_local] > 0).mean() * 100
            best = {'name': feat['name'], 'idx': int(fi),
                    'ic_sign': 1 if feat['ic'] > 0 else -1,
                    'direction': direction, 'pct': int(pi), 'value': float(pval),
                    'wr': float(wr), 'ev': float(ev), 'n_pass': int(n_pass),
                    'pass_rate': float(n_pass / valid.sum())}
    return best


def search_cell(features, pnl, years, resolved, col_names):
    mid_mask = np.isin(years, MID_YEARS) & resolved
    if mid_mask.sum() < 1000:
        return None
    stats = compute_all_feature_stats(features[mid_mask], pnl[mid_mask],
                                      years[mid_mask], col_names=col_names)
    stable = [f for f in stats if f['stab'] >= STAB_MIN and f['n_yr'] >= 2
              and f['abs_ic'] >= IC_MIN and f['name'] not in EXCLUDED_FEATURES]
    deduped = dedup_1h_4h(stable)
    indep = select_independent(deduped, features[mid_mask],
                               np.isfinite(pnl[mid_mask]))
    if len(indep) < 3:
        return None

    opt_feats = []
    for f in indep:
        opt = optimize_feature_threshold(features, pnl, resolved, f)
        if opt is not None:
            opt_feats.append(opt)
    if len(opt_feats) < 3:
        return None

    yr_mask_arr = np.stack([(years == y) for y in IS_YEARS])
    wf_masks = []
    for _, _, val_yrs in FWD_WF:
        wf_masks.append((True, np.isin(years, val_yrs)))
    for _, _, val_yrs in REV_WF:
        wf_masks.append((False, np.isin(years, val_yrs)))
    pnl_pos = (pnl > 0).astype(np.uint8)
    best_any = None

    feat_masks = []
    for g in opt_feats:
        col = features[:, g['idx']].astype(np.float64)
        valid = np.isfinite(col)
        if g['direction'] == 'gt':
            feat_masks.append(valid & (col > g['value']))
        else:
            feat_masks.append(valid & (col < g['value']))

    failed_subsets = set()
    for k in K_RANGE:
        if k > len(opt_feats):
            break
        for cidx in combinations(range(len(opt_feats)), k):
            skip = False
            if k > 3:
                for sub_k in range(3, k):
                    for sub in combinations(cidx, sub_k):
                        if sub in failed_subsets:
                            skip = True
                            break
                    if skip:
                        break
            if skip:
                continue

            combined = resolved.copy()
            for i in cidx:
                combined &= feat_masks[i]
            n_kept = combined.sum()
            if n_kept < MIN_YR_TRADES * len(IS_YEARS):
                failed_subsets.add(cidx)
                continue
            yr_counts = yr_mask_arr[:, combined].sum(axis=1)
            if int(yr_counts.min()) < MIN_YR_TRADES:
                failed_subsets.add(cidx)
                continue

            p = pnl[combined]
            ev = p.mean()
            if ev <= 0:
                continue
            wr = pnl_pos[combined].sum() / n_kept * 100

            fwd_pass = rev_pass = 0
            for is_fwd, wf_m in wf_masks:
                vm = combined & wf_m
                if vm.sum() < 50:
                    continue
                if pnl[vm].mean() > 0:
                    if is_fwd:
                        fwd_pass += 1
                    else:
                        rev_pass += 1
            total = fwd_pass + rev_pass

            yr_data = {}
            for yi, y in enumerate(IS_YEARS):
                if yr_counts[yi] > 0:
                    ym = combined & yr_mask_arr[yi]
                    yr_data[y] = {'n': int(yr_counts[yi]),
                                  'wr': float(pnl_pos[ym].sum() / yr_counts[yi] * 100),
                                  'ev': float(pnl[ym].mean())}

            worst_yr_ev = min((v['ev'] for v in yr_data.values()), default=-999)
            worst_yr_wr = min((v['wr'] for v in yr_data.values()), default=0)

            rec = {'gates': [opt_feats[i] for i in cidx], 'k': int(k),
                   'n_kept': int(n_kept), 'wr': float(wr), 'ev': float(ev),
                   'fwd': int(fwd_pass), 'rev': int(rev_pass),
                   'total': int(total), 'yr': yr_data,
                   'worst_yr_ev': float(worst_yr_ev),
                   'worst_yr_wr': float(worst_yr_wr)}
            score = (total, worst_yr_ev, ev)
            if best_any is None or score > best_any.get('_score', (-999,)):
                rec['_score'] = score
                best_any = rec
    return best_any


def verdict(fwd, rev):
    if fwd == 3 and rev == 3:
        return 'STRONG'
    if fwd == 3 or rev == 3:
        return 'GOOD'
    if fwd >= 2 and rev >= 2:
        return 'OK'
    if fwd >= 2 or rev >= 2:
        return 'WEAK'
    return 'FAIL'


def _process_cell_data(d, direction, cell, col_names):
    """SL/TP grid + combinatorial gate search for ONE already-loaded cell."""
    features = d['features']
    years = d['years']
    cell_results = []
    for sl in SL_LEVELS:
        for tp in TP_LEVELS:
            if tp < sl:
                continue
            cp = d.get(f'pnl_{sl}_{tp}')
            if cp is None:
                continue
            cr = np.isfinite(cp) & (cp != 0)
            if cr.sum() < MIN_YR_TRADES * len(IS_YEARS):
                continue
            res = search_cell(features, cp, years, cr, col_names)
            if res is None:
                continue
            v = verdict(res['fwd'], res['rev'])
            if v != 'STRONG':
                continue
            cell_results.append({
                'direction': direction, 'cell': cell, 'sl': sl, 'tp': tp,
                'verdict': v, 'wr': round(res['wr'], 2),
                'ev': round(res['ev'], 4), 'n': res['n_kept'],
                'worst_yr_ev': round(res['worst_yr_ev'], 4),
                'worst_yr_wr': round(res['worst_yr_wr'], 2),
                'fwd': res['fwd'], 'rev': res['rev'],
                'yr': res['yr'], 'gates': res['gates'],
            })
    sym = [c for c in cell_results if c['sl'] == c['tp']]
    asym = [c for c in cell_results if c['sl'] != c['tp']]
    sym.sort(key=lambda x: (x['worst_yr_wr'], x['worst_yr_ev']), reverse=True)
    asym.sort(key=lambda x: (x['worst_yr_ev'], x['ev']), reverse=True)
    return sym + asym


_CELL_EST = {}

def _cell_est_bytes(code, cell):
    """Estimated in-RAM load of one (direction, cell): the cell's IS
    monthly-shard bytes x 2.2 (unpickle + vstack transient). Shard files
    hold BOTH directions, so this overestimates ~2x — extra safety."""
    key = (code, cell)
    if key in _CELL_EST:
        return _CELL_EST[key]
    inst = load_builder(code)._instance
    cache_base = str(inst.CACHE_FILE).replace('_raw_cache.pkl', '')
    cdir = Path(cache_base).parent
    name = Path(cache_base).name
    t = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
    v_ = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
    sz = sum(f.stat().st_size for yr in IS_YEARS
             for f in cdir.glob(f'{name}_{yr}-??_t{t}v{v_}.pkl'))
    # shards hold BOTH directions (load = ~sz/2), x2.2 unpickle+vstack peak
    est = (sz / 2) * 2.2 if sz else 7e9
    _CELL_EST[key] = est
    return est


def _cell_worker(args):
    """Process worker: stream-load ONE (direction, cell) and search it.

    q (Manager queue): signals 'memory settled' right after the load
    returns — the parent measures RAM at that moment and admits the
    next cell (event-driven admission, no timers).
    """
    code, direction, cell, col_names, q = args
    builder = load_builder(code)
    inst = getattr(builder, '_instance')
    t_idx = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
    v_idx = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
    try:
        cache = inst.load_lite(direction, years=IS_YEARS, cell=(t_idx, v_idx))
    finally:
        if q is not None:
            q.put('loaded')   # RAM settled -> parent admits next
    try:
        if cache is None:
            return direction, cell, []
        d = cache[direction]
        if len(d['years']) < 5000:
            return direction, cell, []
        return direction, cell, _process_cell_data(d, direction, cell,
                                                   col_names)
    finally:
        if q is not None:
            q.put('done')     # RAM freed -> parent re-checks immediately


def run_parallel(strategy_code, cells=None, col_names=None, out_path=None,
                 max_workers=None):
    """Parallel mode: each (direction, cell) is an independent process with
    a cell-filtered streaming load. MUST reproduce run()'s results exactly
    (re-acceptance against the serial run required after any engine change).
    """
    code = strategy_code.upper()
    rel, _sig, _cat = STRATEGIES[code]
    strat_dir = BACKTEST / Path(rel).parent
    cells = cells or ALL_CELLS
    col_names = col_names or ALL_COL_NAMES
    json_path = Path(out_path) if out_path else strat_dir / 'gate_sweep_results.json'

    t0 = time.time()
    all_results = {}
    if json_path.exists():
        all_results = json.load(open(json_path))

    n_cpu = max_workers or max(1, (os.cpu_count() or 8) - 2)
    todo = [(code, d, c) for d in ['SHORT', 'LONG'] for c in cells
            if f'{d}_{c}' not in all_results]
    # largest-first: big cells claim RAM while it's most free
    sized = sorted(((t, _cell_est_bytes(code, t[2])) for t in todo),
                   key=lambda x: -x[1])
    print(f'  GATE SWEEP [{code}] parallel: {len(todo)} cells, '
          f'event-driven RAM admission (cpu cap {n_cpu})', flush=True)

    import psutil, multiprocessing, queue as _queue
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    RESERVE = 3e9        # OS + parent headroom

    mgr = multiprocessing.Manager()
    q = mgr.Queue()
    pending = deque(sized)
    running = {}          # future -> (label, est)
    loading_fut = None    # at most ONE worker in its load phase
    with ProcessPoolExecutor(max_workers=n_cpu) as ex:
        while pending or running:
            # admission: no load in flight + MEASURED RAM covers next cell
            if pending and loading_fut is None and len(running) < n_cpu:
                (c_, d_, cell_), est = pending[0]
                avail = psutil.virtual_memory().available
                if (avail - RESERVE) > est or not running:
                    pending.popleft()
                    fut = ex.submit(_cell_worker,
                                    (c_, d_, cell_, col_names, q))
                    running[fut] = (f'{d_}_{cell_}', est)
                    loading_fut = fut
                    print(f'  + admit {d_}_{cell_} (est {est/1e9:.1f}GB, '
                          f'avail {avail/1e9:.1f}GB, active {len(running)})',
                          flush=True)
            # events: 'loaded' (RAM settled) / 'done' (RAM freed) -> both
            # wake the parent instantly for a fresh measure+admit cycle
            try:
                ev = q.get(timeout=5)
                if ev == 'loaded':
                    loading_fut = None
            except _queue.Empty:
                pass   # fallback tick (covers hard-crashed workers)
            # harvest finished futures
            for fut in [f for f in running if f.done()]:
                running.pop(fut)
                if fut is loading_fut:   # crashed before 'loaded' event
                    loading_fut = None
                direction, cell, cell_results = fut.result()
                all_results[f'{direction}_{cell}'] = cell_results
                json.dump(all_results, open(json_path, 'w'), indent=2)
                print(f'  {direction}_{cell:<12} -> {len(cell_results)} STRONG '
                      f'({time.time()-t0:.0f}s)', flush=True)

    total = sum(len(v) for v in all_results.values())
    print(f'  Saved {total} configs ({(time.time()-t0)/60:.1f} min)', flush=True)
    return all_results


def run_parallel_multi(strategy_codes, col_names=None, max_workers=None):
    """GLOBAL admission queue across strategies (user design 2026-06-13).

    One pending list of every strategy's missing (direction, cell). The
    admission scan takes the FIRST task that fits measured RAM — so while
    one strategy's giant cells crunch CPU-bound, the next strategies'
    small cells fill the idle RAM/cores instead of waiting at the
    strategy boundary. Queue order: strategy-major, largest cell first
    within each strategy. Per-strategy JSONs written incrementally by
    the single parent (no write races).
    """
    col_names = col_names or ALL_COL_NAMES
    n_cpu = max_workers or max(1, (os.cpu_count() or 8) - 2)
    t0 = time.time()

    json_paths, results = {}, {}
    pending = []          # [(code, d, cell, est)] in admission scan order
    for code in strategy_codes:
        rel, _sig, _cat = STRATEGIES[code]
        jp = BACKTEST / Path(rel).parent / 'gate_sweep_results.json'
        json_paths[code] = jp
        results[code] = json.load(open(jp)) if jp.exists() else {}
        todo = [(d, c) for d in ['SHORT', 'LONG'] for c in ALL_CELLS
                if f'{d}_{c}' not in results[code]]
        sized = sorted(((d, c, _cell_est_bytes(code, c)) for d, c in todo),
                       key=lambda x: -x[2])
        pending.extend((code, d, c, e) for d, c, e in sized)
    print(f'  GATE SWEEP MULTI: {len(strategy_codes)} strategies, '
          f'{len(pending)} cells, global RAM admission (cpu cap {n_cpu})',
          flush=True)

    import psutil, multiprocessing, queue as _queue
    from concurrent.futures import ProcessPoolExecutor
    RESERVE = 3e9

    mgr = multiprocessing.Manager()
    q = mgr.Queue()
    running = {}          # future -> (code, label)
    loading_fut = None
    bad_cells = []
    with ProcessPoolExecutor(max_workers=n_cpu) as ex:
        while pending or running:
            if pending and loading_fut is None and len(running) < n_cpu:
                avail = psutil.virtual_memory().available
                pick = None
                for i, (c_, d_, cell_, est) in enumerate(pending):
                    if (avail - RESERVE) > est or not running:
                        pick = i
                        break   # first fit: preserves strategy-major bias
                if pick is not None:
                    c_, d_, cell_, est = pending.pop(pick)
                    fut = ex.submit(_cell_worker,
                                    (c_, d_, cell_, col_names, q))
                    running[fut] = (c_, f'{d_}_{cell_}')
                    loading_fut = fut
                    print(f'  + admit {c_}:{d_}_{cell_} '
                          f'(est {est/1e9:.1f}GB, avail {avail/1e9:.1f}GB, '
                          f'active {len(running)})', flush=True)
            try:
                ev = q.get(timeout=5)
                if ev == 'loaded':
                    loading_fut = None
            except _queue.Empty:
                pass
            for fut in [f for f in running if f.done()]:
                code, label = running.pop(fut)
                if fut is loading_fut:
                    loading_fut = None
                try:
                    direction, cell, cell_results = fut.result()
                except Exception as e:
                    # corrupt shard etc. — the cell stays missing; a resume
                    # rerun retries it after repair; the queue keeps going
                    bad_cells.append(f'{code}:{label}')
                    print(f'  !! {code}:{label} CELL ERROR: '
                          f'{type(e).__name__}: {str(e)[:100]}', flush=True)
                    continue
                results[code][f'{direction}_{cell}'] = cell_results
                json.dump(results[code], open(json_paths[code], 'w'),
                          indent=2)
                done_n = len(results[code])
                print(f'  {code}:{direction}_{cell:<12} -> '
                      f'{len(cell_results)} STRONG [{done_n}/18] '
                      f'({(time.time()-t0)/60:.0f}min)', flush=True)
                if done_n == 18:
                    print(f'  == {code} GATE DONE ==', flush=True)

    print(f'  MULTI done ({(time.time()-t0)/3600:.1f}h)', flush=True)
    if bad_cells:
        print(f'  FAILED CELLS ({len(bad_cells)}): {bad_cells}',
              flush=True)
        raise RuntimeError(f'{len(bad_cells)} cells failed: {bad_cells}')
    return results


def run(strategy_code, cells=None, col_names=None, out_path=None):
    code = strategy_code.upper()
    rel, _sig, _cat = STRATEGIES[code]
    strat_dir = BACKTEST / Path(rel).parent
    builder = load_builder(code)
    inst = getattr(builder, '_instance')
    cells = cells or ALL_CELLS
    col_names = col_names or ALL_COL_NAMES
    json_path = Path(out_path) if out_path else strat_dir / 'gate_sweep_results.json'

    t0 = time.time()
    print('=' * 120, flush=True)
    print(f'  GATE SL/TP SWEEP [{code}] — common engine', flush=True)
    print(f'  cells={len(cells)} | cols={len(col_names)} | out={json_path.name}', flush=True)
    print('=' * 120, flush=True)

    all_results = {}
    if json_path.exists():
        all_results = json.load(open(json_path))
        print(f'  Loaded {len(all_results)} cached cells', flush=True)

    for direction in ['SHORT', 'LONG']:
        cache = inst.load_lite(direction, years=IS_YEARS)
        d = cache[direction]
        features = d['features']
        years = d['years']
        trends = d['trends']
        vols = d['vols']
        print(f'\n  {direction}: {len(years):,} rows', flush=True)

        for cell in cells:
            cell_key = f'{direction}_{cell}'
            if cell_key in all_results:
                print(f'  {cell:<12} CACHED', flush=True)
                continue
            t_idx = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
            v_idx = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
            mask = (trends == t_idx) & (vols == v_idx)
            if mask.sum() < 5000:
                all_results[cell_key] = []
                print(f'  {cell:<12} skip ({mask.sum()} rows)', flush=True)
                continue

            cf = features[mask]
            cy = years[mask]
            cell_results = []
            for sl in SL_LEVELS:
                for tp in TP_LEVELS:
                    if tp < sl:
                        continue
                    pnl_all = d.get(f'pnl_{sl}_{tp}')
                    if pnl_all is None:
                        continue
                    cp = pnl_all[mask]
                    cr = np.isfinite(cp) & (cp != 0)
                    if cr.sum() < MIN_YR_TRADES * len(IS_YEARS):
                        continue
                    res = search_cell(cf, cp, cy, cr, col_names)
                    if res is None:
                        continue
                    v = verdict(res['fwd'], res['rev'])
                    if v != 'STRONG':
                        continue
                    cell_results.append({
                        'direction': direction, 'cell': cell, 'sl': sl, 'tp': tp,
                        'verdict': v, 'wr': round(res['wr'], 2),
                        'ev': round(res['ev'], 4), 'n': res['n_kept'],
                        'worst_yr_ev': round(res['worst_yr_ev'], 4),
                        'worst_yr_wr': round(res['worst_yr_wr'], 2),
                        'fwd': res['fwd'], 'rev': res['rev'],
                        'yr': res['yr'], 'gates': res['gates'],
                    })

            sym = [c for c in cell_results if c['sl'] == c['tp']]
            asym = [c for c in cell_results if c['sl'] != c['tp']]
            sym.sort(key=lambda x: (x['worst_yr_wr'], x['worst_yr_ev']), reverse=True)
            asym.sort(key=lambda x: (x['worst_yr_ev'], x['ev']), reverse=True)
            all_results[cell_key] = sym + asym
            json.dump(all_results, open(json_path, 'w'), indent=2)
            print(f'  {cell:<12} -> {len(cell_results)} STRONG configs '
                  f'({time.time()-t0:.0f}s)', flush=True)

        del cache, d, features, years, trends, vols
        gc.collect()

    total = sum(len(v) for v in all_results.values())
    print(f'\n  Saved {total} configs to {json_path} ({(time.time()-t0)/60:.1f} min)',
          flush=True)
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--cells', default=None,
                    help='comma-separated cell list (default: all 9)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--parallel', action='store_true',
                    help='cell-process parallelism (re-accept before trusting)')
    ap.add_argument('--max-workers', type=int, default=None)
    args = ap.parse_args()

    cells = args.cells.split(',') if args.cells else None
    col_names = None
    if args.parallel:
        run_parallel(args.strategy, cells=cells, col_names=col_names,
                     out_path=args.out, max_workers=args.max_workers)
    else:
        run(args.strategy, cells=cells, col_names=col_names, out_path=args.out)


if __name__ == '__main__':
    main()
