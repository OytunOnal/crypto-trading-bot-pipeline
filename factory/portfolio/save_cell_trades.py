"""Save cell trades — PORTED to the monthly cell-sharded cache + qs_core.

Pipeline: cell files (build_cell_files) -> top-N configs per cell (CELL_PARAMS)
-> per (code, direction, cell) load the monthly cache, apply gate+block (new
phase2) + QS quintile (CANONICAL qs_core, what the bot trades) -> extract trade
list -> per-cell per-year pickles (trade_pickles/{CELL}_{YEAR}.pkl).

Loading & worker admission mirror qs_to_cellcomp.py: monthly load_lite
(skip_trades=False for entry/exit/coin), MEASURED-RAM event-driven admission
(est-lag reserve, cpu cap), per-task tmp for crash-safe resume.

  python save_cell_trades.py                 # all cells, all years
  python save_cell_trades.py --year 2026     # refresh only 2026
"""
import sys, os, time, pickle, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.data.registry import STRATEGIES, load_builder, BACKTEST
from factory.gates.features_common import ALL_COL_NAMES
from factory.qs.qs_features import _qs_cell_est
from factory.qs.qs_core import qs_avg_rank_pct
from factory.qs.age_filter import age_ok_mask   # min-age coin filter (combo-search)

_WORK = Path(os.environ.get('TM_WORK', ROOT))  # isolated workdir (parallel arms)
CELLS_DIR = _WORK / 'cells'
OUT_DIR = _WORK / 'trade_pickles'

# Optional: restrict the QS-rank stream to the live coin_whitelist (env SCT_WHITELIST=1).
# Default OFF -> min-age universe (combo-search basis). ON -> matches the live bot's
# whitelist universe so the backtest QS-rank == the live QS-rank (reconcile parity).
_WHITELIST = None
if os.environ.get('SCT_WHITELIST'):
    _wlf = ROOT / 'config' / 'coin_whitelist.txt'
    _WHITELIST = frozenset(l.strip() for l in open(_wlf, encoding='utf-8') if l.strip())
    print(f'  [SCT_WHITELIST] QS stream restricted to {len(_WHITELIST)} whitelist coins', flush=True)
QS_DIRS = [ROOT / 'strategies']   # one dir per category (toy set: single)

TREND_MAP = {'BEAR': 0, 'FLAT': 1, 'BULL': 2}
VOL_MAP = {'LOW': 0, 'MED': 1, 'HIGH': 2}
N_ROLL = 100

# Matches trade_combo CELL_PARAMS exactly (top-40, 2/slot per direction)
# so save provides precisely the config pool combo_test looks up in pickles.
CELL_PARAMS = {c: (40, 2, 40, 2) for c in
               ('BEAR_HIGH', 'BEAR_MED', 'BEAR_LOW', 'FLAT_HIGH', 'FLAT_MED',
                'FLAT_LOW', 'BULL_HIGH', 'BULL_MED', 'BULL_LOW')}
ALL_CELLS = ['BEAR_HIGH', 'BEAR_MED', 'BEAR_LOW', 'FLAT_HIGH', 'FLAT_MED',
             'FLAT_LOW', 'BULL_HIGH', 'BULL_MED', 'BULL_LOW']


# ── Parse cell files -> top-N configs per cell ──────────────────────
def parse_all_configs(cells=None):
    """cells/*.txt (build_cell_files) -> {(code,direction,cell): [(label,q,ci,sl,tp)]}.
    Top-N by score with per-(direction,strat) slot caps from CELL_PARAMS.

    Second return is the {(cell,label)} valid set. A label ({dir}:{strat}_q{q}_c{ci})
    is NOT cell-unique -- several cells legitimately carry the same (strat,q,ci)
    index, each a DIFFERENT config. The old {label: cell} map here was last-wins,
    so aggregation piled every cell's same-label trades into ONE cell file (8% of
    all trades landed in the wrong cell's file and 19 configs vanished from their
    own cell's combo pool)."""
    cells = cells or ALL_CELLS
    by_task = defaultdict(list)
    valid_labels = set()
    for cell_name in cells:
        l_tn, l_tps, s_tn, s_tps = CELL_PARAMS.get(cell_name, (25, 2, 25, 2))
        for prefix in ['L', 'S']:
            direction = 'LONG' if prefix == 'L' else 'SHORT'
            tn = l_tn if prefix == 'L' else s_tn
            tps = l_tps if prefix == 'L' else s_tps
            fpath = CELLS_DIR / f'{prefix}_{cell_name}.txt'
            if not fpath.exists():
                continue
            configs = []
            for line in open(fpath, encoding='utf-8'):
                parts = line.strip().split('|')
                if len(parts) < 4:
                    continue
                hdr = parts[0].split()
                if len(hdr) < 6:
                    continue
                try:
                    int(hdr[0])
                    cat, strat, sl_tp = hdr[1], hdr[2], hdr[3]
                    q, ci = int(hdr[4]), int(hdr[5])
                    score = float(parts[3].strip().split()[0])
                except (ValueError, IndexError):
                    continue
                configs.append((score, strat, sl_tp, q, ci))
            configs.sort(key=lambda x: -x[0])
            slots = defaultdict(int)
            for score, strat, sl_tp, q, ci in configs[:tn]:
                if slots[strat] >= tps:
                    continue
                slots[strat] += 1
                sl = float(sl_tp.split('/')[0])
                tp = float(sl_tp.split('/')[1].rstrip('*'))
                label = f'{prefix}:{strat}_q{q}_c{ci}'
                by_task[(strat, direction, cell_name)].append((label, q, ci, sl, tp))
                valid_labels.add((cell_name, label))
    return by_task, valid_labels


def load_qs_features():
    qs = {}
    for d in QS_DIRS:
        f = d / 'unified_qs_features.json'
        if f.exists():
            qs.update(json.load(open(f)))
    return qs


def _phase2_path(code):
    sd = BACKTEST / Path(STRATEGIES[code][0]).parent
    return sd / f'{sd.name}_phase2_results.json'


# ── QS quintile via canonical qs_core (identical to qs_to_cellcomp) ──
def _qs_mask(features, pnl, qs_feats, ic_map, q, entry_times=None):
    resolved = np.isfinite(pnl) & (pnl != 0)
    avg = qs_avg_rank_pct(features, qs_feats, ic_map, ALL_COL_NAMES, N_ROLL,
                          entry_times=entry_times)
    if avg is None:
        return resolved
    cutoff = 100.0 * (1 - 1.0 / q)
    return resolved & np.isfinite(avg) & (avg >= cutoff)


def _gate_block_mask(features, pnl, p2cfg):
    resolved = np.isfinite(pnl) & (pnl != 0)
    gp = resolved.copy()
    for g in p2cfg.get('gate', []):
        if g['name'] not in ALL_COL_NAMES:
            continue
        col = features[:, ALL_COL_NAMES.index(g['name'])].astype(np.float64)
        v = np.isfinite(col)
        gp &= (v & (col > g['value'])) if g['direction'] == 'gt' else (v & (col < g['value']))
    for b in p2cfg.get('blocks', []):
        if b['name'] not in ALL_COL_NAMES:
            continue
        col = features[:, ALL_COL_NAMES.index(b['name'])].astype(np.float64)
        v = np.isfinite(col)
        gp &= ~(v & (col < b['value'])) if b['block_op'] == '<' else ~(v & (col > b['value']))
    return gp


def _process_cell(code, direction, cell, cfgs, cell_data, phase2, qs_data):
    """All configs for one (code, direction, cell) -> {label: [trade dicts]}."""
    features = cell_data['features']
    et_ns, xt_ns, coins = cell_data['et_ns'], cell_data['xt_ns'], cell_data['coins']
    pnl_arrays = cell_data['pnl']
    p2_list = phase2.get(f'{direction}_{cell}', [])
    age_ok = age_ok_mask(coins, pd.to_datetime(et_ns))  # 12mo+ universe (matches qs_to_cellcomp)
    if _WHITELIST is not None:
        age_ok = age_ok & np.array([c in _WHITELIST for c in coins])   # + live-whitelist universe
    out = {}
    for label, q, ci, sl, tp in cfgs:
        pnl = pnl_arrays.get(f'pnl_{sl}_{tp}')
        if pnl is None:
            pnl = pnl_arrays.get(f'pnl_{sl}')
        if pnl is None or ci >= len(p2_list):
            continue
        gp = _gate_block_mask(features, pnl, p2_list[ci])
        gp &= age_ok   # 12mo+ coin-age (combo-search universe)
        if gp.sum() < 50:
            continue
        # qs_core rolling-rank assumes a CHRONOLOGICAL gate-passed stream; load_lite
        # rows are coin-major, so the gate-passed indices must be sorted by entry
        # time before QS or the rolling window mixes coins/times -> garbage quintile
        # (== the live bot, which accumulates the stream bar-by-bar in time order).
        gp_idx = np.where(gp)[0]
        gp_idx = gp_idx[np.argsort(et_ns[gp_idx], kind='stable')]
        if q > 1:
            key = f'{direction}_{cell}_{code}_q{q}_c{ci}'
            e = qs_data.get(key)
            if e:
                sel = _qs_mask(features[gp_idx], pnl[gp_idx], e['features'], e['ic'], q,
                               entry_times=et_ns[gp_idx])
            else:
                p = pnl[gp_idx]
                sel = np.isfinite(p) & (p != 0)
        else:
            p = pnl[gp_idx]
            sel = np.isfinite(p) & (p != 0)
        idx = gp_idx[sel]
        trades = []
        for i in idx:
            if et_ns[i] == 0 or xt_ns[i] == 0:
                continue
            trades.append({'entry_time': pd.Timestamp(et_ns[i]),
                           'exit_time': pd.Timestamp(xt_ns[i]),
                           'coin': coins[i] if coins[i] else '',
                           'pnl_pct': float(pnl[i]),
                           'direction': direction, 'cell': cell})
        out[label] = trades
    return out


def _build_cell_data(d, cfgs):
    trades = d.get('trades', [])
    n = len(d['features'])
    et = np.zeros(n, dtype=np.int64)
    xt = np.zeros(n, dtype=np.int64)
    co = np.empty(n, dtype=object); co[:] = ''
    for i in range(min(len(trades), n)):
        tr = trades[i]
        if isinstance(tr, dict):
            e, x = tr.get('entry_time'), tr.get('exit_time')
            if e is not None: et[i] = e.value
            if x is not None: xt[i] = x.value
            co[i] = tr.get('coin', '')
    pnl = {}
    for _l, _q, _ci, sl, tp in cfgs:
        for pk in (f'pnl_{sl}_{tp}', f'pnl_{sl}'):
            if pk not in pnl and pk in d:
                pnl[pk] = d[pk]
    return {'features': d['features'], 'et_ns': et, 'xt_ns': xt, 'coins': co, 'pnl': pnl}


def _cell_task(args):
    code, direction, cell, cfgs, phase2, qs_code, q_sig, tid, target_year = args
    t = TREND_MAP[cell.split('_')[0]]; v = VOL_MAP[cell.split('_')[1]]
    yrs = ([target_year - 1, target_year] if target_year and target_year > 2021
           else ([target_year] if target_year else None))
    pnl_keys = set()
    for _l, _q, _ci, sl, tp in cfgs:
        pnl_keys.add(f'pnl_{sl}_{tp}'); pnl_keys.add(f'pnl_{sl}')
    try:
        cache = load_builder(code)._instance.load_lite(
            direction, years=yrs, cell=(t, v), skip_trades=False, pnl_keys=pnl_keys)
    finally:
        if q_sig is not None:
            q_sig.put(('loaded', tid))
    if cache is None:
        return {}
    d = cache.get(direction)
    if d is None or d.get('features', np.array([])).size == 0:
        return {}
    cell_data = _build_cell_data(d, cfgs)
    del d['trades']
    return _process_cell(code, direction, cell, cfgs, cell_data, phase2, qs_code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('cells', nargs='*')
    args = ap.parse_args()
    cells = [c for c in args.cells if c in ALL_CELLS] or None
    target_year = args.year

    import psutil, multiprocessing, queue as _queue
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from concurrent.futures.process import BrokenProcessPool

    n_workers = args.workers or max(1, (os.cpu_count() or 8) - 2)
    RESERVE = 3e9
    t0 = time.time()
    print('=' * 100, flush=True)
    print(f'  SAVE CELL TRADES (monthly cache + qs_core) | cells={cells or "ALL"} '
          f'| year={target_year or "ALL"} | cpu cap {n_workers}', flush=True)

    by_task, valid_labels = parse_all_configs(cells)
    qs_all = load_qs_features()
    phase2_cache = {}
    n_cfg = sum(len(v) for v in by_task.values())
    print(f'  {len(by_task)} (code,dir,cell) tasks, {n_cfg} configs, '
          f'{len(qs_all)} QS entry', flush=True)

    TMP = OUT_DIR / '_tmp'
    TMP.mkdir(parents=True, exist_ok=True)

    def tmp_path(code, direction, cell):
        return TMP / f'{code}__{direction}__{cell}.pkl'

    # build pending tasks (skip done -> resume)
    pending, n_skip = deque(), 0
    for (code, direction, cell), cfgs in by_task.items():
        if code not in STRATEGIES:
            print(f'  [{code}] not in registry, skipped', flush=True)
            continue
        if tmp_path(code, direction, cell).exists():
            n_skip += 1
            continue
        if code not in phase2_cache:
            pp = _phase2_path(code)
            phase2_cache[code] = json.load(open(pp)) if pp.exists() else {}
        p2 = phase2_cache[code]
        qs_code = {k: v for k, v in qs_all.items() if f'_{code}_q' in k}
        est = _qs_cell_est(code, cell)
        pending.append((code, direction, cell, cfgs, p2, qs_code, est))
    total = len(pending) + n_skip
    print(f'  {total} tasks ({n_skip} skipped from disk), measured-RAM admission', flush=True)

    done_n, tid_seq = n_skip, 0
    while pending:
        running, loading_est, broke = {}, {}, False
        mgr = multiprocessing.Manager(); q = mgr.Queue()
        ex = ProcessPoolExecutor(max_workers=n_workers)
        try:
            while pending or running:
                if pending and len(running) < n_workers:
                    avail = psutil.virtual_memory().available
                    reserved = sum(loading_est.values())
                    m = pending[0]
                    if (avail - RESERVE - reserved) > m[6] or not running:
                        pending.popleft()
                        tid_seq += 1
                        fut = ex.submit(_cell_task, (m[0], m[1], m[2], m[3], m[4],
                                                     m[5], q, tid_seq, target_year))
                        running[fut] = (m, tid_seq)
                        loading_est[tid_seq] = m[6]
                        continue
                try:
                    while True:
                        ev = q.get_nowait()
                        if isinstance(ev, tuple) and ev[0] == 'loaded':
                            loading_est.pop(ev[1], None)
                except _queue.Empty:
                    pass
                if not running:
                    if not pending:
                        break
                    continue
                fini, _ = wait(running, timeout=2, return_when=FIRST_COMPLETED)
                for fut in fini:
                    m, tid = running.pop(fut)
                    loading_est.pop(tid, None)
                    code, direction, cell = m[0], m[1], m[2]
                    res = fut.result()
                    pickle.dump(res, open(tmp_path(code, direction, cell), 'wb'), protocol=4)
                    done_n += 1
                    nt = sum(len(v) for v in res.values())
                    print(f'  {code}:{direction[0]}_{cell:<10} -> {len(res)} cfg '
                          f'{nt:,} trade [{done_n}/{total}] '
                          f'({(time.time()-t0)/60:.0f}dk)', flush=True)
        except BrokenProcessPool:
            broke = True
            for m, tid in running.values():
                pending.appendleft(m)
            print(f'  !! POOL DIED — {len(running)} tasks requeued, pool rebuilt', flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True); mgr.shutdown()
        if not broke:
            break

    # aggregate tmp -> per-cell per-year pickles
    print(f'\n  Pickle yaziliyor...', flush=True)
    all_ct = defaultdict(lambda: defaultdict(dict))
    for tp in TMP.glob('*.pkl'):
        # tmp file = one (code, direction, cell) task -> the CELL comes from the
        # filename, not from the label (labels collide across cells; routing by a
        # label->cell map merged different cells' same-label configs into one file).
        cell = tp.stem.split('__')[2]
        for label, trades in pickle.load(open(tp, 'rb')).items():
            if (cell, label) not in valid_labels:
                continue
            for tr in trades:
                yr = tr['entry_time'].year
                if target_year and yr != target_year:
                    continue
                all_ct[cell][yr].setdefault(label, []).append(tr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nf = 0
    for cell in sorted(all_ct):
        for yr in sorted(all_ct[cell]):
            ct = all_ct[cell][yr]
            op = OUT_DIR / f'{cell}_{yr}.pkl'
            pickle.dump(ct, open(op, 'wb'), protocol=4)
            nt = sum(len(v) for v in ct.values())
            print(f'    {cell}_{yr}: {len(ct)} cfg, {nt:,} trade '
                  f'({op.stat().st_size/1e6:.1f}MB)', flush=True)
            nf += 1
    print(f'\n  DONE: {nf} dosya ({(time.time()-t0)/60:.1f}dk) — tmp: {TMP}', flush=True)


if __name__ == '__main__':
    main()
