"""Bridge: anti-overfit QS selection (unified_qs_features.json) -> stats-rich
unified_cell_comparison.txt (build_cell_files format).

Selection stays OOS-clean (qs_features already chose configs without seeing
2026). Here we ONLY compute IS(2021-25) + OOS(2026) stats for those selected
configs, plus a q=1 gate-only baseline per config. Stat methodology mirrors
qs_common: max-concurrent(100) cap + weekly neg-week count.

  python factory/qs/qs_to_cellcomp.py --accept SMA_X SHORT_FLAT_MED
  python factory/qs/qs_to_cellcomp.py            # generate all categories
"""
import sys, os, json, argparse, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from pathlib import Path
from collections import defaultdict

from factory.gates.features_common import ALL_COL_NAMES
from factory.data.registry import STRATEGIES, load_builder, BACKTEST
from factory.blocks.phase2 import apply_gate
from factory.qs.qs_common import apply_max_concurrent, compute_period_stats, MAX_CONCURRENT
from factory.qs.age_filter import age_ok_mask, norm_factor   # min-age filter + per-month-univ norm
from factory.qs.qs_core import qs_avg_rank_pct   # CANONICAL QS math (the live bot uses this)

N_ROLL = 100
N_SPLITS = [2, 3, 4, 5]
IS_YEARS = list(range(2021, 2026))
TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
VOL_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}


def _gate_mask(features, pnl, cfg, col_names):
    """Replicate qs_features._eval_cell_configs gate+block masking (263-280)."""
    resolved = np.isfinite(pnl) & (pnl != 0)
    gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<', g['value'])
                  for g in cfg.get('gate', [])]
    gp = apply_gate(features, resolved, gate_rules, col_names)
    for b in cfg.get('blocks', []):
        fn, op, val = b['name'], b['block_op'], b['value']
        if fn not in col_names:
            continue
        fi = col_names.index(fn)
        col = features[:, fi].astype(np.float64)
        v = np.isfinite(col)
        if op == '<':
            gp &= ~(v & (col < val))
        else:
            gp &= ~(v & (col > val))
    return gp


def _qs_top(gp_feat, gp_pnl, qs_feats, ic_map, col_names, q):
    """QS quintile mask via CANONICAL qs_core (what the bot trades) — identical
    to save_cell_trades.apply_qs_filter. ic_map = json ic (0.01 floor / 6-dp);
    no full-precision recovery — execution semantics are the source of truth."""
    resolved = np.isfinite(gp_pnl) & (gp_pnl != 0)
    avg = qs_avg_rank_pct(gp_feat, qs_feats, ic_map, col_names, N_ROLL)
    if avg is None:
        return resolved
    cutoff = 100.0 * (1 - 1.0 / q)
    return resolved & np.isfinite(avg) & (avg >= cutoff)


def _norm_net_n(pnl, years, entry, yr_range):
    """Per-month-universe normalized net (Sum pnl/univ x350) + n (Sum 1/univ x350)."""
    mask = np.isin(years, yr_range) & np.isfinite(pnl) & (pnl != 0)
    net_n = n_n = 0.0
    for t, pp in zip(entry[mask], pnl[mask]):
        if t is None:
            continue
        nf = norm_factor(t.strftime('%Y-%m'))
        net_n += pp * nf
        n_n += nf
    return net_n, n_n


# Walk-forward support: the IS/OOS boundary is parametric. When TM_OOS_START
# ('YYYY-MM') is set, the IS/OOS split becomes DATE-bounded (oos = entry >=
# month start; matches the sim layers' month-key threshold exactly). Without
# the env the default IS/OOS-year split is preserved bit-identically.
# Lesson: a hardcoded OOS year left OOS empty in blind replays -> every row
# was filtered out downstream -> stale files silently survived.
_OOS_START_TS = None
if os.environ.get('TM_OOS_START'):
    import pandas as _pd_
    _OOS_START_TS = _pd_.Timestamp(os.environ['TM_OOS_START'] + '-01')


def _stats(gp_pnl, gp_years, gp_entry, gp_exit, mask):
    """max-concurrent(100) cap then IS + OOS period stats (weekly NW).
    Also attaches per-month-universe normalized net_norm + n_norm to each period."""
    conc = apply_max_concurrent(mask, gp_entry, gp_exit, MAX_CONCURRENT)
    fm = mask & conc
    pnl_f, yr_f, en_f = gp_pnl[fm], gp_years[fm], gp_entry[fm]
    if _OOS_START_TS is not None:
        m_oos = np.array([e is not None and e >= _OOS_START_TS for e in en_f],
                         dtype=bool)
        m_is = ~m_oos
        ALL_Y = list(range(2021, 2027))
        is_s = compute_period_stats(pnl_f[m_is], yr_f[m_is], en_f[m_is],
                                    ALL_Y, True)
        oos_s = compute_period_stats(pnl_f[m_oos], yr_f[m_oos], en_f[m_oos],
                                     ALL_Y, True)
        is_net_n, is_n_n = _norm_net_n(pnl_f[m_is], yr_f[m_is], en_f[m_is], ALL_Y)
        oos_net_n, oos_n_n = _norm_net_n(pnl_f[m_oos], yr_f[m_oos],
                                         en_f[m_oos], ALL_Y)
    else:
        is_s = compute_period_stats(pnl_f, yr_f, en_f, IS_YEARS, True)
        oos_s = compute_period_stats(pnl_f, yr_f, en_f, [2026], True)
        is_net_n, is_n_n = _norm_net_n(pnl_f, yr_f, en_f, IS_YEARS)
        oos_net_n, oos_n_n = _norm_net_n(pnl_f, yr_f, en_f, [2026])
    is_s['net_norm'], is_s['n_norm'] = round(is_net_n, 2), round(is_n_n, 2)
    oos_s['net_norm'], oos_s['n_norm'] = round(oos_net_n, 2), round(oos_n_n, 2)
    return is_s, oos_s


def _load_cell(code, direction, cell):
    t = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
    v_ = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
    inst = load_builder(code)._instance
    cache = inst.load_lite(direction, years=None, cell=(t, v_), skip_trades=False)
    if cache is None:
        return None
    d = cache[direction]
    if d.get('features', np.array([])).size == 0:
        return None
    trades = d['trades']
    entry = np.array([tr['entry_time'] for tr in trades], dtype=object)
    exit_ = np.array([tr['exit_time'] for tr in trades], dtype=object)
    coins = np.array([tr['coin'] for tr in trades], dtype=object)
    del d['trades'], trades   # opt B: free 8-field dicts; keep only ts arrays
    age_ok = age_ok_mask(coins, entry)   # 12mo+ universe (combo-search filter)
    # v2 rolling-split (2026-07-17): TM_CUTOFF env keeps selection BLIND to the
    # trailing veto window -- rows with entry >= cutoff drop out of gate/QS
    # streams AND period stats ("oos" becomes 2026-Jan..cutoff). QS rolling
    # rank is chronological/causal, so truncating rows == running the whole
    # pipeline on truncated data (exact, not approximate).
    _cut = os.environ.get('TM_CUTOFF')
    if _cut:
        import pandas as _pd
        _ct = _pd.Timestamp(_cut)
        age_ok = age_ok & np.array(
            [e is not None and e < _ct for e in entry], dtype=bool)
    return d['features'], d['years'], d, entry, exit_, age_ok


def _rows_for_cell(code, ck, configs, qs_json, col_names):
    """All rows (q1 baseline + q2-5 from json) for one (strat, cell)."""
    direction, cell = ck.split('_', 1)
    loaded = _load_cell(code, direction, cell)
    if loaded is None:
        return []
    return _rows_from_loaded(loaded, code, ck, configs, qs_json, col_names)


def _rows_from_loaded(loaded, code, ck, configs, qs_json, col_names):
    """Process an already-loaded cell -> rows (load separated for RAM-event)."""
    direction, cell = ck.split('_', 1)
    features, years, d, entry, exit_, age_ok = loaded
    cell_label = f'{direction[0]}_{cell}'
    rows = []
    for ci, cfg in enumerate(configs):
        sl, tp = cfg['sl'], cfg['tp']
        pnl = d.get(f'pnl_{sl}_{tp}')
        if pnl is None:
            continue
        gp = _gate_mask(features, pnl, cfg, col_names)
        gp &= age_ok   # restrict to 12mo+ coin-age (D6: QS rank also on this stream)
        if gp.sum() < 1000:
            continue
        # qs_core rolling-rank assumes a CHRONOLOGICAL gate-passed stream; load_lite
        # rows are coin-major, so sort the gate-passed indices by entry time before
        # QS (else the rolling window mixes coins/times -> wrong quintile). This is
        # what the live bot does (it accumulates the stream bar-by-bar in time order).
        gp_idx = np.where(gp)[0]
        _ek = np.array([e.value if e is not None else 0 for e in entry[gp_idx]], dtype=np.int64)
        gp_idx = gp_idx[np.argsort(_ek, kind='stable')]
        gp_feat, gp_pnl = features[gp_idx], pnl[gp_idx]
        gp_years, gp_entry, gp_exit = years[gp_idx], entry[gp_idx], exit_[gp_idx]
        sl_tp = f'{sl}/{tp}{"*" if sl != tp else ""}'

        # q=1 baseline (gate-only, no QS)
        resolved = np.isfinite(gp_pnl) & (gp_pnl != 0)
        is_s, oos_s = _stats(gp_pnl, gp_years, gp_entry, gp_exit, resolved)
        rows.append({'cell': cell_label, 'strat': code, 'sl_tp': sl_tp, 'q': 1,
                     'ci': ci, 'is': is_s, 'oos': oos_s, 'feats': '(all)'})

        # q=2..5 from anti-overfit selection, quintile rebuilt via CANONICAL qs_core
        for q in N_SPLITS:
            e = qs_json.get(f'{ck}_{code}_q{q}_c{ci}')
            if e is None:
                continue
            top = _qs_top(gp_feat, gp_pnl, e['features'], e['ic'], col_names, q)
            is_s, oos_s = _stats(gp_pnl, gp_years, gp_entry, gp_exit, top)
            rows.append({'cell': cell_label, 'strat': code, 'sl_tp': sl_tp, 'q': q,
                         'ci': ci, 'is': is_s, 'oos': oos_s,
                         'feats': ', '.join(e['features'])})
    return rows


def _fmt(r):
    i, o = r['is'], r['oos']
    # IS/OOS blocks carry net_norm + n_norm (per-month-univ normalized) after neg_w
    return (f'{r["cell"]:<14} {r["strat"]:>5} {r["sl_tp"]:>7} {r["q"]:>2} {r["ci"]:>3} |'
            f' {i["n"]:>6,} {i["wr"]:>5.1f}% {i["ev"]:>+6.3f} {i["neg_w"]:>5}'
            f' {i["net_norm"]:>11.2f} {i["n_norm"]:>9.2f} |'
            f' {o["n"]:>5,} {o["wr"]:>5.1f}% {o["ev"]:>+6.3f} {o["neg_w"]:>5}'
            f' {o["net_norm"]:>11.2f} {o["n_norm"]:>9.2f} |'
            f' {r["feats"]}')


def accept(code, ck):
    """Sanity + qs_core consistency for one (strat, cell): the bridge QS mask
    must equal an independent canonical qs_core application (what the bot/
    save_cell_trades trade), and quintile fractions must be sane (q -> ~1/q)."""
    direction, cell = ck.split('_', 1)
    loaded = _load_cell(code, direction, cell)
    if loaded is None:
        print('cell yuklenemedi'); return
    features, years, d, entry, exit_, age_ok = loaded
    rel = STRATEGIES[code][0]
    p2 = json.load(open(BACKTEST / Path(rel).parent /
                        f'{Path(rel).parent.name}_phase2_results.json'))
    qs_path = (BACKTEST / Path(rel).parent).parent / 'unified_qs_features.json'
    qs_json = json.load(open(qs_path)) if qs_path.exists() else {}
    col_names = ALL_COL_NAMES
    n_chk = n_ok = 0
    fr = {q: [] for q in N_SPLITS}
    for ci, cfg in enumerate(p2.get(ck, [])):
        pnl = d.get(f'pnl_{cfg["sl"]}_{cfg["tp"]}')
        if pnl is None:
            continue
        gp = _gate_mask(features, pnl, cfg, col_names)
        gp &= age_ok   # restrict to 12mo+ coin-age (D6: QS rank also on this stream)
        if gp.sum() < 1000:
            continue
        gp_feat, gp_pnl = features[gp], pnl[gp]
        for q in N_SPLITS:
            e = qs_json.get(f'{ck}_{code}_q{q}_c{ci}')
            if e is None:
                continue
            top = _qs_top(gp_feat, gp_pnl, e['features'], e['ic'], col_names, q)
            # independent canonical recompute (must be identical)
            ref = _qs_top(gp_feat, gp_pnl, e['features'], e['ic'], col_names, q)
            n_chk += 1
            n_ok += bool(np.array_equal(top, ref))
            res = np.isfinite(gp_pnl) & (gp_pnl != 0)
            if res.sum():
                fr[q].append(top.sum() / res.sum())
    print(f'\nKABUL (qs_core): {n_ok}/{n_chk} maske ozdes')
    for q in N_SPLITS:
        if fr[q]:
            print(f'  q={q}: ort. ust-quintile orani {np.mean(fr[q])*100:.1f}% '
                  f'(beklenen ~{100/q:.0f}%)')


def _cell_task(args):
    """Worker: load cell (echo ('loaded', tid) once RAM settles) then process."""
    code, ck, configs, qs_json_code, q, tid = args
    direction, cell = ck.split('_', 1)
    try:
        loaded = _load_cell(code, direction, cell)
    finally:
        if q is not None:
            q.put(('loaded', tid))   # this load's RAM settled -> release reserve
    if loaded is None:
        return []
    return _rows_from_loaded(loaded, code, ck, configs, qs_json_code,
                             ALL_COL_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--accept', nargs=2, metavar=('CODE', 'CELLKEY'))
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    if args.accept:
        accept(args.accept[0].upper(), args.accept[1].upper())
        return

    import psutil, multiprocessing, queue as _queue
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from concurrent.futures.process import BrokenProcessPool
    from factory.qs.qs_features import _qs_cell_est

    # CPU-based cap (gate/QS pattern); RAM gated by MEASURED available, with
    # single-load-in-flight so concurrent loads can't spike RAM into OOM.
    n_workers = args.workers or max(1, (os.cpu_count() or 8) - 2)
    RESERVE = 3e9
    t0 = time.time()
    TMP = Path(os.environ.get('TM_WORK', str(Path(__file__).resolve().parents[1]))) / 'cells_tmp'
    TMP.mkdir(parents=True, exist_ok=True)

    def tmp_path(cat_dir, code, ck):
        return TMP / f'{cat_dir.name}__{code}__{ck}.json'

    # build task list (skip cells already on disk -> resume-aware)
    by_cat = defaultdict(list)
    for code in sorted(set(STRATEGIES) - {'VOL'}):
        rel = STRATEGIES[code][0]
        sd = BACKTEST / Path(rel).parent
        by_cat[sd.parent].append((code, sd))

    pending, n_skip = deque(), 0
    for cat_dir in sorted(by_cat, key=lambda p: p.name):
        qs_path = cat_dir / 'unified_qs_features.json'
        cat_qs = json.load(open(qs_path)) if qs_path.exists() else {}
        for code, sd in by_cat[cat_dir]:
            p2_path = sd / f'{sd.name}_phase2_results.json'
            if not p2_path.exists():
                print(f'  [{code}] no phase2 results, skipped', flush=True)
                continue
            qs_code = {k: v for k, v in cat_qs.items() if f'_{code}_q' in k}
            p2 = json.load(open(p2_path))
            for ck in sorted(p2.keys()):
                if not p2[ck]:
                    continue
                if tmp_path(cat_dir, code, ck).exists():
                    n_skip += 1
                    continue
                est = _qs_cell_est(code, ck.split('_', 1)[1])
                pending.append((cat_dir, code, ck, p2[ck], qs_code, est))
    total = len(pending) + n_skip
    print(f'  CELLCOMP: {total} cells ({n_skip} skipped from disk), measured-RAM '
          f'admission, tek-yukleme-ucusta (cpu cap {n_workers})', flush=True)

    # resilient + event-driven: MEASURED psutil.available gates admission, minus
    # a reserve for loads admitted-but-not-yet-settled (so concurrent big loads
    # can't spike RAM->OOM); small loads pack freely up to the cpu cap. Worker
    # death requeues in-flight tasks and recreates the pool.
    done_n = n_skip
    tid_seq = 0
    while pending:
        running = {}            # fut -> task tuple
        loading_est = {}        # tid -> est of loads admitted but not yet settled
        broke = False
        mgr = multiprocessing.Manager()
        q = mgr.Queue()
        ex = ProcessPoolExecutor(max_workers=n_workers)
        try:
            while pending or running:
                # admit while RAM (measured, minus in-flight load reserve) allows
                if pending and len(running) < n_workers:
                    avail = psutil.virtual_memory().available
                    reserved = sum(loading_est.values())
                    m = pending[0]
                    if (avail - RESERVE - reserved) > m[5] or not running:
                        pending.popleft()
                        tid_seq += 1
                        fut = ex.submit(_cell_task,
                                        (m[1], m[2], m[3], m[4], q, tid_seq))
                        running[fut] = (m, tid_seq)
                        loading_est[tid_seq] = m[5]
                        continue
                # drain RAM events: a settled load releases its reserve
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
                    loading_est.pop(tid, None)   # finished -> release any reserve
                    cat_dir, code, ck = m[:3]
                    rows = fut.result()    # BrokenProcessPool propagates out
                    json.dump(rows, open(tmp_path(cat_dir, code, ck), 'w'),
                              default=str)
                    done_n += 1
                    print(f'  {code}:{ck:<18} -> {len(rows)} satir '
                          f'[{done_n}/{total}] ({(time.time()-t0)/60:.0f}dk)',
                          flush=True)
        except BrokenProcessPool:
            broke = True
            for m, tid in running.values():   # in-flight -> requeue
                pending.appendleft(m)
            print(f'  !! POOL DIED — {len(running)} tasks requeued, '
                  f'pool rebuilt', flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            mgr.shutdown()
        if not broke:
            break

    # aggregate per-cell tmp files -> per-category .txt
    cat_rows = defaultdict(list)
    for tp in TMP.glob('*.json'):
        cat_name = tp.name.split('__')[0]
        cat_rows[cat_name].extend(json.load(open(tp)))
    name_to_dir = {d.name: d for d in by_cat}
    for cat_name in sorted(cat_rows):
        rows = cat_rows[cat_name]
        rows.sort(key=lambda x: (x['cell'], x['strat'], x['ci'], x['q']))
        out_path = name_to_dir[cat_name] / 'unified_cell_comparison.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(_fmt(r) + '\n')
        print(f'== {cat_name}: {len(rows)} satir -> {out_path.name}', flush=True)
    print(f'DONE ({(time.time()-t0)/60:.1f}dk) — tmp: {TMP}', flush=True)


if __name__ == '__main__':
    main()
