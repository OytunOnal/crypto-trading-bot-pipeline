"""Common QS feature-selection engine — anti-overfit redesign (2026-06-12).

NOT a pure port. The legacy flow (qs_common.process_one_cell_config) had
three overfit holes, all fixed here:
  LEAK : rows were filtered on OOS-2026 EV >= 0  -> 2026 NEVER touches
         selection here; it is reported, never filtered on
  STAB : sign stability was >=2/5 years          -> >=4/5 required
  LIFT : top-q lift checked in-sample only       -> WF gate: lift > 0 in
         >=2/3 FWD val years {2023,2024,2025} AND >=2/3 REV val years
         {2021,2022,2023}; fails either -> NOT published (q=1 fallback)
  ABS  : top-q stream must ALSO be absolutely positive (EV>0) in every
         IS year with n>=10 -- a variant test showed all winners already
         satisfied it (free insurance); stricter WF/stab gates were
         REJECTED by the same test (forward performance got worse)
Weighting is also validated, not assumed: |IC| weights vs equal weights
compete in the same WF gate; equal weights are encoded as ic=±0.01 so the
live scorer's max(|ic|, 0.01) floor makes them equal — zero live changes.

Ranks use the canonical qs_core math (live parity by construction).

Output contract (unchanged): merged into <category>/unified_qs_features.json
  key  '{D}_{CELL}_{STRAT}_q{q}_c{ci}'
  val  {'features': [...], 'ic': {name: signed}, 'sl': x, 'tp': y}

Usage:
  python factory/qs/qs_features.py --strategy SMA_X
"""
import sys, os, time, gc, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.gates.features_common import ALL_COL_NAMES, _spearman
from factory.data.registry import STRATEGIES, load_builder, BACKTEST
from factory.blocks.phase2 import apply_gate
from factory.qs.qs_core import rolling_rank_exclude_self

EXCLUDED_FEATURES = set()   # feature names to hard-exclude from QS pool
TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
VOL_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}

IS_YEARS = list(range(2021, 2026))
OOS_YEAR = 2026
FWD_VAL_YEARS = [2023, 2024, 2025]   # forward walk-forward validation years
REV_VAL_YEARS = [2021, 2022, 2023]   # backwards (reverse) validation years
WF_MIN_PASS = 2                      # required in EACH direction

IC_MIN = 0.010
SIGN_STAB_MIN = 4          # >=4/5 years same IC sign (legacy was 2)
CORR_THRESH = 0.60
MAX_QS_INDEP = 8
QS_K_RANGE = range(2, 6)   # 2..5 features per combo
N_SPLITS = [2, 3, 4, 5]
N_ROLL = 100
MIN_LIFT = 0.01
WEIGHT_SCHEMES = ['ic', 'eq']


def _vectorized_rank_ic(feat, pnl, mask):
    """Spearman IC per column over masked rows."""
    out = np.full(feat.shape[1], np.nan)
    p = pnl[mask]
    if len(p) < 50:
        return out
    pr = np.argsort(np.argsort(p)).astype(np.float64)
    pr -= pr.mean()
    pden = (pr * pr).sum()
    for fi in range(feat.shape[1]):
        col = feat[mask, fi].astype(np.float64)
        v = np.isfinite(col)
        if v.sum() < 50:
            continue
        cr = np.argsort(np.argsort(col[v])).astype(np.float64)
        cr -= cr.mean()
        d = np.sqrt((cr * cr).sum() * (pr[v] - pr[v].mean()).__pow__(2).sum())
        if d <= 0:
            continue
        out[fi] = float((cr * (pr[v] - pr[v].mean())).sum() / d)
    return out


def select_features(gp_feat, gp_pnl, gp_years, gate_feat_names, col_names):
    """IS-only, sign-stable (>=4/5), deduped, independent feature pool."""
    is_m = np.isin(gp_years, IS_YEARS) & np.isfinite(gp_pnl) & (gp_pnl != 0)
    if is_m.sum() < 500:
        return []
    overall = _vectorized_rank_ic(gp_feat, gp_pnl, is_m)
    yr_ics = {}
    for y in IS_YEARS:
        ym = is_m & (gp_years == y)
        if ym.sum() >= 50:
            yr_ics[y] = _vectorized_rank_ic(gp_feat, gp_pnl, ym)

    stats = []
    for fi in range(gp_feat.shape[1]):
        name = col_names[fi] if fi < len(col_names) else f'col_{fi}'
        if name in gate_feat_names or name in EXCLUDED_FEATURES:
            continue
        ic = overall[fi]
        if not np.isfinite(ic) or abs(ic) < IC_MIN:
            continue
        signs = [yr_ics[y][fi] for y in yr_ics if np.isfinite(yr_ics[y][fi])]
        if len(signs) < 4:
            continue
        stab = sum(1 for s in signs if (s > 0) == (ic > 0))
        if stab < SIGN_STAB_MIN:
            continue
        stats.append({'idx': fi, 'name': name, 'ic': float(ic),
                      'abs_ic': abs(float(ic))})

    # dedup TF variants (keep highest |IC|)
    seen = {}
    deduped = []
    for f in sorted(stats, key=lambda x: x['abs_ic'], reverse=True):
        base = f['name']
        for sf in ('_15m', '_1h', '_4h'):
            if base.endswith(sf):
                base = base[:-len(sf)]
                break
        if base not in seen:
            seen[base] = True
            deduped.append(f)

    indep = []
    for f in deduped:
        col_f = gp_feat[is_m, f['idx']].astype(np.float64)
        ok = True
        for s in indep:
            col_s = gp_feat[is_m, s['idx']].astype(np.float64)
            v = np.isfinite(col_f) & np.isfinite(col_s)
            if v.sum() < 100:
                continue
            if abs(_spearman(col_f[v], col_s[v])) > CORR_THRESH:
                ok = False
                break
        if ok:
            indep.append(f)
            if len(indep) >= MAX_QS_INDEP:
                break
    return indep


def evaluate_config(gp_feat, gp_pnl, gp_years, gate_feat_names, col_names):
    """Return {q: best_entry} for one gate-passed config stream.

    Entry only published when the WF lift gate passes — otherwise the q is
    omitted and downstream falls back to q=1 (no QS). Selection NEVER sees
    2026; OOS lift is attached for reporting only.
    """
    pool = select_features(gp_feat, gp_pnl, gp_years, gate_feat_names, col_names)
    if len(pool) < 2:
        return {}

    resolved = np.isfinite(gp_pnl) & (gp_pnl != 0)
    is_m = np.isin(gp_years, IS_YEARS) & resolved
    oos_m = (gp_years == OOS_YEAR) & resolved
    yr_masks = {y: (gp_years == y) for y in IS_YEARS}

    ranks = {}
    for f in pool:
        col = gp_feat[:, f['idx']].astype(np.float64)
        if f['ic'] < 0:
            col = -col
        ranks[f['idx']] = rolling_rank_exclude_self(col, N_ROLL)

    best = {}
    for k in QS_K_RANGE:
        if k > len(pool):
            break
        for cidx in combinations(range(len(pool)), k):
            feats = [pool[i] for i in cidx]
            for scheme in WEIGHT_SCHEMES:
                ws = [f['abs_ic'] if scheme == 'ic' else 1.0 for f in feats]
                wt = sum(ws)
                wsum = np.zeros(len(gp_pnl))
                valid = np.ones(len(gp_pnl), dtype=bool)
                for f, w in zip(feats, ws):
                    rr = ranks[f['idx']]
                    valid &= np.isfinite(rr)
                    wsum += np.where(np.isfinite(rr), rr * w, 0)
                avg_rank = wsum / wt
                pct = rolling_rank_exclude_self(avg_rank, N_ROLL)
                base_ok = valid & np.isfinite(pct) & resolved
                base_ok[:N_ROLL] = False

                for q in N_SPLITS:
                    cutoff = 100.0 * (1 - 1.0 / q)
                    top = base_ok & (pct >= cutoff)
                    rest = base_ok & ~top

                    # WF lift gate: FWD AND REV validation (IS data only)
                    def _passes(years_list):
                        n_ok = 0
                        for y in years_list:
                            yt = top & yr_masks[y]
                            yr_ = rest & yr_masks[y]
                            if yt.sum() < 30 or yr_.sum() < 100:
                                continue
                            if gp_pnl[yt].mean() > gp_pnl[yr_].mean():
                                n_ok += 1
                        return n_ok
                    fwd_pass = _passes(FWD_VAL_YEARS)
                    if fwd_pass < WF_MIN_PASS:
                        continue
                    rev_pass = _passes(REV_VAL_YEARS)
                    if rev_pass < WF_MIN_PASS:
                        continue
                    wf_pass = fwd_pass + rev_pass

                    ti = top & is_m
                    ri = rest & is_m
                    if ti.sum() < 150 or ri.sum() < 500:
                        continue
                    lift = gp_pnl[ti].mean() - gp_pnl[ri].mean()
                    if lift < MIN_LIFT:
                        continue

                    yr_lifts = []
                    abs_ok = True
                    for y in IS_YEARS:
                        yt = ti & yr_masks[y]
                        yr_ = ri & yr_masks[y]
                        if yt.sum() >= 10 and yr_.sum() >= 30:
                            yr_lifts.append(gp_pnl[yt].mean() - gp_pnl[yr_].mean())
                        # absolute positivity: the top-q stream must be EV>0
                        # on its own in EVERY IS year. A variant test showed
                        # current winners already satisfied it (behavior-neutral)
                        # -- free insurance for future selections.
                        if yt.sum() >= 10 and gp_pnl[yt].mean() <= 0:
                            abs_ok = False
                    if not abs_ok:
                        continue
                    worst_lift = min(yr_lifts) if yr_lifts else -999

                    score = (wf_pass, worst_lift, lift)
                    cur = best.get(q)
                    if cur is None or score > cur['_score']:
                        to, ro = top & oos_m, rest & oos_m
                        oos_lift = (float(gp_pnl[to].mean() - gp_pnl[ro].mean())
                                    if to.sum() >= 30 and ro.sum() >= 100 else None)
                        sign = lambda f: 1 if f['ic'] > 0 else -1
                        if scheme == 'ic':
                            ic_out = {f['name']: round(f['ic'], 6) for f in feats}
                        else:  # equal weights: ±1.0 carrier (float-exact uniform;
                               # 0.01 isn't representable -> qs_core tie reshuffle)
                            ic_out = {f['name']: float(sign(f)) for f in feats}
                        best[q] = {
                            'features': [f['name'] for f in feats],
                            'ic': ic_out, 'scheme': scheme,
                            'wf_pass': wf_pass, 'fwd': fwd_pass, 'rev': rev_pass,
                            'lift': round(float(lift), 4),
                            'worst_yr_lift': round(float(worst_lift), 4),
                            'oos_lift': oos_lift, '_score': score,
                        }
    return best


def _eval_cell_configs(d, direction, cell, code, configs, col_names):
    """One cell's configs -> ({json_key: entry}, n_cfg, log_lines)."""
    features = d['features']
    years = d['years']
    # entry timestamps for CHRONOLOGICAL QS ordering: load_lite rows are coin-major,
    # but qs_core's rolling-rank quintile assumes a time-ordered stream (== the live
    # bot, which accumulates bar-by-bar). Without this the selected QS features are
    # chosen on a garbage quintile.
    _trades = d.get('trades', [])
    _et = np.zeros(len(features), dtype=np.int64)
    for _i in range(min(len(_trades), len(features))):
        _tr = _trades[_i]
        if isinstance(_tr, dict) and _tr.get('entry_time') is not None:
            _et[_i] = _tr['entry_time'].value
    entries, lines, n_cfg = {}, [], 0
    for ci, cfg in enumerate(configs):
        sl, tp = cfg['sl'], cfg['tp']
        pnl = d.get(f'pnl_{sl}_{tp}')
        if pnl is None:
            continue
        resolved = np.isfinite(pnl) & (pnl != 0)
        gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<',
                       g['value']) for g in cfg.get('gate', [])]
        block_rules = [(b['name'], b['block_op'], b['value'])
                       for b in cfg.get('blocks', [])]
        gp = apply_gate(features, resolved, gate_rules, col_names)
        for fn, op, val in block_rules:
            if fn not in col_names:
                continue
            fi = col_names.index(fn)
            col = features[:, fi].astype(np.float64)
            v = np.isfinite(col)
            if op == '<':
                gp &= ~(v & (col < val))
            else:
                gp &= ~(v & (col > val))
        if gp.sum() < 1000:
            continue

        gate_names = set(r[0] for r in gate_rules) | set(
            r[0] for r in block_rules)
        gp_idx = np.where(gp)[0]
        gp_idx = gp_idx[np.argsort(_et[gp_idx], kind='stable')]  # chronological
        res = evaluate_config(features[gp_idx], pnl[gp_idx], years[gp_idx],
                              gate_names, col_names)
        n_cfg += 1
        for q_, e in res.items():
            key = f'{direction}_{cell}_{code}_q{q_}_c{ci}'
            entries[key] = {'features': e['features'], 'ic': e['ic'],
                            'sl': sl, 'tp': tp}
            ol = f"{e['oos_lift']:+.4f}" if e['oos_lift'] is not None else 'n/a'
            lines.append(
                f"  {key:<44} {e['scheme']:<3} F={e['fwd']}/3 R={e['rev']}/3 "
                f"lift={e['lift']:+.4f} wst={e['worst_yr_lift']:+.4f} oos={ol}")
    return entries, n_cfg, lines


def _qs_cell_est(code, cell):
    """Cell load estimate — ALL years (QS loads IS+2026), both-dir/2 x2.2."""
    inst = load_builder(code)._instance
    cache_base = str(inst.CACHE_FILE).replace('_raw_cache.pkl', '')
    cdir = Path(cache_base).parent
    name = Path(cache_base).name
    t = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
    v_ = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
    sz = sum(f.stat().st_size
             for f in cdir.glob(f'{name}_20??-??_t{t}v{v_}.pkl'))
    return (sz / 2) * 2.2 if sz else 7e9


def _qs_cell_worker(args):
    """Load ONE (direction, cell), evaluate its configs (event-driven)."""
    code, cell_key, configs, col_names, q = args
    direction, cell = cell_key.split('_', 1)
    t_idx = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
    v_idx = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]
    inst = load_builder(code)._instance
    try:
        # skip_trades=False: need entry_time to chronologically order the QS stream
        cache = inst.load_lite(direction, years=None, cell=(t_idx, v_idx),
                               skip_trades=False)
    finally:
        if q is not None:
            q.put('loaded')   # RAM settled -> parent admits next
    try:
        if cache is None:
            return cell_key, {}, 0, []
        entries, n_cfg, lines = _eval_cell_configs(
            cache[direction], direction, cell, code, configs, col_names)
        return cell_key, entries, n_cfg, lines
    finally:
        if q is not None:
            q.put('done')     # RAM freed -> parent re-checks


def run(strategy_code, col_names=None, phase2_json=None, max_workers=None):
    """Parallel QS: one worker per (direction, cell), measured-RAM
    admission identical to gate_sweep (loaded/done events)."""
    code = strategy_code.upper()
    rel, _sig, cat = STRATEGIES[code]
    strat_dir = BACKTEST / Path(rel).parent
    cat_dir = strat_dir.parent
    col_names = col_names or ALL_COL_NAMES

    p2_path = (Path(phase2_json) if phase2_json
               else strat_dir / f'{strat_dir.name}_phase2_results.json')
    out_path = cat_dir / 'unified_qs_features.json'

    t0 = time.time()
    print('=' * 120, flush=True)
    print(f'  QS FEATURES [{code}] — anti-overfit engine '
          f'(sign-stab>={SIGN_STAB_MIN}/5, WF lift gate >={WF_MIN_PASS}/3, '
          f'no OOS in selection, RAM admission)', flush=True)
    print('=' * 120, flush=True)

    if not p2_path.exists():
        print(f'  ERROR: phase2 json not found: {p2_path}', flush=True)
        return None
    p2 = json.load(open(p2_path))

    merged = json.load(open(out_path)) if out_path.exists() else {}
    strat_token = f'_{code}_q'
    merged = {k: v for k, v in merged.items() if strat_token not in k}

    todo = [(ck, p2[ck]) for ck in sorted(p2.keys()) if p2[ck]]
    sized = sorted(((ck, cfgs, _qs_cell_est(code, ck.split('_', 1)[1]))
                    for ck, cfgs in todo), key=lambda x: -x[2])
    n_cpu = max_workers or max(1, (os.cpu_count() or 8) - 2)
    print(f'  {len(sized)} cells, event-driven RAM admission '
          f'(cpu cap {n_cpu})', flush=True)

    import psutil, multiprocessing, queue as _queue
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    RESERVE = 3e9

    mgr = multiprocessing.Manager()
    q = mgr.Queue()
    pending = deque(sized)
    running = {}
    loading_fut = None
    n_pub = n_cfg = 0
    with ProcessPoolExecutor(max_workers=n_cpu) as ex:
        while pending or running:
            if pending and loading_fut is None and len(running) < n_cpu:
                ck, cfgs, est = pending[0]
                avail = psutil.virtual_memory().available
                if (avail - RESERVE) > est or not running:
                    pending.popleft()
                    fut = ex.submit(_qs_cell_worker,
                                    (code, ck, cfgs, col_names, q))
                    running[fut] = ck
                    loading_fut = fut
                    print(f'  + admit {ck} (est {est/1e9:.1f}GB, '
                          f'avail {avail/1e9:.1f}GB, active {len(running)})',
                          flush=True)
            try:
                ev = q.get(timeout=5)
                if ev == 'loaded':
                    loading_fut = None
            except _queue.Empty:
                pass
            for fut in [f for f in running if f.done()]:
                running.pop(fut)
                if fut is loading_fut:
                    loading_fut = None
                ck, entries, nc, lines = fut.result()
                merged.update(entries)
                n_pub += len(entries)
                n_cfg += nc
                for ln in lines:
                    print(ln, flush=True)
                json.dump(merged, open(out_path, 'w'), indent=1)
                print(f'  {ck:<22} -> {len(entries)} entry / {nc} config '
                      f'({time.time()-t0:.0f}s)', flush=True)

    print(f'\n  {n_pub} entries published / {n_cfg} configs evaluated '
          f'-> {out_path} ({(time.time()-t0)/60:.1f} min)', flush=True)
    return merged


def run_multi(strategy_codes, col_names=None, max_workers=None):
    """GLOBAL QS queue across strategies — twin of the gate multi runner.

    Category jsons carry multiple strategies; a single parent writes them.
    """
    col_names = col_names or ALL_COL_NAMES
    n_cpu = max_workers or max(1, (os.cpu_count() or 8) - 2)
    t0 = time.time()

    merged_by_path, out_path_of = {}, {}
    pending = []
    for code in strategy_codes:
        rel, _sig, _cat = STRATEGIES[code]
        strat_dir = BACKTEST / Path(rel).parent
        p2_path = strat_dir / f'{strat_dir.name}_phase2_results.json'
        if not p2_path.exists():
            print(f'  [{code}] no phase2 json, skipped', flush=True)
            continue
        out_path = strat_dir.parent / 'unified_qs_features.json'
        out_path_of[code] = out_path
        if out_path not in merged_by_path:
            merged_by_path[out_path] = (json.load(open(out_path))
                                        if out_path.exists() else {})
        # clear this strategy's old entries (regeneration)
        tok = f'_{code}_q'
        merged_by_path[out_path] = {k: v for k, v in
                                    merged_by_path[out_path].items()
                                    if tok not in k}
        p2 = json.load(open(p2_path))
        for ck in sorted(p2.keys()):
            if p2[ck]:
                est = _qs_cell_est(code, ck.split('_', 1)[1])
                pending.append((code, ck, p2[ck], est))
    # strategy-major, largest cell first within each strategy
    print(f'  QS MULTI: {len(strategy_codes)} strategies, {len(pending)} '
          f'cells, global RAM admission (cpu cap {n_cpu})', flush=True)

    import psutil, multiprocessing, queue as _queue
    from concurrent.futures import ProcessPoolExecutor
    RESERVE = 3e9
    mgr = multiprocessing.Manager()
    q = mgr.Queue()
    running = {}
    loading_fut = None
    n_pub = 0
    bad = []
    with ProcessPoolExecutor(max_workers=n_cpu) as ex:
        while pending or running:
            if pending and loading_fut is None and len(running) < n_cpu:
                avail = psutil.virtual_memory().available
                pick = None
                for i, (c_, ck_, cfgs_, est_) in enumerate(pending):
                    if (avail - RESERVE) > est_ or not running:
                        pick = i
                        break
                if pick is not None:
                    c_, ck_, cfgs_, est_ = pending.pop(pick)
                    fut = ex.submit(_qs_cell_worker,
                                    (c_, ck_, cfgs_, col_names, q))
                    running[fut] = (c_, ck_)
                    loading_fut = fut
                    print(f'  + admit {c_}:{ck_} (est {est_/1e9:.1f}GB, '
                          f'avail {avail/1e9:.1f}GB, active {len(running)})',
                          flush=True)
            try:
                ev = q.get(timeout=5)
                if ev == 'loaded':
                    loading_fut = None
            except _queue.Empty:
                pass
            for fut in [f for f in running if f.done()]:
                c_, ck_ = running.pop(fut)
                if fut is loading_fut:
                    loading_fut = None
                try:
                    _ck, entries, nc, lines = fut.result()
                except Exception as e:
                    bad.append(f'{c_}:{ck_}')
                    print(f'  !! {c_}:{ck_} ERROR: {type(e).__name__}: '
                          f'{str(e)[:80]}', flush=True)
                    continue
                op = out_path_of[c_]
                merged_by_path[op].update(entries)
                n_pub += len(entries)
                for ln in lines:
                    print(ln, flush=True)
                json.dump(merged_by_path[op], open(op, 'w'), indent=1)
                print(f'  {c_}:{ck_:<20} -> {len(entries)} entry '
                      f'({(time.time()-t0)/60:.0f}min)', flush=True)

    print(f'\n  QS MULTI done: {n_pub} entries '
          f'({(time.time()-t0)/3600:.1f}h)', flush=True)
    if bad:
        print(f'  FAILED: {bad}', flush=True)
    return merged_by_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--phase2-json', default=None)
    args = ap.parse_args()
    col_names = None
    run(args.strategy, col_names=col_names, phase2_json=args.phase2_json)


if __name__ == '__main__':
    main()
