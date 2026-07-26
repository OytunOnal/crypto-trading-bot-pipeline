"""Common Phase-2 (block rules) engine — works for ANY registry strategy.

One shared engine instead of a copy-pasted per-strategy script.

For each STRONG gate-sweep config: find per-feature block candidates
(worst-tail percentiles, year-stable), select independent ones, exhaustive
combo search k=0..MAX_K, FWD+REV walk-forward, keep STRONG. Uses the
cell-filtered streaming load (peak RAM ~ direction/9).

Usage:
  python factory/blocks/phase2.py --strategy SMA_X
"""
import sys, os, time, gc, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.gates.features_common import ALL_COL_NAMES, _spearman
from factory.data.registry import STRATEGIES, load_builder, BACKTEST

EXCLUDED_FEATURES = set()   # feature names to hard-exclude from block search
TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
VOL_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}

CORR_THRESH = 0.50
MAX_INDEP_BLOCKS = 15
MAX_K = 10
MIN_YR_TRADES = 500
BLOCK_PCTS = [1, 2, 3, 5, 95, 97, 98, 99]
IS_YEARS = list(range(2021, 2026))

FWD_WF = [('FWF1', [2021, 2022], [2023]), ('FWF2', [2021, 2022, 2023], [2024]),
          ('FWF3', [2021, 2022, 2023, 2024], [2025])]
REV_WF = [('RWF1', [2024, 2025], [2023]), ('RWF2', [2023, 2024, 2025], [2022]),
          ('RWF3', [2022, 2023, 2024, 2025], [2021])]

N_WORKERS = 10


def apply_gate(features, resolved, gate_rules, col_names):
    mask = resolved.copy()
    for feat_name, op, value in gate_rules:
        if feat_name not in col_names:
            continue
        fi = col_names.index(feat_name)
        col = features[:, fi].astype(np.float64)
        valid = np.isfinite(col)
        if op == '<':
            mask &= (valid & (col < value))
        else:
            mask &= (valid & (col > value))
    return mask


def find_block_candidates(features, pnl, years, resolved, gate_feat_names,
                          col_names):
    baseline_ev = pnl[resolved].mean()
    candidates = []
    for fi in range(features.shape[1]):
        col_name = col_names[fi] if fi < len(col_names) else f'col_{fi}'
        if col_name in gate_feat_names or col_name in EXCLUDED_FEATURES:
            continue
        col = features[:, fi].astype(np.float64)
        valid = resolved & np.isfinite(col)
        if valid.sum() < 500:
            continue
        col_valid = col[valid]
        for pct in BLOCK_PCTS:
            threshold = float(np.percentile(col_valid, pct))
            if pct <= 50:
                blocked = valid & (col < threshold)
                kept = valid & (col >= threshold)
                block_op = '<'
            else:
                blocked = valid & (col > threshold)
                kept = valid & (col <= threshold)
                block_op = '>'
            n_blocked = blocked.sum()
            n_kept = kept.sum()
            if n_blocked < 100 or n_kept < MIN_YR_TRADES * 5:
                continue
            blocked_ev = pnl[blocked].mean()
            kept_ev = pnl[kept].mean()
            if blocked_ev < baseline_ev - 0.01 and kept_ev > baseline_ev:
                yr_ok = 0
                for y in IS_YEARS:
                    ym_k = kept & (years == y)
                    ym_b = blocked & (years == y)
                    if ym_k.sum() > 50 and ym_b.sum() > 10:
                        if pnl[ym_k].mean() > pnl[ym_b].mean():
                            yr_ok += 1
                if yr_ok >= 3:
                    candidates.append({
                        'name': col_name, 'idx': fi, 'block_op': block_op,
                        'value': threshold, 'pct': pct,
                        'blocked_ev': blocked_ev, 'kept_ev': kept_ev,
                        'n_blocked': n_blocked, 'n_kept': n_kept,
                        'improvement': kept_ev - baseline_ev,
                        'yr_stable': yr_ok,
                    })
    candidates.sort(key=lambda x: -x['improvement'])
    seen = set()
    unique = []
    for c in candidates:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique.append(c)
    return unique


def select_independent_blocks(candidates, features, resolved):
    selected = []
    for c in candidates:
        col_c = features[resolved, c['idx']].astype(np.float64)
        ok = True
        for s in selected:
            col_s = features[resolved, s['idx']].astype(np.float64)
            v = np.isfinite(col_c) & np.isfinite(col_s)
            if v.sum() < 100:
                continue
            if abs(_spearman(col_c[v], col_s[v])) > CORR_THRESH:
                ok = False
                break
        if ok:
            selected.append(c)
            if len(selected) >= MAX_INDEP_BLOCKS:
                break
    return selected


def search_block_combos(features, pnl, years, resolved, block_candidates):
    baseline_ev = pnl[resolved].mean()
    baseline_n = int(resolved.sum())

    yr_masks = {y: (years == y) for y in IS_YEARS}
    yr_mask_arr = np.stack([yr_masks[y] for y in IS_YEARS])
    wf_masks = []
    for _, _, val_yrs in FWD_WF:
        wf_masks.append((True, np.isin(years, val_yrs)))
    for _, _, val_yrs in REV_WF:
        wf_masks.append((False, np.isin(years, val_yrs)))

    baseline_yr = {}
    for y in IS_YEARS:
        ym = resolved & yr_masks[y]
        if ym.sum() > 0:
            baseline_yr[y] = {'n': int(ym.sum()), 'ev': pnl[ym].mean(),
                              'wr': (pnl[ym] > 0).mean() * 100}
    baseline_worst_ev = min((d['ev'] for d in baseline_yr.values()), default=-999)

    best = {
        'k': 0, 'blocks': [], 'wr': (pnl[resolved] > 0).mean() * 100,
        'ev': baseline_ev, 'n': baseline_n, 'fwd': 3, 'rev': 3,
        'improvement': 0, 'yr': baseline_yr,
        'worst_yr_ev': baseline_worst_ev,
        '_score': (6, baseline_worst_ev, baseline_ev),
    }
    if not block_candidates:
        return best

    N = len(pnl)
    n_cands = len(block_candidates)
    keep_matrix = np.ones((N, n_cands), dtype=np.bool_)
    for ci, b in enumerate(block_candidates):
        col = features[:, b['idx']].astype(np.float64)
        valid = np.isfinite(col)
        if b['block_op'] == '<':
            keep_matrix[:, ci] = ~(valid & (col < b['value']))
        else:
            keep_matrix[:, ci] = ~(valid & (col > b['value']))

    pnl_pos = (pnl > 0).astype(np.uint8)
    max_k = min(MAX_K, n_cands)
    best_score = best['_score']
    failed_frozen = set()

    for k in range(1, max_k + 1):
        if k > n_cands:
            break
        for cidx in combinations(range(n_cands), k):
            skip = False
            if k > 1:
                for sub_k in range(1, k):
                    for sub in combinations(cidx, sub_k):
                        if sub in failed_frozen:
                            skip = True
                            break
                    if skip:
                        break
            if skip:
                continue

            combined = resolved.copy()
            for i in cidx:
                combined &= keep_matrix[:, i]
            n_kept = combined.sum()
            if n_kept < MIN_YR_TRADES * 5:
                failed_frozen.add(cidx)
                continue
            yr_counts = yr_mask_arr[:, combined].sum(axis=1)
            if int(yr_counts.min()) < MIN_YR_TRADES:
                failed_frozen.add(cidx)
                continue

            ev = pnl[combined].mean()
            if ev <= 0:
                continue
            wr = pnl_pos[combined].sum() / n_kept * 100

            yr_evs = []
            for yi in range(len(IS_YEARS)):
                ym = combined & yr_mask_arr[yi]
                yr_evs.append(pnl[ym].mean() if ym.sum() > 0 else -999)
            worst_yr_ev = min(yr_evs)

            candidate_score = (6, worst_yr_ev, ev)
            if candidate_score <= best_score:
                continue

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
            if total < 4:
                continue

            score = (total, worst_yr_ev, ev)
            if score > best_score:
                yr_data = {}
                for yi, y in enumerate(IS_YEARS):
                    if yr_counts[yi] > 0:
                        ym = combined & yr_mask_arr[yi]
                        yr_data[y] = {'n': int(yr_counts[yi]), 'ev': yr_evs[yi],
                                      'wr': pnl_pos[ym].sum() / yr_counts[yi] * 100}
                best = {
                    'k': k, 'blocks': [block_candidates[i] for i in cidx],
                    'wr': wr, 'ev': ev, 'n': int(n_kept),
                    'fwd': fwd_pass, 'rev': rev_pass,
                    'improvement': ev - baseline_ev,
                    'yr': yr_data, 'worst_yr_ev': worst_yr_ev, '_score': score,
                }
                best_score = score
    return best


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


def process_one_config(args):
    (ci, cfg, gp_feat, gp_pnl, gp_years, direction, cell, sl, tp,
     col_names) = args

    gp_resolved = np.isfinite(gp_pnl) & (gp_pnl != 0)
    baseline_ev = float(gp_pnl[gp_resolved].mean())

    gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<', g['value'])
                  for g in cfg.get('gates', [])]
    gate_feat_names = set(r[0] for r in gate_rules)

    candidates = find_block_candidates(gp_feat, gp_pnl, gp_years, gp_resolved,
                                       gate_feat_names, col_names)
    indep = select_independent_blocks(candidates, gp_feat, gp_resolved)
    result = search_block_combos(gp_feat, gp_pnl, gp_years, gp_resolved, indep)
    v = verdict(result['fwd'], result['rev'])
    if v != 'STRONG':
        return ci, sl, tp, v, None

    save_data = {
        'direction': direction, 'cell': cell, 'sl': sl, 'tp': tp,
        'verdict': v, 'k': result['k'], 'n': int(result['n']),
        'wr': round(float(result['wr']), 2), 'ev': round(float(result['ev']), 3),
        'worst_yr_ev': round(float(result['worst_yr_ev']), 3),
        'worst_yr_wr': round(float(min((d['wr'] for d in result['yr'].values()),
                                       default=0)), 1),
        'fwd': int(result['fwd']), 'rev': int(result['rev']),
        'improvement': round(float(result['improvement']), 3),
        'yr': result['yr'], 'baseline_ev': round(baseline_ev, 3),
        'gate': [{'name': g['name'], 'direction': g['direction'],
                  'value': float(g['value'])} for g in cfg.get('gates', [])],
        'blocks': [{'name': b['name'], 'block_op': b['block_op'],
                    'value': float(b['value']), 'pct': int(b['pct'])}
                   for b in result.get('blocks', [])],
    }
    return ci, sl, tp, v, save_data


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(f'{type(o).__name__} not JSON serializable')


def run_multi(strategy_codes, col_names=None, max_workers=None, out_dir=None):
    """GLOBAL pipelined phase2 across strategies (user design 2026-06-13).

    One worker pool for everything. The parent walks (strategy, cell)
    tasks strategy-major: loads a cell (RAM-admission-gated via the
    cell's shard estimate), submits its config slices (exact-cost
    admission), and moves to the NEXT cell without waiting — the pool
    never starves at cell/strategy boundaries. A cell's JSON is written
    when its last config completes.

    out_dir: redirect result JSONs here (test isolation — leaves the real
    <dir>_phase2_results.json untouched). Default None = real per-dir path.
    """
    from factory.gates.gate_sweep import _cell_est_bytes
    import psutil
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    col_names = col_names or ALL_COL_NAMES
    n_workers = max_workers or N_WORKERS
    RESERVE = 3e9
    t0 = time.time()

    # task discovery (resume-aware)
    res_files, completed, cell_tasks = {}, {}, deque()
    for code in strategy_codes:
        rel, _sig, _cat = STRATEGIES[code]
        strat_dir = BACKTEST / Path(rel).parent
        gate_path = strat_dir / 'gate_sweep_results.json'
        if not gate_path.exists():
            print(f'  [{code}] no gate json, skipped', flush=True)
            continue
        rf = ((Path(out_dir) / f'{code}_phase2_results.json') if out_dir
              else strat_dir / f'{strat_dir.name}_phase2_results.json')
        res_files[code] = rf
        completed[code] = json.load(open(rf)) if rf.exists() else {}
        gate_configs = json.load(open(gate_path))
        for cell_key in sorted(gate_configs.keys()):
            if cell_key in completed[code]:
                continue
            cfgs = [c for c in gate_configs[cell_key]
                    if c.get('verdict') == 'STRONG']
            cell_tasks.append((code, cell_key, cfgs))
    print(f'  PHASE2 MULTI: {len(strategy_codes)} strategies, '
          f'{len(cell_tasks)} cells, pipelined global queue '
          f'({n_workers} worker)', flush=True)

    def _write_cell(code, cell_key, results_list):
        results_list.sort(key=lambda x: (x.get('worst_yr_wr', 0),
                                         x.get('worst_yr_ev', 0)),
                          reverse=True)
        completed[code][cell_key] = results_list
        json.dump(completed[code], open(res_files[code], 'w'), indent=2,
                  default=_json_default)
        print(f'  {code}:{cell_key:<20} -> {len(results_list)} STRONG '
              f'({(time.time()-t0)/60:.0f}min)', flush=True)

    running = {}            # fut -> (code, cell_key)
    cell_state = {}         # (code, cell_key) -> {'res': [], 'left': n,
                            #                      'submitted_all': bool}
    cur = None              # cell currently being loaded/fed
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        while cell_tasks or cur or running:
            # 1) load next cell (if RAM allows) — pipelined: loads while
            #    the previous cell's configs are still running
            if cur is None and cell_tasks:
                code, cell_key, cfgs = cell_tasks[0]
                if not cfgs:
                    cell_tasks.popleft()
                    _write_cell(code, cell_key, [])
                    continue
                est = _cell_est_bytes(code, cell_key.split('_', 1)[1])
                avail = psutil.virtual_memory().available
                if (avail - RESERVE) > est or not running:
                    cell_tasks.popleft()
                    direction, cell = cell_key.split('_', 1)
                    t_i = [k for k, v in TREND_NAMES.items()
                           if v == cell.split('_')[0]][0]
                    v_i = [k for k, v in VOL_NAMES.items()
                           if v == cell.split('_')[1]][0]
                    inst = load_builder(code)._instance
                    cache = inst.load_lite(direction, years=IS_YEARS,
                                           cell=(t_i, v_i))
                    if cache is None:
                        _write_cell(code, cell_key, [])
                    else:
                        d = cache[direction]
                        cell_state[(code, cell_key)] = {'res': [], 'left': 0,
                                                        'submitted_all': False}
                        cur = {'code': code, 'cell_key': cell_key,
                               'direction': direction, 'cell': cell,
                               'd': d, 'cfgs': deque(enumerate(cfgs)),
                               'is_mask': np.isin(d['years'], IS_YEARS)}
                        del cache
            # 2) submit active cell's configs with exact-cost admission
            if cur is not None:
                d = cur['d']
                features, years = d['features'], d['years']
                while cur['cfgs'] and len(running) < n_workers * 2:
                    ci, cfg = cur['cfgs'][0]
                    sl, tp = cfg['sl'], cfg['tp']
                    pnl_arr = d.get(f'pnl_{sl}_{tp}')
                    if pnl_arr is None:
                        cur['cfgs'].popleft()
                        continue
                    resolved = np.isfinite(pnl_arr) & (pnl_arr != 0)
                    g_rules = [(g['name'],
                                '>' if g['direction'] == 'gt' else '<',
                                g['value']) for g in cfg.get('gates', [])]
                    gp = apply_gate(features, resolved, g_rules, col_names)
                    gp &= cur['is_mask']
                    n_gp = int(gp.sum())
                    if n_gp < MIN_YR_TRADES * 5:
                        cur['cfgs'].popleft()
                        continue
                    est = n_gp * (features.shape[1] * 2 + 8 + 2) * 2.5
                    avail = psutil.virtual_memory().available
                    if (avail - RESERVE) > est or not running:
                        cur['cfgs'].popleft()
                        key = (cur['code'], cur['cell_key'])
                        fut = ex.submit(
                            process_one_config,
                            (ci, cfg, features[gp].copy(),
                             pnl_arr[gp].copy(), years[gp].copy(),
                             cur['direction'], cur['cell'], sl, tp,
                             col_names))
                        running[fut] = key
                        cell_state[key]['left'] += 1
                    else:
                        break
                if not cur['cfgs']:
                    key = (cur['code'], cur['cell_key'])
                    cell_state[key]['submitted_all'] = True
                    if cell_state[key]['left'] == 0:
                        _write_cell(cur['code'], cur['cell_key'],
                                    cell_state.pop(key)['res'])
                    cur = None      # cell data freed -> next load
                    gc.collect()
                    continue
            # 3) harvest finished
            if not running:
                continue
            done, _ = wait(running, timeout=5, return_when=FIRST_COMPLETED)
            for fut in done:
                key = running.pop(fut)
                st = cell_state.get(key)
                try:
                    _ci, _sl, _tp, _v, save_data = fut.result()
                    if save_data is not None and st is not None:
                        st['res'].append(save_data)
                except Exception as e:
                    print(f'  !! {key} config error: '
                          f'{type(e).__name__}: {str(e)[:80]}', flush=True)
                if st is not None:
                    st['left'] -= 1
                    if st['submitted_all'] and st['left'] == 0:
                        _write_cell(key[0], key[1],
                                    cell_state.pop(key)['res'])

    print(f'  PHASE2 MULTI done ({(time.time()-t0)/3600:.1f}h)',
          flush=True)
    return completed


def run(strategy_code, col_names=None, gate_json=None, out_path=None,
        max_workers=None):
    code = strategy_code.upper()
    rel, _sig, _cat = STRATEGIES[code]
    strat_dir = BACKTEST / Path(rel).parent
    builder = load_builder(code)
    inst = getattr(builder, '_instance')
    col_names = col_names or ALL_COL_NAMES
    n_workers = max_workers or N_WORKERS

    gate_path = Path(gate_json) if gate_json else strat_dir / 'gate_sweep_results.json'
    # output name follows the per-strategy convention: <dir>_phase2_results.json
    results_file = (Path(out_path) if out_path
                    else strat_dir / f'{strat_dir.name}_phase2_results.json')

    t0 = time.time()
    print('=' * 120, flush=True)
    print(f'  PHASE 2 [{code}] — common engine '
          f'(max_k={MAX_K}, max_indep={MAX_INDEP_BLOCKS}, workers={n_workers})',
          flush=True)
    print('=' * 120, flush=True)

    if not gate_path.exists():
        print(f'  ERROR: gate sweep JSON not found: {gate_path}', flush=True)
        return None
    gate_configs = json.load(open(gate_path))
    print(f'  Loaded {sum(len(v) for v in gate_configs.values())} configs / '
          f'{len(gate_configs)} cells', flush=True)

    completed = json.load(open(results_file)) if results_file.exists() else {}
    if completed:
        print(f'  Skipping {len(completed)} completed cells', flush=True)

    for cell_key in sorted(gate_configs.keys()):
        if cell_key in completed:
            continue
        configs = [c for c in gate_configs[cell_key]
                   if c.get('verdict') == 'STRONG']
        if not configs:
            completed[cell_key] = []
            json.dump(completed, open(results_file, 'w'), indent=2,
                      default=_json_default)
            continue

        direction, cell = cell_key.split('_', 1)
        t_idx = [k for k, v in TREND_NAMES.items() if v == cell.split('_')[0]][0]
        v_idx = [k for k, v in VOL_NAMES.items() if v == cell.split('_')[1]][0]

        # cell-filtered streaming load: peak RAM ~ direction/9
        cache = inst.load_lite(direction, years=IS_YEARS, cell=(t_idx, v_idx))
        if cache is None:
            completed[cell_key] = []
            json.dump(completed, open(results_file, 'w'), indent=2,
                      default=_json_default)
            continue
        d = cache[direction]
        features = d['features']
        years = d['years']

        # Measured-RAM admission (same pattern as gate_sweep): each
        # config's slice size is known BEFORE submit; if free RAM (psutil)
        # does not cover reserve+cost the submit waits, and a finishing
        # worker's freed RAM admits the next one.
        cfg_specs = []
        for ci, cfg in enumerate(configs):
            sl, tp = cfg['sl'], cfg['tp']
            if d.get(f'pnl_{sl}_{tp}') is None:
                continue
            cfg_specs.append((ci, cfg, sl, tp))

        if not cfg_specs:
            completed[cell_key] = []
            json.dump(completed, open(results_file, 'w'), indent=2,
                      default=_json_default)
            del cache, d, features, years
            gc.collect()
            continue

        t_cell = time.time()
        print(f'  {cell_key}: {len(cfg_specs)}/{len(configs)} configs -> '
              f'parallel ({n_workers} workers, RAM admission)...', end='',
              flush=True)
        import psutil
        from collections import deque
        from concurrent.futures import wait, FIRST_COMPLETED
        RESERVE = 3e9
        is_mask_all = np.isin(years, IS_YEARS)
        cell_results = []
        pending = deque(cfg_specs)
        running = set()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            while pending or running:
                while pending and len(running) < n_workers:
                    ci, cfg, sl, tp = pending[0]
                    cell_pnl = d[f'pnl_{sl}_{tp}']
                    resolved = np.isfinite(cell_pnl) & (cell_pnl != 0)
                    gate_rules = [(g['name'],
                                   '>' if g['direction'] == 'gt' else '<',
                                   g['value']) for g in cfg.get('gates', [])]
                    gp_mask = apply_gate(features, resolved, gate_rules,
                                         col_names)
                    gp_mask &= is_mask_all
                    n_gp = int(gp_mask.sum())
                    if n_gp < MIN_YR_TRADES * 5:
                        pending.popleft()
                        continue
                    # exact cost: slice bytes x2.5 (parent copy + pickle
                    # transfer + child copy)
                    est = n_gp * (features.shape[1] * 2 + 8 + 2) * 2.5
                    avail = psutil.virtual_memory().available
                    if (avail - RESERVE) > est or not running:
                        pending.popleft()
                        fut = pool.submit(
                            process_one_config,
                            (ci, cfg, features[gp_mask].copy(),
                             cell_pnl[gp_mask].copy(), years[gp_mask].copy(),
                             direction, cell, sl, tp, col_names))
                        running.add(fut)
                    else:
                        break
                done, running = wait(running, timeout=5,
                                     return_when=FIRST_COMPLETED)
                for fut in done:
                    _ci, _sl, _tp, _v, save_data = fut.result()
                    if save_data is not None:
                        cell_results.append(save_data)
                gc.collect()

        del cache, d, features, years
        gc.collect()

        cell_results.sort(key=lambda x: (x.get('worst_yr_wr', 0),
                                         x.get('worst_yr_ev', 0)), reverse=True)
        completed[cell_key] = cell_results
        json.dump(completed, open(results_file, 'w'), indent=2,
                  default=_json_default)
        print(f' {len(cell_results)} STRONG ({time.time()-t_cell:.0f}s)',
              flush=True)

    print(f'\n  DONE ({(time.time()-t0)/60:.1f} min) -> {results_file.name}',
          flush=True)
    return completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--gate-json', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-workers', type=int, default=None)
    args = ap.parse_args()

    col_names = None
    run(args.strategy, col_names=col_names, gate_json=args.gate_json,
        out_path=args.out, max_workers=args.max_workers)


if __name__ == '__main__':
    main()
