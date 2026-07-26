"""Common QS (Quantile Sweep) utilities for unified_cell_comparison scripts.

Shared functions: rolling rank, gate application, concurrent position filter,
period stats, vectorized IC, QS combo search, strategy processing, output.
Every category's unified_cell_comparison.py imports from here.
"""
import sys, os, time, json, gc
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
import numpy as np
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Constants ────────────────────────────────────────────────────────
TREND_MAP = {'BEAR': 0, 'FLAT': 1, 'BULL': 2}
VOL_MAP = {'LOW': 0, 'MED': 1, 'HIGH': 2}
N_ROLL = 100
IC_MIN = 0.010
STAB_MIN = 2
CORR_THRESH = 0.60
QS_K_RANGE = range(2, 6)
N_SPLITS = [2, 3, 4, 5]
MAX_CONCURRENT = 100
MAX_QS_INDEP = 15
TOTAL_RAM_GB = 32
RAM_RESERVE_GB = 8


# ── Dynamic worker count ─────────────────────────────────────────────
def calc_workers(cache_dir_size_gb):
    """Calculate safe worker count based on single-direction cache size.

    Calibrated: 10 workers fits at 9.5GB single dir (19GB total, ~28GB RAM).
    Memory model: total_used = cache*2.5 + n * per_worker_cost
    At 9.5GB single dir, 10 workers -> 28GB used. So per_worker ~(28-9.5*2.5)/10 = 0.4GB
    """
    cache_ram = cache_dir_size_gb * 1.7  # calibrated on a 9.5GB single-direction cache
    per_worker = 0.4  # calibrated: (28 - 8 - 9.5*1.7) / 10
    avail = TOTAL_RAM_GB - RAM_RESERVE_GB - cache_ram
    n = int(avail / per_worker) if per_worker > 0 else 2
    return max(1, min(10, n))


# ── Rolling rank (numba) ──────────────────────────────────────────────
import os
os.environ['NUMBA_NUM_THREADS'] = '1'  # prevent thread contention with ProcessPool

from numba import njit

@njit(cache=True)
def _rolling_rank_numba(vals, window, min_periods):
    """Numba-accelerated rolling rank percentile."""
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        start = i - window + 1
        current = vals[i]
        if np.isnan(current):
            continue
        n_valid = 0
        less = 0
        equal = 0
        for j in range(start, i + 1):
            v = vals[j]
            if np.isnan(v):
                continue
            n_valid += 1
            if v < current:
                less += 1
            elif v == current:
                equal += 1
        if n_valid < min_periods or n_valid <= 1:
            continue
        avg_rank = less + (equal + 1.0) / 2.0
        result[i] = (avg_rank - 1.0) / (n_valid - 1.0) * 100.0
    return result


def rolling_rank_exclude_self(vals, n_roll):
    """Rolling percentile rank. Numba-accelerated."""
    vals = np.asarray(vals, dtype=np.float64)
    result = _rolling_rank_numba(vals, n_roll + 1, 21)
    result[:n_roll] = np.nan
    return result


# ── Gate / block application ─────────────────────────────────────────
def apply_gates(features, resolved, gate_rules, block_rules, all_col_names):
    mask = resolved.copy()
    for feat_name, op, value in gate_rules:
        if feat_name not in all_col_names: continue
        fi = all_col_names.index(feat_name)
        col = features[:, fi].astype(np.float64); valid = np.isfinite(col)
        if op == '<': mask &= (valid & (col < value))
        else: mask &= (valid & (col > value))
    for feat_name, op, value in block_rules:
        if feat_name not in all_col_names: continue
        fi = all_col_names.index(feat_name)
        col = features[:, fi].astype(np.float64); valid = np.isfinite(col)
        if op == '<': mask &= ~(valid & (col < value))
        else: mask &= ~(valid & (col > value))
    return mask


# ── Max concurrent positions ─────────────────────────────────────────
def apply_max_concurrent(trades_mask, entry_times, exit_times, max_conc):
    indices = np.where(trades_mask)[0]
    if len(indices) == 0: return trades_mask

    valid_time = np.array([entry_times[i] is not None and exit_times[i] is not None for i in indices])
    indices = indices[valid_time]
    if len(indices) == 0: return trades_mask

    etimes = np.array([entry_times[i] for i in indices])
    order = np.argsort(etimes)
    sorted_idx = indices[order]

    kept = np.zeros(len(trades_mask), dtype=bool)
    open_exits = []

    for idx in sorted_idx:
        et = entry_times[idx]; xt = exit_times[idx]
        open_exits = [x for x in open_exits if x > et]
        if len(open_exits) < max_conc:
            kept[idx] = True
            open_exits.append(xt)

    return kept


# ── Period stats ──────────────────────────────────────────────────────
def compute_period_stats(pnl, years, entry_times, yr_range, weekly=True):
    mask = np.isin(years, yr_range) & np.isfinite(pnl) & (pnl != 0)
    n = mask.sum()
    if n == 0:
        return {'n': 0, 'wr': 0, 'ev': 0, 'neg_w': 0}
    p = pnl[mask]
    ev = float(p.mean())
    wr = float((p > 0).mean() * 100)

    neg_w = 0
    if weekly and entry_times is not None:
        wk_pnl = {}
        et = entry_times[mask]
        for t, pp in zip(et, p):
            if t is None: continue
            iso = t.isocalendar(); wk = f'{iso[0]}-W{iso[1]:02d}'
            wk_pnl[wk] = wk_pnl.get(wk, 0) + pp
        neg_w = sum(1 for v in wk_pnl.values() if v < 0)

    return {'n': int(n), 'wr': round(wr, 1), 'ev': round(ev, 3), 'neg_w': int(neg_w)}


# ── Vectorized rank IC (pure numpy) ──────────────────────────────────
def _vectorized_rank_ic(feat_matrix, pnl, valid_mask):
    """Vectorized Spearman IC: rank once, pearson on ranks. Pure numpy."""
    n = valid_mask.sum()
    if n < 200:
        return np.full(feat_matrix.shape[1], np.nan)
    pnl_valid = pnl[valid_mask].astype(np.float64)
    pnl_rank = np.argsort(np.argsort(pnl_valid)).astype(np.float64)
    ics = np.full(feat_matrix.shape[1], np.nan)
    for fi in range(feat_matrix.shape[1]):
        col = feat_matrix[valid_mask, fi].astype(np.float64)
        col_valid = np.isfinite(col)
        if col_valid.sum() < 200: continue
        jv = col_valid
        if jv.sum() < 200: continue
        col_rank = np.argsort(np.argsort(col[jv])).astype(np.float64)
        pnl_r = pnl_rank[jv]
        c = np.corrcoef(col_rank, pnl_r)[0, 1]
        ics[fi] = c if np.isfinite(c) else 0.0
    return ics


# ── Process one cell config (runs in worker) ─────────────────────────
def process_one_cell_config(args):
    """Process one (cell, config, strategy) combination."""
    import warnings
    warnings.filterwarnings('ignore')

    (gp_feat, gp_pnl, gp_years, gp_entry, gp_exit, gp_resolved,
     n_gp, all_col_names, excluded, gate_feat_names,
     direction, cell, strat_name, sl, tp, ci) = args

    rows = []

    # q=1 baseline
    conc_mask = apply_max_concurrent(gp_resolved, gp_entry, gp_exit, MAX_CONCURRENT)
    is_conc = compute_period_stats(gp_pnl[conc_mask], gp_years[conc_mask],
                                    gp_entry[conc_mask], list(range(2021, 2026)), True)
    oos_conc = compute_period_stats(gp_pnl[conc_mask], gp_years[conc_mask],
                                     gp_entry[conc_mask], [2026], True)

    cell_label = f'{direction[0]}_{cell}'
    sl_tp_str = f'{sl}/{tp}{"*" if sl != tp else ""}'

    if oos_conc.get('ev', 0) >= 0:
        rows.append({
            'cell': cell_label, 'strat': strat_name,
            'sl_tp': sl_tp_str, 'q': 1, 'ci': ci,
            **{f'is_{k}': v for k, v in is_conc.items()},
            **{f'oos_{k}': v for k, v in oos_conc.items()},
        })

    # QS features (IC computed on 2021-2025 only)
    is_mask = (gp_years <= 2025) & gp_resolved
    if is_mask.sum() < 500:
        return rows

    gp_feat_is = gp_feat[is_mask]; gp_pnl_is = gp_pnl[is_mask]
    gp_years_is = gp_years[is_mask]
    resolved_is = np.isfinite(gp_pnl_is) & (gp_pnl_is != 0)
    unique_yrs = sorted(set(gp_years_is[resolved_is]))

    overall_ics = _vectorized_rank_ic(gp_feat_is, gp_pnl_is, resolved_is)

    yr_ic_matrix = {}
    for y in unique_yrs:
        ym = resolved_is & (gp_years_is == y)
        if ym.sum() >= 50:
            yr_ic_matrix[y] = _vectorized_rank_ic(gp_feat_is, gp_pnl_is, ym)

    feat_stats = []
    for fi in range(gp_feat_is.shape[1]):
        col_name = all_col_names[fi] if fi < len(all_col_names) else f'col_{fi}'
        if col_name in gate_feat_names or col_name in excluded: continue
        ic = overall_ics[fi]
        if not np.isfinite(ic) or abs(ic) < IC_MIN: continue
        yr_ics = [float(yr_ic_matrix[y][fi]) for y in yr_ic_matrix
                  if np.isfinite(yr_ic_matrix[y][fi])]
        if len(yr_ics) < 2: continue
        stab = sum(1 for yic in yr_ics if (yic > 0) == (ic > 0))
        if stab < STAB_MIN: continue
        feat_stats.append({'idx': fi, 'name': col_name, 'ic': float(ic),
                          'abs_ic': abs(float(ic)), 'stab': stab, 'n_yr': len(yr_ics)})

    # Dedup 1h/4h
    seen_bases = {}; deduped = []
    for f in sorted(feat_stats, key=lambda f: f['abs_ic'], reverse=True):
        name = f['name']
        for sf in ['_15m', '_1h', '_4h']:
            if name.endswith(sf): base = name[:-len(sf)]; break
        else: base = name
        if base not in seen_bases: seen_bases[base] = True; deduped.append(f)

    # Select independent (top MAX_QS_INDEP)
    from factory.gates.features_common import _spearman
    indep = []
    for f in deduped:
        col_f = gp_feat_is[resolved_is, f['idx']].astype(np.float64)
        ok = True
        for s in indep:
            col_s = gp_feat_is[resolved_is, s['idx']].astype(np.float64)
            v2 = np.isfinite(col_f) & np.isfinite(col_s)
            if v2.sum() < 100: continue
            if abs(_spearman(col_f[v2], col_s[v2])) > CORR_THRESH: ok = False; break
        if ok:
            indep.append(f)
            if len(indep) >= MAX_QS_INDEP: break

    if len(indep) < 2:
        return rows

    qs_candidates = [(f['idx'], 1 if f['ic'] > 0 else -1, f['name'], abs(f['ic'])) for f in indep]

    precomp_ranks = {}
    for fi, dsign, fname, ic in qs_candidates:
        col = gp_feat[:, fi].astype(np.float64)
        if dsign == -1: col = -col
        precomp_ranks[(fi, dsign)] = rolling_rank_exclude_self(col, N_ROLL)

    # Pre-compute year masks
    is_yr_mask = (gp_years >= 2021) & (gp_years <= 2025)
    yr_masks = {y: (gp_years == y) for y in range(2021, 2026)}

    # Combo outer, n_split inner — rolling_rank computed once per combo (4x faster)
    cutoffs = {ns: 100.0 * (1 - 1.0 / ns) for ns in N_SPLITS}
    best_per_split = {ns: None for ns in N_SPLITS}

    for k in QS_K_RANGE:
        if k > len(qs_candidates): break
        for cidx in combinations(range(len(qs_candidates)), k):
            feat_keys = [qs_candidates[i][:2] for i in cidx]
            feat_ics = [qs_candidates[i][3] for i in cidx]
            ic_total = sum(feat_ics)

            weighted_sum = np.zeros(n_gp, dtype=np.float64)
            valid = np.ones(n_gp, dtype=bool)
            for j, fkey in enumerate(feat_keys):
                rr = precomp_ranks[fkey]
                valid &= np.isfinite(rr)
                weighted_sum += np.where(np.isfinite(rr), rr * feat_ics[j], 0)
            avg_rank = weighted_sum / ic_total
            avg_rank_pct = rolling_rank_exclude_self(avg_rank, N_ROLL)  # ONCE per combo
            base_valid = valid & np.isfinite(avg_rank_pct) & gp_resolved
            base_valid[:N_ROLL] = False

            for n_split in N_SPLITS:
                cutoff = cutoffs[n_split]
                top_mask = base_valid & (avg_rank_pct >= cutoff)
                top_n = top_mask.sum()
                if top_n < 50: continue

                is_top = top_mask & is_yr_mask
                is_rest = base_valid & ~top_mask & is_yr_mask
                if is_top.sum() < 30 or is_rest.sum() < 100: continue
                if gp_pnl[is_top].mean() <= gp_pnl[is_rest].mean(): continue

                top_best = 0
                for y in range(2021, 2026):
                    yt = is_top & yr_masks[y]; yr_ = is_rest & yr_masks[y]
                    if yt.sum() < 10 or yr_.sum() < 30: continue
                    if gp_pnl[yt].mean() > gp_pnl[yr_].mean(): top_best += 1
                if top_best < 4: continue

                lift = gp_pnl[is_top].mean() - gp_pnl[is_rest].mean()
                if lift < 0.01: continue

                wk_pnl = {}
                et_top = gp_entry[is_top]; pp_top = gp_pnl[is_top]
                for t, p in zip(et_top, pp_top):
                    if t is None: continue
                    iso = t.isocalendar(); wk = f'{iso[0]}-W{iso[1]:02d}'
                    wk_pnl[wk] = wk_pnl.get(wk, 0) + p
                nw = sum(1 for v2 in wk_pnl.values() if v2 < 0)

                yr_evs = []
                for y in range(2021, 2026):
                    ym = is_top & yr_masks[y]
                    if ym.sum() >= 10: yr_evs.append(gp_pnl[ym].mean())
                worst_yr_ev = min(yr_evs) if yr_evs else -999

                score = (top_best + 1, -nw, worst_yr_ev, lift)
                best_c = best_per_split[n_split]
                if best_c is None or score > best_c['_score']:
                    top_conc = apply_max_concurrent(top_mask, gp_entry, gp_exit, MAX_CONCURRENT)
                    final_mask = top_mask & top_conc

                    is_s = compute_period_stats(gp_pnl[final_mask], gp_years[final_mask],
                                                 gp_entry[final_mask], list(range(2021, 2026)), True)
                    oos_s = compute_period_stats(gp_pnl[final_mask], gp_years[final_mask],
                                                  gp_entry[final_mask], [2026], True)

                    feat_names = [qs_candidates[i][2] for i in cidx]
                    feat_ics_dict = {qs_candidates[i][2]: qs_candidates[i][3] * qs_candidates[i][1]
                                     for i in cidx}
                    best_per_split[n_split] = {'is': is_s, 'oos': oos_s, 'feats': feat_names,
                                               'feat_ics': feat_ics_dict, '_score': score}

    for n_split in N_SPLITS:
        best_c = best_per_split[n_split]
        if best_c and best_c['oos'].get('ev', 0) >= 0:
            rows.append({
                'cell': cell_label, 'strat': strat_name,
                'sl_tp': sl_tp_str, 'q': n_split, 'ci': ci,
                **{f'is_{k}': v for k, v in best_c['is'].items()},
                **{f'oos_{k}': v for k, v in best_c['oos'].items()},
                'feats': ', '.join(best_c['feats']),
                'feat_list': best_c['feats'],
                'feat_ics': best_c['feat_ics'],
            })

    return rows


# ── Process one strategy ──────────────────────────────────────────────
def process_strategy(strat_name, cfg):
    """Process all STRONG cells for a strategy. Year-by-year + temp files.

    Phase 1: Load 1 year -> extract per-config masked data -> write to temp pkl -> free year
    Phase 2: Per config: read temp files -> build task -> submit to pool -> free
    Peak RAM: ~6GB (1 year + 1 config in flight)
    """
    import tempfile, pickle as pkl, shutil
    from collections import defaultdict
    import pandas as pd

    strat_path = cfg['path']
    excluded = cfg['excluded']

    sys.path.insert(0, str(strat_path))
    if 'extra_path' in cfg:
        sys.path.insert(0, str(cfg['extra_path']))

    feat_mod = __import__(cfg['features_module'])
    ALL_COL_NAMES = feat_mod.ALL_COL_NAMES

    v2_mod = __import__(cfg['v2_module'])
    cache_obj = getattr(v2_mod, cfg['v2_class'])()

    p2_path = strat_path / cfg['phase2']
    if not p2_path.exists():
        print(f'  {strat_name}: phase2 not found', flush=True)
        return []
    phase2 = json.load(open(p2_path))

    all_rows = []

    # Group cell_keys by direction
    dir_cells = {}
    for cell_key in sorted(phase2.keys()):
        configs = phase2[cell_key]
        if not isinstance(configs, list) or not configs:
            continue
        direction = cell_key.split('_', 1)[0]
        dir_cells.setdefault(direction, []).append(cell_key)

    for direction in sorted(dir_cells.keys()):
        # Build config specs
        config_specs = []
        for cell_key in dir_cells[direction]:
            configs = phase2[cell_key]
            cell = cell_key.split('_', 1)[1]
            t_idx = TREND_MAP[cell.split('_')[0]]; v_idx = VOL_MAP[cell.split('_')[1]]
            for ci, p2_cfg in enumerate(configs):
                sl = p2_cfg['sl']; tp = p2_cfg.get('tp', sl)
                gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<', g['value'])
                              for g in p2_cfg.get('gate', [])]
                block_rules = [(b['name'], b['block_op'], b['value'])
                               for b in p2_cfg.get('blocks', [])]
                gate_feat_names = set(r[0] for r in gate_rules) | set(r[0] for r in block_rules)
                config_specs.append({
                    'cell': cell, 'ci': ci, 'sl': sl, 'tp': tp,
                    'gate_rules': gate_rules, 'block_rules': block_rules,
                    'gate_feat_names': gate_feat_names,
                    't_idx': t_idx, 'v_idx': v_idx,
                })

        if not config_specs:
            continue

        t_dir = time.time()
        temp_dir = tempfile.mkdtemp(prefix='qs_')
        config_has_data = set()

        print(f'  {strat_name} {direction}: {len(config_specs)} configs, extracting to disk...', flush=True)

        # ── Phase 1: Year-by-year extraction to temp files ──────────
        for yr in range(2021, 2027):
            yr_data = cache_obj.load_year(yr)
            if yr_data is None: continue
            other = 'LONG' if direction == 'SHORT' else 'SHORT'
            if other in yr_data: del yr_data[other]
            d = yr_data.get(direction)
            if d is None or d.get('features', np.array([])).size == 0:
                del yr_data; gc.collect(); continue

            feat_yr = d['features']; yrs_yr = d['years']
            trends_yr = d['trends']; vols_yr = d['vols']
            trades_yr = d.get('trades', [])

            # Entry/exit as int64 nanoseconds (compact)
            n_rows = len(feat_yr)
            entry_yr = np.zeros(n_rows, dtype=np.int64)
            exit_yr = np.zeros(n_rows, dtype=np.int64)
            for ti in range(min(len(trades_yr), n_rows)):
                tr = trades_yr[ti]
                if isinstance(tr, dict):
                    e = tr.get('entry_time'); x = tr.get('exit_time')
                    if e is not None: entry_yr[ti] = e.value
                    if x is not None: exit_yr[ti] = x.value
            del trades_yr; d.pop('trades', None)

            # Group by cell -> SL/TP to avoid redundant mask computation
            cell_groups = defaultdict(list)
            for ci_idx, spec in enumerate(config_specs):
                cell_groups[(spec['t_idx'], spec['v_idx'])].append((ci_idx, spec))

            n_written = 0
            for (t_i, v_i), group in cell_groups.items():
                cell_mask = (trends_yr == t_i) & (vols_yr == v_i)
                if not cell_mask.any(): continue

                sltp_groups = defaultdict(list)
                for ci_idx, spec in group:
                    sltp_groups[(spec['sl'], spec['tp'])].append((ci_idx, spec))

                for (sl, tp), sltp_grp in sltp_groups.items():
                    cell_pnl = d.get(f'pnl_{sl}_{tp}')
                    if cell_pnl is None: cell_pnl = d.get(f'pnl_{sl}')
                    if cell_pnl is None: continue
                    resolved_base = cell_mask & np.isfinite(cell_pnl) & (cell_pnl != 0)

                    for ci_idx, spec in sltp_grp:
                        gp_mask = apply_gates(feat_yr, resolved_base, spec['gate_rules'],
                                              spec['block_rules'], ALL_COL_NAMES)
                        gp_mask &= (yrs_yr >= 2021)
                        if gp_mask.sum() == 0: continue

                        temp_path = os.path.join(temp_dir, f'c{ci_idx}_y{yr}.pkl')
                        with open(temp_path, 'wb') as f:
                            pkl.dump((feat_yr[gp_mask], cell_pnl[gp_mask],
                                      yrs_yr[gp_mask], entry_yr[gp_mask],
                                      exit_yr[gp_mask]), f, protocol=4)
                        config_has_data.add(ci_idx)
                        n_written += 1

            del yr_data, d, feat_yr, yrs_yr, trends_yr, vols_yr, entry_yr, exit_yr
            gc.collect()
            print(f'    Year {yr}: {n_written} chunks written ({time.time()-t_dir:.0f}s)', flush=True)

        # ── Phase 2: Load per-config from disk, submit to pool ──────
        valid_configs = []
        for ci_idx in sorted(config_has_data):
            tfiles = [os.path.join(temp_dir, f'c{ci_idx}_y{yr}.pkl')
                      for yr in range(2021, 2027)
                      if os.path.exists(os.path.join(temp_dir, f'c{ci_idx}_y{yr}.pkl'))]
            if tfiles:
                valid_configs.append((ci_idx, config_specs[ci_idx], tfiles))

        n_total = len(valid_configs)
        if n_total == 0:
            shutil.rmtree(temp_dir, ignore_errors=True); continue
        print(f'  {strat_name} {direction}: {n_total} valid configs -> pool(10)...', flush=True)

        def _load_and_build_task(vi):
            """Load one config from temp files, return task args or None."""
            ci_idx, spec, tfiles = valid_configs[vi]
            ff, pp, yy, ee, xx = [], [], [], [], []
            for tf in tfiles:
                with open(tf, 'rb') as fh:
                    f_, p_, y_, e_, x_ = pkl.load(fh)
                ff.append(f_); pp.append(p_); yy.append(y_)
                ee.append(e_); xx.append(x_)
                os.unlink(tf)
            feat = np.vstack(ff); pnl = np.concatenate(pp)
            yrs = np.concatenate(yy)
            et_ns = np.concatenate(ee); xt_ns = np.concatenate(xx)
            del ff, pp, yy, ee, xx
            n_gp = len(feat)
            if n_gp < 500:
                return None
            # Convert int64 ns -> Timestamp objects
            et = np.empty(n_gp, dtype=object); et[:] = None
            xt = np.empty(n_gp, dtype=object); xt[:] = None
            v_et = et_ns != 0
            if v_et.any(): et[v_et] = list(pd.DatetimeIndex(et_ns[v_et]))
            v_xt = xt_ns != 0
            if v_xt.any(): xt[v_xt] = list(pd.DatetimeIndex(xt_ns[v_xt]))
            del et_ns, xt_ns
            resolved = np.isfinite(pnl) & (pnl != 0)
            return (feat, pnl, yrs, et, xt, resolved, n_gp,
                    ALL_COL_NAMES, excluded, spec['gate_feat_names'],
                    direction, spec['cell'], strat_name,
                    spec['sl'], spec['tp'], spec['ci'])

        dir_rows = []; done = 0; submitted = 0
        vi_iter = iter(range(n_total))

        with ProcessPoolExecutor(max_workers=10) as pool:
            active = {}
            # Initial fill (2x workers for overlap)
            while len(active) < min(20, n_total):
                try: vi = next(vi_iter)
                except StopIteration: break
                task = _load_and_build_task(vi)
                if task is not None:
                    future = pool.submit(process_one_cell_config, task)
                    del task
                    active[future] = vi; submitted += 1

            while active:
                for future in as_completed(active):
                    result_rows = future.result()
                    dir_rows.extend(result_rows)
                    del active[future]; done += 1

                    # Submit next
                    while len(active) < min(20, n_total - done):
                        try: vi = next(vi_iter)
                        except StopIteration: break
                        task = _load_and_build_task(vi)
                        if task is not None:
                            nf = pool.submit(process_one_cell_config, task)
                            del task
                            active[nf] = vi; submitted += 1
                        break  # submit one at a time to keep pipeline steady

                    if done % 10 == 0 or done == submitted:
                        print(f'    [{done}/{submitted}] {len(dir_rows)} rows ({time.time()-t_dir:.0f}s)', flush=True)
                    break  # re-enter as_completed

        shutil.rmtree(temp_dir, ignore_errors=True)
        all_rows.extend(dir_rows)
        n_qs = sum(1 for r in dir_rows if r['q'] > 1)
        print(f'  {strat_name} {direction}: {len(dir_rows)} rows ({n_qs} QS) ({time.time()-t_dir:.0f}s)', flush=True)
        gc.collect()

    return all_rows


# ── Run comparison (main logic) ───────────────────────────────────────
def run_comparison(strategies, title, out_dir):
    """Run unified cell comparison for a group of strategies."""
    t0 = time.time()
    strat_names = ' + '.join(strategies.keys())
    print(f'{"="*160}', flush=True)
    print(f'  UNIFIED CELL COMPARISON: {title} ({strat_names})', flush=True)
    print(f'  q=1 baseline + q=2..5 QS | Max {MAX_CONCURRENT} concurrent | IS 2021-2025 | OOS 2026', flush=True)
    print(f'  Parallel: year-by-year loading, 10 workers', flush=True)
    print(f'{"="*160}', flush=True)

    out_path = out_dir / 'unified_cell_comparison.txt'

    all_rows = []
    for strat_name, cfg in strategies.items():
        print(f'\n  Loading {strat_name}...', flush=True)
        rows = process_strategy(strat_name, cfg)
        all_rows.extend(rows)
        print(f'  {strat_name}: {len(rows)} rows', flush=True)
        gc.collect()

        # Incremental save
        _save_txt(all_rows, out_path)
        print(f'  Incremental save: {len(all_rows)} total rows -> {out_path.name}', flush=True)

    # Print table
    _print_table(all_rows)

    # Final save
    _save_txt(all_rows, out_path)

    # Save QS features JSON
    qs_json = {}
    for r in all_rows:
        if r['q'] <= 1: continue
        if not r.get('feat_list'): continue
        direction = 'SHORT' if r['cell'].startswith('S_') else 'LONG'
        cell = r['cell'][2:]
        ci = r.get('ci', 0)
        key = f'{direction}_{cell}_{r["strat"]}_q{r["q"]}_c{ci}'
        qs_json[key] = {
            'features': r['feat_list'],
            'ic': {k: round(v, 6) for k, v in r['feat_ics'].items()},
            'sl': float(r['sl_tp'].split('/')[0]),
            'tp': float(r['sl_tp'].split('/')[1].rstrip('*')),
        }
    qs_path = out_dir / 'unified_qs_features.json'
    json.dump(qs_json, open(qs_path, 'w'), indent=2)

    print(f'\n  Saved {out_path.name} + {qs_path.name} ({len(qs_json)} QS entries)', flush=True)
    print(f'  DONE ({time.time()-t0:.0f}s)', flush=True)


def _save_txt(all_rows, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in sorted(all_rows, key=lambda x: (x['cell'], x['strat'], x.get('ci', 0), x['q'])):
            feats = r.get('feats', '(all)') if r['q'] > 1 else '(all)'
            f.write(f'{r["cell"]:<14} {r["strat"]:>5} {r["sl_tp"]:>7} {r["q"]:>2} {r.get("ci",0):>3} |'
                    f' {r.get("is_n",0):>6,} {r.get("is_wr",0):>5.1f}% {r.get("is_ev",0):>+6.3f} {r.get("is_neg_w",0):>5} |'
                    f' {r.get("oos_n",0):>5,} {r.get("oos_wr",0):>5.1f}% {r.get("oos_ev",0):>+6.3f} {r.get("oos_neg_w",0):>5} |'
                    f' {feats}\n')


def _print_table(all_rows):
    print(f'\n{"="*160}', flush=True)
    print(f'  {"Cell":<14} {"S":>5} {"SL/TP":>7} {"q":>2} {"ci":>3} |'
          f' {"N_IS":>6} {"WR_IS":>6} {"EV_IS":>7} {"NW_IS":>5} |'
          f' {"N_26":>5} {"WR_26":>6} {"EV_26":>7} {"NW_26":>5} |'
          f' Features', flush=True)
    print(f'  {"-"*155}', flush=True)

    prev_cell_strat = None
    for r in sorted(all_rows, key=lambda x: (x['cell'], x['strat'], x.get('ci', 0), x['q'])):
        cs = (r['cell'], r['strat'], r.get('ci', 0))
        if cs != prev_cell_strat and prev_cell_strat is not None:
            print(f'  {"-"*155}', flush=True)
        prev_cell_strat = cs

        feats = r.get('feats', '(all)') if r['q'] > 1 else '(all)'
        print(f'  {r["cell"]:<14} {r["strat"]:>5} {r["sl_tp"]:>7} {r["q"]:>2} {r.get("ci",0):>3} |'
              f' {r.get("is_n",0):>6,} {r.get("is_wr",0):>5.1f}% {r.get("is_ev",0):>+6.3f} {r.get("is_neg_w",0):>5} |'
              f' {r.get("oos_n",0):>5,} {r.get("oos_wr",0):>5.1f}% {r.get("oos_ev",0):>+6.3f} {r.get("oos_neg_w",0):>5} |'
              f' {feats}', flush=True)
