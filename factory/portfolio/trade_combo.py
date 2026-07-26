"""Trade-level combo selector (portfolio assembly).

Reads pre-saved trade pickles (save_cell_trades), brute-forces config combos
per cell with fast_sim for real NW, DD, net profit. Combos evaluated in
PARALLEL (cpu+ram-aware); writes top combos to results/trade_combo_{cell}.txt.
"""
import sys, os, time, json
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Rule B (close-on-opposite + skip-both-fire) is the production default and the ONLY
# mode now: writes trade_combo_<cell>.txt (canonical un-suffixed) and builds a cache
# price map for the early exit. The old A (1/coin skip) / _B env-gated split is gone.
import os as _os
_RULE = 'B'   # Rule B (close-on-opposite + skip-both-fire) is the production default
_CACHE_DIR = ROOT / 'data' / 'cache'
_GPRICE = {}   # worker global: {cell: {coin: {et: close}}}

from factory.qs.age_filter import norm_factor   # per-month-universe normalization
from factory.portfolio.budget_merge import sv_budget, round_robin_merge, BUDGETS  # DD-diverse retention
import os
_WORK = Path(os.environ.get('TM_WORK', ROOT))  # isolated workdir (parallel arms)

# Blind walk-forward: the OOS boundary is parametric. Default = the production
# path bit-identically; quarter replays with an earlier cutoff pass
# TM_OOS_START/TM_OOS_WSTART.
_OOS_MS = os.environ.get('TM_OOS_START', '2026')    # ay-anahtari esigi
_OOS_WS = os.environ.get('TM_OOS_WSTART', '2026')   # hafta-anahtari esigi
_OOS_YR = int(_OOS_MS[:4])

# Strategy registry: cat/strat -> (path, p2_file, feat_module, cache_module)
STRAT_REGISTRY = {
    'TOYSET/SMA_X': (ROOT/'strategies'/'momentum_sma_cross', 'momentum_sma_cross_phase2_results.json', 'sma_cross_features', 'sma_cross_cache_builder'),
    'TOYSET/RSI_MR': (ROOT/'strategies'/'meanrev_rsi', 'meanrev_rsi_phase2_results.json', 'rsi_mr_features', 'rsi_mr_cache_builder'),
    'TOYSET/CH_BRK': (ROOT/'strategies'/'breakout_channel', 'breakout_channel_phase2_results.json', 'channel_brk_features', 'channel_brk_cache_builder'),
}

TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
MOM_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}
TREND_MAP = {'BEAR': 0, 'FLAT': 1, 'BULL': 2}
VOL_MAP = {'LOW': 0, 'MED': 1, 'HIGH': 2}

N_ROLL = 100
FEE_RT = 0.08
MAX_CONCURRENT = 100

# Cache for loaded strategy data
_cache_store = {}
_feat_store = {}


def rolling_rank_exclude_self(vals, n_roll):
    s = pd.Series(vals)
    rr = s.rolling(n_roll + 1, min_periods=21).rank()
    n_in = s.rolling(n_roll + 1, min_periods=21).count()
    result = np.where(n_in > 1, (rr - 1) / (n_in - 1) * 100, np.nan)
    result[:n_roll] = np.nan
    return result


def load_strategy(cat_strat):
    """Load cache + phase2 for a strategy. Cached."""
    if cat_strat in _cache_store:
        return _cache_store[cat_strat], _feat_store[cat_strat]

    if cat_strat not in STRAT_REGISTRY:
        print(f'    WARNING: {cat_strat} not in registry')
        return None, None

    path, p2_file, feat_mod_name, cache_mod_name = STRAT_REGISTRY[cat_strat]

    sys.path.insert(0, str(path))

    feat_mod = __import__(feat_mod_name)
    ALL_COL_NAMES = feat_mod.ALL_COL_NAMES

    cache_mod = __import__(cache_mod_name)
    cache = cache_mod.load_or_build()

    phase2 = json.load(open(path / p2_file))

    # Find QS features files (one per strategy category)
    qs_files = []
    for cat_dir in [ROOT / 'strategies']:
        qf = cat_dir / 'unified_qs_features.json'
        if qf.exists():
            qs_files.append(qf)

    data = {'cache': cache, 'phase2': phase2, 'qs_files': qs_files, 'path': path}
    _cache_store[cat_strat] = data
    _feat_store[cat_strat] = ALL_COL_NAMES
    return data, ALL_COL_NAMES


def build_cell_trades(direction, cell, cat_strat, q_level, ci_target=None):
    """Build trade list for one config. Returns list of trade dicts."""
    data, ALL_COL_NAMES = load_strategy(cat_strat)
    if data is None:
        return []

    cache = data['cache']
    phase2 = data['phase2']

    cell_key = f'{direction}_{cell}'
    p2_list = phase2.get(cell_key, [])
    if not isinstance(p2_list, list) or not p2_list:
        return []

    # Pick the right config (by ci index)
    if ci_target is not None and ci_target < len(p2_list):
        p2 = p2_list[ci_target]
    else:
        p2 = p2_list[0]

    sl = p2['sl']; tp = p2.get('tp', sl)

    gate_rules = [(g['name'], '>' if g['direction'] == 'gt' else '<', g['value'])
                  for g in p2.get('gate', [])]
    block_rules = [(b['name'], b['block_op'], b['value'])
                   for b in p2.get('blocks', [])]

    d = cache[direction]
    features = d['features']; years = d['years']
    trends = d['trends']; vols = d['vols']; trades = d['trades']

    t_idx = TREND_MAP[cell.split('_')[0]]
    v_idx = VOL_MAP[cell.split('_')[1]]
    cell_mask = (trends == t_idx) & (vols == v_idx)

    if sl == tp:
        cell_pnl = d.get(f'pnl_{sl}')
    else:
        cell_pnl = d.get(f'pnl_{sl}_{tp}', d.get(f'pnl_{sl}'))
    if cell_pnl is None:
        return []

    resolved = cell_mask & np.isfinite(cell_pnl) & (cell_pnl != 0)
    gp_mask = resolved.copy()

    # Apply gates
    for feat_name, op, value in gate_rules:
        if feat_name not in ALL_COL_NAMES: continue
        fi = ALL_COL_NAMES.index(feat_name)
        col = features[:, fi].astype(np.float64); valid = np.isfinite(col)
        if op == '<': gp_mask &= (valid & (col < value))
        else: gp_mask &= (valid & (col > value))

    # Apply blocks
    for feat_name, op, value in block_rules:
        if feat_name not in ALL_COL_NAMES: continue
        fi = ALL_COL_NAMES.index(feat_name)
        col = features[:, fi].astype(np.float64); valid = np.isfinite(col)
        if op == '<': gp_mask &= ~(valid & (col < value))
        else: gp_mask &= ~(valid & (col > value))

    gp_mask &= (years >= 2021)
    n_gp = gp_mask.sum()
    if n_gp < 50:
        return []

    # chronological order for the QS rolling: load_lite rows are coin-major, but
    # qs_core's rolling-rank quintile assumes a time-ordered stream (== the live bot,
    # which accumulates bar-by-bar). Sort gate-passed indices by entry time.
    _ridx = np.where(np.isfinite(d['pnl']))[0]
    et_full = np.zeros(len(features), dtype=np.int64)
    for _ti, _fi in enumerate(_ridx):
        if _ti < len(trades):
            _tr = trades[_ti]
            if isinstance(_tr, dict) and _tr.get('entry_time') is not None:
                et_full[_fi] = _tr['entry_time'].value
    gp_indices = np.where(gp_mask)[0]
    gp_indices = gp_indices[np.argsort(et_full[gp_indices], kind='stable')]

    # QS filtering
    if q_level > 1:
        gp_feat = features[gp_indices]; gp_pnl = cell_pnl[gp_indices]
        gp_years = years[gp_indices]
        gp_resolved = np.isfinite(gp_pnl) & (gp_pnl != 0)

        # Find QS features from unified files
        strat_short = cat_strat.split('/')[1]
        qs_key_prefix = f'{direction}_{cell}_{strat_short}_q{q_level}'
        qs_feats = None
        ic_map = None

        for qf in data['qs_files']:
            qdata = json.load(open(qf))
            for k, v in qdata.items():
                if k.startswith(qs_key_prefix):
                    if isinstance(v, dict) and 'features' in v:
                        qs_feats = v['features']
                        ic_map = v.get('ic', {})
                    break
            if qs_feats:
                break

        if qs_feats is None:
            # Fallback: auto-select
            from scipy.stats import spearmanr as _spr
            is_mask = gp_years <= 2025
            is_feat = gp_feat[is_mask]; is_pnl = gp_pnl[is_mask]
            is_resolved = np.isfinite(is_pnl) & (is_pnl != 0)
            feat_mod = __import__(STRAT_REGISTRY[cat_strat][2])
            stats = feat_mod.compute_all_feature_stats(is_feat, is_pnl, gp_years[is_mask])
            stable = [f for f in stats if f.get('stab', 0) >= 2 and f.get('n_yr', 0) >= 2 and f.get('abs_ic', 0) >= 0.01]
            seen = set(); deduped = []
            for f in sorted(stable, key=lambda x: x['abs_ic'], reverse=True):
                base = f['name'].rsplit('_', 1)[0] if f['name'].rsplit('_', 1)[-1] in ('1h', '4h', '15m') else f['name']
                if base not in seen: deduped.append(f); seen.add(base)
            indep = []
            for f in deduped:
                col_f = is_feat[is_resolved, f['idx']].astype(np.float64)
                ok = True
                for s in indep:
                    col_s = is_feat[is_resolved, s['idx']].astype(np.float64)
                    v = np.isfinite(col_f) & np.isfinite(col_s)
                    if v.sum() < 100: continue
                    rho, _ = _spr(col_f[v], col_s[v])
                    if abs(rho) > 0.6: ok = False; break
                if ok: indep.append(f)
            qs_feats = [f['name'] for f in indep[:5]]
            ic_map = {f['name']: f['ic'] for f in stats}

        if qs_feats:
            feat_ranks = {}; ic_weights = {}
            for fname in qs_feats:
                if fname not in ALL_COL_NAMES: continue
                fi = ALL_COL_NAMES.index(fname)
                ic = ic_map.get(fname, 0) if ic_map else 0
                dsign = 1 if ic > 0 else -1
                col = gp_feat[:, fi].astype(np.float64)
                if dsign == -1: col = -col
                feat_ranks[fname] = rolling_rank_exclude_self(col, N_ROLL)
                ic_weights[fname] = abs(ic) if abs(ic) > 0 else 0.01

            if feat_ranks:
                ic_total = sum(ic_weights[f] for f in feat_ranks)
                weighted_sum = np.zeros(n_gp, dtype=np.float64)
                for fname, rr in feat_ranks.items():
                    weighted_sum += np.where(np.isfinite(rr), rr * ic_weights[fname], 0)
                avg_rank = weighted_sum / ic_total
                avg_rank_pct = rolling_rank_exclude_self(avg_rank, N_ROLL)
                cutoff = 100.0 * (1 - 1.0 / q_level)
                sel_mask = gp_resolved & np.isfinite(avg_rank_pct) & (avg_rank_pct >= cutoff)
            else:
                sel_mask = gp_resolved
        else:
            sel_mask = gp_resolved
    else:
        sel_mask = np.isfinite(cell_pnl[gp_indices]) & (cell_pnl[gp_indices] != 0)

    # Extract trades (gp_indices already chronological)
    sel_indices = gp_indices[sel_mask]
    default_pnl = d['pnl']
    resolved_all = np.isfinite(default_pnl)
    resolved_idx = np.where(resolved_all)[0]
    trade_idx_map = {}
    for ti, fi in enumerate(resolved_idx):
        if ti < len(trades):
            trade_idx_map[fi] = ti

    result = []
    for fi in sel_indices:
        ti = trade_idx_map.get(fi)
        if ti is not None:
            t = trades[ti]
            if t.get('entry_time') and t.get('exit_time'):
                result.append({
                    'entry_time': t['entry_time'], 'exit_time': t['exit_time'],
                    'coin': t['coin'], 'pnl_pct': float(cell_pnl[fi]),
                    'direction': direction, 'cell': cell,
                })
    return result


def build_master_timeline(trade_sets):
    """Build optimized per-key timelines."""
    per_key_tg = {}
    per_key_times = {}
    for label, trades in trade_sets.items():
        tg = defaultdict(list)
        for t in trades:
            et = t['entry_time']
            iso = et.isocalendar()
            tg[et].append((t['coin'], t['pnl_pct'], t['exit_time'],
                           et.strftime('%Y-%m'), f'{iso[0]}-W{iso[1]:02d}',
                           t.get('direction', '')))   # [5] direction (rule B; A ignores)
        per_key_tg[label] = dict(tg)
        per_key_times[label] = set(tg.keys())
    return per_key_tg, per_key_times


def fast_sim(per_key_tg, per_key_times, active_keys, price=None):
    """Run simulation with real trade overlap/concurrent control.

    rule B (price given + _RULE=='B'): a held coin's OPPOSITE-direction trade
    closes the position early (cache close-to-close); BOTH long+short on a coin
    with no position is skipped. Otherwise (A): 1 position/coin, skip if held."""
    rule_b = (_RULE == 'B' and price is not None)
    all_times = set()
    for k in active_keys:
        all_times |= per_key_times[k]
    sorted_times = sorted(all_times)

    cash = 10000.0; wd = 0.0; opos = {}
    opened = 0; skipped = 0
    peak_eq = 10000.0; max_dd = 0.0
    mo = defaultdict(float); wk = defaultdict(float)
    mo_tr = defaultdict(int)   # monthly trade count (per-month-universe norm)
    yr_dd = defaultdict(float)  # per-year max DD (for recent-regime score)
    fee_frac = FEE_RT / 100
    # Track executed trade stats
    exec_is_w = exec_is_t = exec_oos_w = exec_oos_t = 0
    exec_is_pnl = exec_oos_pnl = 0.0

    for et in sorted_times:
        expired = [c for c, p in opos.items() if p[0] <= et]
        for c in expired:
            p = opos.pop(c); ps = p[1]; pnl = p[2]   # p[0]=xt; works for 3- and 5-tuple
            cash += ps + ps * (pnl / 100) - ps * fee_frac
        lk = sum(p[1] for p in opos.values()); eq = cash + lk
        if eq > 10000 and cash > 0:
            w = min(eq - 10000, cash); cash -= w; wd += w; eq = cash + lk
        if eq > peak_eq: peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100 if peak_eq > 0 else 0
        if dd > max_dd: max_dd = dd
        if dd > yr_dd[et.year]: yr_dd[et.year] = dd

        cs = []
        for k in active_keys:
            kt = per_key_tg[k].get(et)
            if kt:
                for tr in kt:
                    if rule_b or tr[0] not in opos:   # B: include held (detect opposite)
                        cs.append(tr)
        both = set()
        if rule_b:
            _ds = defaultdict(set)
            for tr in cs:
                _ds[tr[0]].add(tr[5])
            both = {c for c, s in _ds.items() if len(s) >= 2}
        for tr in cs:
            c0 = tr[0]
            if c0 in opos:
                if not rule_b:
                    continue                          # A: dup coin -> capital-leak fix
                pos = opos[c0]                        # (xt,ps,pnl,dir,entry_et,entry_mo,entry_wk)
                if pos[3] == tr[5]:
                    continue                          # same dir -> skip (1/coin)
                pm = price.get(c0, {}) if price else {}
                ep = pm.get(pos[4]); pxe = pm.get(et)   # price lookup ONLY on conflict (~3%)
                if ep is None or pxe is None or ep <= 0:
                    continue                          # can't price -> hold
                early = (pxe / ep - 1) * 100 if pos[3] == 'LONG' else (1 - pxe / ep) * 100
                p = opos.pop(c0); ps2 = p[1]
                early_real = ps2 * (early / 100) - ps2 * fee_frac
                full_booked = ps2 * (p[2] / 100) - ps2 * fee_frac   # p[2]=held pnl_pct
                cash += max(ps2 + early_real, 0)
                # correct ENTRY-bar booking: replace assumed-full with early-realized
                mo[p[5]] += early_real - full_booked   # p[5]=entry month
                wk[p[6]] += early_real - full_booked   # p[6]=entry week
                # fix executed-WR if the realized sign flipped vs assumed-full
                if (p[2] > 0) != (early > 0):
                    if p[5] >= _OOS_MS: exec_oos_w += 1 if early > 0 else -1
                    else:                        exec_is_w += 1 if early > 0 else -1
                continue
            if rule_b and c0 in both:
                continue                              # both-fire, no position -> skip
            if len(opos) >= MAX_CONCURRENT: skipped += 1; continue
            ps = eq * 0.01
            if ps < 2.5: skipped += 1; continue
            if ps > cash: ps = cash
            if ps < 2.5: skipped += 1; continue
            cash -= ps
            if rule_b:
                # store entry_et (price looked up lazily on conflict) + entry month/week
                opos[c0] = (tr[2], ps, tr[1], tr[5], et, tr[3], tr[4])
            else:
                opos[c0] = (tr[2], ps, tr[1])
            pnl = ps * (tr[1] / 100) - ps * fee_frac
            mo[tr[3]] += pnl; wk[tr[4]] += pnl
            mo_tr[tr[3]] += 1
            opened += 1
            # Track executed trade WR/EV by period
            is_2026 = tr[3] >= _OOS_MS
            if is_2026:
                exec_oos_t += 1; exec_oos_pnl += tr[1]
                if tr[1] > 0: exec_oos_w += 1
            else:
                exec_is_t += 1; exec_is_pnl += tr[1]
                if tr[1] > 0: exec_is_w += 1

    for c, p in opos.items():
        cash += p[1] + p[1] * (p[2] / 100) - p[1] * fee_frac
    eq = cash
    if eq > 10000: wd += eq - 10000
    net = wd + min(eq, 10000) - 10000
    nm = sum(1 for v in mo.values() if v < 0)
    nw = sum(1 for v in wk.values() if v < 0)
    nw26 = sum(1 for k, v in wk.items() if k >= _OOS_WS and v < 0)
    net26 = sum(v for k, v in mo.items() if k >= _OOS_MS)
    # per-month-universe normalized magnitudes (x350/univ_month)
    net_norm = sum(v * norm_factor(k) for k, v in mo.items())
    net26_norm = sum(v * norm_factor(k) for k, v in mo.items() if k >= _OOS_MS)
    oos_n_norm = sum(c * norm_factor(k) for k, c in mo_tr.items() if k >= _OOS_MS)
    # Robustness scalars (behavior-neutral record keys):
    # worst_yr = worst booking-net year; h2_share = second-half net / |total|
    # (anti-recency); top5wk_share = share of the best 5 weeks in positive sum.
    _yrn = {}
    for _k, _v in mo.items():
        _yrn[_k[:4]] = _yrn.get(_k[:4], 0.0) + _v
    worst_yr = min(_yrn.values()) if _yrn else 0.0
    # Universe-growth correction: the monthly series is NORM-weighted so the
    # h2/worst month-week terms are free of the coin-count artifact.
    _mon = {_k: _v * norm_factor(_k) for _k, _v in mo.items()}
    _mks = sorted(_mon)
    _tot = sum(_mon.values())
    h2_share = (sum(_mon[_k] for _k in _mks[len(_mks) // 2:]) / abs(_tot)) \
        if _tot else 0.0
    worst_mo_n = min(_mon.values()) if _mon else 0.0
    def _wk_nf(_k):
        # 'YYYY-W07' -> approximate month (ISO week/4.33) -> norm_factor
        _y, _w = _k.split('-W')
        return norm_factor(f'{_y}-{min(12, int(_w) // 4 + 1):02d}')
    _wkn = {_k: _v * _wk_nf(_k) for _k, _v in wk.items()}
    worst_wk_n = min(_wkn.values()) if _wkn else 0.0
    _wpos = sorted((v for v in _wkn.values() if v > 0), reverse=True)
    top5wk_share = (sum(_wpos[:5]) / sum(_wpos)) if _wpos else 1.0
    # recent-regime (2024-26, large universe) magnitudes for the v2 score variant
    _R = (_OOS_YR - 2, _OOS_YR - 1, _OOS_YR)
    net_recent = sum(v for k, v in mo.items() if int(k[:4]) in _R)
    recent_n = sum(c for k, c in mo_tr.items() if int(k[:4]) in _R)
    dd_recent = max((yr_dd[y] for y in _R if y in yr_dd), default=max_dd)
    return {'net': round(net), 'nm': nm, 'nw': nw, 'nw26': nw26, 'net26': round(net26),
            'worst_yr': worst_yr, 'h2_share': h2_share, 'top5wk_share': top5wk_share,
            'worst_mo_n': worst_mo_n, 'worst_wk_n': worst_wk_n,
            'net_norm': net_norm, 'net26_norm': net26_norm, 'oos_n_norm': oos_n_norm,
            'net_recent': round(net_recent), 'recent_n': recent_n,
            'dd_recent': round(dd_recent, 2),
            'opened': opened, 'skipped': skipped, 'max_dd': round(max_dd, 2),
            'exec_is_w': exec_is_w, 'exec_is_t': exec_is_t,
            'exec_oos_w': exec_oos_w, 'exec_oos_t': exec_oos_t,
            'exec_is_pnl': exec_is_pnl, 'exec_oos_pnl': exec_oos_pnl}


def parse_cell_configs(cell_name, top_per_dir=8):
    """Read cell file, get top configs grouped by direction+strategy.
    Returns dict of slot_key -> list of (label, config_info) sorted by score."""
    cell_path = _WORK / 'cells' / f'{cell_name}.txt'
    if not cell_path.exists():
        return {}

    slots = {}  # (direction, cat_strat) -> [(score, cat, strat, sl_tp, q, ci)]
    with open(cell_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 4: continue
            hdr = parts[0].split()
            if len(hdr) < 6: continue
            try:
                rank = int(hdr[0])
            except ValueError:
                continue
            cat = hdr[1]; strat = hdr[2]; sl_tp = hdr[3]; q = int(hdr[4]); ci = int(hdr[5])
            score = float(parts[3].strip().split()[0])

            # Direction from cell name prefix
            direction = 'LONG' if cell_name.startswith('L_') else 'SHORT'
            cat_strat = f'{cat}/{strat}'
            slot_key = (direction, cat_strat)

            if slot_key not in slots:
                slots[slot_key] = []
            slots[slot_key].append((score, cat, strat, sl_tp, q, ci, cat_strat))

    # Sort each slot by score, keep top 2
    for key in slots:
        slots[key].sort(key=lambda x: -x[0])
        slots[key] = slots[key][:2]

    # Separate L and S, take top N per direction
    l_slots = {k: v for k, v in slots.items() if k[0] == 'LONG'}
    s_slots = {k: v for k, v in slots.items() if k[0] == 'SHORT'}

    # Sort slots by best score
    l_sorted = sorted(l_slots.items(), key=lambda x: -x[1][0][0])[:top_per_dir]
    s_sorted = sorted(s_slots.items(), key=lambda x: -x[1][0][0])[:top_per_dir]

    return dict(l_sorted + s_sorted)


def parse_top_n_configs(cell_arg, top_n=50, top_per_slot=2,
                        l_top_n=None, l_tps=None, s_top_n=None, s_tps=None):
    """Read L_ and S_ cell files separately, take top N per direction,
    group into slots (direction+strategy), keep top_per_slot per slot.

    Per-direction overrides: l_top_n/l_tps for LONG, s_top_n/s_tps for SHORT.
    Falls back to top_n/top_per_slot if not specified.
    """
    all_slots = {}
    for prefix in ['L', 'S']:
        cell_file = f'{prefix}_{cell_arg}'
        direction = 'LONG' if prefix == 'L' else 'SHORT'
        tn = (l_top_n if prefix == 'L' else s_top_n) or top_n
        tps = (l_tps if prefix == 'L' else s_tps) or top_per_slot
        cell_path = _WORK / 'cells' / f'{cell_file}.txt'
        if not cell_path.exists():
            continue
        configs = []
        with open(cell_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 4: continue
                hdr = parts[0].split()
                if len(hdr) < 6: continue
                try:
                    rank = int(hdr[0])
                except ValueError:
                    continue
                cat = hdr[1]; strat = hdr[2]; sl_tp = hdr[3]
                q = int(hdr[4]); ci = int(hdr[5])
                score = float(parts[3].strip().split()[0])
                cat_strat = f'{cat}/{strat}'
                configs.append((score, direction, cat_strat, sl_tp, q, ci))

        configs.sort(key=lambda x: -x[0])
        top_configs = configs[:tn]

        for score, direction, cat_strat, sl_tp, q, ci in top_configs:
            slot_key = (direction, cat_strat)
            if slot_key not in all_slots:
                all_slots[slot_key] = []
            if len(all_slots[slot_key]) < tps:
                all_slots[slot_key].append((score, direction, cat_strat, sl_tp, q, ci))

    return all_slots


ALL_CELLS = ['BEAR_HIGH', 'BEAR_MED', 'BEAR_LOW',
             'FLAT_HIGH', 'FLAT_MED', 'FLAT_LOW',
             'BULL_HIGH', 'BULL_MED', 'BULL_LOW']


def auto_workers(per_worker_data, label='veri'):
    """cpu AND ram: each spawned worker gets a pickled copy of per_worker_data."""
    import pickle, psutil
    sz = max(1, len(pickle.dumps(per_worker_data, protocol=4)))
    avail = psutil.virtual_memory().available
    by_ram = max(1, int((avail - 3e9) // sz))
    by_cpu = max(1, (os.cpu_count() or 8) - 2)
    n = max(1, min(by_cpu, by_ram))
    hard = os.environ.get('TM_MAX_WORKERS')   # paralel turnuva: chain-basi kısıt
    if hard:
        n = max(1, min(n, int(hard)))
    print(f'  worker: cpu={by_cpu}, ram-limit={by_ram} '
          f'({label} {sz/1e6:.0f}MB/worker, bos {avail/1e9:.0f}GB) -> {n}', flush=True)
    return n


_w_tg = _w_times = None

def _init_combo_worker(per_key_tg, per_key_times):
    global _w_tg, _w_times
    _w_tg, _w_times = per_key_tg, per_key_times

def _combo_eval(active_tuple):
    """Worker: fast_sim one combo (shared timeline via initializer)."""
    active = list(active_tuple)
    r = fast_sim(_w_tg, _w_times, active)
    if r['net'] <= 0 or r['net26'] <= 0:
        return None
    r['n_active'] = len(active)
    r['combo'] = active_tuple
    r['is_wr'] = round(r['exec_is_w'] / r['exec_is_t'] * 100, 1) if r['exec_is_t'] else 0
    r['oos_wr'] = round(r['exec_oos_w'] / r['exec_oos_t'] * 100, 1) if r['exec_oos_t'] else 0
    r['is_ev'] = round(r['exec_is_pnl'] / r['exec_is_t'], 3) if r['exec_is_t'] else 0
    r['oos_ev'] = round(r['exec_oos_pnl'] / r['exec_oos_t'], 3) if r['exec_oos_t'] else 0
    r['is_n'] = r['exec_is_t']; r['oos_n'] = r['exec_oos_t']
    return r


# ── GLOBAL combo queue (cells overlap, all cores busy) ──────────────
import math as _math
MAX_SLOTS_PER_DIR = 6
MAX_ACTIVE_PER_DIR = 4

# A/B arm flags — default OFF = bit-identical production path.
#  TM_JK_BEAM=1   : cell menu ranked by year-jackknife MEDIAN-RANK (the
#                   DD-diverse budget-heap pool is re-simulated under 5
#                   drop-year variants; pool source unchanged).
#  TM_DIV_QUOTA=1 : top-100 menu round-robins style x dir x DD buckets.
_JK_BEAM = os.environ.get('TM_JK_BEAM') == '1'
_DIV_QUOTA = os.environ.get('TM_DIV_QUOTA') == '1'
_DIVCRASH = os.environ.get('TM_DIVCRASH') == '1'   # co-crash penalty rerank
_JK_YEARS = (2021, 2022, 2023, 2024, 2025)
_CELL_PARAMS = {c: (40, 2, 40, 2) for c in
                ('BEAR_HIGH', 'BEAR_MED', 'BEAR_LOW', 'FLAT_HIGH', 'FLAT_MED',
                 'FLAT_LOW', 'BULL_HIGH', 'BULL_MED', 'BULL_LOW')}


def combo_score_fn(r):
    # magnitude per-month-universe NORMALIZED; significance floors stay RAW
    net_norm = r.get('net_norm', 0); net26_norm = r.get('net26_norm', 0)
    if net_norm <= 0 or net26_norm <= 0:
        return -999
    dd = r.get('max_dd', 0); oos_wr = r.get('oos_wr', 0)
    is_wr = r.get('is_wr', 0); oos_n = r.get('oos_n', 0)
    oos_n_norm = r.get('oos_n_norm', 0)
    nw, nw26 = r['nw'], r['nw26']
    if oos_n < 50 or oos_wr <= 50 or is_wr <= 50:
        return -999
    # TEMPLATE FORMULA — plug your own combo score; constants are EXAMPLES.
    q = (oos_wr / 100) * (is_wr / 100)
    base = ((_math.sqrt(net_norm) / (1 + dd / 3)) * q
            * (1 / (1 + nw / 15 + nw26 * 1)) * _math.log2(max(oos_n_norm, 50)))
    return base


def _gen_combos(slot_options, slot_dirs, individual_net, max_active):
    """Yield valid combo tuples (structural pruning: max_active, per-dir, neg).

    v2 (2026-07-17): STYLE CAP -- a combo whose dominant style (momentum vs
    mean-rev, common.style_map) exceeds MAX_STYLE_FRAC of actives is rejected
    at yield. Disabled automatically when the cell's whole pool is single-style
    (else the cell would produce zero combos). TM_STYLE_CAP=1 disables."""
    import math as _m
    from factory.portfolio.style_map import style_cap_ok, style_of, MAX_STYLE_FRAC
    n_slots = len(slot_options)
    _styles = [{o: style_of(o) for o in opts if o is not None}
               for opts in slot_options]
    _pool_styles = {st for d_ in _styles for st in d_.values()}
    _pool_styles.discard('OTHER')
    _cap_on = len(_pool_styles) >= 2

    def rec(idx, active, na, nl, ns, sc):
        if idx == n_slots:
            if na > 0 and (not _cap_on or style_cap_ok(active)):
                yield tuple(active)
            return
        rem = n_slots - idx - 1
        for opt in slot_options[idx]:
            if opt is None:
                if na > 0 or rem > 0:
                    yield from rec(idx + 1, active, na, nl, ns, sc)
            else:
                if na + 1 > max_active:
                    continue
                il = slot_dirs[idx] == 'L'
                if il and nl >= MAX_ACTIVE_PER_DIR:
                    continue
                if not il and ns >= MAX_ACTIVE_PER_DIR:
                    continue
                if individual_net.get(opt, 0) < -1000:
                    continue
                st = _styles[idx].get(opt, 'OTHER')
                if _cap_on and st != 'OTHER':
                    # branch pruning: if this style cannot fit the cap even
                    # at the largest REACHABLE combo size, it is a guaranteed
                    # fail. (max_na>2 guards the <=2-active exemption.)
                    ns_ = sc.get(st, 0) + 1
                    max_na = min(max_active, na + 1 + rem)
                    if max_na > 2 and ns_ > _m.ceil(MAX_STYLE_FRAC * max_na):
                        continue
                    sc[st] = ns_
                active.append(opt)
                yield from rec(idx + 1, active, na + 1,
                               nl + (1 if il else 0), ns + (0 if il else 1), sc)
                active.pop()
                if _cap_on and st != 'OTHER':
                    sc[st] -= 1
    yield from rec(0, [], 0, 0, 0, {})


_GTG = {}

def _init_global(tg_by_cell, price_by_cell=None):
    global _GTG, _GPRICE
    _GTG = tg_by_cell
    _GPRICE = price_by_cell or {}

def _combo_eval_g(args):
    """Worker: fast_sim one (cell, combo) using that cell's shared timeline."""
    cell, combo = args
    tg, times = _GTG[cell]
    r = fast_sim(tg, times, list(combo), price=_GPRICE.get(cell))
    if r['net'] <= 0 or r['net26'] <= 0:
        return (cell, None)
    r['n_active'] = len(combo); r['combo'] = combo
    r['is_wr'] = round(r['exec_is_w'] / r['exec_is_t'] * 100, 1) if r['exec_is_t'] else 0
    r['oos_wr'] = round(r['exec_oos_w'] / r['exec_oos_t'] * 100, 1) if r['exec_oos_t'] else 0
    r['is_ev'] = round(r['exec_is_pnl'] / r['exec_is_t'], 3) if r['exec_is_t'] else 0
    r['oos_ev'] = round(r['exec_oos_pnl'] / r['exec_oos_t'], 3) if r['exec_oos_t'] else 0
    r['is_n'] = r['exec_is_t']; r['oos_n'] = r['exec_oos_t']
    return (cell, r)


# ── year-jackknife median-rank rerank (TM_JK_BEAM=1) ────────────────
_JKV = _JKPRICE = None

def _init_jk(var_tg, price):
    global _JKV, _JKPRICE
    _JKV, _JKPRICE = var_tg, price


def _jk_eval(args):
    y, combo = args
    tg, times = _JKV[y]
    r = fast_sim(tg, times, list(combo), price=_JKPRICE)
    if r['net'] <= 0 or r['net26'] <= 0:
        return (y, combo, -999.0)
    r['is_wr'] = round(r['exec_is_w'] / r['exec_is_t'] * 100, 1) if r['exec_is_t'] else 0
    r['oos_wr'] = round(r['exec_oos_w'] / r['exec_oos_t'] * 100, 1) if r['exec_oos_t'] else 0
    r['oos_n'] = r['exec_oos_t']
    return (y, combo, combo_score_fn(r))


def _jackknife_rerank(cell, prep, pool):
    """Re-simulate the (DD-diverse union) pool under 5 drop-year variants;
    ordering = the MEDIAN rank across 6 variants (full + 5 jackknife). The
    pool source is untouched — only the final ordering changes."""
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    tg, times = prep['tg'], prep['times']
    var_tg = {}
    for y in _JK_YEARS:
        ftg = {k: {et: v for et, v in d.items() if et.year != y}
               for k, d in tg.items()}
        var_tg[y] = (ftg, {k: set(d.keys()) for k, d in ftg.items()})
    scores = {y: {} for y in _JK_YEARS}
    tasks = [(y, tuple(r['combo'])) for y in _JK_YEARS for r in pool]
    nw = min(len(_JK_YEARS) * 2, max(2, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=nw, initializer=_init_jk,
                             initargs=(var_tg, prep.get('price'))) as ex:
        for y, combo, s in ex.map(_jk_eval, tasks, chunksize=16):
            scores[y][combo] = s
    full_rank = {tuple(r['combo']): i for i, r in enumerate(
        sorted(pool, key=lambda r: -combo_score_fn(r)))}
    ranks = {tuple(r['combo']): [full_rank[tuple(r['combo'])]] for r in pool}
    for y in _JK_YEARS:
        order = sorted(pool, key=lambda r: -scores[y].get(tuple(r['combo']), -999))
        for i, r in enumerate(order):
            ranks[tuple(r['combo'])].append(i)
    med = {c: float(np.median(v)) for c, v in ranks.items()}
    out = sorted(pool, key=lambda r: (med[tuple(r['combo'])],
                                      full_rank[tuple(r['combo'])]))
    print(f'  [{cell}] JK-rerank: {len(pool)} aday x {len(_JK_YEARS)} varyant '
          f'({time.time()-t0:.0f}s)', flush=True)
    return out


# ── co-crash penalty rerank (TM_DIVCRASH=1) ─────────────────────────
def _divcrash_rerank(cell, prep, ranked):
    """Corrects the pool ranking with a MEASURED pairwise co-crash /
    complementarity matrix. PRE-REGISTERED form:
      new_score = score / (1 + 1.0*avg_overlap - 0.5*avg_complementarity)
    overlap = pairwise intersection ratio of members' worst-10-week sets;
    complementarity = share of those weeks where the other member's median >= 0.
    CELL layer only; does not solve same-day signal clustering (known limit)."""
    from itertools import combinations as _comb
    wkvec = {}
    for label, tg in prep['tg'].items():
        wv = {}
        for lst in tg.values():
            for t in lst:
                wv[t[4]] = wv.get(t[4], 0.0) + t[1]
        wkvec[label] = wv
    worst10 = {l: set(sorted(wv, key=wv.get)[:10]) for l, wv in wkvec.items()}

    def pair(la, lb):
        ov = len(worst10[la] & worst10[lb]) / 10
        ta = float(np.median([wkvec[lb].get(w, 0.0) for w in worst10[la]]) >= 0)
        tb = float(np.median([wkvec[la].get(w, 0.0) for w in worst10[lb]]) >= 0)
        return ov, (ta + tb) / 2

    pcache = {}
    out = []
    for r in ranked:
        mems = list(r['combo'])
        if len(mems) < 2:
            out.append((combo_score_fn(r), r))
            continue
        ovs, tls = [], []
        for la, lb in _comb(sorted(mems), 2):
            k = (la, lb)
            if k not in pcache:
                pcache[k] = pair(la, lb)
            o, t_ = pcache[k]
            ovs.append(o); tls.append(t_)
        adj = 1 / (1 + 1.0 * float(np.mean(ovs)) - 0.5 * float(np.mean(tls)))
        out.append((combo_score_fn(r) * adj, r))
    out.sort(key=lambda x: -x[0])
    print(f'  [{cell}] DIVCRASH-rerank: {len(ranked)} aday, '
          f'{len(pcache)} cift', flush=True)
    return [r for _, r in out]


def _build_price_map(trade_sets):
    """{coin: {entry_time: close}} for all trade entries -- early-exit lookups (rule B)."""
    coin_times = defaultdict(set)
    for trades in trade_sets.values():
        for t in trades:
            coin_times[t['coin']].add(t['entry_time'])
    pm = {}
    for c, qset in coin_times.items():
        parts = []
        for f in sorted(_CACHE_DIR.glob(f'{c}_5m_*.parquet')):
            d = pd.read_parquet(f, columns=['close'])
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            parts.append(d['close'])
        if not parts:
            continue
        s = pd.concat(parts); s = s[~s.index.duplicated(keep='last')].sort_index()
        pm[c] = s.reindex(pd.DatetimeIndex(sorted(qset)), method='ffill').to_dict()
    print(f'  [price map] {len(pm)} coins priced (rule B)', flush=True)
    return pm


def prep_cell(cell_arg, max_active):
    """Load pickles + timeline + slots + individual_net for one cell."""
    import pickle as _pkl
    l_tn, l_tps, s_tn, s_tps = _CELL_PARAMS.get(cell_arg, (25, 2, 25, 2))
    all_slots = parse_top_n_configs(cell_arg, l_top_n=l_tn, l_tps=l_tps,
                                    s_top_n=s_tn, s_tps=s_tps)
    l_slots = {k: v for k, v in all_slots.items() if k[0] == 'LONG'}
    s_slots = {k: v for k, v in all_slots.items() if k[0] == 'SHORT'}
    if len(l_slots) > MAX_SLOTS_PER_DIR:
        l_slots = dict(sorted(l_slots.items(), key=lambda x: -x[1][0][0])[:MAX_SLOTS_PER_DIR])
    if len(s_slots) > MAX_SLOTS_PER_DIR:
        s_slots = dict(sorted(s_slots.items(), key=lambda x: -x[1][0][0])[:MAX_SLOTS_PER_DIR])
    all_slots = {**l_slots, **s_slots}

    pdir = _WORK / 'trade_pickles'
    all_pkl = {}
    for yr in range(2021, 2027):
        yp = pdir / f'{cell_arg}_{yr}.pkl'
        if yp.exists():
            for label, trades in _pkl.load(open(yp, 'rb')).items():
                all_pkl.setdefault(label, []).extend(trades)
    if not all_pkl:
        print(f'  [{cell_arg}] no pickles, skipped', flush=True)
        return None

    trade_sets, slot_labels = {}, {}
    sltp_map = {}
    for slot_key, configs in all_slots.items():
        direction, cat_strat = slot_key
        for (score, d2, cs2, sl_tp, q, ci) in configs:
            label = f'{direction[0]}:{cat_strat.split("/")[1]}_q{q}_c{ci}'
            if all_pkl.get(label):
                trade_sets[label] = all_pkl[label]
                slot_labels.setdefault(slot_key, []).append(label)
                sltp_map[label] = sl_tp   # diversity-quota 4th axis (SLTP)
    if not trade_sets:
        print(f'  [{cell_arg}] no matching configs, skipped', flush=True)
        return None

    tg, times = build_master_timeline(trade_sets)
    price_map = _build_price_map(trade_sets) if _RULE == 'B' else None
    individual_net = {lbl: fast_sim(tg, times, [lbl])['net'] for lbl in trade_sets}
    slot_options, slot_dirs = [], []
    for slot_key in sorted(slot_labels.keys()):
        slot_options.append([None] + slot_labels[slot_key])
        slot_dirs.append('L' if slot_key[0] == 'LONG' else 'S')
    total = 1
    for o in slot_options:
        total *= len(o)
    print(f'  [{cell_arg}] {len(trade_sets)} config, {len(slot_options)} slot, '
          f'~{total:,} combo', flush=True)
    return {'cell': cell_arg, 'tg': tg, 'times': times, 'slot_options': slot_options,
            'slot_dirs': slot_dirs, 'individual_net': individual_net, 'total': total,
            'price': price_map, 'sltp': sltp_map}


def _save_cell(cell, results, checked, elapsed):
    """results = top-K (already score-ranked). Print top-5, write top-100."""
    for r in results:
        r['score_v5'] = round(combo_score_fn(r), 1)
    valid = [r for r in results if r['score_v5'] > 0]
    print(f'  {cell}: {len(valid)} valid / {checked:,} combo. TOP5:', flush=True)
    for i, r in enumerate(results[:5]):
        print(f'    #{i+1} N={r["n_active"]} Net=${r["net"]:+,} Net26=${r["net26"]:+,} '
              f'DD={r["max_dd"]}% Score={r["score_v5"]:.0f} | {", ".join(r["combo"])}', flush=True)
    _sfx = ''   # Rule B is main now -> write to the canonical trade_combo_<cell>.txt
    out = _WORK / 'results' / f'trade_combo_{cell}{_sfx}.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'{cell} | {checked:,} combos evaluated | {elapsed:.0f}s\n\n')
        for i, r in enumerate(valid[:100]):
            f.write(f'#{i+1} N={r["n_active"]} NW={r["nw"]} NW26={r["nw26"]} '
                    f'Net=${r["net"]:+,} Net26=${r["net26"]:+,} DD={r["max_dd"]}% '
                    f'Score={r["score_v5"]:.0f} | IS:{r["is_n"]:,}/{r["is_wr"]}%/EV={r["is_ev"]:+.3f} '
                    f'OOS:{r["oos_n"]:,}/{r["oos_wr"]}%/EV={r["oos_ev"]:+.3f} | '
                    f'{", ".join(r["combo"])}\n')
    return len(valid)


def run_all_global(cells, max_active):
    """GLOBAL queue: all cells' combos in ONE pool (workers hold every cell's
    timeline ~8MB each), so cells OVERLAP and all cores stay busy. Per-cell
    bounded top-K heap keeps parent memory tiny."""
    import heapq
    from itertools import islice
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    preps = {}
    for cell in cells:
        p = prep_cell(cell, max_active)
        if p is not None:
            preps[cell] = p
    if not preps:
        print('  no prep'); return
    tg_by_cell = {c: (p['tg'], p['times']) for c, p in preps.items()}
    price_by_cell = {c: p.get('price') for c, p in preps.items()} if _RULE == 'B' else None
    nw = auto_workers(tg_by_cell, f'{len(tg_by_cell)} cell timeline')
    total = sum(p['total'] for p in preps.values())
    print(f'  GLOBAL combo: {len(preps)} cell, ~{total:,} ham combo -> '
          f'ProcessPool({nw})', flush=True)

    # DD-diverse retention: keep top-KEEP_PER_B per cell PER budget (so cap-bound
    # low-DD champions, which score low on the single metric, survive the merge).
    KEEP_PER_B = 150
    heaps = {c: {B: [] for B in BUDGETS} for c in preps}
    ctr = {c: 0 for c in preps}
    checked = {c: 0 for c in preps}

    def all_tasks():
        for c, p in preps.items():
            for combo in _gen_combos(p['slot_options'], p['slot_dirs'],
                                     p['individual_net'], max_active):
                yield (c, combo)

    t1 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=nw, initializer=_init_global,
                             initargs=(tg_by_cell, price_by_cell)) as pool:
        gen = all_tasks()
        while True:
            batch = list(islice(gen, 30000))
            if not batch:
                break
            for cell, r in pool.map(_combo_eval_g, batch, chunksize=300):
                done += 1; checked[cell] += 1
                if r is None:
                    continue
                ctr[cell] += 1; cid = ctr[cell]
                for B in BUDGETS:
                    sb = sv_budget(r, B)
                    if sb <= 0:
                        continue
                    h = heaps[cell][B]
                    if len(h) < KEEP_PER_B:
                        heapq.heappush(h, (sb, cid, r))
                    elif sb > h[0][0]:
                        heapq.heapreplace(h, (sb, cid, r))
            el = time.time() - t1
            print(f'    {done:,} combo islendi ({el:.0f}s, {done/max(el,1):.0f}/s)',
                  flush=True)
    el = time.time() - t1
    for cell in preps:
        # union of the 3 budget heaps (dedup), then round-robin merge -> top-100
        union = {}
        for B in BUDGETS:
            for sb, cid, r in heaps[cell][B]:
                union[tuple(r['combo'])] = r
        pool_ = list(union.values())
        if _JK_BEAM:                      # jackknife median-rank ordering
            ranked = _jackknife_rerank(cell, preps[cell], pool_)
        else:
            ranked = round_robin_merge(
                pool_, key_fn=lambda r: tuple(r['combo']),
                target=(len(pool_) if (_DIV_QUOTA or _DIVCRASH) else 100))
        if _DIVCRASH:                     # co-crash penalty rerank
            ranked = _divcrash_rerank(cell, preps[cell], ranked)
        if _DIV_QUOTA:                    # style x dir x DD x SLTP quota selection
            from factory.portfolio.beam_diversity import quota_select
            ranked = quota_select(ranked, target=100,
                                  sltp_map=preps[cell].get('sltp'))
        else:
            ranked = ranked[:100]
        _save_cell(cell, ranked, checked[cell], el)
    print(f'  GLOBAL DONE ({(time.time()-t0)/60:.1f}min)', flush=True)


def run_cell(cell_arg, max_active=10):
    """Run trade-level combo test for a single cell. Returns (cell, n_valid, elapsed)."""
    t0 = time.time()
    CELL_PARAMS = {
        'BEAR_HIGH': (40, 2, 40, 2),
        'BEAR_MED':  (40, 2, 40, 2),
        'BEAR_LOW':  (40, 2, 40, 2),
        'FLAT_HIGH': (40, 2, 40, 2),
        'FLAT_MED':  (40, 2, 40, 2),
        'FLAT_LOW':  (40, 2, 40, 2),
        'BULL_HIGH': (40, 2, 40, 2),
        'BULL_MED':  (40, 2, 40, 2),
        'BULL_LOW':  (40, 2, 40, 2),
    }

    params = CELL_PARAMS.get(cell_arg, (25, 2, 25, 2))
    l_tn, l_tps, s_tn, s_tps = params

    print(f'{"="*150}')
    print(f'  TRADE-LEVEL COMBO TEST: {cell_arg}')
    print(f'  L: top {l_tn} / {l_tps} per slot, S: top {s_tn} / {s_tps} per slot, max {max_active} active')
    print(f'{"="*150}')

    all_slots = parse_top_n_configs(cell_arg, l_top_n=l_tn, l_tps=l_tps,
                                    s_top_n=s_tn, s_tps=s_tps)

    # Limit to max 6 slots per direction, max 4 active per direction
    MAX_SLOTS_PER_DIR = 6
    MAX_ACTIVE_PER_DIR = 4
    l_slots = {k: v for k, v in all_slots.items() if k[0] == 'LONG'}
    s_slots = {k: v for k, v in all_slots.items() if k[0] == 'SHORT'}
    if len(l_slots) > MAX_SLOTS_PER_DIR:
        l_sorted = sorted(l_slots.items(), key=lambda x: -x[1][0][0])[:MAX_SLOTS_PER_DIR]
        l_slots = dict(l_sorted)
    if len(s_slots) > MAX_SLOTS_PER_DIR:
        s_sorted = sorted(s_slots.items(), key=lambda x: -x[1][0][0])[:MAX_SLOTS_PER_DIR]
        s_slots = dict(s_sorted)
    all_slots = {**l_slots, **s_slots}

    n_slots = len(all_slots)
    n_configs = sum(len(v) for v in all_slots.values())
    print(f'  {n_slots} unique slots, {n_configs} configs (max {MAX_SLOTS_PER_DIR} slots/dir, max {MAX_ACTIVE_PER_DIR} active/dir)')

    # Try loading from year-split pickles first
    import pickle as _pkl
    pickle_dir = _WORK / 'trade_pickles'
    trade_sets = {}
    slot_labels = {}

    # Load year-split pickles: {cell}_{year}.pkl
    all_trades_pkl = {}
    has_pickles = False
    for yr in range(2021, 2027):
        yr_path = pickle_dir / f'{cell_arg}_{yr}.pkl'
        if yr_path.exists():
            has_pickles = True
            with open(yr_path, 'rb') as f:
                yr_trades = _pkl.load(f)
            for label, trades in yr_trades.items():
                if label not in all_trades_pkl: all_trades_pkl[label] = []
                all_trades_pkl[label].extend(trades)

    if has_pickles:
        print(f'\n  Loaded {len(all_trades_pkl)} configs from year-split pickles', flush=True)

        # Match pickle labels to slots
        for slot_key, configs in all_slots.items():
            direction, cat_strat = slot_key
            for idx, (score, direction2, cat_strat2, sl_tp, q, ci) in enumerate(configs):
                strat_short = cat_strat.split('/')[1]
                label = f'{direction[0]}:{strat_short}_q{q}_c{ci}'
                if label in all_trades_pkl and all_trades_pkl[label]:
                    trade_sets[label] = all_trades_pkl[label]
                    if slot_key not in slot_labels:
                        slot_labels[slot_key] = []
                    slot_labels[slot_key].append(label)
                    print(f'    {label:<30} {len(all_trades_pkl[label]):>6,} trades  (score={score:.0f})', flush=True)
        del all_trades_pkl
    else:
        # Fall back to cache loading
        print(f'\n  Building trades from cache...', flush=True)
        for slot_key, configs in all_slots.items():
            direction, cat_strat = slot_key
            for idx, (score, direction2, cat_strat2, sl_tp, q, ci) in enumerate(configs):
                strat_short = cat_strat.split('/')[1]
                label = f'{direction[0]}:{strat_short}_q{q}_c{ci}'
                t1 = time.time()
                trades = build_cell_trades(direction, cell_arg, cat_strat, q, ci)
                elapsed = time.time() - t1

                if trades:
                    trade_sets[label] = trades
                    if slot_key not in slot_labels:
                        slot_labels[slot_key] = []
                    slot_labels[slot_key].append(label)
                    print(f'    {label:<30} {len(trades):>6,} trades  (score={score:.0f}, {elapsed:.1f}s)', flush=True)
                else:
                    print(f'    {label:<30} NO TRADES  ({elapsed:.1f}s)', flush=True)

    print(f'\n  Total: {len(trade_sets)} active configs, {sum(len(v) for v in trade_sets.values()):,} trades')

    if not trade_sets:
        print('  No trades found!')
        return

    # OPT 1: Pre-compute WR + EV per config
    print(f'  Pre-computing WR/EV...', flush=True)
    wr_cache = {}
    for label, trades in trade_sets.items():
        is_w = is_t = oos_w = oos_t = 0
        is_pnl_sum = oos_pnl_sum = 0.0
        for t in trades:
            if t['entry_time'].year <= 2025:
                is_t += 1
                is_pnl_sum += t['pnl_pct']
                if t['pnl_pct'] > 0: is_w += 1
            else:
                oos_t += 1
                oos_pnl_sum += t['pnl_pct']
                if t['pnl_pct'] > 0: oos_w += 1
        is_ev = is_pnl_sum / is_t if is_t else 0
        oos_ev = oos_pnl_sum / oos_t if oos_t else 0
        wr_cache[label] = (is_w, is_t, oos_w, oos_t, is_ev, oos_ev)

    # Build master timeline
    print(f'  Building master timeline...', flush=True)
    per_key_tg, per_key_times = build_master_timeline(trade_sets)

    # OPT: Pre-compute individual sim for pruning
    print(f'  Pre-computing individual sims for pruning...', flush=True)
    individual_net = {}
    for label in trade_sets:
        r = fast_sim(per_key_tg, per_key_times, [label])
        individual_net[label] = r['net']
        print(f'    {label:<30} net=${r["net"]:>+7,}', flush=True)

    # Generate combos: for each slot, pick None or one of the labels
    slot_options = []
    slot_names = []
    for slot_key in sorted(slot_labels.keys()):
        labels = slot_labels[slot_key]
        slot_options.append([None] + labels)
        slot_names.append(f'{slot_key[0][0]}:{slot_key[1].split("/")[1]}')

    n_slots = len(slot_options)
    total_combos = 1
    for opts in slot_options:
        total_combos *= len(opts)
    print(f'  {n_slots} slots, {total_combos:,} total combos (before max_active filter)')

    # Scoring function (used for ranking + pruning)
    import math
    def combo_score_fn(r):
        net = r['net']
        net26 = r['net26']
        if net <= 0 or net26 <= 0:
            return -999
        dd = r.get('max_dd', 0)
        oos_wr = r.get('oos_wr', 0)
        is_wr = r.get('is_wr', 0)
        oos_n = r.get('oos_n', 0)
        nw = r['nw']
        nw26 = r['nw26']
        if oos_n < 50 or oos_wr <= 50 or is_wr <= 50:
            return -999

        risk_adj_net = math.sqrt(net) / (1 + dd / 3)
        quality = (oos_wr / 100) * (is_wr / 100)
        stability = 1 / (1 + nw / 15 + nw26 * 1)
        n_bonus = math.log2(max(oos_n, 50))

        return risk_adj_net * quality * stability * n_bonus

    # Build slot direction map for per-direction active limit
    slot_dirs = []  # 'L' or 'S' for each slot
    for slot_key in sorted(slot_labels.keys()):
        slot_dirs.append('L' if slot_key[0] == 'LONG' else 'S')

    # Enumerate valid combos (structural pruning only), then fast_sim them in
    # PARALLEL — combos are independent (the old min_score threshold was
    # display-only, never pruned branches). Timeline shared via initializer;
    # cpu+ram-aware pool. Generator + batches keep peak memory bounded.
    from itertools import islice
    from concurrent.futures import ProcessPoolExecutor

    def gen_combos(idx, active, n_active, n_long, n_short):
        if idx == n_slots:
            if n_active > 0:
                yield tuple(active)
            return
        remaining = n_slots - idx - 1
        for opt in slot_options[idx]:
            if opt is None:
                if n_active > 0 or remaining > 0:
                    yield from gen_combos(idx + 1, active, n_active, n_long, n_short)
            else:
                if n_active + 1 > max_active:
                    continue
                is_long = slot_dirs[idx] == 'L'
                if is_long and n_long >= MAX_ACTIVE_PER_DIR:
                    continue
                if not is_long and n_short >= MAX_ACTIVE_PER_DIR:
                    continue
                if individual_net.get(opt, 0) < -1000:
                    continue
                active.append(opt)
                yield from gen_combos(idx + 1, active, n_active + 1,
                                      n_long + (1 if is_long else 0),
                                      n_short + (0 if is_long else 1))
                active.pop()

    nw = auto_workers((per_key_tg, per_key_times), 'timeline')
    print(f'  Running combos (max {max_active} total, max {MAX_ACTIVE_PER_DIR}/dir) '
          f'-> ProcessPool({nw})...', flush=True)
    results = []
    checked = 0
    t1 = time.time()
    gen = gen_combos(0, [], 0, 0, 0)
    with ProcessPoolExecutor(max_workers=nw, initializer=_init_combo_worker,
                             initargs=(per_key_tg, per_key_times)) as pool:
        while True:
            batch = list(islice(gen, 20000))
            if not batch:
                break
            for r in pool.map(_combo_eval, batch, chunksize=200):
                checked += 1
                if r is not None:
                    results.append(r)
            elapsed = time.time() - t1
            print(f'    {checked:,} combos, {len(results):,} valid '
                  f'({elapsed:.0f}s, {checked/max(elapsed,1):.0f}/s)', flush=True)
    elapsed = time.time() - t1
    print(f'  Evaluated {checked:,} combos in {elapsed:.0f}s ({checked/max(elapsed,1):.0f}/s)')

    # Sort by combo score
    results.sort(key=lambda r: -combo_score_fn(r))
    for r in results:
        r['score_v5'] = round(combo_score_fn(r), 1)

    # Print top 30
    print(f'\n  TOP 30 COMBOS (Score v5):')
    print(f'  {"#":>3} {"N":>3} | {"NW":>4} {"NW26":>4} {"Net$":>8} {"Net26$":>7} {"DD%":>5} {"Score":>10} | '
          f'{"IS_N":>6} {"IS_WR":>5} {"IS_EV":>6} {"OOS_N":>5} {"OOS_WR":>5} {"OOS_EV":>6} | Configs')
    print(f'  {"-"*155}')

    for i, r in enumerate(results[:30]):
        combo_str = ', '.join(r['combo'])
        print(f'  {i+1:>3} {r["n_active"]:>3} | {r["nw"]:>4} {r["nw26"]:>4} '
              f'${r["net"]:>+7,} ${r["net26"]:>+6,} {r["max_dd"]:>4.1f}% {r["score_v5"]:>10.0f} | '
              f'{r["is_n"]:>6,} {r["is_wr"]:>4.1f}% {r["is_ev"]:>+5.3f} '
              f'{r["oos_n"]:>5,} {r["oos_wr"]:>4.1f}% {r["oos_ev"]:>+5.3f} | '
              f'{combo_str}')

    # Save
    out_path = _WORK / 'results' / f'trade_combo_{cell_arg}.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'{cell_arg} | {checked:,} combos evaluated | {elapsed:.0f}s\n\n')
        valid = [r for r in results if r['score_v5'] > 0]
        for i, r in enumerate(valid[:100]):
            combo_str = ', '.join(r['combo'])
            f.write(f'#{i+1} N={r["n_active"]} NW={r["nw"]} NW26={r["nw26"]} '
                    f'Net=${r["net"]:+,} Net26=${r["net26"]:+,} DD={r["max_dd"]}% Score={r["score_v5"]:.0f} | '
                    f'IS:{r["is_n"]:,}/{r["is_wr"]}%/EV={r["is_ev"]:+.3f} '
                    f'OOS:{r["oos_n"]:,}/{r["oos_wr"]}%/EV={r["oos_ev"]:+.3f} | '
                    f'{combo_str}\n')

    print(f'\n  Saved: {out_path}')
    elapsed = time.time() - t0
    n_valid = len([r for r in results if r.get('score_v5', 0) > 0])
    print(f'  DONE ({elapsed:.0f}s)')
    return cell_arg, n_valid, elapsed


def main():
    args = sys.argv[1:]

    if args and args[0] == '--all':
        max_active = int(args[1]) if len(args) > 1 else 10
        print(f'{"="*120}')
        print(f'  TRADE COMBO: ALL CELLS (GLOBAL queue — cells overlap, all '
              f'cores, max_active={max_active})')
        print(f'{"="*120}', flush=True)
        run_all_global(ALL_CELLS, max_active)
    else:
        cell_arg = args[0] if args else 'BEAR_HIGH'
        max_active = int(args[1]) if len(args) > 1 else 10
        run_all_global([cell_arg], max_active)


if __name__ == '__main__':
    main()
