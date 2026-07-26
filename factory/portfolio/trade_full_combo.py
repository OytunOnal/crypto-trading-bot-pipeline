"""Full bot combo: BEAR + FLAT + BULL regime combos.

Reads top N regime combos from trade_regime_*.txt,
loads trade pickles, pre-builds int64-indexed tg dicts per regime combo,
ProcessPool(10) for parallel fast_sim on N^3 full combos.
"""
import sys, os, time, pickle, math
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
from collections import defaultdict
from itertools import product as _product
from concurrent.futures import ProcessPoolExecutor

COMBO_DIR = Path(__file__).resolve().parents[2]
import os as _osw
_WORK = Path(_osw.environ.get('TM_WORK', COMBO_DIR))  # isolated workdir (parallel arms)
PICKLE_DIR = _WORK / 'trade_pickles'  # (TM_WORK)
FEE_RT = 0.08
MAX_CONCURRENT = 100

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.qs.age_filter import norm_factor   # per-month-universe normalization

# Blind walk-forward: parametric OOS boundary (default = production, bit-identical)
_OOS_MS = os.environ.get('TM_OOS_START', '2026')
_OOS_WS = os.environ.get('TM_OOS_WSTART', '2026')

from factory.portfolio.budget_merge import round_robin_merge  # DD-diverse retention

# Rule B (close-on-opposite + skip-both-fire) is the production default and the ONLY
# mode now: reads trade_regime_<regime>.txt, writes trade_full_combo.txt (canonical
# un-suffixed), builds a cache price map for the early exit. Old A / _B split is gone.
_RULE = 'B'   # Rule B (close-on-opposite + skip-both-fire) is the production default
_SFX = ''     # write to / read from the canonical (un-suffixed) result files
_CACHE_DIR = Path(__file__).resolve().parents[3] / 'data' / 'backtest' / 'cache'


def _build_price_map(all_trades_by_coin, coin_to_idx):
    """{(coin_idx, et_int64): close} for all trade entries (rule B early exit)."""
    import pandas as pd
    pm = {}
    for coin, ets in all_trades_by_coin.items():
        ci = coin_to_idx.get(coin)
        if ci is None:
            continue
        parts = []
        for f in sorted(_CACHE_DIR.glob(f'{coin}_5m_*.parquet')):
            d = pd.read_parquet(f, columns=['close'])
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            parts.append(d['close'])
        if not parts:
            continue
        s = pd.concat(parts); s = s[~s.index.duplicated(keep='last')].sort_index()
        qt = pd.DatetimeIndex(sorted(ets))
        vals = s.reindex(qt, method='ffill')
        for ts, v in zip(qt, vals.values):
            if v == v:
                pm[(ci, ts.value)] = float(v)
    print(f'  [price map] {len(pm):,} (coin,bar) priced (rule B)', flush=True)
    return pm


def auto_workers(per_worker_data, label='veri'):
    """Worker count from cpu AND ram. Each worker gets a pickled copy of
    per_worker_data, but its RUNTIME peak is far above that pickle: fast_sim
    builds per-worker int64 index/sim scratch on top of the initargs. Measured
    on this pipeline: pickle ~316MB -> resident ~1.8GB/worker (~5.7x). So we
    size by PEAK = sz*PEAK_MULT, and reserve MAIN_RESERVE for the main process
    (it keeps all pre-built regime-combo tg dicts resident, ~6GB, while the pool
    runs). TM_MAX_WORKERS env hard-caps for manual control."""
    import psutil
    PEAK_MULT = float(os.environ.get('TM_PEAK_MULT', '6'))
    MAIN_RESERVE = float(os.environ.get('TM_MAIN_RESERVE_GB', '6')) * 1e9
    sz = max(1, len(pickle.dumps(per_worker_data, protocol=4)))
    peak = sz * PEAK_MULT
    avail = psutil.virtual_memory().available
    by_ram = max(1, int((avail - MAIN_RESERVE) // peak))
    by_cpu = max(1, (os.cpu_count() or 8) - 2)
    n = max(1, min(by_cpu, by_ram))
    hard = os.environ.get('TM_MAX_WORKERS')
    if hard:
        n = max(1, min(n, int(hard)))
    print(f'  worker: cpu={by_cpu}, ram-limit={by_ram} '
          f'({label} pickle {sz/1e6:.0f}MB x{PEAK_MULT:.0f}={peak/1e6:.0f}MB/worker, '
          f'bos {avail/1e9:.0f}GB, main-rsv {MAIN_RESERVE/1e9:.0f}GB) -> {n}', flush=True)
    return n

REGIME_CELLS = {
    'BEAR': ['BEAR_LOW', 'BEAR_MED', 'BEAR_HIGH'],
    'FLAT': ['FLAT_LOW', 'FLAT_MED', 'FLAT_HIGH'],
    'BULL': ['BULL_LOW', 'BULL_MED', 'BULL_HIGH'],
}


def load_cell_trades(cell_name):
    """Load year-split trade pickles for a cell, merge per config."""
    merged = {}
    for yr in range(2021, 2027):
        fpath = PICKLE_DIR / f'{cell_name}_{yr}.pkl'
        if not fpath.exists(): continue
        with open(fpath, 'rb') as f:
            yr_trades = pickle.load(f)
        for label, trades in yr_trades.items():
            if label not in merged: merged[label] = []
            merged[label].extend(trades)
    if not merged:
        print(f'  WARNING: no trade pickles for {cell_name}')
    return merged


def parse_regime_combo_file(regime, top_n=30):
    fpath = _WORK / 'results' / f'trade_regime_{regime}{_SFX}.txt'
    if not fpath.exists(): return []
    combos = []
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'): continue
            parts = line.split('|')
            if len(parts) < 3: continue
            try:
                tokens = parts[0].strip().split()
                rank = int(tokens[0].lstrip('#'))
                cell_info = parts[2].strip()
                cell_refs = [c.strip() for c in cell_info.split('+')]
                combos.append({'rank': rank, 'cell_refs': cell_refs})
            except (ValueError, IndexError):
                continue
            if len(combos) >= top_n: break
    return combos


def parse_cell_combo_file(cell_name, rank):
    fpath = _WORK / 'results' / f'trade_combo_{cell_name}.txt'
    if not fpath.exists(): return []
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            if not line.startswith(f'#{rank} '): continue
            parts = line.split('|')
            if len(parts) < 3: continue
            config_str = parts[2].strip()
            return [c.strip() for c in config_str.split(',')]
    return []


def combo_score_fn(r):
    # magnitude per-month-universe NORMALIZED; significance floors stay RAW
    net_norm = r.get('net_norm', 0); net26_norm = r.get('net26_norm', 0)
    if net_norm <= 0 or net26_norm <= 0: return -999
    dd = r.get('max_dd', 0)
    oos_wr = r.get('oos_wr', 0); is_wr = r.get('is_wr', 0)
    oos_n = r.get('oos_n', 0); oos_n_norm = r.get('oos_n_norm', 0)
    nw = r['nw']; nw26 = r['nw26']
    if oos_n < 50 or oos_wr <= 50 or is_wr <= 50: return -999
    risk_adj_net = math.sqrt(net_norm) / (1 + dd / 3)
    # TEMPLATE FORMULA — plug your own; constants are EXAMPLES.
    quality = (oos_wr / 100) * (is_wr / 100)
    stability = 1 / (1 + nw / 15 + nw26 * 1)
    n_bonus = math.log2(max(oos_n_norm, 50))
    return risk_adj_net * quality * stability * n_bonus


# ── Parallel workers (module-level, int64 indexed) ──────────────────
_w_regime_tg = None
_w_regimes = None
_w_regime_combos = None
_w_oos_month_start = None
_w_oos_week_start = None
_w_month_nf = None   # month_idx -> norm_factor (350/univ_month)
_w_price = None      # {(coin_idx, et_int64): close} for rule B early exit

def _init_full_worker(regime_tg, regimes, regime_combos, oos_month_start, oos_week_start,
                      month_nf, price=None):
    global _w_regime_tg, _w_regimes, _w_regime_combos, _w_oos_month_start, _w_oos_week_start
    global _w_month_nf, _w_price
    _w_regime_tg = regime_tg
    _w_regimes = regimes
    _w_regime_combos = regime_combos
    _w_oos_month_start = oos_month_start
    _w_oos_week_start = oos_week_start
    _w_month_nf = month_nf
    _w_price = price or {}

def _run_full_sim(combo_tuple):
    """Worker: merge 3 regime tg dicts, run fast_sim."""
    tg_list = []
    regime_info = []
    for ri, regime in enumerate(_w_regimes):
        idx = combo_tuple[ri]
        combo = _w_regime_combos[regime][idx]
        key = (regime, combo['rank'])
        tg_list.append(_w_regime_tg[key])
        regime_info.append(f'{regime}#{combo["rank"]}')

    # Merge tg dicts
    merged = defaultdict(list)
    for tg in tg_list:
        for et, trades in tg.items():
            merged[et].extend(trades)
    sorted_times = sorted(merged.keys())
    if not sorted_times: return None

    # fast_sim (int64 keys, int indices)
    cash = 10000.0; wd = 0.0; opos = {}
    opened = 0; skipped = 0
    peak_eq = 10000.0; max_dd = 0.0
    mo = defaultdict(float); wk = defaultdict(float)
    mo_tr = defaultdict(int)   # monthly trade count (per-month-universe norm)
    fee_frac = FEE_RT / 100
    exec_is_w = exec_is_t = exec_oos_w = exec_oos_t = 0
    exec_is_pnl = exec_oos_pnl = 0.0
    rule_b = (_RULE == 'B')

    for et in sorted_times:
        expired = [c for c, p in opos.items() if p[0] <= et]
        for c in expired:
            p = opos.pop(c); ps = p[1]; pnl = p[2]   # p[0]=xt; 3- and 5-tuple compatible
            cash += ps + ps * (pnl / 100) - ps * fee_frac
        lk = sum(p[1] for p in opos.values()); eq = cash + lk
        if eq > 10000 and cash > 0:
            w = min(eq - 10000, cash); cash -= w; wd += w; eq = cash + lk
        if eq > peak_eq: peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100 if peak_eq > 0 else 0
        if dd > max_dd: max_dd = dd

        cs = merged[et] if rule_b else [tr for tr in merged[et] if tr[0] not in opos]
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
                    continue                  # A: dup coin same et -> capital-leak fix
                pos = opos[c0]                # (xt,ps,pnl,dir,entry_et,entry_mo,entry_wk)
                if pos[3] == tr[5]:
                    continue                  # same dir -> skip (1/coin)
                ep = _w_price.get((c0, pos[4])); pxe = _w_price.get((c0, et))  # lookup on conflict
                if ep is None or pxe is None or ep <= 0:
                    continue
                early = (pxe / ep - 1) * 100 if pos[3] == 'LONG' else (1 - pxe / ep) * 100
                p = opos.pop(c0); ps2 = p[1]
                early_real = ps2 * (early / 100) - ps2 * fee_frac
                full_booked = ps2 * (p[2] / 100) - ps2 * fee_frac   # p[2]=held pnl_pct
                cash += max(ps2 + early_real, 0)
                # correct ENTRY-bar booking: replace assumed-full with early-realized
                mo[p[5]] += early_real - full_booked   # p[5]=entry month_idx
                wk[p[6]] += early_real - full_booked   # p[6]=entry week_idx
                if (p[2] > 0) != (early > 0):          # fix WR on sign flip
                    if p[5] >= _w_oos_month_start: exec_oos_w += 1 if early > 0 else -1
                    else:                          exec_is_w += 1 if early > 0 else -1
                continue
            if rule_b and c0 in both:
                continue                      # both-fire, no position -> skip
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
            pnl_v = ps * (tr[1] / 100) - ps * fee_frac
            mo[tr[3]] += pnl_v; wk[tr[4]] += pnl_v
            mo_tr[tr[3]] += 1
            opened += 1
            if tr[3] >= _w_oos_month_start:
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
    if net <= 0: return None
    net26 = sum(v for mi, v in mo.items() if mi >= _w_oos_month_start)
    if net26 <= 0: return None
    nm = sum(1 for v in mo.values() if v < 0)
    nw = sum(1 for v in wk.values() if v < 0)
    nw26 = sum(1 for wi, v in wk.items() if wi >= _w_oos_week_start and v < 0)
    # per-month-universe normalized magnitudes (month_idx -> norm_factor)
    nf = _w_month_nf
    net_norm = sum(v * nf.get(mi, 0.0) for mi, v in mo.items())
    net26_norm = sum(v * nf.get(mi, 0.0) for mi, v in mo.items() if mi >= _w_oos_month_start)
    oos_n_norm = sum(c * nf.get(mi, 0.0) for mi, c in mo_tr.items() if mi >= _w_oos_month_start)
    is_wr = round(exec_is_w / exec_is_t * 100, 1) if exec_is_t else 0
    oos_wr = round(exec_oos_w / exec_oos_t * 100, 1) if exec_oos_t else 0
    is_ev = round(exec_is_pnl / exec_is_t, 3) if exec_is_t else 0
    oos_ev = round(exec_oos_pnl / exec_oos_t, 3) if exec_oos_t else 0
    return {'net': round(net), 'nm': nm, 'nw': nw, 'nw26': nw26, 'net26': round(net26),
            'net_norm': net_norm, 'net26_norm': net26_norm, 'oos_n_norm': oos_n_norm,
            'opened': opened, 'skipped': skipped, 'max_dd': round(max_dd, 2),
            'is_wr': is_wr, 'oos_wr': oos_wr, 'is_ev': is_ev, 'oos_ev': oos_ev,
            'is_n': exec_is_t, 'oos_n': exec_oos_t, 'regime_info': ' + '.join(regime_info)}


# ── Main ────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    regimes = ['BEAR', 'FLAT', 'BULL']

    print(f'{"="*150}')
    print(f'  FULL BOT COMBO: BEAR + FLAT + BULL (ProcessPool(10), int64 indexed)')
    print(f'  Top {top_n} regime combos per regime')
    print(f'{"="*150}')

    # Load all 9 cell trade pickles
    all_cell_trades = {}
    for regime in regimes:
        for cell in REGIME_CELLS[regime]:
            trades = load_cell_trades(cell)
            all_cell_trades[cell] = trades
            total = sum(len(v) for v in trades.values())
            print(f'  {cell}: {len(trades)} configs, {total:,} trades', flush=True)

    # Load regime combo results
    regime_combos = {}
    for regime in regimes:
        combos = parse_regime_combo_file(regime, top_n)
        regime_combos[regime] = combos
        print(f'  {regime}: {len(combos)} regime combos loaded', flush=True)

    # Build index tables from all trades
    print(f'\n  Building index tables...', flush=True)
    all_coins = set(); all_months = set(); all_weeks = set()
    for cell_trades in all_cell_trades.values():
        for trades in cell_trades.values():
            for t in trades:
                all_coins.add(t['coin'])
                et = t['entry_time']
                iso = et.isocalendar()
                all_months.add(et.strftime('%Y-%m'))
                all_weeks.add(f'{iso[0]}-W{iso[1]:02d}')

    coin_to_idx = {c: i for i, c in enumerate(sorted(all_coins))}
    month_to_idx = {m: i for i, m in enumerate(sorted(all_months))}
    week_to_idx = {w: i for i, w in enumerate(sorted(all_weeks))}
    oos_month_start = min((v for k, v in month_to_idx.items() if k >= _OOS_MS),
                          default=len(month_to_idx))
    oos_week_start = min((v for k, v in week_to_idx.items() if k >= _OOS_WS),
                         default=len(week_to_idx))
    # month_idx -> per-month-universe norm factor (350/univ_month) for normalized score
    month_nf = {idx: norm_factor(ym) for ym, idx in month_to_idx.items()}
    print(f'    {len(coin_to_idx)} coins, {len(month_to_idx)} months, {len(week_to_idx)} weeks', flush=True)

    # Pre-build int64-indexed tg dicts for each regime combo
    print(f'  Pre-building regime combo tg dicts...', flush=True)
    t_tg = time.time()
    regime_tg = {}
    for regime in regimes:
        for combo in regime_combos[regime]:
            trades = []
            for cell_ref in combo['cell_refs']:
                cell_name, rank_str = cell_ref.rsplit('#', 1)
                cell_rank = int(rank_str)
                configs = parse_cell_combo_file(cell_name, cell_rank)
                for cfg in configs:
                    if cfg in all_cell_trades.get(cell_name, {}):
                        trades.extend(all_cell_trades[cell_name][cfg])
            # Build int64-indexed tg
            tg = defaultdict(list)
            for t in trades:
                et = t['entry_time']
                iso = et.isocalendar()
                tg[et.value].append((
                    coin_to_idx.get(t['coin'], 0),
                    t['pnl_pct'],
                    t['exit_time'].value,
                    month_to_idx.get(et.strftime('%Y-%m'), 0),
                    week_to_idx.get(f'{iso[0]}-W{iso[1]:02d}', 0),
                    t.get('direction', ''),   # [5] direction (rule B; A ignores)
                ))
            regime_tg[(regime, combo['rank'])] = dict(tg)

    price_map = None
    if _RULE == 'B':
        _tbc = defaultdict(set)
        for cell_trades in all_cell_trades.values():
            for cfg_trades in cell_trades.values():
                for t in cfg_trades:
                    _tbc[t['coin']].add(t['entry_time'])
        price_map = _build_price_map(_tbc, coin_to_idx)
    del all_cell_trades
    print(f'  tg dicts built in {time.time()-t_tg:.1f}s', flush=True)

    # Run N^3 full combos with ProcessPool(10)
    total = 1
    for regime in regimes:
        total *= len(regime_combos[regime])
    nw = auto_workers(regime_tg, 'regime_tg')
    print(f'  Total full combos: {total:,} -> ProcessPool({nw})', flush=True)

    all_combo_tuples = list(_product(*[range(len(regime_combos[r])) for r in regimes]))

    t1 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=nw, initializer=_init_full_worker,
                             initargs=(regime_tg, regimes, regime_combos,
                                       oos_month_start, oos_week_start, month_nf,
                                       price_map)) as pool:
        batch_size = 2000
        for batch_start in range(0, len(all_combo_tuples), batch_size):
            batch = all_combo_tuples[batch_start:batch_start + batch_size]
            batch_results = list(pool.map(_run_full_sim, batch))
            for r in batch_results:
                if r is not None:
                    results.append(r)
            elapsed = time.time() - t1
            done = batch_start + len(batch)
            print(f'    {done:,}/{total:,} ({len(results):,} valid, {elapsed:.0f}s, '
                  f'{done/max(elapsed,1):.0f}/s)', flush=True)

    elapsed = time.time() - t1
    print(f'  Evaluated {total:,} combos in {elapsed:.0f}s ({total/max(elapsed,1):.0f}/s, {len(results):,} valid)')

    for r in results:
        r['score_v5'] = round(combo_score_fn(r), 1)
    # DD-diverse selection: round-robin merge of B=10/15/20 leverage budgets
    # (replaces single-score sort so the full pool spans the DD spectrum -> deploy picks)
    results = round_robin_merge(results, key_fn=lambda r: r['regime_info'], target=100)

    # Print top 20
    print(f'\n  TOP 20 FULL BOT COMBOS (round-robin B=10/15/20):')
    print(f'  {"#":>3} | {"NW":>4} {"NW26":>4} {"Net$":>8} {"Net26$":>7} {"DD%":>5} {"Score":>10} | '
          f'{"IS_N":>6} {"IS_WR":>5} {"IS_EV":>6} {"OOS_N":>5} {"OOS_WR":>5} {"OOS_EV":>6} | Regimes')
    print(f'  {"-"*140}')

    for i, r in enumerate(results[:20]):
        print(f'  {i+1:>3} | {r["nw"]:>4} {r["nw26"]:>4} '
              f'${r["net"]:>+7,} ${r["net26"]:>+6,} {r["max_dd"]:>4.1f}% {r["score_v5"]:>10.0f} | '
              f'{r["is_n"]:>6,} {r["is_wr"]:>4.1f}% {r["is_ev"]:>+5.3f} '
              f'{r["oos_n"]:>5,} {r["oos_wr"]:>4.1f}% {r["oos_ev"]:>+5.3f} | '
              f'{r["regime_info"]}')

    # Save
    out_path = _WORK / 'results' / f'trade_full_combo{_SFX}.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'FULL BOT | {total:,} combos | {elapsed:.0f}s\n\n')
        valid = [r for r in results if r['score_v5'] > 0]
        for i, r in enumerate(valid[:100]):
            f.write(f'#{i+1} NW={r["nw"]} NW26={r["nw26"]} '
                    f'Net=${r["net"]:+,} Net26=${r["net26"]:+,} DD={r["max_dd"]}% Score={r["score_v5"]:.0f} | '
                    f'IS:{r["is_n"]:,}/{r["is_wr"]}%/EV={r["is_ev"]:+.3f} '
                    f'OOS:{r["oos_n"]:,}/{r["oos_wr"]}%/EV={r["oos_ev"]:+.3f} | '
                    f'{r["regime_info"]}\n')

    print(f'\n  Saved: {out_path}')
    print(f'  DONE ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
