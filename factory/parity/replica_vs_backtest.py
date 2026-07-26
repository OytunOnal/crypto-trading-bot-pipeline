"""Weekly replica-vs-backtest diff (4 layers).

Replays the live bot's decision pipeline (current local modules: signals,
features with windowed OBV, gates, regime trackers, 1200-bar buffer, live sig
trims) over a date window on official klines, then compares per (coin, bar):

  L1 SIGNAL: replica signal fire  vs  cache raw signal rows
  L2 GATE:   replica gate+block   vs  gate+block on cache feature rows
             (same CELL_CONFIG rules — isolates FEATURE parity)
  L3 FINAL:  gate-passed rows     vs  trade_pickles membership (QS narrowing)
  L4 QS:     optional --qs-replay: chronological qs_core replay seeded from
             the bootstrap parquet, per-signal score + pass/fail

Usage:
  python replica_vs_backtest.py --start 2026-06-04 --end 2026-06-11
"""
import sys, os, argparse, pickle, time, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / 'data' / 'backtest' / 'cache'
HERE = Path(__file__).resolve().parent
PICKLE_DIR = Path(__file__).resolve().parents[1] / 'trade_pickles'

from factory.qs.live_features import compute_all_features, merge_config_needs
from factory.qs.regime_gate import (
    BtcTrendTracker, BtcMomentumTracker,
    check_cell_gate, check_block_gate, SIGNAL_FUNCS,
)
from factory.qs.live_config import (
    CELL_CONFIG, MIN_COIN_BARS, needed_signals_for_regime, configs_for_regime,
)
from factory.gates.features_common import ALL_COL_NAMES
from factory.qs.age_filter import load_listing, MIN_AGE_DAYS   # min-age gate (mirror live)
_LST = load_listing()   # {coin: pd.Timestamp} futures onboardDate (tz-naive)

TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
MOM_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}
TRIM_BARS = 4400   # scanner buffer (ema_a EWM convergence)
FEAT_BARS = 1200   # feature window (converged; mirrors brain)

# Single source — same registry the live brain uses
from factory.qs.regime_gate import SIGNAL_FUNCS

# signal -> cache directory + file prefix (toy set)
SIG_CACHE = {
    'sma_x': ('momentum_sma_cross', 'sma_cross'),
    'rsi_mr': ('meanrev_rsi', 'rsi_mr'),
    'ch_brk': ('breakout_channel', 'channel_brk'),
}


# ── Multiprocessing worker state (Windows spawn-safe) ──
_G = {}


def _worker_init(warm_start_iso, end_iso, universe):
    """Per-worker data load (spawn: each process loads its own copy)."""
    warm_start = pd.Timestamp(warm_start_iso)
    end = pd.Timestamp(end_iso)
    btc = pd.read_parquet(CACHE_DIR / 'BTCUSDT_5m_20260101_20261231.parquet')
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)
    _G['btc'] = btc.loc[warm_start - pd.Timedelta(days=4):end]
    dfs = {}
    for c in universe:
        hits = list(CACHE_DIR.glob(f'{c}_5m_2026*.parquet'))
        if not hits:
            continue
        df = pd.read_parquet(hits[0])
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.loc[warm_start:end]
        if len(df) >= 100:
            dfs[c] = df
    _G['coin_dfs'] = dfs


def _replica_bars(bar_list):
    """Process a chunk of bars; returns (fire, gate, cells, signals_needed)."""
    btc = _G['btc']
    coin_dfs = _G['coin_dfs']
    fire = {}
    gate = {}
    cells = {}
    sigs_needed = set()

    for T in bar_list:
        b = btc.loc[:T]
        if len(b) < 2100:
            continue
        b1h = b.resample('1h').agg({'close': 'last'}).dropna()
        if len(b1h) > 1:
            b1h = b1h.iloc[:-1]
        c1 = b1h['close']
        trend = BtcTrendTracker().update(c1)
        mom = BtcMomentumTracker().update(c1)
        if trend is None or mom is None:
            continue
        cells[T] = f'{TREND_NAMES[trend]}_{MOM_NAMES[mom]}'

        needed = needed_signals_for_regime(trend, mom)
        sigs_needed |= set(needed)
        regime_configs = configs_for_regime(trend, mom)

        r1c = b['close'].copy(); r1c.iloc[:11] = np.nan
        r1h_high = b['high'].rolling(12).max()
        r1h_low = b['low'].rolling(12).min()
        tr = pd.concat([r1h_high - r1h_low, (r1h_high - r1c.shift(1)).abs(),
                        (r1h_low - r1c.shift(1)).abs()], axis=1).max(axis=1)
        btc_kwargs = dict(
            btc_1h_ret=c1.pct_change(),
            btc_5m_roc10=float(r1c.pct_change(10).iloc[-1]),
            btc_5m_atr_pct=float(tr.rolling(14).mean().iloc[-1] / r1c.iloc[-1]),
            btc_roc_24h=float(c1.pct_change(24).iloc[-1]),
            btc_roc_4h=float(c1.pct_change(4).iloc[-1]),
            btc_roc_slope=float(c1.pct_change(10).diff(5).iloc[-1]),
        )

        for sym, df in coin_dfs.items():
            buf = df.loc[:T]
            if len(buf) < MIN_COIN_BARS or buf.index[-1] != T:
                continue
            # 365d coin-age gate (mirror live brain): unknown coin -> excluded
            ls = _LST.get(sym)
            if ls is None or (T - ls).days < MIN_AGE_DAYS:
                continue
            buf = buf.iloc[-TRIM_BARS:]

            sigs = {}
            for sig_name in needed:
                entry = SIGNAL_FUNCS.get(sig_name)
                if entry is None:
                    continue
                func, trim_s = entry
                ohlcv_sig = buf.iloc[-trim_s:] if len(buf) > trim_s else buf
                try:
                    s, l = func(ohlcv_sig)
                    if s or l:
                        sigs[sig_name] = (s, l)
                except Exception:
                    pass
            if not sigs:
                continue

            candidates = []
            for direction, cell, signal, cfg in regime_configs:
                sp = sigs.get(signal)
                if sp is None:
                    continue
                if (direction == 'SHORT' and sp[0]) or (direction == 'LONG' and sp[1]):
                    candidates.append((direction, cell, signal, cfg))
            if not candidates:
                continue

            bases, btcn = merge_config_needs(candidates)
            buf_feat = buf.iloc[-FEAT_BARS:] if len(buf) > FEAT_BARS else buf
            features = compute_all_features(buf_feat, needed_bases=bases,
                                            needed_btc=btcn, **btc_kwargs)
            # Record EVERY fired candidate (signal-level) — the cache also
            # holds one row per signal. Live's first-match `break` is a
            # composition nuance handled at the QS layer, not here.
            for direction, cell, signal, cfg in candidates:
                fire[(direction, sym, T, signal)] = cell
                ok = (check_cell_gate(features, cfg['gate']) and
                      check_block_gate(features, cfg.get('block', [])))
                gate[(direction, sym, T, signal)] = ok

    return fire, gate, cells, sigs_needed


def load_universe():
    """Live universe = whitelist file (HEAD = deployed)."""
    import subprocess
    wl = subprocess.run(['git', 'show', 'HEAD:config/coin_whitelist.txt'],
                        capture_output=True, text=True, cwd=str(ROOT),
                        encoding='utf-8', errors='ignore').stdout
    return sorted(set(l.strip() for l in wl.splitlines() if l.strip()))


def stream_cache_rows(signal, start, end):
    """Stream a signal's 2026 cache chunks, return window rows:
    {(direction, coin, bar_ts): (cell, feat_row float16[73])}"""
    subdir, prefix = SIG_CACHE[signal]
    cache_dir = ROOT / 'backtest' / subdir / 'cache'
    files = sorted(cache_dir.glob(f'{prefix}_5m_2026*.pkl'))
    out = {}
    for fpath in files:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        for direction in ('SHORT', 'LONG'):
            sd = data.get(direction)
            if not sd:
                continue
            trades = sd['trades']
            feats = sd['features']
            trends = sd['trends']
            vols = sd['vols']
            for i, t in enumerate(trades):
                if not isinstance(t, dict):
                    continue
                et = t['entry_time']
                if not (start <= et <= end):
                    continue
                cell = f"{TREND_NAMES[int(trends[i])]}_{MOM_NAMES[int(vols[i])]}"
                out[(direction, t['coin'], et)] = (cell, np.array(feats[i]))
        del data
    return out


def gate_pass_row(feat_row, cfg):
    """Backtest-style gate+block on a cache feature row with CELL_CONFIG rules."""
    for feat_name, op, value in cfg.get('gate', []):
        if feat_name not in ALL_COL_NAMES:
            continue
        fi = ALL_COL_NAMES.index(feat_name)
        v = float(feat_row[fi])
        if not np.isfinite(v):
            return False
        if op == '<' and not (v < value):
            return False
        if op == '>' and not (v > value):
            return False
    for feat_name, op, value in cfg.get('block', []):
        if feat_name not in ALL_COL_NAMES:
            continue
        fi = ALL_COL_NAMES.index(feat_name)
        v = float(feat_row[fi])
        if not np.isfinite(v):
            continue
        if op == '<' and v < value:
            return False
        if op == '>' and v > value:
            return False
    return True


def load_pickle_final(start, end):
    """Final pickle membership: {(cell, config_label, coin, bar)}"""
    final = set()
    for cell_file in PICKLE_DIR.glob('*_2026.pkl'):
        cell = cell_file.stem.replace('_2026', '')
        with open(cell_file, 'rb') as f:
            data = pickle.load(f)
        for label, trades in data.items():
            for t in trades:
                et = t['entry_time']
                if start <= et <= end:
                    final.add((cell, label, t['coin'], et))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)  # inclusive end day

    t0 = time.time()
    universe = load_universe()
    print(f'Universe: {len(universe)} coins | window {start} -> {end}')

    # ── Load data in main too (universe-gap set + serial fallback) ──
    # 17 days: 4400-bar buffer (15.3d) + margin, so ema_a's window is full
    warm_start = start - pd.Timedelta(days=17)
    _worker_init(str(warm_start), str(end), universe)
    coin_dfs = _G['coin_dfs']
    print(f'Coin data: {len(coin_dfs)} loaded ({time.time()-t0:.0f}s)')

    # ── Replica pass (multiprocess over bar chunks) ──
    bars = pd.date_range(start, end - pd.Timedelta(minutes=5), freq='5min')
    replica_fire = {}    # (d, coin, bar) -> (cell, signal)
    replica_gate = {}    # (d, coin, bar) -> gate+block passed (bool)
    cells_by_bar = {}
    signals_needed = set()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    rep_cache = HERE / f'_replica_{args.start}_{args.end}.pkl'
    if rep_cache.exists():
        with open(rep_cache, 'rb') as f:
            replica_fire, replica_gate, cells_by_bar, signals_needed = pickle.load(f)
        print(f'Replica loaded from cache: {len(replica_fire)} fires')
    else:
        # CPU+RAM-aware (mirror pipeline auto_workers): each worker RELOADS all
        # coin_dfs in _worker_init, so RAM ~= (n_workers+1) x coin_dfs. Cap by
        # (avail - 3GB reserve) / coin_dfs_size to avoid swap.
        import psutil
        cdf_bytes = sum(int(df.memory_usage(deep=True).sum()) for df in coin_dfs.values())
        # per-worker footprint = coin_dfs (reloaded in worker) + ~1.0GB overhead
        # (python interpreter + read_parquet transient + feature-compute buffers)
        per_worker = cdf_bytes + 1.0e9
        avail = psutil.virtual_memory().available
        by_ram = max(1, int((avail - 3e9) // per_worker))
        by_cpu = max(1, (os.cpu_count() or 8) - 2)
        n_workers = max(1, min(by_cpu, by_ram))
        env = os.environ.get('REPLICA_WORKERS')
        if env:
            n_workers = min(n_workers, int(env))
        print(f'  replica workers: cpu={by_cpu} ram-limit={by_ram} '
              f'(per-worker {per_worker/1e9:.2f}GB, avail {avail/1e9:.1f}GB) '
              f'-> {n_workers}', flush=True)
        if len(bars) < 100 or n_workers == 1:
            f, g, c, s = _replica_bars(list(bars))
            replica_fire.update(f); replica_gate.update(g)
            cells_by_bar.update(c); signals_needed |= s
        else:
            chunks = np.array_split(np.array(bars, dtype='datetime64[ns]'), n_workers)
            chunks = [[pd.Timestamp(x) for x in ch] for ch in chunks if len(ch)]
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(str(warm_start), str(end), universe)) as ex:
                futs = [ex.submit(_replica_bars, ch) for ch in chunks]
                for fut in as_completed(futs):
                    f, g, c, s = fut.result()
                    replica_fire.update(f); replica_gate.update(g)
                    cells_by_bar.update(c); signals_needed |= s
        with open(rep_cache, 'wb') as f:
            pickle.dump((replica_fire, replica_gate, cells_by_bar,
                         signals_needed), f)
        print(f'Replica done: {len(replica_fire)} fires ({time.time()-t0:.0f}s)')

    # ── Cache rows for needed signals ──
    cache_rows = {}
    for sig in sorted(signals_needed):
        if sig not in SIG_CACHE:
            continue
        rows = stream_cache_rows(sig, start, end)
        for k, v in rows.items():
            cache_rows.setdefault(sig, {})[k] = v
        print(f'  cache {sig}: {len(rows)} window rows ({time.time()-t0:.0f}s)',
              flush=True)

    final = load_pickle_final(start, end)
    print(f'Pickle final rows in window: {len(final)}')

    # ── L1+L2+L3 classification ──
    stats = defaultdict(int)
    examples = defaultdict(list)
    cov = defaultdict(lambda: [0, 0, 0])  # (dir,cell,sig) -> [compared, gate_pass, kept]

    for (d, sym, T, sig), cell in replica_fire.items():
        crows = cache_rows.get(sig, {})
        ck = (d, sym, T)
        if ck not in crows:
            if not sym.isascii() or sym == 'XMRUSDT':
                stats['L1_replica_only_excluded_universe'] += 1
            else:
                stats['L1_replica_only'] += 1
                stats[f'L1_replica_only::{sig}'] += 1
                if len(examples['L1_replica_only']) < 8:
                    examples['L1_replica_only'].append(f'{d} {sym} {T} {cell}/{sig}')
            continue
        c_cell, feat_row = crows[ck]
        if c_cell != cell:
            stats['L1_cell_mismatch'] += 1
            if len(examples['L1_cell_mismatch']) < 8:
                examples['L1_cell_mismatch'].append(
                    f'{d} {sym} {T} replica={cell} cache={c_cell}')
            continue
        stats['L1_match'] += 1

        cfg = CELL_CONFIG.get((d, cell, sig))
        if cfg is None:
            continue
        rep_ok = replica_gate[(d, sym, T, sig)]
        bt_ok = gate_pass_row(feat_row, cfg)
        ck = f'{d}_{cell}_{sig}'
        cov[ck][0] += 1   # compared (L2)
        if rep_ok == bt_ok:
            stats['L2_match'] += 1
        else:
            stats['L2_diverge'] += 1
            if len(examples['L2_diverge']) < 10:
                examples['L2_diverge'].append(
                    f'{d} {sym} {T} {cell}/{sig} replica={rep_ok} backtest={bt_ok}')

        if rep_ok and bt_ok:
            cov[ck][1] += 1   # gate+block passed (pre-QS)
            in_final = any(fk[2] == sym and fk[3] == T and fk[0] == cell
                           for fk in final)
            if in_final:
                cov[ck][2] += 1   # kept (would open, post-QS)
            stats['L3_qs_kept' if in_final else 'L3_qs_dropped'] += 1

    # Cache rows that replica never fired (reverse flips + universe gap)
    for sig, crows in cache_rows.items():
        for (d, sym, T), (c_cell, _fr) in crows.items():
            if (d, sym, T, sig) in replica_fire:
                continue
            if T not in cells_by_bar or cells_by_bar[T] != c_cell:
                continue  # different regime context — not a comparable row
            if (d, c_cell, sig) not in CELL_CONFIG:
                continue  # not an active combo config
            if sym not in coin_dfs:
                stats['L1_cache_only_universe'] += 1
                continue
            ls = _LST.get(sym)   # 365d age gate: replica correctly skips young coins
            if ls is None or (T - ls).days < MIN_AGE_DAYS:
                stats['L1_cache_only_age_gated'] += 1
                continue
            stats['L1_cache_only'] += 1
            stats[f'L1_cache_only::{sig}'] += 1
            if len(examples['L1_cache_only']) < 8:
                examples['L1_cache_only'].append(f'{d} {sym} {T} {c_cell}/{sig}')

    # ── Report (built into a string, printed AND persisted to file) ──
    n_l1 = stats['L1_match'] + stats['L1_replica_only'] + stats['L1_cache_only']
    n_l2 = stats['L2_match'] + stats['L2_diverge']
    l1_rate = stats['L1_match'] / max(n_l1, 1) * 100
    l2_div_pct = stats['L2_diverge'] / max(n_l2, 1) * 100
    rep = []
    rep.append(f'{"="*72}\nSUMMARY {args.start} -> {args.end}\n{"="*72}')
    rep.append(f"L1 SIGNAL : match={stats['L1_match']} "
               f"replica_only={stats['L1_replica_only']} "
               f"cache_only={stats['L1_cache_only']} "
               f"(universe_gap={stats['L1_cache_only_universe']}, "
               f"age_gated={stats['L1_cache_only_age_gated']}, "
               f"cell_mismatch={stats['L1_cell_mismatch']})")
    if n_l1:
        rep.append(f"            match_rate={l1_rate:.2f}%")
    rep.append(f"L2 GATE   : match={stats['L2_match']} diverge={stats['L2_diverge']}"
               f" ({l2_div_pct:.2f}% diverge)")
    rep.append(f"L3 FINAL  : qs_kept={stats['L3_qs_kept']} "
               f"qs_dropped={stats['L3_qs_dropped']} (QS narrowing — by design)")
    rep.append(f"            replica_only excluded-universe (non-ascii/XMR): "
               f"{stats['L1_replica_only_excluded_universe']}")
    per_sig = sorted((k, v) for k, v in stats.items() if '::' in k)
    if per_sig:
        rep.append('\nPer-signal breakdown:')
        for k, v in per_sig:
            rep.append(f'  {k}: {v}')
    for k in ('L1_replica_only', 'L1_cache_only', 'L1_cell_mismatch', 'L2_diverge'):
        if examples[k]:
            rep.append(f'\n{k} examples:')
            for e in examples[k]:
                rep.append(f'  {e}')
    report_text = '\n'.join(rep)
    print('\n' + report_text)

    # persist: append human-readable block + merge machine-readable per-window stats
    with open(HERE / '_diff_report.txt', 'a', encoding='utf-8') as f:
        f.write('\n' + report_text + '\n')
    sp = HERE / '_diff_stats.json'
    allstats = json.loads(sp.read_text()) if sp.exists() else {}
    allstats[f'{args.start}_{args.end}'] = {
        'L1_match': stats['L1_match'], 'L1_replica_only': stats['L1_replica_only'],
        'L1_cache_only': stats['L1_cache_only'], 'L1_cell_mismatch': stats['L1_cell_mismatch'],
        'L1_match_rate': round(l1_rate, 3),
        'L2_match': stats['L2_match'], 'L2_diverge': stats['L2_diverge'],
        'L2_diverge_pct': round(l2_div_pct, 4),
        'L3_qs_kept': stats['L3_qs_kept'], 'L3_qs_dropped': stats['L3_qs_dropped'],
        'L2_diverge_examples': list(examples['L2_diverge']),
        'L1_cell_mismatch_examples': list(examples['L1_cell_mismatch']),
    }
    sp.write_text(json.dumps(allstats, indent=2))

    # ── per-config coverage: merge into persistent JSON (accumulates across windows) ──
    cov_path = HERE / '_config_coverage.json'
    acc = json.loads(cov_path.read_text()) if cov_path.exists() else {}
    for ck, (cmp_, gp, kept) in cov.items():
        a = acc.get(ck, [0, 0, 0])
        acc[ck] = [a[0] + cmp_, a[1] + gp, a[2] + kept]
    cov_path.write_text(json.dumps(acc, indent=0))
    print(f'\nconfig coverage merged -> {cov_path.name} ({len(acc)} configs seen)')

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
