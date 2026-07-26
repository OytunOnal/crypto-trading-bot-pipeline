"""Common year-split cache builder with base class interface.

Each strategy implements StrategyCache:
    class SmaCrossCache(StrategyCache):
        CACHE_FILE = Path(...)
        DESCRIPTION = 'SMA Cross (toy example)'
        def compute_signals(self, df): return long_mask, short_mask

Usage:
    cache_obj = SmaCrossCache()
    cache_obj.build()           # all years
    cache_obj.build(2026)       # single year
    data = cache_obj.load()     # all years
    data = cache_obj.load([2026])  # specific years
"""
import sys, os, time, pickle, gc
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'data' / 'cache'

sys.path.insert(0, str(ROOT))
from factory.gates.features_common import (
    _ROC10_IDX, _ATR_PCT_IDX, N_FEAT, N_COLS, ALL_COL_NAMES,
    compute_features, simulate_sltp, rolling_resample_ohlcv,
    build_btc_trend, load_btc_5m_features,
)
from factory.data.data_io import load_ohlcv, load_ohlcv_years, resolve_coins

MAX_BARS = 288
WARMUP_5M = 1200
TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
MOM_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}
BTC_TREND_FIXED = (-2.1, 2.7)
BTC_MOM_ROC_HOURS = 24
BTC_MOM_ROLL_WINDOW = 150
SL_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


class StrategyCache:
    """Base class for strategy cache builders.

    Subclasses must set:
        CACHE_FILE: Path to cache file
        DESCRIPTION: human label

    Subclasses must implement:
        compute_signals(df) -> (long_mask, short_mask)
    """
    CACHE_FILE = None
    DESCRIPTION = ''

    def compute_signals(self, df):
        """Override: compute signal masks from 5m DataFrame.
        Returns (long_mask, short_mask) boolean numpy arrays.
        """
        raise NotImplementedError

    def build(self, target_year=None):
        _build(self, target_year)

    def load(self, years=None):
        return _load(self.CACHE_FILE, years)

    def load_lite(self, direction, years=None, skip_trades=True, cell=None,
                  pnl_keys=None):
        """Load single direction only, pnl native float64.
        skip_trades=True: gate/phase2. skip_trades=False: QS/portfolio.
        cell=(trend_idx, vol_idx): stream-filter to one regime cell.
        pnl_keys: load ONLY these pnl_* arrays (None = all ~35). Saves RAM
        when a consumer needs just a few sl/tp (e.g. save_cell_trades).
        """
        return _load_lite(self.CACHE_FILE, direction, years,
                          skip_trades=skip_trades, cell=cell, pnl_keys=pnl_keys)

    def load_year(self, year):
        return _load_year(self.CACHE_FILE, year)


def _load_btc_1h():
    """BTC 1h series resampled from 5m (1h parquets removed — single source)."""
    df5 = load_ohlcv('BTCUSDT', '5m', cache_dir=str(CACHE_DIR))
    if df5.index.tz is not None:
        df5.index = df5.index.tz_localize(None)
    df = df5.resample('1h').agg({'close': 'last'}).dropna()
    return {'close': df['close'], 'ret': df['close'].pct_change(), 'index': df.index}


def _build_btc_trend_5m():
    df5 = load_ohlcv('BTCUSDT', '5m', cache_dir=str(CACHE_DIR))
    if df5.index.tz is not None:
        df5.index = df5.index.tz_localize(None)
    df1h = df5.resample('1h').agg({'close': 'last'}).dropna()
    trend_1h = build_btc_trend(df1h['close'], fixed_terciles=BTC_TREND_FIXED)
    # Hour H's value only becomes known at its close (H+1h). Shift so 5m bars
    # in hour H use hour H-1's value — matches live (scanner drops last 1h bar).
    trend_1h.index = trend_1h.index + pd.Timedelta(hours=1)
    return trend_1h.reindex(df5.index, method='ffill').values, df5.index


def _build_btc_mom_5m():
    df5 = load_ohlcv('BTCUSDT', '5m', cache_dir=str(CACHE_DIR))
    if df5.index.tz is not None:
        df5.index = df5.index.tz_localize(None)
    df1h = df5.resample('1h').agg({'close': 'last'}).dropna()
    roc = df1h['close'].pct_change(BTC_MOM_ROC_HOURS)
    rank = roc.rolling(BTC_MOM_ROLL_WINDOW, min_periods=20).rank(pct=True)
    mom = pd.Series(np.nan, index=df1h.index)
    mom[rank <= 0.3333] = 0
    mom[(rank > 0.3333) & (rank <= 0.6667)] = 1
    mom[rank > 0.6667] = 2
    # Hour H's value only becomes known at its close (H+1h). Shift so 5m bars
    # in hour H use hour H-1's value — matches live (scanner drops last 1h bar).
    mom.index = mom.index + pd.Timedelta(hours=1)
    return mom.reindex(df5.index, method='ffill').values, df5.index


def _new_dd():
    """Create empty accumulator dict for one year's data."""
    return {'SHORT': {'f': [], 'y': [], 't': [], 'v': [], 'trades_25': []},
            'LONG':  {'f': [], 'y': [], 't': [], 'v': [], 'trades_15': []}}


def _build(strategy, target_year=None):
    t0 = time.time()
    cache_file = strategy.CACHE_FILE
    print(f'  Building {strategy.DESCRIPTION}...', flush=True)

    btc = _load_btc_1h()
    btc_trend = build_btc_trend(btc['close'], fixed_terciles=BTC_TREND_FIXED)
    btc_5m = load_btc_5m_features(load_ohlcv, CACHE_DIR)
    print(f'  Building BTC trend 5m...', flush=True)
    btc_trend_5m_arr, btc_5m_idx = _build_btc_trend_5m()
    print(f'  Building BTC momentum 5m...', flush=True)
    btc_mom_5m_arr, _ = _build_btc_mom_5m()
    print(f'  BTC regime ready ({time.time()-t0:.0f}s)', flush=True)

    years_to_build = [target_year] if target_year else list(range(2021, 2027))
    coins = resolve_coins('all', cache_dir=str(CACHE_DIR))
    cache_base = str(cache_file).replace('_raw_cache.pkl', '')

    FLUSH_EVERY = 50  # flush accumulated data to temp file every N coins

    for yr in years_to_build:
        print(f'\n  === Year {yr} ===', flush=True)
        dd = _new_dd()
        n_ok = 0
        load_years = [yr - 1, yr] if yr > 2021 else [yr]
        chunk_files = []

        for ci, coin in enumerate(coins):
            df = load_ohlcv_years(coin, '5m', years=load_years, cache_dir=str(CACHE_DIR))
            if df is None or df.empty or len(df) < WARMUP_5M + MAX_BARS + 100:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            c5, o5 = df['close'].values, df['open'].values
            h5, l5 = df['high'].values, df['low'].values
            n5, ts5 = len(df), df.index

            long_mask, short_mask = strategy.compute_signals(df)

            r15, r1h, r4h = rolling_resample_ohlcv(df)
            f15, f1h, f4h = compute_features(r15), compute_features(r1h), compute_features(r4h)

            d1h = df.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            if len(d1h) < 210:
                continue

            btc_mom_at_coin = pd.Series(btc_mom_5m_arr, index=btc_5m_idx).reindex(ts5, method='ffill')
            bim = btc['index']

            for dr, mask in [('SHORT', short_mask), ('LONG', long_mask)]:
                sa = np.where(mask)[0]
                # MIN_COIN_AGE (=210x1h=2520) matches live MIN_COIN_BARS: a
                # newly listed coin becomes tradeable for BOTH sides at the
                # same bar. Costs nothing for old coins (warmup year covers it).
                sa = sa[(sa >= max(WARMUP_5M, 2520)) & (ts5[sa].year == yr)]
                if len(sa) == 0:
                    continue
                st = ts5[sa]
                m1 = st.floor('1h') - pd.Timedelta(hours=1)
                i1 = d1h.index.get_indexer(m1, method='ffill')
                bm = bim.get_indexer(m1, method='ffill')
                # floor-1h: last completed hour at signal time (no intra-hour look-ahead)
                tv = btc_trend.reindex(st.floor('1h') - pd.Timedelta(hours=1), method='ffill').values
                vv = btc_mom_at_coin.reindex(st).values
                ok = ((sa >= WARMUP_5M) & (i1 >= 0) & (i1 < len(d1h)) &
                      (bm >= 0) & np.isfinite(tv) & np.isfinite(vv) &
                      (tv >= 0) & (vv >= 0))
                w = np.where(ok)[0]
                if len(w) < 10:
                    continue
                si = sa[w]; Ne = len(w)

                # Features
                feat = np.full((Ne, N_COLS), np.nan, dtype=np.float16)
                feat[:, :N_FEAT] = f15[si].astype(np.float16)
                feat[:, N_FEAT:2*N_FEAT] = f1h[si].astype(np.float16)
                feat[:, 2*N_FEAT:3*N_FEAT] = f4h[si].astype(np.float16)
                cb = 3 * N_FEAT
                bi = btc_5m['index'].get_indexer(st[w], method='ffill')
                bv = (bi >= 0) & (bi < len(btc_5m['roc10']))
                feat[:, cb] = (f1h[si, _ROC10_IDX] - np.where(
                    bv, btc_5m['roc10'][np.clip(bi, 0, len(btc_5m['roc10'])-1)], np.nan
                )).astype(np.float16)
                ba = np.where(bv, btc_5m['atr_pct'][np.clip(bi, 0, len(btc_5m['atr_pct'])-1)], np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    feat[:, cb+1] = (f1h[si, _ATR_PCT_IDX] / (ba + 1e-10)).astype(np.float16)
                cr = d1h['close'].pct_change()
                c20 = cr.rolling(20).corr(btc['ret'].reindex(d1h.index))
                feat[:, cb+2] = c20.values[np.clip(i1[w], 0, len(c20)-1)].astype(np.float16)
                bc = btc['close']
                r24 = bc.pct_change(24).reindex(d1h.index, method='ffill')
                r4 = bc.pct_change(4).reindex(d1h.index, method='ffill')
                r10 = bc.pct_change(10)
                sl_ = (r10 - r10.shift(5)).reindex(d1h.index, method='ffill')
                feat[:, cb+3] = r24.values[np.clip(i1[w], 0, len(r24)-1)].astype(np.float16)
                feat[:, cb+4] = r4.values[np.clip(i1[w], 0, len(r4)-1)].astype(np.float16)
                feat[:, cb+5] = sl_.values[np.clip(i1[w], 0, len(sl_)-1)].astype(np.float16)

                # SL/TP sim
                ep = c5[si]
                fi_arr = np.clip(si[:, None] + np.arange(1, MAX_BARS+1)[None, :], 0, n5-1)
                if dr == 'SHORT':
                    mfe = np.maximum.accumulate((1-l5[fi_arr]/ep[:, None])*100, axis=1)
                    mae = np.maximum.accumulate((h5[fi_arr]/ep[:, None]-1)*100, axis=1)
                else:
                    mfe = np.maximum.accumulate((h5[fi_arr]/ep[:, None]-1)*100, axis=1)
                    mae = np.maximum.accumulate((1-l5[fi_arr]/ep[:, None])*100, axis=1)
                pnl_dict = {}
                for slv in SL_LEVELS:
                    for tpv in SL_LEVELS:
                        if tpv >= slv:
                            pnl_dict[(slv, tpv)] = simulate_sltp(mfe, mae, slv, tpv).astype(np.float64)

                # Resolve unresolved
                exit_close = c5[np.clip(si+MAX_BARS, 0, n5-1)]
                tp_ = ((1-exit_close/ep)*100 if dr == 'SHORT' else (exit_close/ep-1)*100)
                for slv in SL_LEVELS:
                    for tpv in SL_LEVELS:
                        if tpv >= slv:
                            arr = pnl_dict[(slv, tpv)]
                            bad = ~np.isfinite(arr)
                            if bad.any(): arr[bad] = tp_[bad]

                dsl = 2.5 if dr == 'SHORT' else 1.5
                pnl = pnl_dict[(dsl, dsl)]
                sl_hit = mae >= dsl; tp_hit = mfe >= dsl
                sl_bar = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), MAX_BARS)
                tp_bar = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), MAX_BARS)
                exit_bars = np.minimum(sl_bar, tp_bar) + 1

                d = dd[dr]
                d['f'].append(feat)
                for slv in SL_LEVELS:
                    for tpv in SL_LEVELS:
                        if tpv >= slv:
                            d.setdefault(f'p_{slv}_{tpv}', []).append(pnl_dict[(slv, tpv)])
                d['y'].append(st[w].year.values.astype(np.int16))
                d['t'].append(tv[w].astype(np.int8))
                d['v'].append(vv[w].astype(np.int8))
                tkey = 'trades_25' if dr == 'SHORT' else 'trades_15'
                for j in range(Ne):
                    eidx = min(si[j]+exit_bars[j], n5-1)
                    d[tkey].append({
                        'coin': coin, 'direction': dr,
                        'entry_time': st[w[j]], 'exit_time': ts5[eidx],
                        'cell': f'{TREND_NAMES[int(tv[w[j]])]}_{MOM_NAMES[int(vv[w[j]])]}',
                        'pnl_pct': float(pnl[j]),
                        'trend': int(tv[w[j]]), 'vol': int(vv[w[j]]),
                    })

            n_ok += 1
            if (ci+1) % 50 == 0 or ci+1 == len(coins):
                ns = sum(len(a) for a in dd['SHORT']['f'])
                nl = sum(len(a) for a in dd['LONG']['f'])
                print(f'  [{ci+1:>3}/{len(coins)}] {n_ok} coins S={ns:,} L={nl:,} ({time.time()-t0:.0f}s)', flush=True)

            # Flush chunk to disk every FLUSH_EVERY coins (no merge needed later)
            if (ci+1) % FLUSH_EVERY == 0 and any(dd[dr]['f'] for dr in ['SHORT', 'LONG']):
                _save_year_chunk(cache_file, yr, dd, len(chunk_files), t0)
                chunk_files.append(len(chunk_files))
                dd = _new_dd()
                gc.collect()

        # Save remaining data as final chunk
        if any(dd[dr]['f'] for dr in ['SHORT', 'LONG']):
            _save_year_chunk(cache_file, yr, dd, len(chunk_files), t0)
            chunk_files.append(len(chunk_files))
            del dd; gc.collect()

        # If only 1 chunk, rename to simple year file (no chunking overhead on load)
        cache_base = str(cache_file).replace('_raw_cache.pkl', '')
        if len(chunk_files) == 1:
            src = Path(f'{cache_base}_{yr}_c0.pkl')
            dst = Path(f'{cache_base}_{yr}.pkl')
            if src.exists():
                if dst.exists(): dst.unlink()
                src.rename(dst)
        else:
            # Remove old single file if chunked now
            old_single = Path(f'{cache_base}_{yr}.pkl')
            if old_single.exists():
                old_single.unlink()
            ns = nl = 0
            total_sz = 0
            for ci in range(len(chunk_files)):
                cp = Path(f'{cache_base}_{yr}_c{ci}.pkl')
                if cp.exists():
                    total_sz += cp.stat().st_size
            print(f'  Year {yr}: {len(chunk_files)} chunks ({total_sz/1e6:.0f} MB, {time.time()-t0:.0f}s)',
                  flush=True)

    # Save metadata
    _save_meta(cache_file, years_to_build, btc_trend_5m_arr, btc_mom_5m_arr, btc_5m_idx)
    print(f'  Done: {years_to_build} ({time.time()-t0:.0f}s)', flush=True)


def _save_year_chunk(cache_file, yr, dd, chunk_idx, t0):
    """Save one chunk of year data. Low memory: only this chunk in RAM."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_base = str(cache_file).replace('_raw_cache.pkl', '')
    cache_yr = {}
    for dr in ['SHORT', 'LONG']:
        d = dd[dr]
        tkey = 'trades_25' if dr == 'SHORT' else 'trades_15'
        dsl = 2.5 if dr == 'SHORT' else 1.5
        if not d['f']:
            cache_yr[dr] = {'features': np.array([]).reshape(0,0), 'years': np.array([]),
                            'trends': np.array([]), 'vols': np.array([]),
                            'trades': [], 'pnl': np.array([])}
            continue
        entry = {
            'features': np.vstack(d['f']), 'years': np.concatenate(d['y']),
            'trends': np.concatenate(d['t']), 'vols': np.concatenate(d['v']),
            'trades': d[tkey],
        }
        for slv in SL_LEVELS:
            for tpv in SL_LEVELS:
                if tpv >= slv:
                    k = f'p_{slv}_{tpv}'
                    if k in d and d[k]:
                        if slv == tpv: entry[f'pnl_{slv}'] = np.concatenate(d[k]).astype(np.float64)
                        entry[f'pnl_{slv}_{tpv}'] = np.concatenate(d[k]).astype(np.float64)
        entry['pnl'] = entry.get(f'pnl_{dsl}_{dsl}', np.array([]))
        cache_yr[dr] = entry

    chunk_path = Path(f'{cache_base}_{yr}_c{chunk_idx}.pkl')
    with open(chunk_path, 'wb') as f:
        pickle.dump(cache_yr, f, protocol=4)
    sz = chunk_path.stat().st_size / 1e6
    ns = len(cache_yr.get('SHORT', {}).get('trades', []))
    nl = len(cache_yr.get('LONG', {}).get('trades', []))
    print(f'  Chunk {chunk_idx}: S={ns:,} L={nl:,} ({sz:.0f} MB, {time.time()-t0:.0f}s)', flush=True)
    del cache_yr; gc.collect()


def _save_meta(cache_file, years_built, btc_trend_5m, btc_mom_5m, btc_5m_idx):
    cache_base = str(cache_file).replace('_raw_cache.pkl', '')
    meta_path = Path(f'{cache_base}_meta.pkl')
    old_years = []
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            old_years = pickle.load(f).get('years', [])
    meta = {
        'btc_trend_5m': btc_trend_5m, 'btc_mom_5m': btc_mom_5m,
        'btc_5m_idx': btc_5m_idx, 'col_names': ALL_COL_NAMES,
        'years': sorted(set(old_years + years_built)),
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=4)


def _load_year(cache_file, year):
    """Load year data. Supports both single file and chunked files."""
    cache_base = str(cache_file).replace('_raw_cache.pkl', '')

    # Try single file first
    yr_path = Path(f'{cache_base}_{year}.pkl')
    if yr_path.exists():
        with open(yr_path, 'rb') as f:
            return pickle.load(f)

    # Try chunked files
    chunk_paths = sorted(Path(cache_base).parent.glob(
        f'{Path(cache_base).name}_{year}_c*.pkl'))
    if not chunk_paths:
        return None

    # Merge chunks on load
    merged = None
    for cp in chunk_paths:
        with open(cp, 'rb') as f:
            chunk = pickle.load(f)
        if merged is None:
            merged = chunk
            continue
        for dr in ['SHORT', 'LONG']:
            if dr not in chunk or dr not in merged:
                continue
            m, c = merged[dr], chunk[dr]
            if c.get('features', np.array([])).size == 0:
                continue
            if m.get('features', np.array([])).size == 0:
                merged[dr] = c
                continue
            m['features'] = np.vstack([m['features'], c['features']])
            m['years'] = np.concatenate([m['years'], c['years']])
            m['trends'] = np.concatenate([m['trends'], c['trends']])
            m['vols'] = np.concatenate([m['vols'], c['vols']])
            m['trades'] = m.get('trades', []) + c.get('trades', [])
            for k in c:
                if k.startswith('pnl') and isinstance(c[k], np.ndarray) and c[k].size > 0:
                    if k in m and isinstance(m[k], np.ndarray) and m[k].size > 0:
                        m[k] = np.concatenate([m[k], c[k]])
                    else:
                        m[k] = c[k]
        del chunk
    return merged


def _load_lite(cache_file, direction, years=None, skip_trades=True, cell=None,
               pnl_keys=None):
    """Load single direction, optionally skip trades. pnl stays float64.

    pnl dtype is the cache's NATIVE float64 — never cast down. The legacy
    per-strategy diags were split (some full-load float64, some lite
    float32) and a float32 cast flips near-tie config selections in the
    gate/phase2 greedy searches. Canon since 2026-06-12: float64.

    Peak RAM = one year/chunk pickle at a time (other direction deleted
    immediately). Supports chunked years (older code silently SKIPPED them).
    cell=(trend_idx, vol_idx): stream-filter to a single regime cell — peak
    RAM drops to ~1/9 of the direction (enables per-cell worker processes).
    skip_trades=True: gate sweep / phase2; False: QS / portfolio sim.
    """
    meta_path = Path(str(cache_file).replace('_raw_cache.pkl', '_meta.pkl'))
    if not meta_path.exists(): return None
    with open(meta_path, 'rb') as f: meta = pickle.load(f)
    avail = meta['years']
    yrs = [y for y in (years or avail) if y in avail]
    if not yrs: return None
    mode = 'LITE' if skip_trades else 'LITE+TRADES'
    print(f'  Loading {mode} [{direction}] years {yrs}...', flush=True)
    t0 = time.time()

    p = {'features': [], 'years': [], 'trends': [], 'vols': [], 'trades': []}
    pp = {}
    other_dir = 'LONG' if direction == 'SHORT' else 'SHORT'

    cache_base = str(cache_file).replace('_raw_cache.pkl', '')

    def _append_entry(e):
        if e.get('features', np.array([])).size == 0:
            return
        if cell is not None:
            m = (e['trends'] == cell[0]) & (e['vols'] == cell[1])
            if not m.any():
                return
            p['features'].append(e['features'][m])
            p['years'].append(e['years'][m])
            p['trends'].append(e['trends'][m])
            p['vols'].append(e['vols'][m])
            if not skip_trades:
                tr = e.get('trades', [])
                p['trades'].extend([t for t, keep in zip(tr, m) if keep])
            for k, v in e.items():
                if k.startswith('pnl') and (pnl_keys is None or k in pnl_keys):
                    pp.setdefault(k, []).append(v[m])
        else:
            p['features'].append(e['features'])
            p['years'].append(e['years'])
            p['trends'].append(e['trends'])
            p['vols'].append(e['vols'])
            if not skip_trades:
                p['trades'].extend(e.get('trades', []))
            for k, v in e.items():
                if k.startswith('pnl') and (pnl_keys is None or k in pnl_keys):
                    pp.setdefault(k, []).append(v)

    for yr in yrs:
        yr_path = Path(f'{cache_base}_{yr}.pkl')
        yr_chunks = sorted(Path(cache_base).parent.glob(
            f'{Path(cache_base).name}_{yr}_c*.pkl'))

        # Monthly cell-sharded fast path: {base}_{YYYY-MM}_t{T}v{V}.pkl.
        # Used ONLY when the year has no legacy file (full-pipeline format);
        # legacy year files always win to avoid silent partial loads in a
        # mixed-format cache dir.
        if not yr_path.exists() and not yr_chunks:
            if cell is not None:
                m_paths = sorted(Path(cache_base).parent.glob(
                    f'{Path(cache_base).name}_{yr}-??_t{cell[0]}v{cell[1]}.pkl'))
            else:
                m_paths = sorted(Path(cache_base).parent.glob(
                    f'{Path(cache_base).name}_{yr}-??_t?v?.pkl'))
        else:
            m_paths = []
        if m_paths:
            for cp in m_paths:
                with open(cp, 'rb') as f:
                    yc = pickle.load(f)
                if direction in yc:
                    e = yc[direction]
                    if skip_trades:
                        e.pop('trades', None)
                    _append_entry(e)
                del yc
            gc.collect()
            continue

        paths = [yr_path] if yr_path.exists() else yr_chunks
        for cp in paths:
            with open(cp, 'rb') as f:
                yc = pickle.load(f)
            if other_dir in yc:
                del yc[other_dir]
            if skip_trades and direction in yc:
                yc[direction].pop('trades', None)
            if direction in yc:
                _append_entry(yc[direction])
            del yc
            gc.collect()

    if not p['features']:
        return None

    result = {direction: {
        'features': np.vstack(p['features']),
        'years': np.concatenate(p['years']),
        'trends': np.concatenate(p['trends']),
        'vols': np.concatenate(p['vols']),
        'trades': p['trades'],
    }}
    del p; gc.collect()
    for k, vl in pp.items():
        result[direction][k] = np.concatenate(vl)
    del pp; gc.collect()

    for k in ['btc_trend_5m', 'btc_mom_5m', 'btc_5m_idx', 'col_names']:
        result[k] = meta.get(k)

    n = len(result[direction]['features'])
    nt = len(result[direction]['trades'])
    elapsed = time.time() - t0
    print(f'  Loaded {mode} [{direction}] N={n:,} T={nt:,} ({elapsed:.1f}s)', flush=True)
    return result


def _load(cache_file, years=None):
    meta_path = Path(str(cache_file).replace('_raw_cache.pkl', '_meta.pkl'))
    if meta_path.exists():
        with open(meta_path, 'rb') as f: meta = pickle.load(f)
        avail = meta['years']
        yrs = [y for y in (years or avail) if y in avail]
        if not yrs: return None
        print(f'  Loading cache years {yrs}...', flush=True)
        t0 = time.time()
        combined = {}
        for dr in ['SHORT', 'LONG']:
            p = {'features': [], 'years': [], 'trends': [], 'vols': [], 'trades': []}
            pp = {}
            for yr in yrs:
                yc = _load_year(cache_file, yr)
                if not yc or dr not in yc or yc[dr].get('features', np.array([])).size == 0: continue
                e = yc[dr]
                p['features'].append(e['features']); p['years'].append(e['years'])
                p['trends'].append(e['trends']); p['vols'].append(e['vols'])
                p['trades'].extend(e['trades'])
                for k, v in e.items():
                    if k.startswith('pnl'): pp.setdefault(k, []).append(v)
                del yc
            combined[dr] = {
                'features': np.vstack(p['features']) if p['features'] else np.array([]),
                'years': np.concatenate(p['years']) if p['years'] else np.array([]),
                'trends': np.concatenate(p['trends']) if p['trends'] else np.array([]),
                'vols': np.concatenate(p['vols']) if p['vols'] else np.array([]),
                'trades': p['trades'],
            }
            for k, vl in pp.items(): combined[dr][k] = np.concatenate(vl)
        for k in ['btc_trend_5m', 'btc_mom_5m', 'btc_5m_idx', 'col_names']:
            combined[k] = meta.get(k)
        ns = len(combined['SHORT']['trades']); nl = len(combined['LONG']['trades'])
        print(f'  Loaded S={ns:,} L={nl:,} ({time.time()-t0:.1f}s)', flush=True)
        return combined
    if not cache_file.exists(): return None
    print(f'  Loading legacy cache {cache_file}...', flush=True)
    with open(cache_file, 'rb') as f: return pickle.load(f)
