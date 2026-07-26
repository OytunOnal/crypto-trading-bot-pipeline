"""Monthly CELL-SHARDED cache builder (full-pipeline format, 2026-06-12).

File layout (vs legacy year files):
    {base}_{YYYY-MM}_t{T}v{V}.pkl      one file per (month, regime cell)
    each file: {'SHORT': entry, 'LONG': entry}  — same entry schema as
    legacy year files (features float16, pnl_{sl}_{tp} float64, trades).

Why: engines stream per (direction, cell) — with cell shards a worker
reads ONLY its cell's files (~10-150MB each) instead of unpickling
multi-GB year monoliths (~6GB transient peak per worker). 14 workers
fit easily in RAM; daily incremental update = rebuild current month.

Build window per month M: [M_start - LOOKBACK_DAYS, M_end + 1d].
- LOOKBACK_DAYS=35: EWM warmup. Live computes features on its last
  4400 5m bars (scanner TRIM_BARS) = 15.3d; 35d > 2x live window, so
  every feature that converges for live also converges here. Slow
  features (EWM chain span*6.9 > window) differ from the legacy
  yearly format BY DESIGN — the acceptance test reports them; they are
  candidates for engine exclusion since live can't reproduce them
  either.
- +1d tail: the 288-bar (24h) SL/TP forward window of month-end
  signals resolves inside the build. At month rollover the previous
  month is rebuilt once to finalize.
- Coin age: BAR COUNT since listing (matches live len(ohlcv_5m) >=
  MIN_COIN_BARS=2520) — counted as rows in the coin's full parquet
  before load_start + in-window index. NOT load-window-relative (the
  legacy year builder's count restarts at Jan-1-prev-year; with a 35d
  window that would wrongly re-age every coin each month).

Usage:
  python factory/data/cache_builder_monthly.py --strategy SMA_X --month 2025-06
  python factory/data/cache_builder_monthly.py --strategy SMA_X --from 2021-01 --to 2025-12 --parallel
"""
import sys, os, time, pickle, gc, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.gates.features_common import (
    _ROC10_IDX, _ATR_PCT_IDX, N_FEAT, N_COLS,
    compute_features, simulate_sltp, rolling_resample_ohlcv,
    build_btc_trend, load_btc_5m_features,
)
from factory.data.cache_builder import (
    CACHE_DIR, MAX_BARS, WARMUP_5M, TREND_NAMES, MOM_NAMES,
    BTC_TREND_FIXED, SL_LEVELS,
    _load_btc_1h, _build_btc_trend_5m, _build_btc_mom_5m, _save_meta,
)
from factory.data.registry import load_builder
from factory.data.data_io import load_ohlcv, resolve_coins

LOOKBACK_DAYS = 35
MIN_COIN_AGE = 2520      # bars since listing (== live MIN_COIN_BARS)


# ---------------------------------------------------------------- helpers

def _coin_meta(coin, t_from):
    """(listing_ts, n_before, total_rows) from year-split parquets.

    n_before = bar count before t_from: whole files counted via parquet
    metadata (no read); only the boundary-year file's index is read.
    """
    import pyarrow.parquet as pq

    def _read_idx(f):
        # pd.read_parquet restores the saved DatetimeIndex even with no
        # data columns (pq.read_table(columns=[]) drops it -> 1970 bug)
        idx = pd.read_parquet(f, columns=[]).index
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_localize(None)
        return idx

    files = sorted(Path(CACHE_DIR).glob(f'{coin}_5m_*.parquet'))
    if not files:
        return None
    def _file_year(f):
        return int(f.stem.split('_')[-2][:4])

    total = 0
    n_before = 0
    listing_ts = None
    boundary_yr = t_from.year
    for f in files:
        fy = _file_year(f)
        nr = pq.ParquetFile(f).metadata.num_rows
        total += nr
        if listing_ts is None:
            idx0 = _read_idx(f)
            listing_ts = idx0[0] if len(idx0) else None
            if fy == boundary_yr:
                n_before += int(np.searchsorted(idx0.values,
                                                np.datetime64(t_from), 'left'))
            elif fy < boundary_yr:
                n_before += nr
            continue
        if fy < boundary_yr:
            n_before += nr
        elif fy == boundary_yr:
            n_before += int(np.searchsorted(_read_idx(f).values,
                                            np.datetime64(t_from), 'left'))
    return listing_ts, n_before, total


def _load_window(coin, t_from, t_to):
    """5m OHLCV slice [t_from, t_to) — loads only the needed years."""
    from factory.data.data_io import load_ohlcv_years
    yrs = list(range(t_from.year, t_to.year + 1))
    df = load_ohlcv_years(coin, '5m', years=yrs, cache_dir=str(CACHE_DIR))
    if df is None or df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[(df.index >= t_from) & (df.index < t_to)]


def _shard_path(cache_file, ym, t, v):
    base = str(cache_file).replace('_raw_cache.pkl', '')
    return Path(f'{base}_{ym}_t{t}v{v}.pkl')


def _new_dd():
    return {'SHORT': {'f': [], 'y': [], 't': [], 'v': [], 'trades': []},
            'LONG':  {'f': [], 'y': [], 't': [], 'v': [], 'trades': []}}


# ---------------------------------------------------------------- build

def build_month(strategy_code, ym, btc_ctx=None, verbose=True):
    """Build all 9 cell shards of one month for one strategy.

    ym: 'YYYY-MM'. btc_ctx: precomputed BTC regime context (shared across
    months in a serial loop); None -> computed here.
    """
    t0 = time.time()
    builder = load_builder(strategy_code.upper())
    strategy = builder._instance
    # Test isolation: redirect output (workers re-import, so in-process
    # CACHE_FILE patches don't reach them — env survives spawn).
    ov = os.environ.get('MONTHLY_CACHE_OVERRIDE')
    if ov:
        strategy.CACHE_FILE = Path(ov)
    cache_file = strategy.CACHE_FILE

    m_start = pd.Timestamp(f'{ym}-01')
    m_end = m_start + pd.offsets.MonthBegin(1)
    t_from = m_start - pd.Timedelta(days=LOOKBACK_DAYS)
    t_to = m_end + pd.Timedelta(days=1)
    yr = m_start.year

    if btc_ctx is None:
        btc_ctx = build_btc_context()
    btc = btc_ctx['btc_1h']
    btc_trend = btc_ctx['btc_trend_1h']
    btc_5m = btc_ctx['btc_5m']
    btc_mom_5m = btc_ctx['btc_mom_5m']      # pd.Series on full 5m index

    coins = resolve_coins('all', cache_dir=str(CACHE_DIR))
    dd = _new_dd()
    n_ok = 0

    for coin in coins:
        cm = _coin_meta(coin, t_from)
        if cm is None:
            continue
        listing_ts, n_before, total_rows = cm
        if listing_ts is None or total_rows < MIN_COIN_AGE:
            continue
        # Coins listed near/inside the window: load from LISTING so the
        # in-window index sa equals bars-since-listing exactly (live
        # len(ohlcv_5m) semantics). Adds at most ~LOOKBACK extra bars.
        t_from_coin = t_from
        if 0 < n_before <= LOOKBACK_DAYS * 288:
            t_from_coin = listing_ts
            n_before = 0
        df = _load_window(coin, t_from_coin, t_to)
        if df is None or len(df) < 300:
            continue

        c5 = df['close'].values
        h5, l5 = df['high'].values, df['low'].values
        n5, ts5 = len(df), df.index

        long_mask, short_mask = strategy.compute_signals(df)

        r15, r1h, r4h = rolling_resample_ohlcv(df)
        f15, f1h, f4h = (compute_features(r15), compute_features(r1h),
                         compute_features(r4h))

        d1h = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(d1h) < 30:
            continue

        btc_mom_at_coin = btc_mom_5m.reindex(ts5, method='ffill')
        bim = btc['index']

        for dr, mask in [('SHORT', short_mask), ('LONG', long_mask)]:
            sa = np.where(mask)[0]
            if len(sa) == 0:
                continue
            # in-month only + bar-count age since LISTING (live
            # MIN_COIN_BARS semantics). Loaded-window warmup is implied:
            # young coins load from listing (sa == age >= 2520 > WARMUP),
            # old coins carry the full 35d window (sa >= ~10080).
            in_month = (ts5[sa] >= m_start) & (ts5[sa] < m_end)
            age_ok = (n_before + sa) >= max(WARMUP_5M, MIN_COIN_AGE)
            sa = sa[in_month & age_ok]
            if len(sa) == 0:
                continue
            st = ts5[sa]
            m1 = st.floor('1h') - pd.Timedelta(hours=1)
            i1 = d1h.index.get_indexer(m1, method='ffill')
            bm = bim.get_indexer(m1, method='ffill')
            tv = btc_trend.reindex(st.floor('1h') - pd.Timedelta(hours=1),
                                   method='ffill').values
            vv = btc_mom_at_coin.reindex(st).values
            ok = ((i1 >= 0) & (i1 < len(d1h)) & (bm >= 0) &
                  np.isfinite(tv) & np.isfinite(vv) & (tv >= 0) & (vv >= 0))
            w = np.where(ok)[0]
            if len(w) == 0:
                continue
            si = sa[w]; Ne = len(w)

            feat = np.full((Ne, N_COLS), np.nan, dtype=np.float16)
            feat[:, :N_FEAT] = f15[si].astype(np.float16)
            feat[:, N_FEAT:2*N_FEAT] = f1h[si].astype(np.float16)
            feat[:, 2*N_FEAT:3*N_FEAT] = f4h[si].astype(np.float16)
            cb = 3 * N_FEAT
            bi = btc_5m['index'].get_indexer(st, method='ffill')
            bv = (bi >= 0) & (bi < len(btc_5m['roc10']))
            feat[:, cb] = (f1h[si, _ROC10_IDX] - np.where(
                bv, btc_5m['roc10'][np.clip(bi, 0, len(btc_5m['roc10'])-1)],
                np.nan)).astype(np.float16)
            ba = np.where(bv, btc_5m['atr_pct'][np.clip(bi, 0, len(btc_5m['atr_pct'])-1)], np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                feat[:, cb+1] = (f1h[si, _ATR_PCT_IDX] / (ba + 1e-10)).astype(np.float16)
            cr = d1h['close'].pct_change()
            c20 = cr.rolling(20).corr(btc['ret'].reindex(d1h.index))
            feat[:, cb+2] = c20.values[np.clip(i1[w], 0, len(c20)-1)].astype(np.float16)
            bc = btc['close']
            r24 = bc.pct_change(24).reindex(d1h.index, method='ffill')
            r4_ = bc.pct_change(4).reindex(d1h.index, method='ffill')
            r10 = bc.pct_change(10)
            sl_ = (r10 - r10.shift(5)).reindex(d1h.index, method='ffill')
            feat[:, cb+3] = r24.values[np.clip(i1[w], 0, len(r24)-1)].astype(np.float16)
            feat[:, cb+4] = r4_.values[np.clip(i1[w], 0, len(r4_)-1)].astype(np.float16)
            feat[:, cb+5] = sl_.values[np.clip(i1[w], 0, len(sl_)-1)].astype(np.float16)

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
            exit_close = c5[np.clip(si+MAX_BARS, 0, n5-1)]
            tp_ = ((1-exit_close/ep)*100 if dr == 'SHORT' else (exit_close/ep-1)*100)
            for k_, arr in pnl_dict.items():
                bad = ~np.isfinite(arr)
                if bad.any():
                    arr[bad] = tp_[bad]

            dsl = 2.5 if dr == 'SHORT' else 1.5
            pnl = pnl_dict[(dsl, dsl)]
            sl_hit = mae >= dsl; tp_hit = mfe >= dsl
            sl_bar = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), MAX_BARS)
            tp_bar = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), MAX_BARS)
            exit_bars = np.minimum(sl_bar, tp_bar) + 1

            d = dd[dr]
            d['f'].append(feat)
            for k_, arr in pnl_dict.items():
                d.setdefault(f'p_{k_[0]}_{k_[1]}', []).append(arr)
            d['y'].append(st[w].year.values.astype(np.int16))
            d['t'].append(tv[w].astype(np.int8))
            d['v'].append(vv[w].astype(np.int8))
            for j in range(Ne):
                eidx = min(si[j]+exit_bars[j], n5-1)
                d['trades'].append({
                    'coin': coin, 'direction': dr,
                    'entry_time': st[w[j]], 'exit_time': ts5[eidx],
                    'cell': f'{TREND_NAMES[int(tv[w[j]])]}_{MOM_NAMES[int(vv[w[j]])]}',
                    'pnl_pct': float(pnl[j]),
                    'trend': int(tv[w[j]]), 'vol': int(vv[w[j]]),
                })
        n_ok += 1
        del df
    gc.collect()

    # concat per direction, split by cell, write 9 shards
    n_rows = _write_shards(cache_file, ym, dd)
    if verbose:
        print(f'  [{strategy_code} {ym}] {n_ok} coins, {n_rows:,} rows '
              f'({time.time()-t0:.0f}s)', flush=True)
    return n_rows


def _write_shards(cache_file, ym, dd):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    full = {}
    for dr in ['SHORT', 'LONG']:
        d = dd[dr]
        if not d['f']:
            full[dr] = None
            continue
        entry = {
            'features': np.vstack(d['f']), 'years': np.concatenate(d['y']),
            'trends': np.concatenate(d['t']), 'vols': np.concatenate(d['v']),
            'trades': d['trades'],
        }
        for slv in SL_LEVELS:
            for tpv in SL_LEVELS:
                if tpv >= slv:
                    k = f'p_{slv}_{tpv}'
                    if k in d and d[k]:
                        if slv == tpv:
                            entry[f'pnl_{slv}'] = np.concatenate(d[k])
                        entry[f'pnl_{slv}_{tpv}'] = np.concatenate(d[k])
        dsl = 2.5 if dr == 'SHORT' else 1.5
        entry['pnl'] = entry.get(f'pnl_{dsl}_{dsl}', np.array([]))
        full[dr] = entry

    n_total = 0
    for t in range(3):
        for v in range(3):
            shard = {}
            for dr in ['SHORT', 'LONG']:
                e = full[dr]
                if e is None or len(e['years']) == 0:
                    shard[dr] = {'features': np.array([]).reshape(0, 0),
                                 'years': np.array([]), 'trends': np.array([]),
                                 'vols': np.array([]), 'trades': [],
                                 'pnl': np.array([])}
                    continue
                m = (e['trends'] == t) & (e['vols'] == v)
                sub = {'features': e['features'][m],
                       'years': e['years'][m],
                       'trends': e['trends'][m], 'vols': e['vols'][m],
                       'trades': [tr for tr, keep in zip(e['trades'], m) if keep]}
                for k, arr in e.items():
                    if k.startswith('pnl') and isinstance(arr, np.ndarray) and arr.size:
                        sub[k] = arr[m]
                shard[dr] = sub
                n_total += int(m.sum())
            with open(_shard_path(cache_file, ym, t, v), 'wb') as f:
                pickle.dump(shard, f, protocol=4)
    return n_total


def build_btc_context():
    btc = _load_btc_1h()
    btc_trend_1h = build_btc_trend(btc['close'], fixed_terciles=BTC_TREND_FIXED)
    btc_5m = load_btc_5m_features(load_ohlcv, CACHE_DIR)
    mom_arr, idx5 = _build_btc_mom_5m()
    trend_arr, _ = _build_btc_trend_5m()
    return {'btc_1h': btc, 'btc_trend_1h': btc_trend_1h, 'btc_5m': btc_5m,
            'btc_mom_5m': pd.Series(mom_arr, index=idx5),
            'btc_trend_5m_arr': trend_arr, 'btc_mom_5m_arr': mom_arr,
            'btc_5m_idx': idx5}


# ---------------------------------------------------------------- parallel

_WORKER_CTX = None

def _month_worker(args):
    code, ym = args
    global _WORKER_CTX
    try:
        if _WORKER_CTX is None:
            _WORKER_CTX = build_btc_context()
        n = build_month(code, ym, btc_ctx=_WORKER_CTX, verbose=False)
        return code, ym, n, None
    except Exception as e:
        return code, ym, 0, f'{type(e).__name__}: {e}'


def month_range(ym_from, ym_to):
    cur = pd.Timestamp(f'{ym_from}-01')
    end = pd.Timestamp(f'{ym_to}-01')
    out = []
    while cur <= end:
        out.append(cur.strftime('%Y-%m'))
        cur += pd.offsets.MonthBegin(1)
    return out


def build_months(strategy_code, months, parallel=False, max_workers=None):
    code = strategy_code.upper()
    t0 = time.time()
    print(f'  MONTHLY BUILD [{code}]: {len(months)} months '
          f'({months[0]}..{months[-1]}) parallel={parallel}', flush=True)
    if not parallel:
        ctx = build_btc_context()
        for ym in months:
            build_month(code, ym, btc_ctx=ctx)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_workers = max_workers or max(1, (os.cpu_count() or 8) - 2)
        print(f'  workers={n_workers}', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_month_worker, (code, ym)) for ym in months]
            for fut in as_completed(futs):
                _c, ym, n, err = fut.result()
                msg = err if err else f'{n:,} rows'
                print(f'  [{code} {ym}] {msg} ({time.time()-t0:.0f}s)', flush=True)

    # meta: _load_lite resolves years from it (monthly shards included)
    ctx = build_btc_context()
    ov = os.environ.get('MONTHLY_CACHE_OVERRIDE')
    cache_file = Path(ov) if ov else load_builder(code)._instance.CACHE_FILE
    years_built = sorted(set(int(ym[:4]) for ym in months))
    _save_meta(cache_file, years_built, ctx['btc_trend_5m_arr'],
               ctx['btc_mom_5m_arr'], ctx['btc_5m_idx'])
    print(f'  meta updated: years += {years_built}', flush=True)
    print(f'  DONE ({(time.time()-t0)/60:.1f} min)', flush=True)


def build_all(strategy_codes, months, max_workers=None):
    """One pool over the full (strategy, month) task list; per-strategy
    meta written as each strategy's last month completes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    tasks = [(c, ym) for c in strategy_codes for ym in months]
    n_workers = max_workers or max(1, (os.cpu_count() or 8) - 2)
    print(f'  MONTHLY BUILD ALL: {len(strategy_codes)} strategies x '
          f'{len(months)} ay = {len(tasks)} gorev, {n_workers} worker', flush=True)
    remaining = {c: len(months) for c in strategy_codes}
    n_err = 0
    ctx = None
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_month_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            code, ym, n, err = fut.result()
            done += 1
            if err:
                n_err += 1
                print(f'  [{code} {ym}] ERROR: {err}', flush=True)
            elif done % 25 == 0 or remaining[code] == 1:
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f'  [{code} {ym}] {n:,} rows | {done}/{len(tasks)} '
                      f'({el/60:.0f}dk, ETA {eta/60:.0f}dk)', flush=True)
            remaining[code] -= 1
            if remaining[code] == 0:
                if ctx is None:
                    ctx = build_btc_context()
                cf = load_builder(code)._instance.CACHE_FILE
                yrs = sorted(set(int(ym[:4]) for ym in months))
                _save_meta(cf, yrs, ctx['btc_trend_5m_arr'],
                           ctx['btc_mom_5m_arr'], ctx['btc_5m_idx'])
                print(f'  == {code} TAMAM, meta yazildi ==', flush=True)
    print(f'  ALL DONE: {len(tasks)} gorev, {n_err} hata '
          f'({(time.time()-t0)/3600:.1f} saat)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--month', default=None, help='YYYY-MM (single month)')
    ap.add_argument('--from', dest='ym_from', default=None)
    ap.add_argument('--to', dest='ym_to', default=None)
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--max-workers', type=int, default=None)
    args = ap.parse_args()

    if args.month:
        months = [args.month]
    elif args.ym_from and args.ym_to:
        months = month_range(args.ym_from, args.ym_to)
    else:
        ap.error('--month YYYY-MM or --from/--to required')
    if args.strategy.upper() == 'ALL':
        from factory.data.registry import STRATEGIES
        codes = sorted(STRATEGIES)
        build_all(codes, months, max_workers=args.max_workers)
    else:
        build_months(args.strategy, months, parallel=args.parallel,
                     max_workers=args.max_workers)


if __name__ == '__main__':
    main()
