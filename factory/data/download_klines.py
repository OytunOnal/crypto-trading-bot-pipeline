"""Download OHLCV klines for Binance USDT perpetual futures.

5m for all futures + BTC 1h (used by cache builders for regime detection).
Append mode: only fetches bars after last existing bar.
Parallel: 3 workers + token-bucket rate limiter.

Usage:
    python stages/01_data/download_klines.py                  # all symbols, all years
    python stages/01_data/download_klines.py --year 2026      # only 2026 (append mode)
    python stages/01_data/download_klines.py --symbols BTCUSDT,ETHUSDT
    python stages/01_data/download_klines.py --resume SOLUSDT
    python stages/01_data/download_klines.py --serial          # disable parallel
"""
import os, sys, time, argparse, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'data' / 'backtest' / 'cache'

API_MAX = 1500
BAR_MS_5M = 5 * 60_000
BAR_MS_1H = 60 * 60_000
ALL_YEARS = [2026, 2025, 2024, 2023, 2022, 2021]

WORKERS = 3
MAX_CALLS_PER_SEC = 3.5  # 2100 weight/min, under 2400 budget

_rate_lock = threading.Lock()
_last_call_ts = [0.0]


def _rate_limit():
    with _rate_lock:
        now = time.time()
        min_gap = 1.0 / MAX_CALLS_PER_SEC
        wait = _last_call_ts[0] + min_gap - now
        if wait > 0:
            time.sleep(wait)
            _last_call_ts[0] += min_gap
        else:
            _last_call_ts[0] = time.time()


def get_futures_symbols_with_listing():
    """Get all USDT perpetual symbols + listing year.

    Client('', '') pings on construct and futures_exchange_info() is a network
    call -- both can hit transient read/handshake timeouts. Retry with backoff so
    a single blip doesn't crash the caller (download main OR the verify _scan)."""
    from binance import Client
    info = None
    for att in range(4):
        try:
            info = Client('', '').futures_exchange_info()
            break
        except Exception as e:
            if att == 3:
                raise RuntimeError(f'get_futures_symbols_with_listing failed after retries: {e}')
            time.sleep(2 * (att + 1))
    result = []
    for s in info['symbols']:
        if (s['contractType'] == 'PERPETUAL'
                and s['quoteAsset'] == 'USDT'
                and s['status'] == 'TRADING'
                and not s['symbol'].startswith('BTCDOM')):
            onboard_ms = s.get('onboardDate', 0)
            listing_year = (datetime.fromtimestamp(onboard_ms / 1000, tz=timezone.utc).year
                            if onboard_ms else 2020)
            result.append((s['symbol'], listing_year))
    result.sort(key=lambda x: x[0])
    return result


def _fetch_paginated(client, symbol, interval, bar_ms, start_ms, end_ms):
    """Fetch klines between start_ms and end_ms via pagination."""
    all_klines = []
    cursor = start_ms
    while cursor < end_ms:
        _rate_limit()
        try:
            klines = client.futures_klines(
                symbol=symbol, interval=interval,
                startTime=cursor, endTime=end_ms, limit=API_MAX,
            )
        except Exception as e:
            err = str(e)
            if 'Invalid symbol' in err or '-1121' in err:
                return []
            if '429' in err or 'Too Many' in err:
                print(f'    [{symbol}] rate limited, sleep 60s', flush=True)
                time.sleep(60); continue
            if '-1003' in err:
                print(f'    [{symbol}] WAI limit, sleep 30s', flush=True)
                time.sleep(30); continue
            raise
        if not klines:
            break
        all_klines.extend(klines)
        cursor = klines[-1][0] + bar_ms
        if len(klines) < API_MAX:
            break
    return all_klines


def _klines_to_df(klines):
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore',
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    return df


def _download_append(client, symbol, interval, bar_ms, year):
    """Download/append klines for one symbol, one interval, one year.
    Returns number of new bars added.

    OVERLAP RE-FETCH (content verify): resume does NOT start at last_bar+1 — the
    trailing DL_OVERLAP_HOURS (default 24h) are ALWAYS re-fetched and diffed
    against the stored parquet. Exchange klines can be transiently wrong right
    after bar close (2026-07-06 09:30 incident: Binance REST served deficient
    bars minutes after a market-wide event, later revised them; incremental
    resume made the bad bars permanent, 258/367 coins). Revised bars are logged
    (REVISED n) and replaced via keep='last'. If even the OLDEST overlap bar
    differs, the window doubles and refetches until an error-free region is
    reached (no cap — natural bound is the year start).
    """
    start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    year_start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    now_ms = int(time.time() * 1000)
    if year_start_ms > now_ms:
        return 0
    end_ms = min(end_ms, now_ms)

    out_path = CACHE_DIR / f'{symbol}_{interval}_{year}0101_{year}1231.parquet'

    existing_df = None
    resume_ms = year_start_ms
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        if not existing_df.empty:
            if existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)
            last_ts = existing_df.index[-1]
            ts_val = last_ts.timestamp() if hasattr(last_ts, 'timestamp') else pd.Timestamp(last_ts).timestamp()
            resume_ms = int(ts_val * 1000) + bar_ms

    overlap_ms = int(float(os.environ.get('DL_OVERLAP_HOURS', '24')) * 3_600_000)
    depth = overlap_ms if (existing_df is not None and not existing_df.empty) else 0
    revised_total = 0

    while True:
        fetch_start = max(resume_ms - depth, year_start_ms)
        if fetch_start >= end_ms:
            return 0
        klines = _fetch_paginated(client, symbol, interval, bar_ms, fetch_start, end_ms)
        if not klines:
            return 0
        df = _klines_to_df(klines)
        if df.empty:
            return 0
        df = df[(df.index >= str(year)) & (df.index < str(year + 1))]
        if df.empty:
            return 0
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Drop the FORMING bar (close_time > now): the kline API returns the
        # in-progress candle too; storing it froze a partial bar at every
        # historical run's tail (~25 bars/coin since May, incl. 07-06 09:30).
        cutoff = pd.Timestamp(now_ms, unit='ms') - pd.Timedelta(milliseconds=bar_ms)
        df = df[df.index <= cutoff]
        if df.empty:
            return 0

        if existing_df is None or existing_df.empty or depth == 0:
            break
        # content diff on the overlap region
        common = df.index.intersection(existing_df.index)
        if len(common) == 0:
            break
        cols = ['open', 'high', 'low', 'close', 'volume']
        a = df.loc[common, cols].astype('float64')
        b = existing_df.loc[common, cols].astype('float64')
        neq = ~np.isclose(a.values, b.values, rtol=1e-9, equal_nan=True)
        revised_mask = neq.any(axis=1)
        revised_total = int(revised_mask.sum())
        oldest_revised = bool(revised_mask[0]) if len(revised_mask) else False
        if oldest_revised and fetch_start > year_start_ms:
            depth *= 2   # revision reached the window edge -> widen until an error-free region
            continue
        break

    if revised_total:
        rts = common[revised_mask]
        print(f'    REVISED {symbol} {year}: {revised_total} bars '
              f'(overlap {depth / 3_600_000:.0f}h) span={rts.min()}..{rts.max()}', flush=True)

    if existing_df is not None and not existing_df.empty:
        n_before = len(existing_df)
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
    else:
        n_before = 0

    df.to_parquet(out_path)
    return len(df) - n_before


# ── Parallel worker ─────────────────────────────────────────────────
def _worker_5m(sym_listing, years, counters, lock):
    from binance import Client
    sym, listing_year = sym_listing
    # Client('', '') pings on init -> a transient ping read-timeout would crash the
    # worker (and the whole pool). Retry a few times; if still failing, count err+done
    # and bail -> the coin stays un-fetched and the verify+retry pass picks it up.
    client = None
    for _att in range(3):
        try:
            client = Client('', ''); break
        except Exception as e:
            if _att == 2:
                with lock:
                    counters['err'] += 1; counters['done'] += 1
                print(f'    CLIENT-INIT FAIL {sym}: {e}', flush=True)
                return sym, 0
            time.sleep(2)
    sym_new = 0

    for year in years:
        if year < listing_year:
            with lock: counters['skip'] += 1
            continue
        # Past years: skip if file exists (complete)
        out = CACHE_DIR / f'{sym}_5m_{year}0101_{year}1231.parquet'
        if year < datetime.now(timezone.utc).year and out.exists():
            with lock: counters['cached'] += 1
            continue
        try:
            bars = _download_append(client, sym, '5m', BAR_MS_5M, year)
            with lock:
                if bars > 0:
                    counters['dl'] += 1
                    counters['bars'] += bars
                    sym_new += bars
                else:
                    counters['cached'] += 1
        except Exception as e:
            with lock: counters['err'] += 1
            print(f'    ERROR {sym} {year}: {e}', flush=True)

    with lock: counters['done'] += 1
    return sym, sym_new


# ── Freshness verify (folded in: download == verify == targeted retry) ──
def _last_bar(sym, year):
    f = CACHE_DIR / f'{sym}_5m_{year}0101_{year}1231.parquet'
    if not f.exists():
        return None
    try:
        idx = pd.read_parquet(f, columns=[]).index
        return idx[-1] if len(idx) else None
    except Exception:
        return None


def _scan(year, stale_hours, gap_tol_min=5.0):
    """(trading_syms, stale[(sym,last)], missing[sym], gappy[(sym,maxgap)]).

    stale  = last bar older than the freshness cutoff (append will catch it up).
    missing= no/empty parquet.
    gappy  = a FRESH coin whose 5m series has an INTERNAL hole > gap_tol_min. A
             download failure / exchange halt can leave a mid-series gap that
             append-mode CANNOT fill (it only fetches after the last bar) -> the
             replica computes different features across the hole. These need a full
             re-fetch (the verify loop deletes their parquet before retrying).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    yend = datetime(year, 12, 31, 23, 55)
    cutoff = min(now, yend) - pd.Timedelta(hours=stale_hours)
    syms = [s for s, ly in get_futures_symbols_with_listing() if ly <= year]
    stale, missing, gappy = [], [], []
    for s in syms:
        f = CACHE_DIR / f'{s}_5m_{year}0101_{year}1231.parquet'
        if not f.exists():
            missing.append(s); continue
        try:
            idx = pd.read_parquet(f, columns=[]).index
        except Exception:
            missing.append(s); continue
        if len(idx) == 0:
            missing.append(s); continue
        if pd.Timestamp(idx[-1]) < cutoff:
            stale.append((s, str(idx[-1])[:16])); continue        # stale -> append retry
        if len(idx) > 1:                                            # internal-gap check
            mx = idx.to_series().diff().dropna().max().total_seconds() / 60
            if mx > gap_tol_min:
                gappy.append((s, '%.0fmin' % mx))
    return syms, stale, missing, gappy


def run_5m_download(sym_list, years, parallel):
    """Download/append 5m for sym_list over years. Returns the counters dict."""
    total = len(sym_list)
    t0 = time.time()
    counters = {'done': 0, 'dl': 0, 'cached': 0, 'skip': 0, 'err': 0, 'bars': 0}
    lock = threading.Lock()
    print(f'\n  --- 5m download ({total} symbols) ---', flush=True)
    if parallel:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(_worker_5m, sl, years, counters, lock): sl
                       for sl in sym_list}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:   # one worker dying must NOT crash the pool
                    with lock:
                        counters['err'] += 1; counters['done'] += 1
                    print(f'    WORKER ERROR {futures[fut][0]}: {e}', flush=True)
                with lock:
                    done = counters['done']
                if done % 25 == 0 or done == total:
                    with lock:
                        dl = counters['dl']; bars = counters['bars']
                        cached = counters['cached']; err = counters['err']
                    print(f'  [{done}/{total}] dl={dl} bars={bars:,} '
                          f'cached={cached} err={err} ({time.time()-t0:.0f}s)', flush=True)
    else:
        from binance import Client
        client = Client('', '')
        for si, (sym, listing_year) in enumerate(sym_list, 1):
            for year in years:
                if year < listing_year:
                    counters['skip'] += 1; continue
                out = CACHE_DIR / f'{sym}_5m_{year}0101_{year}1231.parquet'
                if year < datetime.now(timezone.utc).year and out.exists():
                    counters['cached'] += 1; continue
                try:
                    bars = _download_append(client, sym, '5m', BAR_MS_5M, year)
                    if bars > 0:
                        counters['dl'] += 1; counters['bars'] += bars
                    else:
                        counters['cached'] += 1
                except Exception as e:
                    counters['err'] += 1
                    print(f'    ERROR {sym} {year}: {e}', flush=True)
            counters['done'] += 1
            if si % 25 == 0 or si == total:
                print(f'  [{si}/{total}] dl={counters["dl"]} bars={counters["bars"]:,} '
                      f'cached={counters["cached"]} err={counters["err"]} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  5m done: {counters["dl"]} files, {counters["bars"]:,} new bars ({time.time()-t0:.0f}s)', flush=True)
    return counters


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Download OHLCV klines (verified)')
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--symbols', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--serial', action='store_true')
    # verify+retry (folded in from the old download_verify wrapper). Active only on a
    # full single-year run (--year, no --symbols): scan freshness, RE-download the
    # stale/missing-but-listed subset until fresh or no-progress, then SURFACE (exit 1)
    # anything still stale instead of leaving silent gaps. --symbols / all-years runs
    # keep the old plain behavior (the retry itself re-enters here with --symbols).
    parser.add_argument('--overlap-hours', type=float, default=24.0,
                        help='trailing window ALWAYS re-fetched + content-diffed vs stored '
                             '(exchange kline revisions); deepens x2 while the oldest overlap '
                             'bar still differs (no cap, bounded by year start)')
    parser.add_argument('--stale-hours', type=float, default=2.0)   # tight: an actively
    #   trading coin has a bar within minutes; >2h behind = the download missed its bars
    #   (a 24h window let ~24h-stale coins pass -> the reconcile replica diverged on them)
    parser.add_argument('--gap-tol', type=float, default=5.0)   # minutes: flag a fresh
    #   coin as gappy if its 5m series has any internal hole > this (5 == any missing bar)
    parser.add_argument('--max-retries', type=int, default=4)
    parser.add_argument('--no-verify', action='store_true', help='skip the verify+retry pass')
    parser.add_argument('--verify-only', action='store_true',
                        help='skip the initial download, run verify+retry only')
    args = parser.parse_args()
    # env (not a global): spawned pool workers inherit os.environ on Windows,
    # module globals set after import do NOT.
    os.environ['DL_OVERLAP_HOURS'] = str(args.overlap_hours)

    years = [args.year] if args.year else ALL_YEARS
    parallel = not args.serial

    print('=' * 80)
    print(f'  DOWNLOAD KLINES (5m all futures, verified)')
    print(f'  Years: {years} | Mode: {"parallel" if parallel else "serial"} | Cache: {CACHE_DIR}')
    print('=' * 80)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print('\n  Fetching symbol list...', flush=True)
    if args.symbols:
        sym_list = [(s.strip(), 2020) for s in args.symbols.split(',')]
    else:
        sym_list = get_futures_symbols_with_listing()
    if args.resume:
        idx = next((i for i, (s, _) in enumerate(sym_list) if s == args.resume), 0)
        sym_list = sym_list[idx:]
        print(f'  Resuming from {args.resume} (index {idx})')
    print(f'  Total: {len(sym_list)} symbols', flush=True)

    if not args.verify_only:
        c = run_5m_download(sym_list, years, parallel)
        # BTC 1h no longer downloaded — all 1h series are resampled from 5m
        # (single source of truth, see cache_builder_common._load_btc_1h).
        print(f'\n{"="*80}')
        print(f'  DOWNLOAD PASS DONE | 5m: {c["dl"]} files, {c["bars"]:,} new bars | '
              f'cached={c["cached"]} skip={c["skip"]} err={c["err"]}')

    # ── verify + targeted retry (folded-in) ──
    if args.no_verify or args.symbols or not args.year:
        return
    year = args.year
    prev_bad = None
    for rnd in range(args.max_retries + 1):
        syms, stale, missing, gappy = _scan(year, args.stale_hours, args.gap_tol)
        gap_syms = [s for s, _ in gappy]
        bad = sorted(set([s for s, _ in stale] + missing + gap_syms))
        print('\n[verify round %d] TRADING=%d fresh=%d stale=%d missing=%d gappy=%d'
              % (rnd, len(syms), len(syms) - len(bad), len(stale), len(missing), len(gappy)), flush=True)
        if not bad:
            break
        if prev_bad is not None and len(bad) >= prev_bad:
            print('  NO PROGRESS -> remaining stuck (halted/illiquid but still listed):', flush=True)
            break
        prev_bad = len(bad)
        # gappy coins: delete the parquet so the re-download refetches a CONTIGUOUS
        # year (append-mode alone can't fill a mid-series hole).
        for s in gap_syms:
            p = CACHE_DIR / f'{s}_5m_{year}0101_{year}1231.parquet'
            if p.exists():
                p.unlink()
        print('  retrying %d coins (%d gappy full-refetch): %s'
              % (len(bad), len(gap_syms), ','.join(bad)), flush=True)
        run_5m_download([(s, 2020) for s in bad], [year], parallel)

    syms, stale, missing, gappy = _scan(year, args.stale_hours, args.gap_tol)
    if not stale and not missing and not gappy:
        print('\nDONE: all %d TRADING symbols fresh + gap-free for %d.' % (len(syms), year))
        return
    print('\nWARNING: %d TRADING symbols NOT clean after %d retries (surfaced, not silent):'
          % (len(stale) + len(missing) + len(gappy), args.max_retries))
    for s, lb in stale:
        print('  STALE   %-16s last=%s' % (s, lb))
    for s in missing:
        print('  MISSING %-16s (no parquet)' % s)
    for s, g in gappy:
        print('  GAPPY   %-16s max-gap=%s (internal hole)' % (s, g))
    sys.exit(1)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    main()
