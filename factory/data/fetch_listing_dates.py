"""Fetch true Binance USDT-perp listing dates (onboardDate) for the universe.

Single futures_exchange_info() call -> onboardDate per symbol = the futures
contract listing date. This fixes the "data-horizon artifact": dating an old
coin by its first cached bar wrongly clamps it to the cache start, which then
corrupts any age-based filtering downstream.

Writes config/coin_listing_dates.json {symbol: "YYYY-MM-DD"}.
Used by BOTH the backtest age filter (min-age combo-search gate) and the
live-side age gate — one source of truth, so live and backtest agree.
"""
import os, sys, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHITELIST = ROOT / 'config' / 'coin_whitelist.txt'
OUT = ROOT / 'config' / 'coin_listing_dates.json'
CACHE = ROOT / 'data' / 'cache'


def main():
    import glob
    import pandas as pd
    from binance import Client
    wl = [l.strip() for l in open(WHITELIST, encoding='utf-8', errors='ignore') if l.strip()]
    # every coin that appears in the cache (trades can only come from here)
    cache_coins = {Path(f).name.split('_5m_')[0] for f in glob.glob(str(CACHE / '*_5m_*.parquet'))}
    universe = sorted(set(wl) | cache_coins)
    print(f'Whitelist {len(wl)} + cache coins {len(cache_coins)} -> union {len(universe)}', flush=True)

    c = Client('', '')
    info = c.futures_exchange_info()
    onboard = {}
    for s in info['symbols']:
        ms = s.get('onboardDate', 0)
        if ms:
            onboard[s['symbol']] = ms
    print(f'Futures exchangeInfo: {len(onboard)} symbols with onboardDate', flush=True)

    out = {}
    n_onboard = n_fallback = 0
    fb_list = []
    for sym in universe:
        ms = onboard.get(sym)
        if ms:
            out[sym] = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            n_onboard += 1
            continue
        # delisted / not on futures now -> fall back to cache first 5m bar
        hits = sorted(glob.glob(str(CACHE / f'{sym}_5m_*.parquet')))
        if hits:
            idx = pd.read_parquet(hits[0], columns=['close']).index
            out[sym] = str(idx[0].date())
            n_fallback += 1
            fb_list.append(sym)

    out = dict(sorted(out.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=0)
    print(f'\nWrote {len(out)} listing dates -> {OUT}')
    print(f'  onboardDate: {n_onboard} | cache-first-bar fallback: {n_fallback}')
    if fb_list:
        print('  fallback (delisted/not-on-futures): ' + ', '.join(fb_list[:25]))

    # sanity cross-check vs cache first-bar for a few
    print('\nCross-check (onboardDate vs cache first 5m bar):')
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        if sym not in out:
            continue
        hits = sorted(glob.glob(str(CACHE / f'{sym}_5m_*.parquet')))
        fb = '?'
        if hits:
            idx = pd.read_parquet(hits[0], columns=['close']).index
            fb = str(idx[0].date())
        print(f'  {sym:12} onboard={out[sym]}  cache_first_bar={fb}')

    # age distribution at today (example: 365d minimum-age gate)
    today = datetime.now(timezone.utc)
    buckets = {'<6mo': 0, '6-12mo': 0, '12mo+': 0}
    for sym, d in out.items():
        age = (today - datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc)).days
        if age < 180: buckets['<6mo'] += 1
        elif age < 365: buckets['6-12mo'] += 1
        else: buckets['12mo+'] += 1
    print(f'\nUniverse age @ today: {buckets}  (12mo+ = tradeable under an example 365d gate)')


if __name__ == '__main__':
    main()
