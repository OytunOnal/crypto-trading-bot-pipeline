"""Coin age gate — ONE source of truth for live AND backtest.

Reads config/coin_listing_dates.json (written by factory/data/
fetch_listing_dates.py; futures onboardDate, pre-cache-start for old coins —
fixes the data-horizon artifact where the first cached bar wrongly dates a
coin to the cache start).

Live side:      coin_age_ok(symbol, at_time)      scalar gate
Backtest side:  age_ok_mask(coins, entry)         vectorized row mask
Normalization:  norm_factor('YYYY-MM')            per-month universe scaler
                (net & trade counts are universe-normalized identically
                 across build_cell_files + the combo layers)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LISTING_JSON = ROOT / 'config' / 'coin_listing_dates.json'
MIN_AGE_DAYS = 365   # example gate; production calibrates this
NORM_REF = 350       # reference universe size for per-month normalization
_CACHE_DIR = ROOT / 'data' / 'cache'

_LISTING = None
_UNIV = None


def load_listing():
    """{coin: pd.Timestamp} of futures onboardDate, cached at module level."""
    global _LISTING
    if _LISTING is None:
        raw = (json.load(open(LISTING_JSON, encoding='utf-8'))
               if LISTING_JSON.exists() else {})
        _LISTING = {k: pd.Timestamp(v) for k, v in raw.items()}
    return _LISTING


def coin_age_ok(symbol: str, at_time) -> bool:
    """Live-side scalar gate. Fail-closed: unknown coin -> not ok."""
    d = load_listing().get(symbol)
    if d is None:
        return False
    return (pd.Timestamp(at_time) - d).days >= MIN_AGE_DAYS


def age_ok_mask(coins, entry, min_days=MIN_AGE_DAYS):
    """Bool mask, len == len(coins): True where (entry - listing) >= min_days.

    coins: array-like of symbol strings.
    entry: array-like of entry timestamps (pd.Timestamp / datetime64).
    Unknown coin or NaT -> False (excluded).
    """
    lst = load_listing()
    listing = pd.to_datetime(pd.Series(list(coins)).map(lst))   # NaT for unknown
    entry_s = pd.to_datetime(pd.Series(list(entry)))
    age = entry_s.values - listing.values                       # timedelta64, NaT->unknown
    # NaT comparisons are False -> unknown coin / bad entry excluded
    return age >= np.timedelta64(int(min_days), 'D')


# ── Per-month available universe (normalization denominator) ─────────
def _build_univ():
    import re
    from collections import defaultdict
    lst = load_listing()
    year_coins = defaultdict(set)
    for p in _CACHE_DIR.glob('*_5m_20*.parquet'):
        m = re.match(r'(.+)_5m_(\d{4})\d{4}_', p.name)
        if m:
            year_coins[int(m.group(2))].add(m.group(1))
    univ = {}
    for yr, coins in year_coins.items():
        for mo in range(1, 13):
            ym = '%04d-%02d' % (yr, mo)
            ms = pd.Timestamp(ym + '-01')
            univ[ym] = sum(1 for c in coins
                           if (ls := lst.get(c)) is not None
                           and (ms - ls).days >= MIN_AGE_DAYS)
    return univ


def month_universe(ym):
    """Available min-age+ coin count for 'YYYY-MM' (0 if unknown month)."""
    global _UNIV
    if _UNIV is None:
        _UNIV = _build_univ()
    return _UNIV.get(ym, 0)


def norm_factor(ym):
    """NORM_REF / month_universe(ym); 0 if universe unknown (drops the trade)."""
    u = month_universe(ym)
    return NORM_REF / u if u else 0.0
