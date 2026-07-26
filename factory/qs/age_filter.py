"""Coin age gate — one source of truth for live AND backtest.

Reads config/coin_listing_dates.json (written by factory/data/
fetch_listing_dates.py). A coin is tradeable when it has been listed for at
least MIN_AGE_DAYS at decision time. Fail-closed: unknown coin -> not ok.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LISTING_JSON = ROOT / 'config' / 'coin_listing_dates.json'
MIN_AGE_DAYS = 365   # example gate; production calibrates this

_dates = None


def _load():
    global _dates
    if _dates is None:
        _dates = (json.load(open(LISTING_JSON, encoding='utf-8'))
                  if LISTING_JSON.exists() else {})
    return _dates


def coin_age_ok(symbol: str, at_time) -> bool:
    d = _load().get(symbol)
    if d is None:
        return False   # fail-closed
    listed = pd.Timestamp(d)
    return (pd.Timestamp(at_time) - listed).days >= MIN_AGE_DAYS
