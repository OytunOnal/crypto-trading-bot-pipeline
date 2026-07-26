"""Minimal OHLCV loading utilities for the pipeline.

Cache layout (written by download_klines.py):
    data/cache/{SYMBOL}_{interval}_{YYYY}0101_{YYYY}1231.parquet

In the source system this lives in a shared data-loader package; the template
ships a compact self-contained version with the same call signatures so every
downstream stage imports cleanly.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / 'data' / 'cache'


def load_ohlcv(symbol: str, interval: str = '5m', cache_dir=None) -> pd.DataFrame:
    """Load ALL cached years for one symbol/interval as a single frame."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    parts = []
    for f in sorted(cache.glob(f'{symbol}_{interval}_*.parquet')):
        df = pd.read_parquet(f)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def load_ohlcv_years(symbol: str, interval: str, years, cache_dir=None):
    """Load specific years for one symbol/interval (None if nothing cached)."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    parts = []
    for yr in years:
        f = cache / f'{symbol}_{interval}_{yr}0101_{yr}1231.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            if not df.empty:
                parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def resolve_coins(preset: str = 'all', cache_dir=None):
    """Resolve a coin universe.

    'all'      -> every symbol present in the cache dir
    'whitelist'-> config/coin_whitelist.txt
    otherwise  -> comma-separated symbol list
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    if preset == 'all':
        return sorted({f.name.split('_5m_')[0] for f in cache.glob('*_5m_*.parquet')})
    if preset == 'whitelist':
        wl = ROOT / 'config' / 'coin_whitelist.txt'
        return [l.strip() for l in open(wl, encoding='utf-8') if l.strip()]
    return [s.strip() for s in preset.split(',') if s.strip()]
