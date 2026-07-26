"""Minimal live kline cache (TEMPLATE stand-in for the bot's KlineCache).

Holds per-symbol 5m OHLCV frames in memory, loads them from a parquet
directory, and supports the trim contract qs_daily relies on. The real bot's
cache also streams bars from the exchange connector; the template only needs
the disk/replay path.
"""
from pathlib import Path
import pandas as pd


class MiniKlineCache:
    def __init__(self, max_bars: int = 4600):
        self.max_bars = max_bars
        self._frames = {}   # symbol -> DataFrame (5m OHLCV)

    def frames(self, interval: str = '5m'):
        return self._frames

    def load_from_disk(self, cache_dir) -> int:
        """Load {SYMBOL}*.parquet files into per-symbol frames."""
        d = Path(cache_dir)
        if not d.exists():
            return 0
        for f in sorted(d.glob('*.parquet')):
            sym = f.stem.split('_')[0]
            df = pd.read_parquet(f)
            if df.empty:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cur = self._frames.get(sym)
            df = pd.concat([cur, df]) if cur is not None else df
            df = df[~df.index.duplicated(keep='last')].sort_index()
            self._frames[sym] = df.iloc[-self.max_bars:]
        return len(self._frames)

    def trim_to(self, max_bars: int, floor_bars: int = 0) -> int:
        """Trim every frame to max(max_bars, floor_bars); return bars removed."""
        keep = max(max_bars, floor_bars)
        removed = 0
        for sym, df in self._frames.items():
            if len(df) > keep:
                removed += len(df) - keep
                self._frames[sym] = df.iloc[-keep:]
        return removed

    def cache_health(self, interval: str = '5m'):
        """Log-friendly gap report: symbols with internal 5m holes."""
        gaps = {}
        for sym, df in self._frames.items():
            if len(df) > 1:
                mx = df.index.to_series().diff().dropna().max().total_seconds() / 60
                if mx > 5.0:
                    gaps[sym] = mx
        return gaps
