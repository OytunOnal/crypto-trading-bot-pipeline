"""SMA Cross 5m cache builder (year-split, common interface). TOY example."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sma_cross_features import compute_sma_cross_signals
from factory.data.cache_builder import StrategyCache


class SmaCrossCache(StrategyCache):
    CACHE_FILE = Path(__file__).resolve().parent / 'cache' / 'sma_cross_5m_raw_cache.pkl'
    DESCRIPTION = 'SMA Cross 5m (toy)'

    def compute_signals(self, df):
        long_mask, short_mask = compute_sma_cross_signals(df['close'].values)
        return long_mask, short_mask


_instance = SmaCrossCache()
build = lambda target_year=None: _instance.build(target_year)
load = lambda years=None: _instance.load(years)
load_year = lambda yr: _instance.load_year(yr)

if __name__ == '__main__':
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _instance.build(yr)
