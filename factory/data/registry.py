"""Strategy registry — SINGLE SOURCE OF TRUTH for the pipeline engines.

Maps each strategy's short code to its cache-builder module, live signal
name and category. Consumed by gate_sweep / phase2 / qs_features / run_all,
so adding a strategy here plugs it into the whole factory.

Cache builder modules expose the StrategyCache interface:
    build(year) / load(years) / load_lite(direction, years) / load_year(yr)

TEMPLATE NOTE: this repo ships three toy strategies to demonstrate the
plumbing. In a real deployment this table is where your proprietary
strategy set lives (one row per substrategy variant).
"""
from pathlib import Path

BACKTEST = Path(__file__).resolve().parents[2]   # repo root

# code -> (builder module path relative to repo root, live signal name, category)
STRATEGIES = {
    'SMA_X':  ('strategies/momentum_sma_cross/sma_cross_cache_builder.py',   'sma_x',  'MOMENTUM'),
    'RSI_MR': ('strategies/meanrev_rsi/rsi_mr_cache_builder.py',             'rsi_mr', 'MEAN_REV'),
    'CH_BRK': ('strategies/breakout_channel/channel_brk_cache_builder.py',   'ch_brk', 'BREAKOUT'),
}


def load_builder(code: str):
    """Import a strategy's cache-builder module by registry code."""
    import importlib.util
    rel, _sig, _cat = STRATEGIES[code]
    path = BACKTEST / rel
    spec = importlib.util.spec_from_file_location(f'builder_{code.lower()}', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
