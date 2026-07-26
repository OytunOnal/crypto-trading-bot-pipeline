"""Unified Cell Comparison across the toy strategy set.

Per-category wrapper: lists each strategy's modules and its SELF-SIGNAL
features (excluded from QS ranking so a strategy cannot rank itself by the
same feature that generated its entries). The real work happens in
factory.qs.qs_common.run_comparison.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factory.qs.qs_common import run_comparison

BASE = Path(__file__).resolve().parent

STRATEGIES = {
    'SMA_X': {
        'path': BASE / 'momentum_sma_cross',
        'features_module': 'sma_cross_features',
        'v2_module': 'sma_cross_cache_builder',
        'v2_class': 'SmaCrossCache',
        'phase2': 'momentum_sma_cross_phase2_results.json',
        'excluded': {'ema20_slope_15m'},     # self-signal family
    },
    'RSI_MR': {
        'path': BASE / 'meanrev_rsi',
        'features_module': 'rsi_mr_features',
        'v2_module': 'rsi_mr_cache_builder',
        'v2_class': 'RsiMrCache',
        'phase2': 'meanrev_rsi_phase2_results.json',
        'excluded': {'rsi_14_15m'},          # self-signal family
    },
    'CH_BRK': {
        'path': BASE / 'breakout_channel',
        'features_module': 'channel_brk_features',
        'v2_module': 'channel_brk_cache_builder',
        'v2_class': 'ChannelBrkCache',
        'phase2': 'breakout_channel_phase2_results.json',
        'excluded': set(),
    },
}

if __name__ == '__main__':
    run_comparison(STRATEGIES, 'ToySet', BASE)
