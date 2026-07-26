"""Live-side regime trackers, signal dispatch and gate checks (TEMPLATE).

Mirrors the backtest regime math (factory.gates.features_common) so that a
bar processed live lands in the SAME (trend, momentum) cell the backtest
would assign — regime parity by construction.
"""
from typing import Dict, List, Tuple
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.qs.live_config import (
    BTC_TREND_TERCILES, BTC_MOM_ROC_HOURS, BTC_MOM_ROLL_WINDOW,
)


class BtcTrendTracker:
    """BTC ROC(168h) fixed-tercile trend: 0=BEAR 1=FLAT 2=BULL."""

    def update(self, btc_1h_close: pd.Series):
        if len(btc_1h_close) < 169:
            return None
        roc = (btc_1h_close.iloc[-1] / btc_1h_close.iloc[-169] - 1) * 100
        if not np.isfinite(roc):
            return None
        t33, t67 = BTC_TREND_TERCILES
        return 0 if roc <= t33 else (1 if roc <= t67 else 2)


class BtcMomentumTracker:
    """BTC ROC(24h) rolling-percentile tercile: 0=LOW 1=MED 2=HIGH."""

    def update(self, btc_1h_close: pd.Series):
        roc = btc_1h_close.pct_change(BTC_MOM_ROC_HOURS)
        rank = roc.rolling(BTC_MOM_ROLL_WINDOW, min_periods=20).rank(pct=True)
        r = rank.iloc[-1]
        if not np.isfinite(r):
            return None
        return 0 if r <= 0.3333 else (1 if r <= 0.6667 else 2)


def check_cell_gate(features: Dict[str, float],
                    gate_rules: List[Tuple[str, str, float]]) -> bool:
    """Check if all AND gate rules pass. Missing/NaN -> blocked."""
    for feat_name, op, value in gate_rules:
        val = features.get(feat_name)
        if val is None or not np.isfinite(val):
            return False
        if op == '<' and val >= value:
            return False
        if op == '>' and val <= value:
            return False
    return True


def check_block_gate(features: Dict[str, float],
                     block_rules: List[Tuple[str, str, float]]) -> bool:
    """Phase-2 block gate: eliminate rows matching ANY block rule.

    Returns False (blocked) if any rule matches. Missing/NaN -> not blocked
    (blocks are risk trims, not entry requirements).
    """
    for feat_name, op, value in block_rules:
        val = features.get(feat_name)
        if val is None or not np.isfinite(val):
            continue
        if op == '<' and val < value:
            return False
        if op == '>' and val > value:
            return False
    return True


# ── Signal dispatch (toy strategies) ──────────────────────────────────
# SIGNAL_FUNCS: name -> (per-bar fn(ohlcv_df) -> (short_fired, long_fired), trim_bars)
# SIGNAL_SERIES_FUNCS: name -> vectorized fn(ohlcv_df) -> (short_mask, long_mask)
sys.path.insert(0, str(ROOT / 'strategies' / 'momentum_sma_cross'))
sys.path.insert(0, str(ROOT / 'strategies' / 'meanrev_rsi'))
sys.path.insert(0, str(ROOT / 'strategies' / 'breakout_channel'))
from sma_cross_features import compute_sma_cross_signals
from rsi_mr_features import compute_rsi_mr_signals
from channel_brk_features import compute_channel_brk_signals


def _sma_x_bar(df):
    l, s = compute_sma_cross_signals(df['close'].values)
    return bool(s[-1]), bool(l[-1])


def _rsi_mr_bar(df):
    l, s = compute_rsi_mr_signals(df['close'].values)
    return bool(s[-1]), bool(l[-1])


def _ch_brk_bar(df):
    l, s = compute_channel_brk_signals(df['high'].values, df['low'].values,
                                       df['close'].values)
    return bool(s[-1]), bool(l[-1])


SIGNAL_FUNCS = {
    'sma_x': (_sma_x_bar, 300),
    'rsi_mr': (_rsi_mr_bar, 300),
    'ch_brk': (_ch_brk_bar, 300),
}


def _sma_x_series(df):
    l, s = compute_sma_cross_signals(df['close'].values)
    return s, l


def _rsi_mr_series(df):
    l, s = compute_rsi_mr_signals(df['close'].values)
    return s, l


def _ch_brk_series(df):
    l, s = compute_channel_brk_signals(df['high'].values, df['low'].values,
                                       df['close'].values)
    return s, l


SIGNAL_SERIES_FUNCS = {
    'sma_x': _sma_x_series,
    'rsi_mr': _rsi_mr_series,
    'ch_brk': _ch_brk_series,
}
