"""Live-side configuration (TEMPLATE values).

In the source system this is the deployed bot's config: the selected
(direction, cell, signal) -> {gate, block, qs_key} table produced by the
portfolio stage. The template ships a small illustrative table wired to the
three toy strategies so the QS replay / daily-refresh / parity code paths
run end to end.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TREND_NAMES = {0: 'BEAR', 1: 'FLAT', 2: 'BULL'}
MOM_NAMES = {0: 'LOW', 1: 'MED', 2: 'HIGH'}

MIN_COIN_BARS = 2520          # bars since listing before a coin is tradeable
N_ROLL = 100                  # QS rolling-rank window
CACHE_BARS = 4600             # live kline cache depth per coin
QS_WARMUP_FLOOR_BARS = 4400   # never trim below the longest signal warmup
QS_RECOMPUTE_DAYS = 3         # daily self-healing recompute window
QS_BOOTSTRAP_ROWS = 2 * N_ROLL + 60   # per-config history depth

QS_PERSIST_PATH = str(ROOT / 'data' / 'live' / 'qs_trade_history.parquet')
QS_FEATURES_JSON = str(ROOT / 'config' / 'qs_features.json')

# BTC regime boundaries (EXAMPLE values — production fixes these from the
# backtest's fixed-tercile calibration).
BTC_TREND_TERCILES = (-2.1, 2.7)      # ROC(168h) % boundaries
BTC_MOM_ROC_HOURS = 24
BTC_MOM_ROLL_WINDOW = 150

# ── Deployed config table (EXAMPLE) ──────────────────────────────────
# (direction, cell, signal) -> cfg. Gates/blocks reference template-battery
# feature names; values are illustrative. A real deployment generates this
# table from the portfolio stage output.
CELL_CONFIG = {
    ('LONG', 'BULL_MED', 'sma_x'): {
        'gate': [('rsi_14_1h', '>', 45.0), ('atr_pct_1h', '<', 0.02)],
        'block': [('vol_zscore_20_15m', '>', 4.0)],
        'qs_key': 'LONG_BULL_MED_SMA_X_q3_c0',
    },
    ('SHORT', 'BEAR_MED', 'sma_x'): {
        'gate': [('rsi_14_1h', '<', 55.0)],
        'block': [],
        'qs_key': 'SHORT_BEAR_MED_SMA_X_q3_c0',
    },
    ('LONG', 'FLAT_LOW', 'rsi_mr'): {
        'gate': [('bb_pct_b_15m', '<', 0.2)],
        'block': [('atr_pct_4h', '>', 0.03)],
        'qs_key': 'LONG_FLAT_LOW_RSI_MR_q4_c0',
    },
    ('SHORT', 'FLAT_HIGH', 'rsi_mr'): {
        'gate': [('bb_pct_b_15m', '>', 0.8)],
        'block': [],
        'qs_key': 'SHORT_FLAT_HIGH_RSI_MR_q4_c0',
    },
    ('LONG', 'BULL_HIGH', 'ch_brk'): {
        'gate': [('vol_zscore_20_1h', '>', 0.5)],
        'block': [],
        'qs_key': 'LONG_BULL_HIGH_CH_BRK_q2_c0',
    },
}


def configs_for_regime(trend_idx: int, mom_idx: int) -> list:
    """Return list of (direction, cell, signal, cfg) for current regime."""
    cell_str = f'{TREND_NAMES[trend_idx]}_{MOM_NAMES[mom_idx]}'
    result = []
    for (direction, cell, signal), cfg in CELL_CONFIG.items():
        if cell == cell_str:
            result.append((direction, cell, signal, cfg))
    return result


def needed_signals_for_regime(trend_idx: int, mom_idx: int) -> set:
    """Return set of signal types needed for the current regime."""
    cell_str = f'{TREND_NAMES[trend_idx]}_{MOM_NAMES[mom_idx]}'
    return {signal for (_d, cell, signal) in CELL_CONFIG if cell == cell_str}
