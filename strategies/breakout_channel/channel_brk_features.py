"""Channel-breakout signals (TOY strategy — pipeline demonstration only)."""
import numpy as np
import pandas as pd

WINDOW = 100


def compute_channel_brk_signals(high: np.ndarray, low: np.ndarray,
                                close: np.ndarray):
    """(long_mask, short_mask) — close breaking the prior N-bar channel."""
    h = pd.Series(high).rolling(WINDOW).max().shift(1).values
    l = pd.Series(low).rolling(WINDOW).min().shift(1).values
    with np.errstate(invalid='ignore'):
        long_mask = np.isfinite(h) & (close > h)
        short_mask = np.isfinite(l) & (close < l)
    return long_mask, short_mask
