"""SMA-cross signal computation (TOY strategy — pipeline demonstration only).

A real deployment replaces this module with a proprietary signal; the
pipeline never needs to know the difference — it only consumes the two
boolean masks.
"""
import numpy as np
import pandas as pd

FAST, SLOW = 20, 50


def compute_sma_cross_signals(close: np.ndarray):
    """(long_mask, short_mask) — fast SMA crossing the slow SMA on 5m bars."""
    c = pd.Series(close)
    fast = c.rolling(FAST).mean()
    slow = c.rolling(SLOW).mean()
    above = (fast > slow).values
    prev = np.roll(above, 1)
    prev[0] = above[0]
    long_mask = above & ~prev          # cross up
    short_mask = ~above & prev         # cross down
    return long_mask, short_mask
