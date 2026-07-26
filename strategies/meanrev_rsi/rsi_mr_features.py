"""RSI mean-reversion signals (TOY strategy — pipeline demonstration only)."""
import numpy as np
import pandas as pd

PERIOD, LO, HI = 14, 30.0, 70.0


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def compute_rsi_mr_signals(close: np.ndarray):
    """(long_mask, short_mask) — RSI re-entering from oversold / overbought."""
    r = _rsi(pd.Series(close), PERIOD).values
    prev = np.roll(r, 1)
    prev[0] = r[0]
    long_mask = (prev < LO) & (r >= LO)     # exit oversold upward
    short_mask = (prev > HI) & (r <= HI)    # exit overbought downward
    return long_mask, short_mask
