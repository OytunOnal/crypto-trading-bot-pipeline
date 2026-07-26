"""Strategy style map + concentration cap for combo assembly.

Motivation: a portfolio whose members share one behavior family moves as ONE
correlated bet in the wrong month. The style CAP in trade_combo rejects
combos whose dominant style exceeds MAX_STYLE_FRAC of active configs.

Unknown strategies -> 'OTHER' (never capped). Keep this map in sync with the
strategy registry when adding strategies.
"""
import math
import os

# TEMPLATE mapping for the toy set — extend with your own strategy codes.
MOMENTUM = {'SMA_X', 'CH_BRK'}
MEANREV = {'RSI_MR'}

MAX_STYLE_FRAC = float(os.environ.get('TM_STYLE_CAP', '0.67'))


def style_of(label: str) -> str:
    """'L:SMA_X_q3_c0' -> 'MOMENTUM' | 'MEANREV' | 'OTHER'."""
    strat = label.split(':', 1)[-1].rsplit('_q', 1)[0]
    if strat in MOMENTUM:
        return 'MOMENTUM'
    if strat in MEANREV:
        return 'MEANREV'
    return 'OTHER'


def style_cap_ok(labels, frac: float = None) -> bool:
    """True when no single style exceeds `frac` of the active configs.

    Combos of <=2 actives are exempt (a 2-config combo is trivially 100%).
    OTHER is never counted against the cap.
    """
    if len(labels) <= 2:
        return True
    frac = MAX_STYLE_FRAC if frac is None else frac
    counts = {}
    for l in labels:
        s = style_of(l)
        if s != 'OTHER':
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return True
    limit = math.ceil(frac * len(labels))
    return max(counts.values()) <= limit
