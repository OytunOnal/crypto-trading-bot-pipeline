"""Canonical QS (quality score) math — single source of truth.

Both the backtest (factory/portfolio/save_cell_trades.py) and the live bot
MUST compute QS through these functions — live/backtest parity by construction.
The semantics are the original backtest semantics (all gate/combo optimization
was done with them):

  - pandas rolling rank (average tie method), window = n_roll + 1 incl. self
  - min_periods = 21
  - positional warmup: first n_roll rows of the stream -> NaN
  - per-feature sign: +1 if IC > 0 else -1 (IC == 0 counts as negative)
  - IC weight floor 0.01; missing/NaN feature rank contributes 0 to the
    weighted sum while its weight STAYS in the denominator (drags avg down)
  - meta-rank: rolling exclude-self rank of the weighted avg_rank stream
"""
import numpy as np
import pandas as pd


def assert_chronological(entry_times, where=''):
    """Guard: the QS rolling-rank is only valid on a time-ordered stream.

    load_lite rows are coin-major; feeding them to qs_avg_rank_pct without an
    entry-time sort silently corrupts the quintile (the coin-major-ordering bug class).
    Call this with the entry timestamps of the EXACT stream about to be ranked.
    Raises AssertionError on any out-of-order row.
    """
    et = np.asarray(entry_times)
    if et.size > 1:
        n_bad = int(np.sum(np.diff(et) < 0))
        if n_bad:
            raise AssertionError(
                f"QS stream NOT chronological{' [' + where + ']' if where else ''}: "
                f"{n_bad}/{et.size - 1} out-of-order rows. rolling-rank requires an "
                f"entry-time-sorted stream (coin-major-ordering bug class).")


def rolling_rank_exclude_self(vals, n_roll):
    """Exclude-self rolling percentile rank, backtest-canonical."""
    s = pd.Series(vals)
    rr = s.rolling(n_roll + 1, min_periods=21).rank()
    n_in = s.rolling(n_roll + 1, min_periods=21).count()
    result = np.where(n_in > 1, (rr - 1) / (n_in - 1) * 100, np.nan)
    result[:n_roll] = np.nan
    return result


def qs_avg_rank_pct(features, qs_feats, ic_map, col_names, n_roll, entry_times=None):
    """Meta-ranked IC-weighted QS percentile per row.

    features: (n, len(col_names)) array — chronological gate-passed stream.
    entry_times: optional timestamps aligned with `features` rows; if given, the
        stream is asserted chronological (catches the coin-major-ordering bug at
        the call site). Backtest callers SHOULD pass it.
    Returns float array of avg_rank_pct (NaN where undefined), or None when
    no configured QS feature exists in col_names.
    """
    if entry_times is not None:
        assert_chronological(entry_times, where='qs_avg_rank_pct')
    n = len(features)
    feat_ranks = {}
    ic_weights = {}
    for fname in qs_feats:
        if fname not in col_names:
            continue
        fi = col_names.index(fname)
        ic = ic_map.get(fname, 0) if ic_map else 0
        dsign = 1 if ic > 0 else -1
        col = features[:, fi].astype(np.float64)
        if dsign == -1:
            col = -col
        feat_ranks[fname] = rolling_rank_exclude_self(col, n_roll)
        ic_weights[fname] = abs(ic) if abs(ic) > 0 else 0.01

    if not feat_ranks:
        return None

    ic_total = sum(ic_weights[f] for f in feat_ranks)
    weighted_sum = np.zeros(n, dtype=np.float64)
    for fname, rr in feat_ranks.items():
        weighted_sum += np.where(np.isfinite(rr), rr * ic_weights[fname], 0)
    avg_rank = weighted_sum / ic_total
    return rolling_rank_exclude_self(avg_rank, n_roll)
