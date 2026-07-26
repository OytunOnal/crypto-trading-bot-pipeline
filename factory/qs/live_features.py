"""Live-side feature computation (TEMPLATE) — mirrors the backtest battery.

The live bot computes, per candidate bar, the SAME feature names the cache
builder wrote (factory.gates.features_common battery x 3 timeframes + 6
BTC-relative columns). Feature parity is what makes live QS ranks equal
backtest QS ranks.

compute_all_features        -> {name: float} for the LAST bar of the buffer
compute_all_features_series -> {name: np.ndarray} over the whole buffer
merge_config_needs          -> union of feature needs across candidate configs
"""
from typing import Dict, Tuple
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.gates.features_common import (
    BASE_FEATURES, TIMEFRAMES, compute_features, rolling_resample_ohlcv,
)

BTC_FEATURES = ['btc_rel_roc10', 'btc_rel_vol', 'btc_corr_20',
                'btc_roc_24h', 'btc_roc_4h', 'btc_roc_slope']
_ROC10 = BASE_FEATURES.index('roc_10')
_ATRP = BASE_FEATURES.index('atr_pct')


def merge_config_needs(candidates) -> Tuple[Dict[str, frozenset], frozenset]:
    """Union of needed base features per TF + needed BTC features across
    candidate (direction, cell, signal, cfg) tuples. The template battery is
    small so engines may simply compute everything; the needs contract is
    kept for signature parity with the source system."""
    bases = {tf: set() for tf in TIMEFRAMES}
    btc = set()
    for _d, _cell, _sig, cfg in candidates:
        rules = list(cfg.get('gate', [])) + list(cfg.get('block', []))
        for name, _op, _val in rules:
            if name in BTC_FEATURES:
                btc.add(name)
                continue
            for tf in TIMEFRAMES:
                sf = f'_{tf}'
                if name.endswith(sf):
                    bases[tf].add(name[:-len(sf)])
                    break
    return {tf: frozenset(s) for tf, s in bases.items()}, frozenset(btc)


def _series_all(ohlcv_5m: pd.DataFrame, btc_1h_ret=None, btc_5m_roc10=None,
                btc_5m_atr_pct=None, btc_roc_24h=None, btc_roc_4h=None,
                btc_roc_slope=None) -> Dict[str, np.ndarray]:
    """Feature arrays over the buffer (all battery columns + BTC-relative)."""
    n = len(ohlcv_5m)
    r15, r1h, r4h = rolling_resample_ohlcv(ohlcv_5m)
    f = {'15m': compute_features(r15), '1h': compute_features(r1h),
         '4h': compute_features(r4h)}
    out = {}
    for tf in TIMEFRAMES:
        for bi, name in enumerate(BASE_FEATURES):
            out[f'{name}_{tf}'] = f[tf][:, bi].astype(np.float64)

    def _as_arr(v):
        if v is None:
            return np.full(n, np.nan)
        a = np.asarray(v, dtype=np.float64)
        return a if a.ndim else np.full(n, float(a))

    b_roc10 = _as_arr(btc_5m_roc10)
    b_atr = _as_arr(btc_5m_atr_pct)
    out['btc_rel_roc10'] = f['1h'][:, _ROC10].astype(np.float64) - b_roc10
    with np.errstate(divide='ignore', invalid='ignore'):
        out['btc_rel_vol'] = f['1h'][:, _ATRP].astype(np.float64) / (b_atr + 1e-10)
    out['btc_roc_24h'] = _as_arr(btc_roc_24h)
    out['btc_roc_4h'] = _as_arr(btc_roc_4h)
    out['btc_roc_slope'] = _as_arr(btc_roc_slope)

    # rolling 20h correlation of coin 1h returns vs BTC 1h returns
    corr = np.full(n, np.nan)
    if btc_1h_ret is not None and len(ohlcv_5m) >= 12:
        d1h = ohlcv_5m['close'].resample('1h').last().dropna()
        cr = d1h.pct_change()
        c20 = cr.rolling(20).corr(pd.Series(btc_1h_ret).reindex(d1h.index))
        corr = c20.reindex(ohlcv_5m.index, method='ffill').values
    out['btc_corr_20'] = corr
    return out


def compute_all_features_series(ohlcv_5m: pd.DataFrame, needed_bases=None,
                                needed_btc=None, **btc_kwargs
                                ) -> Dict[str, np.ndarray]:
    """Series variant used by the vectorized replay. needed_* accepted for
    signature parity; the template computes the full (small) battery."""
    arrs = _series_all(ohlcv_5m, **btc_kwargs)
    # float16 round-trip: parity with the cache's float16 storage
    return {k: np.float16(v).astype(np.float64) for k, v in arrs.items()}


def compute_all_features(ohlcv_5m: pd.DataFrame, needed_bases=None,
                         needed_btc=None, **btc_kwargs) -> Dict[str, float]:
    """Last-bar feature dict (the live decision path)."""
    arrs = _series_all(ohlcv_5m, **btc_kwargs)
    features = {}
    for k, a in arrs.items():
        v = a[-1]
        if np.isfinite(v):
            features[k] = v
    # float16 round-trip: parity with the cache's float16 storage
    return {k: float(np.float16(v)) for k, v in features.items()}
