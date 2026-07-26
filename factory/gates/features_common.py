"""Common feature computation and utilities shared by all strategy cache builders.

Architecture: N base features x 3 timeframes (15m, 1h, 4h) + 6 BTC-relative
columns. Includes: rolling resample, BTC trend/momentum regime, SL/TP
simulation, and per-feature IC analysis used by the gate search.

TEMPLATE NOTE: this repo ships a small illustrative feature battery (6 base
features across momentum / trend / volatility / volume). A production system
typically runs a much larger battery — extend BASE_FEATURES and
compute_features() with your own; every downstream stage (cache build, gate
sweep, phase2, QS) picks the new columns up automatically via N_FEAT /
ALL_COL_NAMES.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ── Feature names (TEMPLATE battery) ──────────────────────────────────
BASE_FEATURES = [
    # Momentum
    'rsi_14', 'roc_10',
    # Trend
    'ema20_slope',
    # Volatility
    'atr_pct', 'bb_pct_b',
    # Volume
    'vol_zscore_20',
]

N_FEAT = len(BASE_FEATURES)
TIMEFRAMES = ['15m', '1h', '4h']
ALL_COL_NAMES = [f'{b}_{tf}' for tf in TIMEFRAMES for b in BASE_FEATURES] + \
                ['btc_rel_roc10', 'btc_rel_vol', 'btc_corr_20',
                 'btc_roc_24h', 'btc_roc_4h', 'btc_roc_slope']
N_COLS = len(ALL_COL_NAMES)

_ROC10_IDX = BASE_FEATURES.index('roc_10')
_ATR_PCT_IDX = BASE_FEATURES.index('atr_pct')

CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cache'


# ── Minimal indicator helpers (self-contained) ────────────────────────
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bollinger(close, period=20, n_std=2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + n_std * sd, mid, mid - n_std * sd


def _spearman(x, y):
    """Spearman rank correlation (scalar) - pure numpy, no scipy overhead.

    Ranks via argsort + scatter (one sort) instead of argsort(argsort)
    (two sorts). Same default sort kind both places -> identical tie
    order -> bit-identical IC (verified incl. ties); ~1.5x faster sort.
    """
    n = len(x)
    if n < 3:
        return 0.0
    ox = np.argsort(x)
    rx = np.empty(n, dtype=np.float64)
    rx[ox] = np.arange(n, dtype=np.float64)
    oy = np.argsort(y)
    ry = np.empty(n, dtype=np.float64)
    ry[oy] = np.arange(n, dtype=np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (rx * rx).sum() * (ry * ry).sum()
    if denom <= 0:
        return 0.0
    rho = float((rx * ry).sum() / np.sqrt(denom))
    return rho if np.isfinite(rho) else 0.0


# ── Feature computation ──────────────────────────────────────────────
def compute_features(df):
    """Compute the base feature battery for an OHLCV DataFrame.

    Returns (n, N_FEAT) float32. Extend here when growing BASE_FEATURES —
    keep column order in sync with the name list above.
    """
    n = len(df)
    feat = np.full((n, N_FEAT), np.nan, dtype=np.float32)

    close_s = df['close']
    high_s = df['high']
    low_s = df['low']
    vol_s = df['volume']

    # Momentum
    feat[:, 0] = _rsi(close_s, 14).values                                # rsi_14
    feat[:, 1] = close_s.pct_change(10).values                           # roc_10

    # Trend
    ema20 = _ema(close_s, 20)
    feat[:, 2] = ((ema20 / ema20.shift(5) - 1) * 100).values             # ema20_slope

    # Volatility
    atr_s = _atr(high_s, low_s, close_s, 14)
    feat[:, 3] = (atr_s / close_s).values                                # atr_pct
    upper, _mid, lower = _bollinger(close_s, 20, 2.0)
    feat[:, 4] = ((close_s - lower) / (upper - lower + 1e-10)).values    # bb_pct_b

    # Volume
    vmean = vol_s.rolling(20).mean()
    vstd = vol_s.rolling(20).std()
    feat[:, 5] = ((vol_s - vmean) / (vstd + 1e-10)).values               # vol_zscore_20

    return feat


# ── SL/TP simulation ──────────────────────────────────────────────────
def simulate_sltp(mfe, mae, sl, tp):
    """Vectorised SL/TP PnL from cumulative MFE / MAE arrays.

    Returns PnL (%) per trade. NaN when neither SL nor TP hit.
    """
    sl_hit = mae >= sl
    tp_hit = mfe >= tp
    sl_bar = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), 999999)
    tp_bar = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), 999999)
    pnl = np.full(len(mfe), np.nan)
    sl_first = (sl_bar <= tp_bar) & (sl_bar < 999999)
    tp_first = (tp_bar < sl_bar) & (tp_bar < 999999)
    pnl[sl_first] = -sl
    pnl[tp_first] = tp
    return pnl


# ── Rolling resample ──────────────────────────────────────────────────
def rolling_resample_ohlcv(df):
    """Rolling-resample 5m bars to 15m, 1h, 4h DataFrames."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    def _build(bars):
        return pd.DataFrame({
            'open': o.shift(bars - 1),
            'high': h.rolling(bars).max(),
            'low': l.rolling(bars).min(),
            'close': c,
            'volume': v.rolling(bars).sum(),
        }, index=df.index)

    return _build(3), _build(12), _build(48)


# ── BTC trend (1h) ────────────────────────────────────────────────────
def build_btc_trend(btc_1h_close, fixed_terciles=None):
    """BTC ROC(168h) tercile -> 0=BEAR 1=FLAT 2=BULL.

    btc_1h_close: pd.Series with DatetimeIndex.
    fixed_terciles: tuple (t33, t67) for fixed boundaries. None = expanding.
    Returns pd.Series with same index.
    """
    roc = ((btc_1h_close / btc_1h_close.shift(168)) - 1) * 100
    trend = pd.Series(np.nan, index=btc_1h_close.index)

    if fixed_terciles is not None:
        t33, t67 = fixed_terciles
        valid = roc.notna()
        trend[valid & (roc <= t33)] = 0
        trend[valid & (roc > t33) & (roc <= t67)] = 1
        trend[valid & (roc > t67)] = 2
        return trend

    # Expanding tercile
    vals = []
    roc_v = roc.values
    for i in range(len(roc_v)):
        v = roc_v[i]
        if not np.isfinite(v):
            continue
        vals.append(v)
        if len(vals) < 100:
            continue
        a = np.array(vals)
        t33, t67 = np.percentile(a, [33.33, 66.67])
        if v <= t33:
            trend.iloc[i] = 0
        elif v <= t67:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = 2
    return trend


# ── Coin vol regime ───────────────────────────────────────────────────
def compute_coin_vol_5m(df):
    """Coin ATR%(14) rolling percentile tercile.

    5m -> 1h resample, then ATR(14) on 1h bars, rolling rank window=150.
    Returns pd.Series (0=LOW, 1=MED, 2=HIGH) at 1h resolution.
    """
    d1h = df.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    if len(d1h) < 20:
        return pd.Series(dtype=float)
    atr14 = _atr(d1h['high'], d1h['low'], d1h['close'], 14)
    atr_pct = atr14 / d1h['close']
    rp = atr_pct.rolling(150, min_periods=20).rank(pct=True)
    vol = pd.Series(np.nan, index=d1h.index)
    valid = rp.notna()
    vol[valid & (rp <= 0.3333)] = 0
    vol[valid & (rp > 0.3333) & (rp <= 0.6667)] = 1
    vol[valid & (rp > 0.6667)] = 2
    return vol


# ── BTC 5m features (for btc-relative columns) ───────────────────────
def load_btc_5m_features(load_ohlcv_fn, cache_dir):
    """Load BTC 5m data and compute roc10 + atr_pct.

    Returns dict with 'index', 'roc10', 'atr_pct' keys.
    """
    df = load_ohlcv_fn('BTCUSDT', '5m', cache_dir=str(cache_dir))
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Rolling-resample to 1h (12 bars)
    r1h = pd.DataFrame({
        'open': df['open'].shift(11),
        'high': df['high'].rolling(12).max(),
        'low': df['low'].rolling(12).min(),
        'close': df['close'],
        'volume': df['volume'].rolling(12).sum(),
    }, index=df.index)

    feats = compute_features(r1h)
    return {
        'index': df.index,
        'roc10': feats[:, _ROC10_IDX],
        'atr_pct': feats[:, _ATR_PCT_IDX],
    }


# ── Feature analysis utilities (for gate / QS search) ────────────────
def compute_all_feature_stats(features, pnl, years, col_names=None):
    """Compute per-feature IC stats.

    col_names: column name list matching the features array width. Defaults
    to the CURRENT ALL_COL_NAMES — pass an explicit snapshot when analysing
    caches built with an older feature set.
    Returns list of dicts with keys: idx, name, ic, abs_ic, stab, n_yr.
    """
    names = col_names if col_names is not None else ALL_COL_NAMES
    resolved = np.isfinite(pnl)
    unique_yrs = sorted(set(years[resolved]))
    stats = []

    for fi in range(features.shape[1]):
        col = features[:, fi].astype(np.float64)
        v = resolved & np.isfinite(col)
        if v.sum() < 200:
            continue
        ic = _spearman(col[v], pnl[v])

        # Per-year stability
        yr_ics = []
        for y in unique_yrs:
            ym = v & (years == y)
            if ym.sum() < 50:
                continue
            yr_ic = _spearman(col[ym], pnl[ym])
            yr_ics.append(yr_ic)

        n_yr = len(yr_ics)
        if n_yr < 2:
            continue
        # Stability = number of years with same sign as overall IC
        stab = sum(1 for yic in yr_ics if (yic > 0) == (ic > 0))

        name = names[fi] if fi < len(names) else f'col_{fi}'

        stats.append({
            'idx': fi, 'name': name,
            'ic': ic, 'abs_ic': abs(ic),
            'stab': stab, 'n_yr': n_yr,
        })
    return stats


def dedup_1h_4h(feats):
    """Remove 15m/1h/4h duplicates of the same base: keep higher abs_ic."""
    keep = []
    seen_bases = {}
    feats = sorted(feats, key=lambda f: f['abs_ic'], reverse=True)
    for f in feats:
        name = f['name']
        for sf in ['_15m', '_1h', '_4h']:
            if name.endswith(sf):
                base = name[: -len(sf)]
                break
        else:
            base = name

        if base in seen_bases:
            continue
        seen_bases[base] = True
        keep.append(f)
    return keep
