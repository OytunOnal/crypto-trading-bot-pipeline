"""Drift-heal test for qs_daily reconcile, on the REAL bootstrap + REAL cache.

Perturb a COPY of the live bootstrap parquet:
  (a) DRIFT row: fake coin, real config, entry_time INSIDE the 1d recompute window
  (b) SENTINEL row: fake coin, fake config, entry_time BEFORE the window
Run a real refresh (whitelist cache -> recompute produces real gate-passed rows),
then assert reconcile:
  - DRIFT (in-window) DROPPED (window replaced by clean recompute)
  - SENTINEL (before-window) PRESERVED
  - recompute produced >0 fresh in-window rows
  - no duplicate (coin, entry_time, config)

  python test_qs_reconcile.py    # exit 0/1
"""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factory.qs.kline_frames import MiniKlineCache as KlineCache
from factory.qs.live_config import CACHE_BARS, QS_PERSIST_PATH
from factory.qs import qs_daily

CACHE = ROOT / 'data' / 'backtest' / 'cache'
SEED = ROOT / QS_PERSIST_PATH
OUT = ROOT / 'data' / 'live' / '_test_reconcile.parquet'

# 1. real bootstrap must exist
if not SEED.exists():
    print('FAIL: no bootstrap parquet at', SEED); sys.exit(1)

# 2. whitelist cache (enough coins for the recompute to gate-pass)
wl = set(l.strip() for l in open(ROOT / 'config' / 'coin_whitelist.txt', encoding='utf-8') if l.strip())
wl.add('BTCUSDT')
kc = KlineCache(connector=None, max_bars=CACHE_BARS)
for s in wl:
    f = CACHE / f'{s}_5m_20260101_20261231.parquet'
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    kc._closed[(s, '5m')] = df.tail(CACHE_BARS)
btc = kc.frames('5m')['BTCUSDT']
now = btc.index[-1]
win_start = (now - pd.Timedelta(days=1)).floor('5min')
win_epoch = win_start.value / 1e9
print('cache: %d coins | now=%s | win_start=%s' % (len(kc.frames('5m')), now, win_start), flush=True)

# 3. copy real bootstrap + inject perturbations
shutil.copy(SEED, OUT)
seed = pd.read_parquet(OUT)
n_before_window_orig = int((seed['entry_time'] < win_epoch).sum())
# feature col present in the seed (use the first QS feature col)
featcol = next(c for c in seed.columns if c not in
               ('direction', 'cell', 'signal', 'entry_time', 'pnl_pct', 'coin'))
drift = {c: np.nan for c in seed.columns}
drift.update({'direction': 'SHORT', 'cell': 'BEAR_MED', 'signal': 'hhhl',
              'coin': 'DRIFTTEST', 'entry_time': (win_start + pd.Timedelta(hours=6)).value / 1e9,
              'pnl_pct': np.nan, featcol: -999.0})
sentinel = {c: np.nan for c in seed.columns}
sentinel.update({'direction': 'LONG', 'cell': 'BEAR_LOW', 'signal': 'faketestsig',
                 'coin': 'OLDSENTINEL', 'entry_time': (now - pd.Timedelta(days=30)).value / 1e9,
                 'pnl_pct': np.nan, featcol: 0.123})
seed = pd.concat([seed, pd.DataFrame([drift, sentinel])], ignore_index=True)
seed.to_parquet(OUT, index=False)
print('injected: DRIFTTEST (in-window) + OLDSENTINEL (30d before, fake config)', flush=True)

# 4. real refresh
res = qs_daily.run_daily_qs_refresh(kc, persist_path=str(OUT), recompute_days=1)
df = pd.read_parquet(OUT)

# 5. assertions
drift_gone = not (df['coin'] == 'DRIFTTEST').any()
sentinel_kept = (df['coin'] == 'OLDSENTINEL').any()
fresh_in_win = int((df['entry_time'] >= win_epoch).sum())
dups = df.duplicated(subset=['direction', 'cell', 'signal', 'coin', 'entry_time']).sum()
recomputed = res.get('recomputed_rows', 0)

print('\n=== reconcile result ===')
print('  status=%s recomputed=%d total=%d' % (res.get('status'), recomputed, len(df)))
print('  DRIFT (in-window) dropped+replaced :', drift_gone)
print('  SENTINEL (before-window) preserved :', sentinel_kept)
print('  fresh in-window rows               : %d (>0 = recompute ran)' % fresh_in_win)
print('  duplicate (coin,time,config) rows  : %d' % dups)
OUT.unlink(missing_ok=True)

ok = drift_gone and sentinel_kept and fresh_in_win > 0 and dups == 0 and recomputed > 0
print('\nRESULT:', 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
