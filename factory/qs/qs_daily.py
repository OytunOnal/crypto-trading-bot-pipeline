"""Daily QS-history refresh (self-healing, drift-correcting bootstrap upkeep).

Each run:
  1. Recompute the trailing QS_RECOMPUTE_DAYS of gate-passed QS rows from the
     live kline_cache via the validated qs_replay core (== live == backtest).
  2. Reconcile into qs_trade_history.parquet: drop existing rows in the recompute
     window, replace with the clean recomputed rows, keep older rows -> heals any
     drift (missed bars, restart-warmup) in the last RECOMPUTE_DAYS.
  3. Trim parquet to QS_BOOTSTRAP_ROWS per config.
  4. Trim kline_cache to CACHE_BARS (never below QS_WARMUP_FLOOR_BARS).
  5. Per-config readiness (ready when >= 2*N_ROLL+1 rows -> QS warm == backtest).

Run by the bot daily, or standalone (loads cache from disk).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factory.qs.live_config import (
    CELL_CONFIG, CACHE_BARS, QS_WARMUP_FLOOR_BARS, QS_RECOMPUTE_DAYS,
    QS_BOOTSTRAP_ROWS, QS_PERSIST_PATH, QS_FEATURES_JSON, N_ROLL,
)
from factory.qs.qs_replay import replay_window_fast
import logging

logger = logging.getLogger(__name__)
BTC_SYMBOL = 'BTCUSDT'
READY_ROWS = 2 * N_ROLL + 1   # rows needed for a full-window QS score (== backtest)


def _rows_to_df(rows):
    """qs_replay rows -> parquet schema (entry_time epoch sec, pnl NaN)."""
    out = []
    for r in rows:
        d = {'direction': r['direction'], 'cell': r['cell'], 'signal': r['signal'],
             'coin': r['coin'], 'entry_time': r['entry_time'].value / 1e9,
             'pnl_pct': np.nan}
        for k, v in r.items():
            if k not in ('direction', 'cell', 'signal', 'coin', 'entry_time'):
                d[k] = v
        out.append(d)
    return pd.DataFrame(out)


def run_daily_qs_refresh(kline_cache, persist_path: str = QS_PERSIST_PATH,
                         recompute_days: int = QS_RECOMPUTE_DAYS, now=None,
                         workers=None) -> dict:
    # `workers` is accepted for caller compatibility but ignored: replay_window_fast
    # (vectorized signals+features, ~25x) is fast enough single-process that the
    # coin process pool is no longer needed.
    frames = kline_cache.frames('5m')
    btc = frames.get(BTC_SYMBOL)
    if btc is None or btc.empty:
        logger.warning('qs_daily: no BTC frame in cache -> skip')
        return {'status': 'no_btc'}

    now = now if now is not None else btc.index[-1]
    win_start = (pd.Timestamp(now) - pd.Timedelta(days=recompute_days)).floor('5min')
    bars = [b for b in btc.index if b >= win_start]
    if not bars:
        logger.warning('qs_daily: no bars in recompute window -> skip')
        return {'status': 'no_bars'}

    qs_json = json.load(open(QS_FEATURES_JSON, encoding='utf-8'))
    rows = replay_window_fast(frames, btc, bars, qs_json, age_gate=True)
    if not rows:
        # A 0-row replay (e.g. transient cold-start: BTC ctx not ready) must NOT wipe
        # the window's existing QS history -> skip reconcile, keep the parquet as-is.
        logger.warning('qs_daily: replay produced 0 rows over %d bars -> skip reconcile '
                       '(existing history kept untouched)' % len(bars))
        return {'status': 'no_rows', 'recomputed_rows': 0}
    new_df = _rows_to_df(rows)
    win_epoch = win_start.value / 1e9

    # reconcile: keep rows OLDER than the window, replace the window with fresh
    path = Path(persist_path)
    if path.exists():
        old = pd.read_parquet(path)
        kept = old[old['entry_time'] < win_epoch] if 'entry_time' in old else old.iloc[0:0]
        merged = pd.concat([kept, new_df], ignore_index=True)
    else:
        merged = new_df

    # trim per (direction, cell, signal) to QS_BOOTSTRAP_ROWS (chronological).
    # Use groupby(...).tail() NOT .apply(lambda g: g.tail()): apply can move the
    # grouping keys into the index in some pandas versions -> the later
    # groupby(['direction',...]) then KeyErrors. .tail() preserves columns + is faster.
    if not merged.empty:
        # within-bar (same entry_time) order MUST be COIN-alphabetical to match the
        # backtest stream (cache == resolve_coins-sorted) and the brain's real-time
        # accumulate (also coin-sorted). entry_time-only sort would leave within-bar in
        # replay coin-major (frames) order -> the daily refresh would overwrite the
        # brain's alphabetical history with a different tie order -> QS-rank drift.
        merged = merged.sort_values(['entry_time', 'coin'])
        merged = merged.groupby(['direction', 'cell', 'signal'],
                                group_keys=False, sort=False).tail(QS_BOOTSTRAP_ROWS)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)

    removed = kline_cache.trim_to(CACHE_BARS, floor_bars=QS_WARMUP_FLOOR_BARS)

    # Daily cache-health log: surface any internal gaps (downtime) that would
    # shift long-window signals (SuperTrend) off the backtest. Best-effort.
    try:
        if hasattr(kline_cache, 'cache_health'):
            kline_cache.cache_health('5m')
    except Exception:
        pass

    # readiness per config
    counts = (merged.groupby(['direction', 'cell', 'signal']).size().to_dict()
              if (not merged.empty and 'direction' in merged.columns) else {})
    ready = warming = 0
    warming_list = []
    for (d, cell, sig) in CELL_CONFIG:
        n = counts.get((d, cell, sig), 0)
        if n >= READY_ROWS:
            ready += 1
        else:
            warming += 1
            warming_list.append(f'{d}/{cell}/{sig}({n})')
    res = {'status': 'ok', 'recomputed_rows': len(new_df), 'total_rows': len(merged),
           'cache_trimmed_bars': removed, 'ready': ready, 'warming': warming,
           'warming_list': warming_list, 'recompute_window_start': str(win_start)}
    logger.info('qs_daily: recomputed=%d total=%d ready=%d warming=%d cache_trim=%d'
                % (len(new_df), len(merged), ready, warming, removed))
    if warming_list:
        logger.info('qs_daily warming configs: %s' % ', '.join(warming_list))
    return res


if __name__ == '__main__':
    # standalone: load the live cache from disk and run one refresh
    from factory.qs.kline_frames import MiniKlineCache
    kc = MiniKlineCache(max_bars=CACHE_BARS)
    n = kc.load_from_disk('data/live/kline_cache')
    print('loaded %d cache keys' % n)
    res = run_daily_qs_refresh(kc)
    print(json.dumps({k: v for k, v in res.items() if k != 'warming_list'}, indent=2))
    if res.get('warming_list'):
        print('warming:', ', '.join(res['warming_list']))
