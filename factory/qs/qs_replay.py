"""Historical pipeline replay -> per-config gate-passed QS-feature rows.

Single source of truth for replaying the live bot pipeline (BTC regime ->
needed signals -> per-coin signal fire -> features -> gate+block -> 365d age)
over a window of 5m bars, recording the QS-feature values of every gate-passed
row. Shared by:
  - cold-start self-seed (bootstrap)
  - daily QS reconcile (qs_daily)
  - parity (replica_vs_backtest can import replay_window)

By construction this uses the SAME bot functions the live decision loop uses,
so self-generated QS history == live == backtest (proven by verify_qs_math).
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from factory.qs.live_config import (
    CELL_CONFIG, TREND_NAMES, MOM_NAMES, MIN_COIN_BARS,
    needed_signals_for_regime, configs_for_regime,
)
from factory.qs.regime_gate import (
    BtcTrendTracker, BtcMomentumTracker, SIGNAL_FUNCS, SIGNAL_SERIES_FUNCS,
    check_cell_gate, check_block_gate,
)
from factory.qs.live_features import (
    compute_all_features, compute_all_features_series, merge_config_needs,
)
from factory.qs.age_filter import coin_age_ok
import logging

logger = logging.getLogger(__name__)

TRIM_BARS = 4400   # scanner buffer (covers the longest signal warmup)
FEAT_BARS = 1200   # feature window (converged; mirrors brain)
BTC_MIN_BARS = 2100


def _btc_context(b):
    """BTC regime + btc_kwargs from 5m history up to T (mirrors brain/replica)."""
    b1h = b.resample('1h').agg({'close': 'last'}).dropna()
    if len(b1h) > 1:
        b1h = b1h.iloc[:-1]
    c1 = b1h['close']
    trend = BtcTrendTracker().update(c1)
    mom = BtcMomentumTracker().update(c1)
    if trend is None or mom is None:
        return None
    r1c = b['close'].copy(); r1c.iloc[:11] = np.nan
    r1h_high = b['high'].rolling(12).max(); r1h_low = b['low'].rolling(12).min()
    tr = pd.concat([r1h_high - r1h_low, (r1h_high - r1c.shift(1)).abs(),
                    (r1h_low - r1c.shift(1)).abs()], axis=1).max(axis=1)
    btc_kwargs = dict(
        btc_1h_ret=c1.pct_change(),
        btc_5m_roc10=float(r1c.pct_change(10).iloc[-1]),
        btc_5m_atr_pct=float(tr.rolling(14).mean().iloc[-1] / r1c.iloc[-1]),
        btc_roc_24h=float(c1.pct_change(24).iloc[-1]),
        btc_roc_4h=float(c1.pct_change(4).iloc[-1]),
        btc_roc_slope=float(c1.pct_change(10).diff(5).iloc[-1]),
    )
    return trend, mom, btc_kwargs


def replay_window(coin_ohlcv: Dict[str, pd.DataFrame], btc: pd.DataFrame,
                  bars: List[pd.Timestamp], qs_json: dict,
                  age_gate: bool = True, log_progress: bool = False,
                  progress_path: str = None) -> List[dict]:
    """Replay `bars` over `coin_ohlcv` (+btc); return gate-passed rows with QS vals.

    Each returned row: {'direction','cell','signal','coin','entry_time', <qs_feat>:val...}.
    Only rows that pass signal-fire + gate + block (+365d age if age_gate) are kept.
    QS feature values are taken from the same features dict the live bot computes.

    log_progress=True -> emit a throttled (~every 20s) bar i/N + rows + ETA line so a
    long single-process replay (the 2-core server path) is followable, not a silent block.
    progress_path -> write 'i n rows' to this file each bar; the process-pool parent
    polls it for progress (spawn workers can't write the parent's log file handler).
    """
    import time as _t
    out = []
    n = len(bars)
    t0 = _t.time(); last_log = t0
    for i, T in enumerate(bars):
        if progress_path is not None:
            try:
                with open(progress_path, 'w') as _pf:
                    _pf.write('%d %d %d' % (i, n, len(out)))
            except Exception:
                pass
        if log_progress:
            now = _t.time()
            if now - last_log >= 20:
                last_log = now
                eta = (now - t0) / max(i, 1) * (n - i)
                logger.info('  replay progress: bar %d/%d (%.0f%%), %d rows, %.0fs, ETA %.0fs'
                            % (i, n, 100.0 * i / max(n, 1), len(out), now - t0, eta))
        b = btc.loc[:T]
        if len(b) < BTC_MIN_BARS:
            continue
        ctx = _btc_context(b)
        if ctx is None:
            continue
        trend, mom, btc_kwargs = ctx
        needed = needed_signals_for_regime(trend, mom)
        regime_configs = configs_for_regime(trend, mom)
        if not regime_configs:
            continue

        for sym, df in coin_ohlcv.items():
            buf = df.loc[:T]
            if len(buf) < MIN_COIN_BARS or buf.index[-1] != T:
                continue
            if age_gate and not coin_age_ok(sym, T):
                continue
            buf = buf.iloc[-TRIM_BARS:]

            sigs = {}
            for sig_name in needed:
                entry = SIGNAL_FUNCS.get(sig_name)
                if entry is None:
                    continue
                func, trim_s = entry
                ohlcv_sig = buf.iloc[-trim_s:] if len(buf) > trim_s else buf
                try:
                    s, l = func(ohlcv_sig)
                    if s or l:
                        sigs[sig_name] = (s, l)
                except Exception:
                    pass
            if not sigs:
                continue

            candidates = []
            for direction, cell, signal, cfg in regime_configs:
                sp = sigs.get(signal)
                if sp is None:
                    continue
                if (direction == 'SHORT' and sp[0]) or (direction == 'LONG' and sp[1]):
                    candidates.append((direction, cell, signal, cfg))
            if not candidates:
                continue

            bases, btcn = merge_config_needs(candidates)
            buf_feat = buf.iloc[-FEAT_BARS:] if len(buf) > FEAT_BARS else buf
            features = compute_all_features(buf_feat, needed_bases=bases,
                                            needed_btc=btcn, **btc_kwargs)

            for direction, cell, signal, cfg in candidates:
                if not (check_cell_gate(features, cfg['gate']) and
                        check_block_gate(features, cfg.get('block', []))):
                    continue
                qs_feats = qs_json.get(cfg.get('qs_key', ''), {}).get('features', [])
                row = {'direction': direction, 'cell': cell, 'signal': signal,
                       'coin': sym, 'entry_time': T}
                for f in qs_feats:
                    v = features.get(f)
                    row[f] = float(v) if (v is not None and np.isfinite(v)) else np.nan
                out.append(row)
    return out


def replay_window_fast(coin_ohlcv: Dict[str, pd.DataFrame], btc: pd.DataFrame,
                       bars: List[pd.Timestamp], qs_json: dict,
                       age_gate: bool = True, gate_only: bool = False) -> List[dict]:
    """Vectorized-signal replay. Returns the SAME rows as replay_window (a set; row
    order may differ -- QS reconcile groups by config + sorts by entry_time, so order
    is irrelevant). Speed: each coin's signal-fire is computed ONCE over its buffer via
    SIGNAL_SERIES_FUNCS instead of re-running the per-bar signal func at every bar.
    Signals without a series variant fall back to the per-bar SIGNAL_FUNCS call.
    Features + gate are unchanged (still per gate-passed candidate)."""
    # 1) Per-bar BTC context (cheap: len(bars) iterations, same as replay_window).
    metas = []
    for T in bars:
        b = btc.loc[:T]
        if len(b) < BTC_MIN_BARS:
            metas.append(None); continue
        ctx = _btc_context(b)
        if ctx is None:
            metas.append(None); continue
        trend, mom, btc_kwargs = ctx
        regime_configs = configs_for_regime(trend, mom)
        if not regime_configs:
            metas.append(None); continue
        metas.append((needed_signals_for_regime(trend, mom), regime_configs, btc_kwargs))

    all_needed = set()
    for m in metas:
        if m:
            all_needed.update(m[0])

    import time as _t
    out = []
    items = list(coin_ohlcv.items())
    nco = len(items)
    t0 = _t.time(); last_log = t0
    for ci, (sym, df) in enumerate(items):
        if _t.time() - last_log >= 20:
            last_log = _t.time()
            el = _t.time() - t0
            eta = el / max(ci, 1) * (nco - ci)
            logger.info('  replay_fast: coin %d/%d (%.0f%%), %d rows, %.0fs, ETA %.0fs'
                        % (ci, nco, 100.0 * ci / max(nco, 1), len(out), el, eta))
        idx = df.index
        positions = idx.get_indexer(list(bars))   # exact-match positions, -1 if absent
        vmask = positions >= 0
        if not vmask.any():
            continue
        vpos = positions[vmask]
        min_pos = int(vpos.min()); max_pos = int(vpos.max())
        # Vectorized signal-fire series, computed ONCE per signal over just the needed
        # range [min_pos - trim_s : max_pos] (warmup + window) -- NOT the full buffer.
        # Stored as (start_offset, short_arr, long_arr); lookup at coin-pos `pos` uses
        # index pos-start_offset. Window-insensitive beyond trim_s -> bit-identical.
        sig_series = {}
        for s in all_needed:
            sf = SIGNAL_SERIES_FUNCS.get(s)
            if sf is None:
                continue
            trim_s = SIGNAL_FUNCS[s][1]
            start_s = max(0, min_pos - trim_s)
            try:
                sh, lo = sf(df.iloc[start_s:max_pos + 1])
                sig_series[s] = (start_s, sh, lo)
            except Exception:
                sig_series[s] = None

        # Pass 1: resolve signal fires -> candidates per bar (signal lookups are O(1)
        # from the precomputed series; roc/vol fall back to per-bar). Collect the
        # candidate bars + the UNION of needed feature bases/btc across them.
        bar_cands = []   # (pos, T, candidates, btc_kwargs)
        u_bases = {'15m': set(), '1h': set(), '4h': set()}
        u_btc = set()
        for bar_i, T in enumerate(bars):
            m = metas[bar_i]
            if m is None:
                continue
            pos = positions[bar_i]
            if pos < 0 or pos + 1 < MIN_COIN_BARS:
                continue
            if age_gate and not coin_age_ok(sym, T):
                continue
            needed, regime_configs, btc_kwargs = m
            sigs = {}
            buf = None   # lazy: only for fallback signals
            for sig_name in needed:
                ser = sig_series.get(sig_name)
                if sig_name in sig_series and ser is not None:
                    start_s, sh, lo = ser
                    j = pos - start_s
                    if 0 <= j < len(sh):
                        s_, l_ = bool(sh[j]), bool(lo[j])
                        if s_ or l_:
                            sigs[sig_name] = (s_, l_)
                else:
                    entry = SIGNAL_FUNCS.get(sig_name)
                    if entry is None:
                        continue
                    func, trim_s = entry
                    if buf is None:
                        buf = df.iloc[max(0, pos + 1 - TRIM_BARS):pos + 1]
                    ohlcv_sig = buf.iloc[-trim_s:] if len(buf) > trim_s else buf
                    try:
                        s_, l_ = func(ohlcv_sig)
                        if s_ or l_:
                            sigs[sig_name] = (s_, l_)
                    except Exception:
                        pass
            if not sigs:
                continue
            candidates = []
            for direction, cell, signal, cfg in regime_configs:
                sp = sigs.get(signal)
                if sp is None:
                    continue
                if (direction == 'SHORT' and sp[0]) or (direction == 'LONG' and sp[1]):
                    candidates.append((direction, cell, signal, cfg))
            if not candidates:
                continue
            bar_cands.append((pos, T, candidates, btc_kwargs))
            bb, bt = merge_config_needs(candidates)
            for tf in u_bases:
                u_bases[tf].update(bb.get(tf, frozenset()))
            u_btc.update(bt)
        if not bar_cands:
            continue

        # Pass 2: compute the feature series ONCE over [min_cand - FEAT_BARS : max_cand]
        # (features are window-insensitive beyond FEAT_BARS + float16-cast -> reading bar
        # i == the per-bar 1200-bar compute_all_features). BTC per-bar scalars become
        # arrays aligned to the buffer (value at each candidate bar, NaN elsewhere).
        cand_pos = [bc[0] for bc in bar_cands]
        fb_start = max(0, min(cand_pos) - FEAT_BARS)
        fbuf = df.iloc[fb_start:max(cand_pos) + 1]
        L = len(fbuf)

        def _bt_arr(key):
            a = np.full(L, np.nan)
            for p, _T, _c, bk in bar_cands:
                v = bk.get(key)
                if v is not None:
                    a[p - fb_start] = v
            return a

        feats = compute_all_features_series(
            fbuf,
            needed_bases={tf: frozenset(s) for tf, s in u_bases.items()},
            needed_btc=frozenset(u_btc),
            btc_1h_ret=bar_cands[-1][3].get('btc_1h_ret'),
            btc_5m_roc10=_bt_arr('btc_5m_roc10'),
            btc_5m_atr_pct=_bt_arr('btc_5m_atr_pct'),
            btc_roc_24h=_bt_arr('btc_roc_24h'),
            btc_roc_4h=_bt_arr('btc_roc_4h'),
            btc_roc_slope=_bt_arr('btc_roc_slope'),
        )

        for pos, T, candidates, btc_kwargs in bar_cands:
            j = pos - fb_start
            features = {}
            for feat, arr in feats.items():
                v = arr[j]
                if np.isfinite(v):
                    features[feat] = float(v)
            for direction, cell, signal, cfg in candidates:
                if not check_cell_gate(features, cfg['gate']):
                    continue                                   # gate fail -> drop always
                _blk = check_block_gate(features, cfg.get('block', []))
                if not (gate_only or _blk):
                    continue                                   # default: gate+block only
                qs_feats = qs_json.get(cfg.get('qs_key', ''), {}).get('features', [])
                row = {'direction': direction, 'cell': cell, 'signal': signal,
                       'coin': sym, 'entry_time': T}
                if gate_only:
                    row['_block'] = bool(_blk)                 # gate_only: keep all gate-passed + flag
                for f in qs_feats:
                    v = features.get(f)
                    row[f] = float(v) if (v is not None and np.isfinite(v)) else np.nan
                out.append(row)
    logger.info('replay_fast DONE: %d rows from %d coins, %d bars in %.0fs'
                % (len(out), nco, len(bars), _t.time() - t0))
    return out


# ── Coin-parallel replay (process pool over coins) ──────────────────────────
# replay_window is single-process; the full universe x a multi-day window is hours.
# The work is CPU-bound with heavy Python orchestration (per coin x bar: signal
# dispatch + feature build), so a THREAD pool does NOT help -- it's GIL-bound and
# measured ~0.73-0.96x (slower). (The live scanner's ThreadPool win is from network
# I/O overlap, which the in-memory replay has none of.) Real parallelism needs
# separate PROCESSES: measured ~1.8x with 2 workers on the 2-core server (cores
# free), ~Nx on a many-core box. The auto worker count (workers=None) is
# conservative (cpu-2, then a RAM cap) and collapses to 1 on a 2-core/4GB box, so
# the live refresh callers pass an explicit `workers` (= cpu_count) to actually use
# the cores. Uses a SPAWN pool (fork would risk deadlock in the live multi-threaded
# bot; main.py is __main__-guarded so spawn re-import is safe).
_W_BTC = _W_BARS = _W_QS = _W_AGE = None


def _replay_init(btc, bars, qs_json, age_gate):
    global _W_BTC, _W_BARS, _W_QS, _W_AGE
    _W_BTC, _W_BARS, _W_QS, _W_AGE = btc, bars, qs_json, age_gate


def _replay_chunk(args):
    coin_chunk, ppath = args
    return replay_window(coin_chunk, _W_BTC, _W_BARS, _W_QS, _W_AGE, progress_path=ppath)


def replay_window_parallel(coin_ohlcv: Dict[str, pd.DataFrame], btc: pd.DataFrame,
                           bars: List[pd.Timestamp], qs_json: dict,
                           age_gate: bool = True, workers: Optional[int] = None) -> List[dict]:
    """Coin-parallel replay_window via a process pool. Result == single-process
    (just split + merged). Uses a SPAWN pool (fork would risk deadlock in the live
    multi-threaded bot)."""
    import os, time as _t
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    coins = list(coin_ohlcv.keys())
    if not coins:
        logger.info('replay_parallel: no coins -> 0 rows')
        return []
    if workers is None:
        workers = max(1, (os.cpu_count() or 8) - 2)
        try:
            import psutil
            avail = psutil.virtual_memory().available
            workers = max(1, min(workers, int((avail - 2e9) // 6e8)))  # ~0.6GB/worker
        except Exception:
            pass
    workers = max(1, min(workers, len(coins)))
    if workers == 1:
        logger.info('replay_parallel: 1 worker -> single-process (%d coins, %d bars)'
                    % (len(coins), len(bars)))
        t0 = _t.time()
        res = replay_window(coin_ohlcv, btc, bars, qs_json, age_gate, log_progress=True)
        logger.info('replay_parallel DONE (single-process): %d rows from %d coins in %.0fs'
                    % (len(res), len(coins), _t.time() - t0))
        return res
    chunks = [{} for _ in range(workers)]
    for i, c in enumerate(coins):
        chunks[i % workers][c] = coin_ohlcv[c]
    logger.info('replay_parallel: %d coins, %d bars, %d workers (spawn) -- starting'
                % (len(coins), len(bars), workers))
    # Progress: chunk 0 writes its bar index to a small file each bar; a parent
    # heartbeat thread polls it (~every 20s) and logs bar i/N + ETA. Needed because
    # spawn workers can't write the parent's log file handler -> they'd be silent.
    import tempfile, threading
    n = len(bars)
    progress_path = os.path.join(tempfile.gettempdir(),
                                 'qs_replay_progress_%d.txt' % os.getpid())
    t0 = _t.time()
    stop = threading.Event()

    def _heartbeat():
        while not stop.wait(20):
            i = 0
            try:
                with open(progress_path) as _pf:
                    i = int(_pf.read().split()[0])
            except Exception:
                pass
            el = _t.time() - t0
            eta = el / max(i, 1) * (n - i) if i else 0
            logger.info('  replay_parallel: bar %d/%d (%.0f%%), %.0fs, ETA %.0fs (%d workers)'
                        % (i, n, 100.0 * i / max(n, 1), el, eta, workers))
    threading.Thread(target=_heartbeat, daemon=True).start()

    out = []
    ctx = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_replay_init,
                                 initargs=(btc, bars, qs_json, age_gate)) as ex:
            # only chunk 0 reports progress (all chunks march the same bars in lockstep)
            futs = [ex.submit(_replay_chunk, (ch, progress_path if i == 0 else None))
                    for i, ch in enumerate(chunks)]
            for j, f in enumerate(as_completed(futs), 1):
                rows = f.result()
                out.extend(rows)
                logger.info('  replay_parallel: chunk %d/%d done (+%d rows, %d total, %.0fs)'
                            % (j, workers, len(rows), len(out), _t.time() - t0))
    finally:
        stop.set()
        try:
            os.remove(progress_path)
        except Exception:
            pass
    logger.info('replay_parallel DONE: %d rows from %d coins in %.0fs (%d workers)'
                % (len(out), len(coins), _t.time() - t0, workers))
    return out
