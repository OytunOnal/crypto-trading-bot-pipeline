"""Bootstrap QS trade history parquet for the live bot (cold-start seed).

For each deployed config:
  - Load v2 year-split cache (load_lite per direction)
  - Apply gate + block from config
  - Extract last N_BOOTSTRAP gate-passed trades with QS feature values

Writes a single parquet at data/live/qs_trade_history.parquet.
Peak RAM: ~2-4GB (one direction of one strategy at a time).
"""
import sys, os, time, json, gc
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.qs.live_config import CELL_CONFIG
from factory.gates.features_common import ALL_COL_NAMES
from factory.data.registry import STRATEGIES, load_builder  # per-cell parallel loader
from factory.qs.qs_features import _qs_cell_est             # per-cell RAM estimate (admission)
from factory.qs.age_filter import age_ok_mask               # min-age gate (matches bot + combo-search)

OUTPUT_PATH = ROOT / 'data' / 'live' / 'qs_trade_history.parquet'
QS_JSON_PATH = ROOT / 'config' / 'qs_features.json'
# Live universe: the seed must contain ONLY rows the bot itself could produce.
_WL_PATH = ROOT / 'config' / 'coin_whitelist.txt'
_WHITELIST = frozenset(
    l.strip() for l in open(_WL_PATH, encoding='utf-8')
    if l.strip()) if _WL_PATH.exists() else frozenset()
# 250 = 2*N_ROLL+50: canonical QS meta-rank saturates at 201 stream rows,
# so live scores are backtest-identical from the very first trade.
N_BOOTSTRAP = 250

TREND_MAP = {'BEAR': 0, 'FLAT': 1, 'BULL': 2}
MOM_MAP = {'LOW': 0, 'MED': 1, 'HIGH': 2}

# Signal name (live_config) -> (strat_path, cache_module, cache_class)
STRAT_MAP = {
    'sma_x':  (ROOT/'strategies'/'momentum_sma_cross', 'sma_cross_cache_builder',   'SmaCrossCache'),
    'rsi_mr': (ROOT/'strategies'/'meanrev_rsi',        'rsi_mr_cache_builder',      'RsiMrCache'),
    'ch_brk': (ROOT/'strategies'/'breakout_channel',   'channel_brk_cache_builder', 'ChannelBrkCache'),
}


def load_qs_features():
    if QS_JSON_PATH.exists():
        return json.load(open(QS_JSON_PATH))
    return {}


def extract_config_trades(d, direction, cell, signal, cfg, qs_feats):
    """Apply gates + blocks, return last N_BOOTSTRAP gate-passed trades with QS features."""
    features = d['features']
    trends = d['trends']
    vols = d['vols']
    trades_list = d.get('trades', [])

    sl = cfg['sl']
    tp = cfg['tp']

    # Try asymmetric pnl key first, then symmetric
    pnl_key = f'pnl_{sl}_{tp}'
    if pnl_key not in d:
        pnl_key = f'pnl_{sl}'
    if pnl_key not in d:
        return None, f'pnl key missing (sl={sl} tp={tp})'
    pnl_arr = d[pnl_key]
    resolved = np.isfinite(pnl_arr)

    # Cell filter
    t_idx = TREND_MAP[cell.split('_')[0]]
    m_idx = MOM_MAP[cell.split('_')[1]]
    cell_mask = (trends == t_idx) & (vols == m_idx)

    gate_mask = cell_mask & resolved

    # Gate rules
    for feat_name, op, value in cfg['gate']:
        if feat_name not in ALL_COL_NAMES:
            continue
        fi = ALL_COL_NAMES.index(feat_name)
        col = features[:, fi].astype(np.float64)
        valid = np.isfinite(col)
        if op == '<':
            gate_mask &= (valid & (col < value))
        else:
            gate_mask &= (valid & (col > value))

    # Block rules
    for feat_name, op, value in cfg.get('block', []):
        if feat_name not in ALL_COL_NAMES:
            continue
        fi = ALL_COL_NAMES.index(feat_name)
        col = features[:, fi].astype(np.float64)
        valid = np.isfinite(col)
        if op == '<':
            gate_mask &= ~(valid & (col < value))
        else:
            gate_mask &= ~(valid & (col > value))

    # min-age filter + WHITELIST -- MUST match the bot's own stream: the
    # seed is the live bot's initial QS history and the live universe is
    # whitelist-only (fail-closed). age_ok alone once leaked delisted
    # non-whitelist rows the bot can NEVER produce -> QS rank drift at
    # q-cutoff boundaries.
    n_tr = len(trades_list)
    pre = np.where(gate_mask)[0]
    pcoins = [trades_list[i].get('coin', '') if i < n_tr and isinstance(trades_list[i], dict)
              else '' for i in pre]
    pents = [trades_list[i].get('entry_time') if i < n_tr and isinstance(trades_list[i], dict)
             else None for i in pre]
    aok = age_ok_mask(pcoins, pents)
    wok = np.array([c in _WHITELIST for c in pcoins], dtype=bool)
    age_mask = np.zeros(len(features), dtype=bool)
    age_mask[pre[aok & wok]] = True
    gate_mask &= age_mask

    gp_indices = np.where(gate_mask)[0]
    n_gp = len(gp_indices)
    if n_gp < 30:
        return None, f'gp={n_gp} too few'

    # Map feature index -> trade index (trades are aligned with resolved signals)
    gp_feat = features[gate_mask]
    gp_pnl = pnl_arr[gate_mask]

    # Build feature-to-trade mapping
    n_trades = len(trades_list)
    trade_valid = np.zeros(len(features), dtype=bool)
    trade_valid[:n_trades] = True

    # load_lite rows are coin-major; the live rolling stream needs the
    # chronologically-most-recent N_BOOTSTRAP rows IN TIME ORDER (qs_core ranks a
    # time-ordered stream). Index slicing [-N:] takes the alphabetically-last
    # coins, not the most-recent trades -> sort gate-passed indices by entry time.
    _ents = np.array([trades_list[i].get('entry_time').value
                      if i < n_tr and isinstance(trades_list[i], dict)
                      and trades_list[i].get('entry_time') is not None else 0
                      for i in gp_indices], dtype=np.int64)
    last_n_idx = gp_indices[np.argsort(_ents, kind='stable')][-N_BOOTSTRAP:]
    rows = []
    for rank, orig_idx in enumerate(last_n_idx):
        pnl_val = float(pnl_arr[orig_idx])

        # Get trade metadata if available
        coin = ''
        entry_ts = 0.0
        if orig_idx < n_trades:
            trade = trades_list[orig_idx]
            if isinstance(trade, dict):
                coin = trade.get('coin', '')
                et = trade.get('entry_time')
                if et is not None:
                    entry_ts = et.timestamp() if hasattr(et, 'timestamp') else float(et)

        row = {
            'direction': direction,
            'cell': cell,
            'signal': signal,
            'coin': coin,
            'entry_time': entry_ts,
            'pnl_pct': pnl_val,
        }

        # QS feature values
        feat_vec = features[orig_idx]
        for fname in qs_feats:
            if fname in ALL_COL_NAMES:
                fi = ALL_COL_NAMES.index(fname)
                val = float(feat_vec[fi])
                row[fname] = val if np.isfinite(val) else np.nan
            else:
                row[fname] = np.nan

        rows.append(row)

    return rows, f'{n_gp} gp -> {len(rows)} bootstrapped'


# signal (config.py, lowercase) -> registry code (uppercase, for load_builder)
SIG2CODE = {sig: code for code, (rel, sig, cat) in STRATEGIES.items()}


def _boot_task(args):
    """Worker: per (signal, direction, cell) — load ONLY that cell (peak ~1/9 of
    the direction), extract last N_BOOTSTRAP gate-passed trades with QS features."""
    code, direction, cell, signal, cfg, qs_feats, q_sig, tid = args
    t = TREND_MAP[cell.split('_')[0]]; v = MOM_MAP[cell.split('_')[1]]
    sl, tp = cfg['sl'], cfg['tp']
    pnl_keys = {f'pnl_{sl}_{tp}', f'pnl_{sl}'}
    try:
        cache = load_builder(code)._instance.load_lite(
            direction, years=[2025, 2026], cell=(t, v),
            skip_trades=False, pnl_keys=pnl_keys)
    finally:
        if q_sig is not None:
            q_sig.put(('loaded', tid))
    if cache is None:
        return None, 'cache none'
    d = cache.get(direction)
    if d is None or d.get('features', np.array([])).size == 0:
        return None, 'empty cell'
    res = extract_config_trades(d, direction, cell, signal, cfg, qs_feats)
    return res if res is not None else (None, 'extract None')


def main():
    import psutil, multiprocessing, queue as _queue
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    t0 = time.time()
    print(f'Bootstrap QS seed ({len(CELL_CONFIG)} configs)', flush=True)
    print(f'Output: {OUTPUT_PATH}', flush=True)

    qs_json = load_qs_features()
    print(f'QS JSON: {len(qs_json)} entries', flush=True)

    # one task per config (per-cell load -> small footprint, parallel)
    tasks = []
    for (direction, cell, signal), cfg in CELL_CONFIG.items():
        code = SIG2CODE.get(signal)
        if code is None:
            print(f'  SKIP {signal}: not in registry', flush=True)
            continue
        qs_entry = qs_json.get(cfg.get('qs_key', ''), {})
        qs_feats = qs_entry.get('features', []) if isinstance(qs_entry, dict) else qs_entry
        est = _qs_cell_est(code, cell)
        tasks.append((code, direction, cell, signal, cfg, qs_feats, est))

    n_workers = max(1, (os.cpu_count() or 8) - 2)
    RESERVE = 3e9
    pending = deque(tasks)
    total = len(pending)
    print(f'  {total} config gorevi, cpu cap {n_workers}, measured-RAM admission', flush=True)

    all_rows = []
    running, loading_est, tid_seq, done_n = {}, {}, 0, 0
    mgr = multiprocessing.Manager(); q = mgr.Queue()
    ex = ProcessPoolExecutor(max_workers=n_workers)
    try:
        while pending or running:
            if pending and len(running) < n_workers:
                avail = psutil.virtual_memory().available
                reserved = sum(loading_est.values())
                m = pending[0]
                if (avail - RESERVE - reserved) > m[6] or not running:
                    pending.popleft(); tid_seq += 1
                    fut = ex.submit(_boot_task, (m[0], m[1], m[2], m[3], m[4], m[5], q, tid_seq))
                    running[fut] = (m, tid_seq); loading_est[tid_seq] = m[6]
                    continue
            try:
                while True:
                    ev = q.get_nowait()
                    if isinstance(ev, tuple) and ev[0] == 'loaded':
                        loading_est.pop(ev[1], None)
            except _queue.Empty:
                pass
            if not running:
                if not pending:
                    break
                continue
            fini, _ = wait(running, timeout=2, return_when=FIRST_COMPLETED)
            for fut in fini:
                m, tid = running.pop(fut); loading_est.pop(tid, None)
                rows, msg = fut.result()
                done_n += 1
                tag = f'{m[3]:>7}:{m[1][0]}_{m[2]}'
                if rows:
                    all_rows.extend(rows)
                    print(f'  {tag}: {msg} (q={m[4]["q"]}) [{done_n}/{total}] '
                          f'({(time.time()-t0)/60:.0f}dk)', flush=True)
                else:
                    print(f'  {tag}: SKIP {msg} [{done_n}/{total}]', flush=True)
    finally:
        ex.shutdown(wait=True); mgr.shutdown()

    if not all_rows:
        print('\n  ERROR: No rows extracted!', flush=True)
        return

    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f'\n{"="*80}')
    print(f'  Total: {len(df)} records, {len(df.columns)} columns', flush=True)
    n_cfg = df.groupby(['direction', 'cell', 'signal']).ngroups
    print(f'  {n_cfg} configs with rows | mean rows/config={len(df)/max(n_cfg,1):.0f}', flush=True)
    print(f'\n  Saved to {OUTPUT_PATH}', flush=True)
    print(f'  DONE ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
