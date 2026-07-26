"""QS-math parity: live HybridMixQualityScore (incremental) vs backtest
qs_avg_rank_pct (batch) over the SAME gate-passed stream.

qs_core is shared code, so the math is identical by construction. This proves
the LIVE call path (sliced 2*window+1 history -> score() -> last row) reproduces
the BACKTEST batch (full-stream qs_avg_rank_pct) row-for-row, AND that the
pass/fail decision (>= cutoff) matches.
"""
import os, sys, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.data.registry import load_builder
from factory.gates.features_common import ALL_COL_NAMES
from factory.qs.live_config import CELL_CONFIG, N_ROLL
from factory.qs.qs_core import qs_avg_rank_pct
from factory.qs.live_scorer import LiveQualityScore as HybridMixQualityScore

CELL_TV = {'BEAR': 0, 'FLAT': 1, 'BULL': 2, 'LOW': 0, 'MED': 1, 'HIGH': 2}


def gate_passed_stream(code, D, C, cfg, years):
    """Return features[gp] gate-passed chronological stream + pnl."""
    sl, tp = cfg['sl'], cfg['tp']
    t, v = CELL_TV[C.split('_')[0]], CELL_TV[C.split('_')[1]]
    pk = f'pnl_{sl}_{tp}'
    inst = load_builder(code)._instance
    cache = inst.load_lite(D, years=years, skip_trades=False, cell=(t, v), pnl_keys=[pk])
    d = cache[D]
    feats = d['features']
    pnl = d.get(pk)
    if pnl is None:
        pnl = d.get(f'pnl_{sl}')
    gp = np.isfinite(pnl) & (pnl != 0)
    for (fn, op, val) in cfg['gate']:
        if fn not in ALL_COL_NAMES:
            continue
        col = feats[:, ALL_COL_NAMES.index(fn)].astype(np.float64)
        fin = np.isfinite(col)
        gp &= (fin & (col < val)) if op == '<' else (fin & (col > val))
    for (fn, op, val) in cfg.get('block', []):
        if fn not in ALL_COL_NAMES:
            continue
        col = feats[:, ALL_COL_NAMES.index(fn)].astype(np.float64)
        fin = np.isfinite(col)
        gp &= ~(fin & (col < val)) if op == '<' else ~(fin & (col > val))
    return feats[gp]


def run(code, D, C, S, years):
    cfg = CELL_CONFIG[(D, C, S)]
    q, qk = cfg['q'], cfg['qs_key']
    cutoff = 100.0 * (1 - 1.0 / q)
    qs_json = json.load(open(ROOT / 'config' / 'qs_features.json'))
    e = qs_json[qk]
    qs_feats, ic_map = e['features'], e['ic']

    stream = gate_passed_stream(code, D, C, cfg, years)
    n = len(stream)

    # BACKTEST batch
    batch = qs_avg_rank_pct(stream, qs_feats, ic_map, ALL_COL_NAMES, N_ROLL)
    batch_pass = np.isfinite(batch) & (batch >= cutoff)

    # LIVE incremental
    live = HybridMixQualityScore()
    live_sc = np.full(n, np.nan)
    live_pass = np.zeros(n, dtype=bool)
    fcols = {fn: ALL_COL_NAMES.index(fn) for fn in qs_feats if fn in ALL_COL_NAMES}
    for i in range(n):
        fd = {fn: float(stream[i, ci]) for fn, ci in fcols.items()}
        passed, sc = live.score(D, C, S, fd)
        live_sc[i] = sc
        live_pass[i] = passed
        live.accumulate(D, C, S, fd)

    # compare (live rounds to 2dp; round batch too for score compare)
    bro = np.round(batch, 2)
    both = np.isfinite(bro) & np.isfinite(live_sc)
    score_diff = np.abs(bro[both] - live_sc[both])
    nan_mis = int((np.isfinite(batch) != np.isfinite(live_sc)).sum())
    pass_mis = int((batch_pass != live_pass).sum())

    print(f'\n{code} {D} {C} {S} (q{q}, cutoff={cutoff:.0f}) | qs_feats={qs_feats}')
    print(f'  gate-passed stream rows: {n}')
    print(f'  defined (both finite):   {int(both.sum())}')
    print(f'  score max|diff| (2dp):   {score_diff.max() if score_diff.size else 0:.4f}')
    print(f'  score rows >0.01 diff:   {int((score_diff > 0.01).sum())}')
    print(f'  NaN-alignment mismatch:  {nan_mis}')
    print(f'  PASS-decision mismatch:  {pass_mis}  / {int(np.isfinite(batch).sum())} defined')
    ok = score_diff.max() <= 0.01 and nan_mis == 0 and pass_mis == 0 if score_diff.size else False
    print(f'  -> {"PARITY OK" if ok else "MISMATCH"}')
    return ok


if __name__ == '__main__':
    years = [2024, 2025, 2026]
    # config-driven: one representative q>1 (QS-scored) config per cell.
    # q1 configs are gate-only baseline (no QS) -> skipped. code parsed from qs_key.
    TARGETS, seen = [], set()
    for (D, C, S), cfg in CELL_CONFIG.items():
        if cfg['q'] <= 1 or C in seen:
            continue
        # qs_key is now cN-keyed (..._q{q}_c{ci}); parse code as the token before '_q'
        code = cfg['qs_key'][len(f'{D}_{C}_'):].split('_q')[0]
        TARGETS.append((code, D, C, S))
        seen.add(C)
    print(f'QS-math parity: {len(TARGETS)} configs (one q>1 per cell)')
    allok = True
    for code, D, C, S in TARGETS:
        try:
            allok &= run(code, D, C, S, years)
        except Exception as ex:
            print(f'\n{code} {D} {C} {S}: ERROR {type(ex).__name__}: {ex}')
            allok = False
    print(f'\n{"="*50}\nRESULT: {"ALL PARITY OK" if allok else "MISMATCHES"}')
