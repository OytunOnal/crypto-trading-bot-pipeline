"""QS safety regression tests (CI-able, exit 0/1).

Guards against the 2026-06 coin-major-ordering bug class:
  1. assert_chronological fires on out-of-order streams, passes on sorted.
  2. qs_avg_rank_pct(entry_times=...) raises on coin-major input.
  3. GOLDEN: a fixed chronological stream -> stable QS pass-set (math regression).
  4. PARITY: the live windowed score() == the backtest full-stream _qs_mask on the
     SAME chronological data (proves the two QS code paths agree given equal order).

  python qs_guard_test.py    # -> "ALL PASS" + exit 0, or failures + exit 1
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factory.qs import qs_core
from factory.qs.qs_core import qs_avg_rank_pct, assert_chronological

N_ROLL = 100
fails = []


def check(name, cond):
    print(('  PASS ' if cond else '  FAIL ') + name)
    if not cond:
        fails.append(name)


# ── deterministic synthetic stream (no Math.random in this env -> index hash) ──
def make_stream(n, n_feat=2, seed=1):
    a = np.empty((n, n_feat), dtype=np.float64)
    for i in range(n):
        for j in range(n_feat):
            a[i, j] = ((i * 1103515245 + (j + 1) * 12345 + seed * 7) % 100003) / 100003.0
    return a


COLS = ['f0', 'f1']
QS_FEATS = ['f0', 'f1']
IC = {'f0': 1.0, 'f1': -1.0}

print('1. assert_chronological')
ok = True
try:
    assert_chronological(np.array([1, 2, 2, 5, 9]))
except AssertionError:
    ok = False
check('sorted stream accepted', ok)
raised = False
try:
    assert_chronological(np.array([1, 5, 2, 9]))
except AssertionError:
    raised = True
check('out-of-order stream rejected', raised)

print('2. qs_avg_rank_pct guard')
feats = make_stream(300)
et_sorted = np.arange(300, dtype=np.int64)
ok = True
try:
    qs_avg_rank_pct(feats, QS_FEATS, IC, COLS, N_ROLL, entry_times=et_sorted)
except AssertionError:
    ok = False
check('sorted entry_times accepted', ok)
et_coinmajor = et_sorted.copy(); et_coinmajor[5], et_coinmajor[6] = 6, 5  # one swap
raised = False
try:
    qs_avg_rank_pct(feats, QS_FEATS, IC, COLS, N_ROLL, entry_times=et_coinmajor)
except AssertionError:
    raised = True
check('coin-major entry_times rejected', raised)

print('3. GOLDEN math regression')
feats = make_stream(400, seed=3)
avg = qs_avg_rank_pct(feats, QS_FEATS, IC, COLS, N_ROLL)
cutoff = 80.0  # q5
n_pass = int(np.sum(np.isfinite(avg) & (avg >= cutoff)))
n_valid = int(np.sum(np.isfinite(avg)))
# stable for this fixed input; if the QS math changes intentionally, update these.
check('valid rows == 300 (positional warmup = first n_roll NaN)', n_valid == 300)
# exact golden for this fixed input; a change here means the QS math moved (intended?)
check('q5 pass count stable (== 106)', n_pass == 106)

print('4. PARITY live windowed score() == backtest full-stream')
feats = make_stream(500, seed=5)
full = qs_avg_rank_pct(feats, QS_FEATS, IC, COLS, N_ROLL)   # backtest: rank every row


def windowed_last(stream_end_idx):
    """live score(): last 2*N_ROLL+1 history + current -> rank of last row."""
    lo = max(0, stream_end_idx - (2 * N_ROLL))
    sub = feats[lo:stream_end_idx + 1]
    pct = qs_avg_rank_pct(sub, QS_FEATS, IC, COLS, N_ROLL)
    return pct[-1] if pct is not None else np.nan


mism = 0
for i in range(2 * N_ROLL + 1, 500):   # deep enough that the window is full
    a, b = full[i], windowed_last(i)
    if not (np.isfinite(a) and np.isfinite(b)):
        if np.isfinite(a) != np.isfinite(b):
            mism += 1
        continue
    if abs(a - b) > 1e-9:
        mism += 1
check('live windowed == backtest full-stream (0 mismatch)', mism == 0)

print()
if fails:
    print('FAILED: ' + ', '.join(fails)); sys.exit(1)
print('ALL PASS'); sys.exit(0)
