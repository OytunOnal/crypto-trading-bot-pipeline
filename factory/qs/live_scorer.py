"""Incremental live QS scorer — the live side of QS parity.

The live bot cannot batch-score a full stream; it scores ONE candidate row at
a time against its trailing per-config history. This class implements that
call path on the SAME canonical math (factory.qs.qs_core), so:

    live score(row over sliced 2*N_ROLL+1 history)  ==  batch qs_avg_rank_pct
    row-for-row — proven by factory/parity/verify_qs_math.py.

Contract: score() computes without mutating state; accumulate() appends the
row afterwards (mirrors the bot's decide-then-record loop).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.qs.qs_core import qs_avg_rank_pct
from factory.qs.live_config import CELL_CONFIG, N_ROLL, QS_FEATURES_JSON


class LiveQualityScore:
    def __init__(self, qs_json_path=None, n_roll=N_ROLL):
        path = Path(qs_json_path) if qs_json_path else Path(QS_FEATURES_JSON)
        self.qs_json = json.load(open(path, encoding='utf-8')) if path.exists() else {}
        self.n_roll = n_roll
        self._hist = defaultdict(list)   # (D, C, S) -> [feature_dict, ...]

    def _entry(self, D, C, S):
        cfg = CELL_CONFIG.get((D, C, S))
        if not cfg:
            return None
        e = self.qs_json.get(cfg.get('qs_key', ''))
        if not e:
            return None
        return cfg, e

    def score(self, D, C, S, feature_dict):
        """(passed, score_2dp) for the candidate row vs trailing history.

        No QS entry configured -> (True, nan): the config trades gate-only.
        Insufficient history -> (False, nan): warming up, do not trade.
        """
        ce = self._entry(D, C, S)
        if ce is None:
            return True, float('nan')
        cfg, e = ce
        qs_feats, ic_map = e['features'], e['ic']
        q = cfg.get('q', 1)
        if q <= 1:
            return True, float('nan')
        cutoff = 100.0 * (1 - 1.0 / q)

        hist = self._hist[(D, C, S)][-(2 * self.n_roll):]
        rows = hist + [feature_dict]
        cols = sorted({f for f in qs_feats})
        stream = np.full((len(rows), len(cols)), np.nan)
        for i, fd in enumerate(rows):
            for j, fn in enumerate(cols):
                v = fd.get(fn)
                if v is not None and np.isfinite(v):
                    stream[i, j] = v
        pct = qs_avg_rank_pct(stream, qs_feats, ic_map, cols, self.n_roll)
        if pct is None:
            return True, float('nan')
        sc = pct[-1]
        if not np.isfinite(sc):
            return False, float('nan')
        sc = round(float(sc), 2)
        return sc >= cutoff, sc

    def accumulate(self, D, C, S, feature_dict):
        self._hist[(D, C, S)].append(feature_dict)
