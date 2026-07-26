"""Beam diversity quota for cell-menu selection.

Cell menu selection fills from style x direction x DD-profile buckets via
round-robin instead of "best N": the upper layer can only build a diverse
portfolio (lesson: a single behavior family = one correlated monthly bet).

Usage: rank-ordered candidate list (best first) -> quota_select(results, 100).
Bucket = (style-profile, direction-profile, DD-tercile). Round-robin order
follows each bucket's best candidate; empty buckets fall back to global order.
OFF by default (enable with TM_DIV_QUOTA=1; built for A/B arm testing).
"""
import os
import numpy as np

DOM = 0.60   # bir profili 'baskin' saymak icin pay esigi
# TM_DIV_NODD=1 (ablation flag): drop the DD-tercile axis from the bucket
# key — outcome-profile spreading is the weakest axis (it does not
# decorrelate causes); the ablation measures whether it earns its place.
_NODD = os.environ.get('TM_DIV_NODD') == '1'


def _style_profile(labels):
    from common.style_map import style_of
    st = [style_of(l) for l in labels]
    st = [s for s in st if s != 'OTHER']
    if not st:
        return 'OTHER'
    for s in ('MOMENTUM', 'MEANREV'):
        if st.count(s) / len(st) >= DOM:
            return s
    return 'MIX'


def _dir_profile(labels):
    nl = sum(1 for l in labels if l.startswith('L'))
    ns = len(labels) - nl
    if not labels:
        return 'BAL'
    if nl / len(labels) >= DOM:
        return 'L'
    if ns / len(labels) >= DOM:
        return 'S'
    return 'BAL'


def quota_select(results, target=100, combo_key='combo', dd_key='max_dd',
                 sltp_map=None):
    """results: rank-ordered dict list (best first). Select `target` records.

    Round-robin over buckets; each bucket yields its best remaining candidate
    in turn. Deterministic: bucket order = global rank of its best candidate.
    sltp_map (4th axis): label -> 'sl/tp' class. Measured: differing SL/TP in
    same-direction pairs cuts pair correlation substantially; the bucket key
    gains an intra-combo SLTP-diversity flag (>=2 distinct classes).
    """
    if len(results) <= target:
        return list(results)
    dds = np.array([r[dd_key] for r in results], dtype=float)
    t1, t2 = np.percentile(dds, [33.4, 66.7])

    buckets = {}
    order = []
    for i, r in enumerate(results):
        labels = list(r[combo_key])
        dd = r[dd_key]
        b = (_style_profile(labels), _dir_profile(labels))
        if not _NODD:
            b = b + (0 if dd <= t1 else (1 if dd <= t2 else 2),)
        if sltp_map:
            n_sltp = len({sltp_map.get(l) for l in labels} - {None})
            b = b + (min(2, n_sltp),)
        if b not in buckets:
            buckets[b] = []
            order.append(b)
        buckets[b].append(i)

    picked, seen = [], set()
    while len(picked) < target:
        moved = False
        for b in order:
            if buckets[b]:
                i = buckets[b].pop(0)
                if i not in seen:
                    seen.add(i)
                    picked.append(results[i])
                    moved = True
                    if len(picked) >= target:
                        break
        if not moved:
            break
    return picked
