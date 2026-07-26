#!/bin/bash
# BLIND WALK-FORWARD: SINGLE-QUARTER REPLAY.
#   bash procedure_wf_quarter.sh <QLABEL> <CUTOFF> <JUDGE_START> <JUDGE_END>
#   e.g.: bash procedure_wf_quarter.sh 2026Q1 2025-10-20 2026-01-01 2026-04-01
#
# The chain runs BLIND up to CUTOFF -> deploy consensus picks a package ->
# pwf_judge measures the pick on the unseen [JS,JE) window -> results/pwf/<Q>/.
# Repeating this over many quarters builds the procedure's EXPECTATION BAND:
# what forward performance the selection procedure actually delivers.
#
# METHOD NOTES (these go into any report):
#  - gate/phase2/QS layers are frozen at their current state -> absolute
#    levels for early quarters are OPTIMISTIC; A/B arm comparisons stay fair
#    (all arms share the same contamination).
#  - the coin whitelist is today's whitelist (a mild, documented anachronism).
#  - OOS boundary: fixed calendar-year OOS breaks for earlier cutoffs (the
#    floors empty out). Standard here: OOS = trailing 26 WEEKS from cutoff
#    (uniform across quarters); passed to the sims via TM_OOS_START /
#    TM_OOS_WSTART. Production runs without the env -> bit-identical default.
set -u -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PD="$ROOT/trade_pickles"
QL="$1"; export TM_CUTOFF="$2"; JS="$3"; JE="$4"
OUT="$ROOT/results/pwf/band/$QL"
HB='{print; fflush()}'
cd "$ROOT"

# OOS = cutoff - 26 weeks (month + iso-week keys)
read -r TM_OOS_START TM_OOS_WSTART <<< "$(python - <<EOF
import pandas as pd
c = pd.Timestamp('$TM_CUTOFF') - pd.Timedelta(weeks=26)
iso = c.isocalendar()
print(c.strftime('%Y-%m'), f'{iso[0]}-W{iso[1]:02d}')
EOF
)"
export TM_OOS_START TM_OOS_WSTART
echo "OOS boundary: month>=$TM_OOS_START week>=$TM_OOS_WSTART"

restore() {
    echo "[restore] starting..."
    if [ -d "$ROOT/trade_pickles_PRE_WF" ]; then
        # the truncated set is archived UNDER THE QUARTER NAME so arms can
        # reuse stages B-E without re-running them
        rm -rf "$ROOT/pickle_sets/TRUNC_$QL"
        [ -f "$PD/_WF_MARKER" ] && mv "$PD" "$ROOT/pickle_sets/TRUNC_$QL"
        [ -d "$PD" ] || mv "$ROOT/trade_pickles_PRE_WF" "$PD"
    fi
    git -C "$ROOT" checkout -- 'strategies/unified_cell_comparison.txt' \
        cells/ results/ 2>/dev/null || true
    echo "[restore] OK"
}
trap restore EXIT

echo "== WF $QL | cutoff=$TM_CUTOFF judge=[$JS,$JE) =="
mkdir -p "$OUT"

echo "== A: canonical pickle backup =="
rm -rf "$ROOT/trade_pickles_PRE_WF"
cp -r "$PD" "$ROOT/trade_pickles_PRE_WF"

echo "== B: qs_to_cellcomp (BLIND, cutoff=$TM_CUTOFF) =="
_B0=$(date +%s)
python "$ROOT/factory/qs/qs_to_cellcomp.py" 2>&1 \
    | awk "/admit|-> .*entry|rows|ERROR|Traceback/ $HB"
_BD=$(( $(date +%s) - _B0 ))
echo "cellcomp took: ${_BD}s"
# sanity floor: a suspiciously fast run usually means a stale resume was
# silently reused (the poisoned-cache class of bugs) — fail LOUD instead.
[ "$_BD" -ge 60 ] || { echo "FATAL: cellcomp ${_BD}s — stale-resume suspicion"; exit 1; }

echo "== C: build_cell_files =="
python "$ROOT/factory/portfolio/build_cell_files.py" 2>&1 | awk "/rows|Written|ERROR|Traceback/ $HB"

echo "== D: save_cell_trades ALL (whitelist) =="
rm -rf "$PD/_tmp"
SCT_WHITELIST=1 python "$ROOT/factory/portfolio/save_cell_trades.py" 2>&1 \
    | awk "/-> |DONE|ERROR|Traceback/ $HB"

echo "== E: FULL copy + pickle truncation (cutoff=$TM_CUTOFF) =="
# full-stream copy under the quarter name (the judge reads forward trades here)
rm -rf "$ROOT/pickle_sets/FULL_$QL"
cp -r "$PD" "$ROOT/pickle_sets/FULL_$QL"
TM_CUTOFF="$TM_CUTOFF" WF_PD="$PD" python - <<'EOF'
import os, pickle
from pathlib import Path
import pandas as pd
PD = Path(os.environ['WF_PD'])
CUT = pd.Timestamp(os.environ['TM_CUTOFF'])
n_cut = 0
for p in sorted(PD.glob('*_20*.pkl')):
    d = pickle.load(open(p, 'rb'))
    out = {}
    for label, trades in d.items():
        kept = [t for t in trades if t['entry_time'] < CUT]
        n_cut += len(trades) - len(kept)
        if kept:
            out[label] = kept
    pickle.dump(out, open(p, 'wb'), protocol=4)
open(PD / '_WF_MARKER', 'w').write(str(CUT.date()))
print(f'rows truncated: {n_cut:,}')
EOF

echo "== F: combo chain (blind) =="
python "$ROOT/factory/portfolio/trade_combo.py" --all 10 2>&1 \
    | awk "/GLOBAL|valid|TOP5|DONE|ERROR|Traceback|\\/s\\)/ $HB"
for R in BEAR FLAT BULL; do
    python "$ROOT/factory/portfolio/trade_regime_combo.py" $R 2>&1 \
        | awk "/Saved|DONE|#1 |ERROR|Traceback/ $HB"
done
python "$ROOT/factory/portfolio/trade_full_combo.py" 30 2>&1 \
    | awk "/Saved|DONE|#1 |Evaluated|ERROR|Traceback/ $HB"
python "$ROOT/factory/portfolio/deploy_sweep.py" 2>&1 \
    | awk "/DEPLOY PICK|LEV-RULE|ERROR|Traceback/ $HB"

echo "== G: deploy consensus (truncated data) =="
DC_RESULTS_DIR="results" DC_TRUNC_DIR="trade_pickles" \
    python "$ROOT/factory/portfolio/deploy_consensus.py" 2>&1 \
    | awk "/simulated|WINNER|gate-passing|GLOBAL|lev[0-9]+:|ERROR|Traceback/ $HB"

echo "== H: archive -> results/pwf/band/$QL (BEFORE the judge — ordering lesson) =="
# archiving after the judge once raced a restore and lost a quarter's files
cp "$ROOT"/results/trade_combo_*.txt "$ROOT"/results/trade_regime_*.txt \
   "$ROOT/results/trade_full_combo.txt" "$ROOT/results/deploy_sweep.txt" \
   "$ROOT/results/deploy_consensus.txt" "$ROOT/results/deploy_consensus_all.tsv" \
   "$ROOT/results/cell_summary.txt" "$OUT/" 2>/dev/null || true
# cell menus into the quarter archive (lets arms start from stage F)
mkdir -p "$OUT/cells"
cp "$ROOT"/cells/*.txt "$OUT/cells/" 2>/dev/null || true
# stage-B unified outputs (lets C+D-re-running arms start from the archive)
mkdir -p "$OUT/unified"
cp "$ROOT/strategies/unified_cell_comparison.txt" "$OUT/unified/toyset.txt" 2>/dev/null || true

echo "== I: JUDGE =="
PICK=$(WF_DC="$ROOT/results/deploy_consensus.txt" python - <<'EOF'
import os, re
for line in open(os.environ['WF_DC'], encoding='utf-8'):
    m = re.search(r'WINNER \(gate-passing.*: (B\d+\+F\d+\+BU\d+ lev\d+ \S+ cyc\d+)', line)
    if m:
        print(m.group(1)); break
EOF
)
if [ -z "$PICK" ]; then
    if grep -q "ABSTAIN" "$ROOT/results/deploy_consensus.txt"; then
        echo "ABSTAIN quarter: the procedure proposed no package (live=incumbent)."
        echo "ABSTAIN" > "$OUT/judge.txt"
        echo "WF $QL DONE"
        exit 0
    fi
    echo "FATAL: could not parse the consensus winner"; exit 1
fi
echo "PICK: $PICK"
PWF_FULL_DIR="pickle_sets/FULL_$QL" \
    python "$ROOT/factory/walkforward/pwf_judge.py" "$PICK" "$JS" "$JE" "$OUT" 2>&1 \
    | awk "/PWF|window|trade-level|sim booking|full-stream|ERROR|Traceback/ $HB"
echo "WF $QL DONE"
