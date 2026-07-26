#!/bin/bash
# ISOLATED ARM RUNNER for blind walk-forward A/B testing.
#   bash pwf_arm_iso.sh <ARM> <QLABEL> <CUTOFF> <JS> <JE> [MAX_WORKERS]
# Output: results/pwf/arms/<ARM>/<QLABEL>/judge.txt
#
# Isolation: TM_WORK redirects every write (cells/, trade_pickles/, results/)
# into an arm-private workspace — the canonical tree is NEVER touched, so
# arms are parallel-safe and a crashed arm cannot corrupt production files.
# The arm reuses the baseline quarter's archived B-E outputs (cells + unified
# + truncated pickles) and re-runs only what its flags change.
set -u -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARM="$1"; QL="$2"; export TM_CUTOFF="$3"; JS="$4"; JE="$5"
export TM_MAX_WORKERS="${6:-0}"   # 0 = auto (cpu-2, RAM-capped)
OUT="$ROOT/results/pwf/arms/${ARM}/${QL}"
WORK="$ROOT/_ws/${QL}_${ARM}"      # isolated workspace
HB='{print; fflush()}'
export TM_WORK="$WORK"
cd "$ROOT"

[ -d "$ROOT/pickle_sets/TRUNC_$QL" ] || { echo "FATAL: pickle_sets/TRUNC_$QL missing (run the baseline quarter first)"; exit 1; }
[ -d "$ROOT/results/pwf/band/$QL/cells" ] || { echo "FATAL: band/$QL/cells missing"; exit 1; }

# OOS = cutoff - 26 weeks (uniform boundary; see procedure_wf_quarter.sh)
read -r TM_OOS_START TM_OOS_WSTART <<< "$(python - <<EOF
import pandas as pd
c = pd.Timestamp('$TM_CUTOFF') - pd.Timedelta(weeks=26)
iso = c.isocalendar()
print(c.strftime('%Y-%m'), f'{iso[0]:d}-W{iso[1]:02d}')
EOF
)"
export TM_OOS_START TM_OOS_WSTART

# ── EXAMPLE ARMS — each toggles ONE mechanism; default (no flag) is the
#    bit-identical baseline path. Add your own hypotheses here.
CD_ARM=0
case "$ARM" in
  ref)   ;;                                          # baseline selection (reference)
  jk)    export TM_JK_BEAM=1 ;;                      # jackknife median-rank menu
  quota) export TM_DIV_QUOTA=1 ;;                    # diversity-quota menu
  crash) export TM_DIVCRASH=1 ;;                     # co-crash penalty rerank
  fixed) export TM_FIX_LEV=6 TM_FIX_CAUT="8/15/22D" TM_FIX_CYC=25 ;;  # fixed risk knobs
  *)     echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac

echo "== ISOLATED ARM $ARM | $QL | WORK=$WORK | maxw=$TM_MAX_WORKERS =="
rm -rf "$WORK"
mkdir -p "$WORK/results" "$WORK/cells" "$OUT"

echo "== A: isolated workspace setup =="
if [ "$CD_ARM" = "1" ]; then
    # C+D-re-running arms: copy the canonical pickles (D overwrites, E cuts)
    cp -r "$ROOT/trade_pickles" "$WORK/trade_pickles"
    # CRITICAL: the canonical copy may carry a stale _tmp resume cache; the
    # resume would silently SKIP whole task groups and starve the menu ~100x
    # (measured incident) — always clear it in the copy.
    rm -rf "$WORK/trade_pickles/_tmp"
    export TM_UNIFIED_DIR="$ROOT/results/pwf/band/$QL/unified"
    echo "== C: build_cell_files ($ARM) =="
    python "$ROOT/factory/portfolio/build_cell_files.py" 2>&1 | awk "/rows|Total:|FATAL|Traceback/ $HB"
    echo "== D: save_cell_trades (full-stream pkl) =="
    SCT_WHITELIST=1 python "$ROOT/factory/portfolio/save_cell_trades.py" 2>&1 | awk "/-> |tasks|DONE|Traceback/ $HB"
    # arm-FULL snapshot BEFORE the cut (the judge reads forward trades here)
    cp -r "$WORK/trade_pickles" "$WORK/armfull"
    FULLDIR="_ws/${QL}_${ARM}/armfull"
    echo "== E: pickle truncation (selection blindness) =="
    TM_CUTOFF="$TM_CUTOFF" WF_PD="$WORK/trade_pickles" python - <<'EOF'
import os, pickle
from pathlib import Path
import pandas as pd
PD = Path(os.environ['WF_PD'])
CUT = pd.Timestamp(os.environ['TM_CUTOFF'])
for p in sorted(PD.glob('*_20*.pkl')):
    d = pickle.load(open(p, 'rb'))
    out = {}
    for label, trades in d.items():
        kept = [t for t in trades if t['entry_time'] < CUT]
        if kept:
            out[label] = kept
    pickle.dump(out, open(p, 'wb'), protocol=4)
EOF
else
    # F-only arms: truncated pickles + archived cell menus into the workspace
    cp -r "$ROOT/pickle_sets/TRUNC_$QL" "$WORK/trade_pickles"
    cp "$ROOT/results/pwf/band/$QL/cells/"*.txt "$WORK/cells/"
    FULLDIR="pickle_sets/FULL_$QL"
fi

echo "== F: combo chain ($ARM) =="
python "$ROOT/factory/portfolio/trade_combo.py" --all 10 2>&1 | awk "/GLOBAL|valid|JK-rerank|DIVCRASH|DONE|Traceback/ $HB"
for R in BEAR FLAT BULL; do
    python "$ROOT/factory/portfolio/trade_regime_combo.py" $R 2>&1 | awk "/Saved|DONE|Traceback/ $HB"
done
python "$ROOT/factory/portfolio/trade_full_combo.py" 30 2>&1 | awk "/Saved|DONE|Evaluated|Traceback/ $HB"

echo "== SWEEP + CONSENSUS =="
python "$ROOT/factory/portfolio/deploy_sweep.py" 2>&1 | awk "/DEPLOY PICK|swept|Traceback/ $HB"
DC_RESULTS_DIR="_ws/${QL}_${ARM}/results" DC_TRUNC_DIR="_ws/${QL}_${ARM}/trade_pickles" \
    python "$ROOT/factory/portfolio/deploy_consensus.py" 2>&1 \
    | awk "/WINNER|gate-passing|FALLBACK|Traceback/ $HB"

echo "== ARCHIVE + JUDGE =="
cp "$WORK/results/"*.txt "$WORK/results/"*.tsv "$OUT/" 2>/dev/null || true
PICK=$(WF_DC="$WORK/results/deploy_consensus.txt" python - <<'EOF'
import os, re
for line in open(os.environ['WF_DC'], encoding='utf-8'):
    m = re.search(r'WINNER \(gate-passing.*: (B\d+\+F\d+\+BU\d+ lev\d+ \S+ cyc\d+)', line)
    if m: print(m.group(1)); break
EOF
)
if [ -z "$PICK" ]; then
    grep -q "ABSTAIN" "$WORK/results/deploy_consensus.txt" && { echo "ABSTAIN" > "$OUT/judge.txt"; echo "ISOLATED ARM $ARM $QL DONE (ABSTAIN)"; exit 0; }
    echo "FATAL: no winner"; exit 1
fi
echo "PICK: $PICK"
PWF_FULL_DIR="$FULLDIR" PWF_BASE_DIR="$WORK/results" \
    python "$ROOT/factory/walkforward/pwf_judge.py" "$PICK" "$JS" "$JE" "$OUT" 2>&1 \
    | awk "/PWF|window|trade-level|sim booking|full-stream|Traceback/ $HB"
echo "ISOLATED ARM $ARM $QL DONE"
