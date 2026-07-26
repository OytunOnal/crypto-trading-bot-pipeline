#!/bin/bash
# SEQUENTIAL ARM TOURNAMENT (example runner).
# Runs each arm over the elimination quarters ONE AT A TIME with full workers.
#
# Why sequential, not parallel: the combo stage is CPU-bound. Splitting cores
# across parallel chains does not shrink total wall-clock (work/cores is the
# floor) — it only DELAYS the first judge signal until all chains finish
# together, and adds scheduler thrash. Sequential + full workers delivers
# judges one by one, hours earlier. (Measured: 3 parallel throttled chains
# made the combo stage ~3x slower each, with zero total-time benefit.)
#
# Resume-safe: a quarter with an existing judge.txt is skipped.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/results/pwf/_logs"
exec > "$ROOT/results/pwf/_logs/tournament.log" 2>&1

# Arms to race (see pwf_arm_iso.sh for definitions). 'ref' = baseline.
ARMS="ref quota jk crash"
# Elimination quarters: pick a DIVERSE set (a calm/golden quarter, a storm
# quarter, a middling one) — a single regime hides failure modes.
declare -A CUT=( [2025Q3]=2025-04-22 [2026Q2]=2026-01-21 [2025Q4]=2025-07-23 )
declare -A JS=(  [2025Q3]=2025-07-01 [2026Q2]=2026-04-01 [2025Q4]=2025-10-01 )
declare -A JE=(  [2025Q3]=2025-10-01 [2026Q2]=2026-07-01 [2025Q4]=2026-01-01 )
QS="2025Q3 2026Q2 2025Q4"

echo "== TOURNAMENT started ($(date '+%m-%d %H:%M')) =="
for ARM in $ARMS; do
    echo "########## ARM $ARM starting ($(date '+%m-%d %H:%M')) ##########"
    for Q in $QS; do
        OUT="$ROOT/results/pwf/arms/${ARM}/${Q}"
        WS="$ROOT/_ws/${Q}_${ARM}"
        if [ -f "$OUT/judge.txt" ]; then echo "  ${Q}_${ARM} already done, skip"; continue; fi
        [ -d "$WS" ] && { echo "  ${Q}_${ARM} clearing stale ws"; rm -rf "$WS"; }
        echo "  ${Q}_${ARM} starting (SEQUENTIAL, full workers) ($(date '+%H:%M'))"
        mkdir -p "$ROOT/results/pwf/_logs/${ARM}"
        bash "$SCRIPT_DIR/pwf_arm_iso.sh" "$ARM" "$Q" "${CUT[$Q]}" "${JS[$Q]}" "${JE[$Q]}" \
            > "$ROOT/results/pwf/_logs/${ARM}/${Q}.log" 2>&1
        echo "  ${Q}_${ARM} rc=$? done ($(date '+%H:%M'))"
    done
    echo "########## ARM $ARM DONE ($(date '+%m-%d %H:%M')) ##########"
done
echo "########## TOURNAMENT DONE ($(date '+%m-%d %H:%M')) ##########"
