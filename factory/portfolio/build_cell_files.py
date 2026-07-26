"""Build cell-level ranked files from all unified results.

Reads each category's unified_cell_comparison.txt, scores every config with
the (TEMPLATE) config formula, filters, writes per-cell txt files sorted by
score. These per-cell menus feed the combo layers.
"""
import sys, os, math
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# One entry per strategy category (the toy set ships a single category).
UNIFIED_FILES = {
    'TOYSET': ROOT / 'strategies' / 'unified_cell_comparison.txt',
}

# TM_WORK: isolated working dir for blind walk-forward arms — redirects all
# writes away from the canonical tree (parallel-safe A/B replays).
_WORK = Path(os.environ.get('TM_WORK', ROOT))
OUT_DIR = _WORK / 'cells'
# TM_UNIFIED_DIR: blind-replay arms read unified files from a quarter ARCHIVE
# instead of the live tree -> read-only, no parallel write collisions.
_UNI = os.environ.get('TM_UNIFIED_DIR')
if _UNI:
    UNIFIED_FILES = {k: Path(_UNI) / f'{k.lower()}.txt' for k in UNIFIED_FILES}

# Config score (TEMPLATE FORMULA) — universe-NORMALIZED magnitudes.
# Significance floors (n, wr, ev) stay on RAW counts; net & n_bonus use
# universe-normalized values. Plug your own formula here; every constant
# below is an EXAMPLE.


def compute_score(is_n, is_wr, is_ev, is_nw, oos_n, oos_wr, oos_ev, oos_nw, sl,
                  is_net_norm, is_n_norm, oos_net_norm, oos_n_norm):
    if is_net_norm <= 0 or oos_net_norm <= 0:
        return None
    if oos_n < 50 or is_wr <= 50:   # significance: RAW counts
        return None
    if oos_wr <= 50:
        return None
    if is_ev < 0.3 or oos_ev < 0.3:
        return None

    risk_adj_net = math.sqrt(is_net_norm) / (1 + sl / 3)
    quality = (oos_wr / 100) * (is_wr / 100)
    stability = 1 / (1 + is_nw / 15 + oos_nw * 1)
    n_bonus = math.log2(max(oos_n_norm, 50))

    return risk_adj_net * quality * stability * n_bonus


def parse_line(line, category):
    """Parse a unified_cell_comparison.txt line."""
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None
    hdr = parts[0].split()
    if len(hdr) < 5:
        return None

    cell = hdr[0]
    strat = hdr[1]
    sl_tp = hdr[2]
    q = int(hdr[3])
    ci = int(hdr[4])

    is_parts = parts[1].split()
    oos_parts = parts[2].split()
    if len(is_parts) < 6 or len(oos_parts) < 6:   # n wr ev nw net_norm n_norm
        return None

    try:
        is_n = int(is_parts[0].replace(',', ''))
        is_wr = float(is_parts[1].rstrip('%'))
        is_ev = float(is_parts[2])
        is_nw = int(is_parts[3])
        is_net_norm = float(is_parts[4])
        is_n_norm = float(is_parts[5])

        oos_n = int(oos_parts[0].replace(',', ''))
        oos_wr = float(oos_parts[1].rstrip('%'))
        oos_ev = float(oos_parts[2])
        oos_nw = int(oos_parts[3])
        oos_net_norm = float(oos_parts[4])
        oos_n_norm = float(oos_parts[5])
    except (ValueError, IndexError):
        return None

    sl = float(sl_tp.split('/')[0])
    tp_str = sl_tp.split('/')[1].rstrip('*')
    tp = float(tp_str)

    feats = parts[3].strip() if len(parts) > 3 else '(all)'

    score = compute_score(is_n, is_wr, is_ev, is_nw, oos_n, oos_wr, oos_ev, oos_nw, sl,
                          is_net_norm, is_n_norm, oos_net_norm, oos_n_norm)
    if score is None:
        return None

    return {
        'cell': cell,
        'category': category,
        'strat': strat,
        'sl_tp': sl_tp,
        'sl': sl,
        'tp': tp,
        'q': q,
        'ci': ci,
        'is_n': is_n,
        'is_wr': is_wr,
        'is_ev': is_ev,
        'is_nw': is_nw,
        'is_net_norm': is_net_norm,
        'is_n_norm': is_n_norm,
        'oos_n': oos_n,
        'oos_wr': oos_wr,
        'oos_ev': oos_ev,
        'oos_nw': oos_nw,
        'oos_net_norm': oos_net_norm,
        'oos_n_norm': oos_n_norm,
        'score': score,
        'feats': feats,
    }


def main():
    print(f'  Loading unified results from {len(UNIFIED_FILES)} categories...', flush=True)

    all_rows = []
    for cat_name, fpath in UNIFIED_FILES.items():
        if not fpath.exists():
            print(f'    {cat_name}: NOT FOUND ({fpath})', flush=True)
            continue
        n = 0
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                row = parse_line(line, cat_name)
                if row is not None:
                    all_rows.append(row)
                    n += 1
        print(f'    {cat_name}: {n} rows', flush=True)

    print(f'\n  Total: {len(all_rows)} scored rows', flush=True)
    # Sentinel: 0 rows = disaster (old cell files would silently survive ->
    # a stale menu leaks downstream). BE LOUD, never write nothing quietly.
    if not all_rows:
        print('  FATAL: 0 scored rows — cells NOT written; a stale menu would '
              'have survived. Check the OOS boundary / input data.', flush=True)
        sys.exit(1)

    # Group by cell
    cells = {}
    for row in all_rows:
        cell = row['cell']
        if cell not in cells:
            cells[cell] = []
        cells[cell].append(row)

    # Sort each cell by score desc
    for cell in cells:
        cells[cell].sort(key=lambda r: -r['score'])

    # Write per-cell files
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    header = (f'{"#":>3} {"Cat":>8} {"Strat":>5} {"SL/TP":>7} {"q":>2} {"ci":>3} |'
              f' {"IS_N":>6} {"IS_WR":>5} {"IS_EV":>7} {"IS_NW":>4} |'
              f' {"OOS_N":>5} {"OOS_WR":>5} {"OOS_EV":>7} {"OOS_NW":>4} |'
              f' {"Score":>10} | Features')

    summary = []
    for cell in sorted(cells.keys()):
        rows = cells[cell]
        fpath = OUT_DIR / f'{cell}.txt'
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(f'Cell: {cell} | {len(rows)} configs\n')
            f.write(f'{header}\n')
            f.write(f'{"="*160}\n')
            for i, r in enumerate(rows):
                f.write(f'{i+1:>3} {r["category"]:>8} {r["strat"]:>5} {r["sl_tp"]:>7} {r["q"]:>2} {r["ci"]:>3} |'
                        f' {r["is_n"]:>6,} {r["is_wr"]:>4.1f}% {r["is_ev"]:>+6.3f} {r["is_nw"]:>4} |'
                        f' {r["oos_n"]:>5,} {r["oos_wr"]:>4.1f}% {r["oos_ev"]:>+6.3f} {r["oos_nw"]:>4} |'
                        f' {r["score"]:>10.1f} | {r["feats"]}\n')

        top_score = rows[0]['score'] if rows else 0
        top_strat = f'{rows[0]["category"]}/{rows[0]["strat"]}' if rows else '-'
        n_cats = len(set(r['category'] for r in rows))
        n_strats = len(set(f'{r["category"]}/{r["strat"]}' for r in rows))
        summary.append((cell, len(rows), n_cats, n_strats, top_score, top_strat))
        print(f'    {cell:<14} {len(rows):>5} rows  {n_cats} cats  {n_strats} strats  top={top_score:>10.1f} ({top_strat})', flush=True)

    # Write summary
    sum_path = OUT_DIR.parent / 'results' / 'cell_summary.txt'
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write(f'{"Cell":<14} {"Rows":>6} {"Cats":>5} {"Strats":>7} {"TopScore":>12} {"TopStrat":<20}\n')
        f.write(f'{"="*70}\n')
        for cell, n_rows, n_cats, n_strats, top_sc, top_st in sorted(summary, key=lambda x: -x[4]):
            f.write(f'{cell:<14} {n_rows:>6} {n_cats:>5} {n_strats:>7} {top_sc:>12.1f} {top_st:<20}\n')

    print(f'\n  Written {len(cells)} cell files to {OUT_DIR}/', flush=True)
    print(f'  Summary: {sum_path}', flush=True)
    print(f'  DONE', flush=True)


if __name__ == '__main__':
    main()
