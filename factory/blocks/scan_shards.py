"""Parallel cache-shard integrity scan: try-unpickle every monthly shard.

Usage: python factory/blocks/scan_shards.py [--workers 12]
Output: bad files list -> factory/blocks/bad_shards.json
"""
import sys, os, time, pickle, json, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _check(path_str):
    try:
        with open(path_str, 'rb') as f:
            pickle.load(f)
        return path_str, None
    except Exception as e:
        return path_str, f'{type(e).__name__}: {str(e)[:80]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    from factory.data.registry import STRATEGIES, BACKTEST
    seen, files = set(), []
    for code, (rel, _sig, _cat) in STRATEGIES.items():
        d = (BACKTEST / Path(rel).parent / 'cache').resolve()
        if d in seen or not d.exists():
            continue
        seen.add(d)
        files.extend(str(f) for f in d.glob('*_t?v?.pkl'))
    print(f'{len(files)} shards to scan ({args.workers} workers)', flush=True)

    t0 = time.time()
    bad = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_check, f) for f in files]
        for i, fut in enumerate(as_completed(futs)):
            p, err = fut.result()
            if err:
                bad.append({'file': p, 'error': err})
                print(f'  CORRUPT: {p} ({err})', flush=True)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(files)} ({time.time()-t0:.0f}s, '
                      f'{len(bad)} corrupt)', flush=True)

    out = Path(__file__).parent / 'bad_shards.json'
    json.dump(bad, open(out, 'w'), indent=1)
    print(f'\nSCAN DONE: {len(bad)} corrupt / {len(files)} files '
          f'({(time.time()-t0)/60:.1f}min) -> {out}', flush=True)


if __name__ == '__main__':
    main()
