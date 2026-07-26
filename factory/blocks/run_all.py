"""Full optimization pipeline orchestrator (2026-06-12).

Per strategy, runs the accepted common engines in order:
    1. gate_sweep   (parallel: per (direction, cell) worker, cell-streamed)
    2. phase2       (serial cells, parallel configs)
    3. qs_features  (anti-overfit QS selection, FWD+REV WF gates)

IS discipline lives inside the engines (2021-2025; 2026 never selected
on). Strategies run SERIALLY — each engine parallelizes internally, so
one strategy already saturates the worker pool; stacking strategies
would only multiply RAM.

Resume: each engine writes per-cell incremental JSON and skips
completed cells, so re-running after an interrupt continues in place.

Usage:
  python factory/blocks/run_all.py --strategies ALL
  python factory/blocks/run_all.py --strategies SMA_X,RSI_MR --steps gate,phase2
  python factory/blocks/run_all.py --strategies ALL --steps qs
"""
import sys, os, time, argparse, traceback
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factory.data.registry import STRATEGIES

ALL_STEPS = ['gate', 'phase2', 'qs']


def run_step(step, code, max_workers):
    if step == 'gate':
        from factory.gates.gate_sweep import run_parallel
        run_parallel(code, max_workers=max_workers)
    elif step == 'phase2':
        from factory.blocks.phase2 import run as p2_run
        p2_run(code, max_workers=max_workers)
    elif step == 'qs':
        from factory.qs.qs_features import run as qs_run
        qs_run(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategies', required=True,
                    help='ALL or comma-separated list (SMA_X,RSI_MR,...)')
    ap.add_argument('--steps', default='gate,phase2,qs')
    ap.add_argument('--max-workers', type=int, default=None)
    args = ap.parse_args()

    if args.strategies.upper() == 'ALL':
        codes = sorted(STRATEGIES)
    else:
        codes = [c.strip().upper() for c in args.strategies.split(',')]
    steps = [s.strip() for s in args.steps.split(',')]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        ap.error(f'unknown step: {bad} (valid: {ALL_STEPS})')

    # STAGE-MAJOR: ALL gates first, then ALL phase2, then ALL QS. A stage
    # boundary is a natural checkpoint; each stage has a homogeneous
    # resource profile.
    t0 = time.time()
    failed = []
    for step in steps:
        print('\n' + '#' * 120, flush=True)
        print(f'  STAGE: {step.upper()} ({len(codes)} strategies)', flush=True)
        print('#' * 120, flush=True)
        if step == 'gate':
            # global queue: while one strategy's giant cells are CPU-bound
            # the next strategies' small cells fill idle RAM/cores
            from factory.gates.gate_sweep import run_parallel_multi
            try:
                run_parallel_multi(codes, max_workers=args.max_workers)
            except Exception:
                # if gate fails, phase2/QS are MEANINGLESS — do NOT continue
                # with missing gates (a corrupt shard once leaked into phase2)
                print(f'  [gate:MULTI] FATAL:\n{traceback.format_exc()}',
                      flush=True)
                sys.exit(2)
            print(f'\n  == STAGE GATE DONE '
                  f'({(time.time()-t0)/3600:.1f}h) ==', flush=True)
            continue
        if step == 'phase2':
            from factory.blocks.phase2 import run_multi as p2_multi
            try:
                p2_multi(codes, max_workers=args.max_workers)
            except Exception:
                print(f'  [phase2:MULTI] FATAL:\n{traceback.format_exc()}',
                      flush=True)
                sys.exit(2)
            print(f'\n  == STAGE PHASE2 DONE '
                  f'({(time.time()-t0)/3600:.1f}h) ==', flush=True)
            continue
        if step == 'qs':
            from factory.qs.qs_features import run_multi as qs_multi
            try:
                qs_multi(codes, max_workers=args.max_workers)
            except Exception:
                print(f'  [qs:MULTI] FATAL:\n{traceback.format_exc()}',
                      flush=True)
                sys.exit(2)
            print(f'\n  == STAGE QS DONE '
                  f'({(time.time()-t0)/3600:.1f}h) ==', flush=True)
            continue
        for i, code in enumerate(codes):
            print(f'\n  >>> {step} {i+1}/{len(codes)}: {code} '
                  f'(elapsed {(time.time()-t0)/60:.0f}min)', flush=True)
            try:
                ts = time.time()
                run_step(step, code, args.max_workers)
                print(f'  [{step}:{code}] OK ({(time.time()-ts)/60:.1f}min)',
                      flush=True)
            except Exception:
                failed.append(f'{step}:{code}')
                print(f'  [{step}:{code}] ERROR:\n{traceback.format_exc()}',
                      flush=True)
        print(f'\n  == STAGE {step.upper()} DONE '
              f'({(time.time()-t0)/3600:.1f}h) ==', flush=True)

    print('\n' + '=' * 120, flush=True)
    print(f'  RUN_ALL done: {len(codes)*len(steps)-len(failed)}/'
          f'{len(codes)*len(steps)} steps OK '
          f'({(time.time()-t0)/3600:.1f}h)', flush=True)
    if failed:
        print(f'  FAILED: {failed}', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
