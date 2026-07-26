"""Daily live-vs-replica reconciliation (post-deploy monitoring, cron-able).

Independent check: every live-opened trade for a day MUST reappear as a gate-passed
candidate when the FRESH cache is replayed through the LIVE pipeline (replay_window
_fast). Catches any live/backtest divergence (signal, gate, universe, ordering) that
unit tests can't -- the reference is recomputed independently, not a stored artifact.

Also reports per-signal opened-count drift (live vs replica gate-passed pool); a large
shift (e.g. the coin-major QS bug: live HHHL 95 vs backtest 39) flags here.

Live trades CSV (one row per opened trade):
    symbol,position_type,entry_time(ISO),cell,signal,config_key
(produced by a 'pull live trades' step from the bot's trade store).

  python daily_live_reconcile.py --date 2026-06-23 --live-csv _live_0623.csv
  exit 0 if match >= threshold, else 1 (alert).
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factory.qs.kline_frames import MiniKlineCache as KlineCache
from factory.qs.live_config import CACHE_BARS, QS_FEATURES_JSON
from factory.qs.qs_replay import replay_window_fast

MATCH_THRESHOLD = 0.98   # alert if < 98% of live trades reproduce in the replica


def naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert('UTC').tz_localize(None) if ts.tzinfo is not None else ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)              # YYYY-MM-DD
    ap.add_argument('--live-csv', required=True)
    ap.add_argument('--cand-csv', default=None,
                    help='live candidate log (candidates_DATE.csv) -> enables the '
                         'bidirectional gate-level reverse check')
    ap.add_argument('--qs-history', default=None,
                    help='qs_trade_history.parquet (pulled from server) -> enables '
                         'the QS-level reverse check (replica QS recomputed from it)')
    ap.add_argument('--backtest-dir',
                    default=str(ROOT / 'trade_pickles'),
                    help='save_cell_trades trade_pickles dir -> adds the BACKTEST '
                         'reference (QS-narrowed). "" to disable. NOTE: ~1 day behind '
                         '(trades need 24h SL/TP resolution) -> run on a resolved day.')
    ap.add_argument('--cache-dir', default=str(ROOT / 'data' / 'backtest' / 'cache'))
    args = ap.parse_args()
    day = pd.Timestamp(args.date)

    live = []
    for line in open(args.live_csv):
        p = line.strip().split(',')
        if len(p) < 6:
            continue
        live.append({'coin': p[0], 'dir': p[1].upper(), 'fill': naive(p[2]),
                     'cell': p[3], 'sig': p[4].lower(), 'bar': naive(p[2]).floor('5min')})
    live = [t for t in live if day <= t['fill'] < day + pd.Timedelta(days=1)]
    print('live opened %s: %d' % (args.date, len(live)), flush=True)
    if not live:
        print('no live trades -> nothing to reconcile'); return 0

    # fresh cache (whitelist + BTC)
    wl = set(l.strip() for l in open(ROOT / 'config' / 'coin_whitelist.txt', encoding='utf-8') if l.strip())
    wl.add('BTCUSDT')
    cdir = Path(args.cache_dir)
    kc = KlineCache(connector=None, max_bars=CACHE_BARS)
    for c in wl:
        f = cdir / f'{c}_5m_20260101_20261231.parquet'
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        kc._closed[(c, '5m')] = df.tail(CACHE_BARS)
    frames = kc.frames('5m'); btc = frames['BTCUSDT']

    # Cap live trades at the replay's data coverage (last cached bar). The bot keeps
    # trading past our last kline download, so trades after the cutoff are DATA LAG,
    # not a parity divergence -- the replica structurally can't reproduce a bar it
    # doesn't have (re-downloading to chase is futile: there's always a few-hours gap).
    cov_hi = btc.index[-1]
    _n0 = len(live)
    live = [t for t in live if t['bar'] <= cov_hi]
    if len(live) < _n0:
        print('  capped %d live trade(s) after cache cutoff %s (data lag, not parity)'
              % (_n0 - len(live), str(cov_hi)[:16]), flush=True)
    if not live:
        print('no live trades within cache coverage -> nothing to reconcile'); return 0

    lo = day - pd.Timedelta(days=12)   # backfill: prior-day rows build the >=N_ROLL
    hi = min(btc.index[-1], day + pd.Timedelta(days=1))   # chronological QS-rank window before `day`
    bars = [b for b in btc.index if lo <= b <= hi]
    qsj = json.load(open(QS_FEATURES_JSON, encoding='utf-8'))
    rows = replay_window_fast(frames, btc, bars, qsj, age_gate=True, gate_only=True)
    rows_gb = [r for r in rows if r.get('_block')]   # gate+block-passed (block flag set by gate_only)
    idx = defaultdict(set)
    for r in rows_gb:
        idx[(r['coin'], r['cell'], r['signal'], r['direction'])].add(r['entry_time'])

    matched, miss = 0, []
    for t in live:
        cands = {t['bar'], t['bar'] - pd.Timedelta(minutes=5), t['bar'] + pd.Timedelta(minutes=5)}
        if idx.get((t['coin'], t['cell'], t['sig'], t['dir']), set()) & cands:
            matched += 1
        else:
            miss.append(t)
    rate = matched / len(live)
    print('LIVE in REPLICA: %d/%d (%.1f%%)' % (matched, len(live), 100 * rate), flush=True)

    # per-signal drift (live opened vs replica gate-passed pool, same day)
    repl_sig = defaultdict(int)
    for r in rows_gb:
        if day <= r['entry_time'] < day + pd.Timedelta(days=1):
            repl_sig[(r['direction'], r['signal'])] += 1
    live_sig = defaultdict(int)
    for t in live:
        live_sig[(t['dir'], t['sig'])] += 1
    print('per-signal (live opened / replica gate-passed pool):')
    for k in sorted(set(live_sig) | set(repl_sig)):
        print('   %-5s %-10s live=%-4d replica_pool=%d' % (k[0], k[1], live_sig.get(k, 0), repl_sig.get(k, 0)))

    for t in miss[:15]:
        print('   MISS', t['coin'], t['dir'], t['sig'], str(t['fill'])[:19])

    # ===== multi-reference parity: live <-> replica <-> backtest =====
    # Forward (above) = live-OPENED ⊆ replica (one-way). Now the full bidirectional
    # SET check at the gate+block and QS levels across three INDEPENDENT references:
    #   live     = deployed bot's candidate log            (--cand-csv)
    #   replica  = live pipeline replayed on fresh klines  (rows, computed above)
    #   backtest = offline save_cell_trades trade_pickles  (--backtest-dir, QS-narrowed)
    # spread/max_conc/held are downstream of gate+QS -> they never false-trigger here.
    rev_alert = False
    _offs = (pd.Timedelta(0), pd.Timedelta(minutes=5), pd.Timedelta(minutes=-5))

    def _cov(k, ref):
        c, ce, s, d, b = k
        return any((c, ce, s, d, b + o) in ref for o in _offs)

    def _bidir(la, sa, lb, sb, thresh):
        """Print A<->B set diff (±1 bar tolerance) + return True if over thresh."""
        a_only = [k for k in sa if not _cov(k, sb)]
        b_only = [k for k in sb if not _cov(k, sa)]
        print('  %-8s=%-4d %-8s=%-4d | %s-only=%d %s-only=%d'
              % (la, len(sa), lb, len(sb), la, len(a_only), lb, len(b_only)))
        for k in sorted(a_only)[:8]:
            print('      %s-only' % la, k[0], k[3], k[2], str(k[4])[:16])
        for k in sorted(b_only)[:8]:
            print('      %s-only' % lb, k[0], k[3], k[2], str(k[4])[:16])
        return (len(a_only) + len(b_only)) / max(len(sa), len(sb), 1) > thresh

    # replica: gate-only (all gate-passed) + gate+block (block-passed) sets
    def _rset(rs):
        return {(r['coin'], r['cell'], r['signal'].lower(), r['direction'],
                 naive(r['entry_time']).floor('5min')) for r in rs
                if day <= r['entry_time'] < day + pd.Timedelta(days=1)}
    repl_gate_only = _rset(rows)      # gate-passed (incl. block-failed)
    repl_gate = _rset(rows_gb)        # gate+block-passed
    # live (from candidate log)
    live_gate = live_qs = None
    if args.cand_csv and Path(args.cand_csv).exists():
        lc = []
        for ln in open(args.cand_csv):
            p = ln.rstrip('\n').split(',')
            if len(p) < 7 or p[0] == 'bar':
                continue
            b = naive(p[0])
            if day <= b < day + pd.Timedelta(days=1) and b <= cov_hi:   # day-bounded + cache-cap
                lc.append((p[1], p[2], p[3].lower(), p[4].upper(), b.floor('5min'), p[5] == '1'))
        live_gate = {(c, ce, s, d, b) for (c, ce, s, d, b, _q) in lc}
        live_qs = {(c, ce, s, d, b) for (c, ce, s, d, b, q) in lc if q}
    # replica QS: self-contained chronological replay over the BACKFILL window. The
    # prior-day rows build the >=N_ROLL rolling-rank history before `day`, so we DON'T
    # seed from the pulled live qs_history -- that file is rolling-window-trimmed, so its
    # <day slice can be <N_ROLL -> score()=NaN -> false QS mismatches. Same full-stream
    # idea as verify_qs_math; bt_qs (pickles) stays the authoritative QS reference.
    from factory.qs.live_scorer import LiveQualityScore as HybridMixQualityScore
    sc = HybridMixQualityScore()
    meta = {'direction', 'cell', 'signal', 'coin', 'entry_time', 'pnl_pct', '_block'}
    repl_qs = set()
    _dhi = day + pd.Timedelta(days=1)
    for r in sorted(rows_gb, key=lambda x: (x['entry_time'], x['coin'])):  # gate+block only,
        feats = {k: v for k, v in r.items() if k not in meta}            # order == backtest
        qp, _ = sc.score(r['direction'], r['cell'], r['signal'], feats)  # (resolve_coins sorted)
        sc.accumulate(r['direction'], r['cell'], r['signal'], feats,
                      entry_time=r['entry_time'].value / 1e9, coin=r['coin'])
        if qp and day <= r['entry_time'] < _dhi:
            repl_qs.add((r['coin'], r['cell'], r['signal'].lower(), r['direction'],
                         naive(r['entry_time']).floor('5min')))
    # backtest QS (trade_pickles, QS-narrowed; ~1 day behind due to SL/TP resolution)
    bt_qs = None
    if args.backtest_dir:
        import pickle as _pkl
        from factory.qs.live_config import CELL_CONFIG
        btd = Path(args.backtest_dir); _pc = {}; bt_qs = set()
        for (d, cell, sig), cfg in CELL_CONFIG.items():
            fn = f'{cell}_{day.year}.pkl'
            if fn not in _pc:
                fp = btd / fn
                _pc[fn] = _pkl.load(open(fp, 'rb')) if fp.exists() else {}
            lbl = ('L' if d == 'LONG' else 'S') + ':' + cfg['qs_key'][len(f'{d}_{cell}_'):]
            for t in _pc[fn].get(lbl, []):
                b = naive(t['entry_time'])
                # save_cell_trades' label ({dir}:{strat}_q{q}_c{ci}) is NOT cell-unique:
                # the same (strat,q,ci) index exists in several cells and its trades all
                # merge (last-wins) into ONE cell file. So this deployed-cell bucket also
                # holds OTHER cells' trades -> require t['cell']==cell so bt_qs counts only
                # trades the live bot would actually fire in THIS cell. (Without it, 07-02
                # BULL_MED S:HURST_q2_c0 showed 50 phantom FLAT_HIGH trades -> bt_qs 98 vs
                # live 49; with it, 48 ~= 49.)
                # restrict backtest to the live WHITELIST universe (pickles span the
                # full 12mo+ backtest universe; the live only trades coin_whitelist.txt,
                # so unfiltered bt_qs shows non-whitelist coins as false backtest-only)
                if (day <= b < day + pd.Timedelta(days=1) and b <= cov_hi
                        and t['coin'] in wl and t['cell'] == cell):
                    # day-bounded (< day+1) AND cache-capped (<= cov_hi): for a PAST day the
                    # day bound limits to the day; for the CURRENT partial day cov_hi trims the
                    # post-cutoff lag. (Using cov_hi alone leaked the next day's trades into a
                    # past-day bt_qs -> 114 vs live's 18.)
                    bt_qs.add((t['coin'], cell, sig.lower(), d, b.floor('5min')))

    # GATE level: live candidate log records gate+block-passed (no gate-only column), so
    # compare it to the replica GATE-ONLY set. live-only here = live gate+block-passed that
    # the replica did NOT even gate-pass -> a real GATE divergence. replica-only = replica
    # gate-passed the live didn't surface (block-filtered in live OR gate divergence).
    print('\n=== GATE level (gate-passed; live=gate+block log vs replica gate-only) ===')
    if live_gate is not None:
        if _bidir('live', live_gate, 'repl_gate', repl_gate_only, 0.02):
            rev_alert = True
    print('  block effect on replica: %d gate -> %d gate+block (%d dropped by block)'
          % (len(repl_gate_only), len(repl_gate), len(repl_gate_only) - len(repl_gate)))

    print('\n=== BLOCK level (gate+block-passed sets, ±1 bar) ===')
    if live_gate is not None:
        if _bidir('live', live_gate, 'replica', repl_gate, 0.02):
            rev_alert = True
    else:
        print('  replica=%d (live gate skipped: pass --cand-csv)' % len(repl_gate))

    print('\n=== QS level (QS-passed sets, ±1 bar) ===')
    if bt_qs is not None and len(bt_qs) == 0:
        print('  NOTE: backtest empty for %s (unresolved day / rebuild cache+save_cell_trades) -> excluded' % args.date)
        bt_qs = None
    qs_refs = [(n, s) for n, s in (('live', live_qs), ('replica', repl_qs), ('backtest', bt_qs)) if s is not None]
    if len(qs_refs) < 2:
        print('  (need >=2 QS refs; have: %s)' % (', '.join(n for n, _ in qs_refs) or 'none'))
    else:
        for i in range(len(qs_refs)):
            for j in range(i + 1, len(qs_refs)):
                if _bidir(qs_refs[i][0], qs_refs[i][1], qs_refs[j][0], qs_refs[j][1], 0.05):
                    rev_alert = True

    fail = rate < MATCH_THRESHOLD or rev_alert
    if fail:
        print('\nALERT: parity issue (fwd match %.1f%%, reverse_alert=%s) -> investigate'
              % (100 * rate, rev_alert))
        return 1
    print('\nOK: live<->replica parity green (fwd %.1f%%, bidirectional clean)' % (100 * rate))
    return 0


if __name__ == '__main__':
    sys.exit(main())
