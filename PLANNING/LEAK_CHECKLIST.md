# LEAK-CHECKLIST — applied to EVERY copied file before commit

## Per-file manual pass
1. **Measured results in comments** — remove/genericize: $ amounts, IC values, DD
   percentages from real runs, "user 2026-.." decision notes, quarter-specific findings.
2. **Strategy identity** — real strategy names, substrategy dirs, signal params,
   SL/TP sets, q/ci config labels → toy equivalents only.
3. **Feature names** — real feature identifiers → generic (feature_a / toy features).
4. **Formulas & constants** — Sv5 and scoring constants → TEMPLATE_FORMULA reference;
   gate thresholds, caution ladders, cyc values → config/example.yaml with EXAMPLE marker.
5. **Universe** — real coin whitelist → BTC/ETH/SOL example trio.
6. **Paths** — `C:/Users/hoyti/...`, OneDrive, absolute paths → relative/`Path(__file__)`.
7. **Ops identity** — server names, Tailscale hosts, panel URLs, Docker/image names,
   account/exchange credentials or refs → OUT.
8. **Language** — Turkish comments → English (rewrite while stripping).
9. **Import-clean** — file must pass `python -m compileall` in the new repo layout
   (fix imports to new module paths; stub missing deps if needed).

## Final automated sweep (before any push)
Grep the whole repo for suspicious tokens; ALL must be zero hits:
```
hoyti | OneDrive | tailscale | ts\.net | totalmix | TD7 | Sv5 | Sv6 | Sv7 | Sv8
SUPTR | HHHL | ZSCORE | VWAP_D | RSI_VOL | FISH | STOCH | DONCH | EMA_A | EMA_AC
VOL_BRK | CVD_D | ICHI | HURST | SWING | ADX(?=:) | \$[0-9]{2,3}[,.]?[0-9]{3}
IC=|IC = | B[0-9]+\+F[0-9]+\+BU[0-9]+ | 8/15/22 | 5/12/20 | pwf_ | PWFTRUNC | PWFFULL
api[_-]?key = | secret
```
(Token list to be extended during cleaning as new identifiers surface.)

## Process rules
- Never copy `.git`, pickles, parquet, logs, results/, `.env`.
- Copy stage-by-stage; each stage gets its own commit AFTER checklist pass.
- Push only when user explicitly approves final review.
