# R297 — frozen-entry second-resolution HOLD/EXIT

R296 confirmed a small relative entry edge on untouched June15,16,17,18,22.
Its negative access utility is not realized P&L. Selection and entry remain
frozen; June is sealed for HOLD development and July-August remain unopened.

## Preregistered design

- Reuse May5-8 R294 OOF entry scores and the exact R296 action thresholds:
  utility >= -2.00886730522822 and turn >= 0.12298633907918213.
- Strong-value override stays disabled. No entry model, feature, or threshold tuning.
- These are cross-fitted development entry scores, not the all-May fitted model
  used to score untouched June. The conditioning experiment avoids in-sample entry
  scores, but upstream training is not fully nested in HOLD folds.
- Inspect every completed active second after the actual frozen entry.
- Features: observed P&L, peak/trough, deterioration/recovery, local returns,
  volume/transactions, event gaps, and frozen entry scores. Future opens, returns,
  maximum favorable excursion and targets are labels only.
- Fit two fixed continuation-advantage regressors: sell after 60s or 300s minus
  EXIT now, accounting for the same position's modeled liquidation value.
  Primary dynamic policy is 300s; 60s is sensitivity, not a winner-selection grid.
- HOLD when predicted liquidation advantage is positive; otherwise EXIT at the
  next observable second open, with maximum 3s execution lag. Re-evaluate every
  observed second. No best-future-peak target and no action threshold search.
- Common risk bound: existing -2.5% modeled net stop, triggered by completed
  closes, executed at following open (gap losses are paid). HOT+60m/session end
  is a research observation cap, not a ten-minute trade timeout.
- Compare immediate, 60s/300s/600s fixed holds, old 10m/2% close trailing,
  terminal hold, and dynamic policies on identical entries. Old rule is replayed
  at second resolution; it is not mislabeled as the original minute replay.

## Reporting and interpretation

Report light/base/stress friction-adjusted returns, daily means, CVaR5,
observed-close MAE, resolution, hold duration, matched gain vs old rule with
trading-day/ticker-day bootstrap intervals, +5/+10/+20 executable-open opportunity
capture and upside left after exit. MFE uses executable opens, not bar highs.
Skipped entries stay cash; unresolved positions never become cash. Candidate
cash-adjusted mean is withheld if any entered outcome is unresolved.

Primary research gate: >=95% position resolution, positive BASE mean, positive
matched gain vs old rule, positive BASE mean on >=3/4 days. A failed gate remains
a successfully completed experiment. No claim about portfolio account returns.

Four reused development days and non-nested upstream OOF scores cannot validate
profitability. June HOLD validation may be scored once only after HOLD is frozen;
no policy changes based on fresh results. No new workflows or date sweeps.
