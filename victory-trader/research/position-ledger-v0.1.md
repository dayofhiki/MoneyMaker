# Frozen position-ledger audit v0.1 — 2026-09-19

Objective: establish honest capital accounting before changing the predictor.
Replay request 48 frozen decisions using the existing January–March state panels.
No training, new months, quote purchases, brokerage actions, or gate tuning.

## Rules fixed before output
- Compare original policies in each original month and forward March separately.
  LOMO January/February remain future-trained development diagnostics.
- Independent $10,000 synthetic account per month/policy/cost scenario; fixed
  $1,000 notional fractional orders; no leverage, top-ups or daily cash reset.
  These are accounting units, not a recommendation or a simulation of the user's
  actual account. Simultaneous signals use ticker alphabetical order.
- Decisions known at bar close, next-minute open as zero-latency entry reference.
  Reservation occurs before entry-reference availability is checked. Missing
  entry references expire the synthetic order; this is NOT evidence of no fill.
- Exit uses the scheduled 10-minute bar close, available one minute after bar
  start. A missing close leaves exit pending until the next observed open.
  No hindsight last-bar liquidation. Preserve unresolved positions and capital
  across sessions through the available monthly panel; do not invent overnight
  observations or silently reset unresolved capital at a new trading day.
- Re-entry in a held ticker is blocked; insufficient cash rejects an order.
  All attempts retained, including rejections and missing entry references.
- Original light/base/stress execution assumptions are unchanged.
- Report realized P&L, unresolved committed capital, stale marked equity,
  observed-mark drawdown, delayed exits, cash/position blocks and turnover.
  Stale marks and bar fills cannot establish realizability or actual drawdown.
  Missing bars do not establish a halt, no trading, or a provider outage.
- Deployment promotion is false for every result. Even complete accounting is
  not validation of alpha. This audit does not enable continuous live trading.

## Next research decision
Use the replay to quantify exit-label selection and capital contention first.
Then construct causal training labels from this explicit exit rule, with missing
entry/unresolved outcomes reported separately; fit January/February and evaluate
March only as reused development. Compare to frozen signals under the SAME
capital and cost assumptions. Do not optimize gates on this audit's outcomes.
A continuous entry model still requires an independently specified decision
schedule and careful execution validation; no profitability claim follows here.

## Local verification
Eight behavioral tests cover cash conservation, delayed exits, overnight capital
lock, same-ticker blocking, missing references, future-price independence of
acceptance, friction, and duplicate data rejection.
