# Request283 — shortlist-conditioned pullback risk audit

## Result

Request283 completed successfully and confirmed the conditioning failure
suspected after Request282.

The stage-2 severe-loss signal is useful on the full pullback population, but
most of that discrimination disappears inside the exact stage-1 top-20
shortlist where the veto is actually applied.

No trading action was changed and no new dates were opened.

## Global pullback population

Resolved pullback trades: 259

- severe-loss prevalence: 52.12%
- severe-loss AUC: 0.6713
- average precision: 0.7024
- AP lift over prevalence: +0.1812
- rejected trade mean: -2.2992%
- accepted trade mean: -1.0252%
- rejected severe rate: 69.64%
- accepted severe rate: 38.78%
- accepted mean advantage: +1.2740pp

Across all pullbacks, the risk score is genuinely useful and approximately
monotonic. The highest-risk quartile averaged -2.81% with a 73.85% severe-loss
rate.

## Conditioned on the stage-1 shortlist

Resolved shortlisted pullback trades: 32

- severe-loss prevalence: 37.50%
- severe-loss AUC: 0.5667
- average precision: 0.4311
- AP lift over prevalence: +0.0561
- rejected trade mean: -0.5871%
- accepted trade mean: -0.6657%
- rejected severe rate: 40.00%
- accepted severe rate: 36.36%
- accepted mean advantage: -0.0786pp

The preregistered conditioned-signal gate failed.

Risk quartiles inside the shortlist were strongly non-monotonic:

- Q1, lowest predicted risk: trade mean -0.9126%, severe 37.5%
- Q2: trade mean +0.0528%, severe 12.5%
- Q3: trade mean -1.6841%, severe 75.0%
- Q4, highest predicted risk: trade mean -0.0206%, severe 25.0%

This explains Request282. The global risk model learned patterns that separate
obvious failures from ordinary pullbacks. Stage 1 already removes many of
those obvious failures, leaving a harder conditional population in which the
same score no longer orders outcomes reliably.

## Interpretation

The problem is not the 50% veto threshold. A different threshold cannot repair
an AUC near 0.57 or the non-monotonic conditioned quartiles.

The next intervention should adapt the stage-2 model to the stage-1 domain
before adding ENTER/WAIT/REJECT complexity.

## Next boundary — Request284

Keep the frozen stage-1 selector and downstream execution policy. Rebuild only
the stage-2 severe-risk representation/model so it knows that it is operating
inside the stage-1 shortlist.

Use the same outer May5-8 folds and compare:

1. Request283 global severe-risk model;
2. a shortlist-emphasized severe-risk model, trained on all resolved pullbacks
   but with higher training weight on stage-1-shortlisted states;
3. include the causal stage-1 predicted value and margin above its threshold as
   stage-2 context.

Evaluate only shortlist-conditioned OOF AUC/AP and risk-quartile economics.
Do not change the trading action yet.

Only after conditioned risk ordering becomes credible should the model add a
causal WAIT/recheck action.
