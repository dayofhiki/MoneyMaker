# Request290 — second-resolution crack potential and recurrent entry value

## Result

Request290 completed successfully on May5-8 only.  The research gate failed.

The experiment moved the target away from the frozen 10-minute trading policy:
- crack target: maximum future one-second close return over 60 minutes;
- timing target: value of ENTER now versus forced entry five seconds later;
- every feature used only seconds completed through decision_t - 1 second;
- the first causal 2% pullback remained only a diagnostic anchor.

## Data

- first-pullback states: 330
- complete 60m crack targets: 328
- ticker-days with second data: 330
- h0 second-feature coverage: 100%
- sequential 0/5/10/.../30s states with an execution reference: 885
- labeled WAIT-5s comparisons: 320
- mean execution-reference lag: 0.74s
- p95 lag: 3s
- API requests: 330, no retries

The sequential support was much smaller than the theoretical 2,310 rows because
one-second aggregate bars are event bars: many exact wall-clock recheck times
have no trade aggregate within the three-second execution-reference allowance.

## Crack-potential prediction at the first 2% pullback

Population future second-close MFE mean over 60m: +4.30%.

Baseline (122 features):
- Spearman: 0.0421
- top-20% MFE mean: +4.72%
- +5% population/top20: 27.44% / 32.84% (1.20x)
- +10% population/top20: 8.23% / 5.97% (0.73x)
- +10 AUC: 0.4241

Completed-second enriched (159 features):
- Spearman: 0.0536
- gain vs baseline: +0.0115
- top-20% MFE mean: +4.14%
- +5% population/top20: 27.44% / 26.87% (0.98x)
- +10% population/top20: 8.23% / 2.99% (0.36x)
- +10 AUC: 0.4075

Interpretation: the current first-pullback representation does not identify
large future winners.  Adding the trailing 60 seconds of second aggregates gives
a tiny rank-correlation improvement but makes +10% enrichment worse.  The
second context should not be treated as a generic crack detector at this anchor.

## Five-second ENTER/WAIT value

Among the 320 labeled adjacent comparisons, waiting five seconds was
counterfactually beneficial 33.4% of the time.

Baseline:
- wait-advantage Spearman: -0.0424
- sign AUC: 0.5006

Second enriched:
- wait-advantage Spearman: -0.0328
- sign AUC: 0.5134

This is essentially no learnable signal in the current fixed-clock state.

The recurrent OOF second-enriched policy:
- waited on only 7.27% of episodes
- mean chosen wait: 0.85s
- mean entry-price improvement: +0.0098%
- mean 60m MFE change vs h0: -0.0038pp
- +5/+10/+20 opportunity rates were unchanged

So the policy was economically neutral, not a useful low-entry controller.

Fixed waits show the same lesson:
- 5s mean price improvement +0.016%, MFE change -0.018pp
- 15s mean price improvement +0.022%, MFE change -0.049pp
- 30s mean price improvement -0.088%, MFE change -0.197pp

Blind waiting is not the answer.

## Main lesson

Request290 reveals two architecture errors rather than disproving the
second-resolution direction.

1. Crack selection and entry timing were still mixed together.

A first 2% pullback is a late, arbitrary anchor for asking whether the whole
setup has +10%/+20% potential.  The crack selector should be trained on the
broader candidate/HOT setup before the final entry decision.

2. Fixed wall-clock 5-second states are a poor fit to aggregate trade data.

Historical one-second aggregates exist when trading occurs.  The final
controller should update on observed active-second events (or, when live,
quotes/trades), not demand a bar at every exact 5-second timestamp.

## Next boundary

Request291 should explicitly separate the hierarchy:

### Crack setup selector
- use the full first-HOT candidate population on May5-8, not only the 330 rows
  that happened to reach a first 2% pullback;
- train directly on continuous 60m upside opportunity;
- use broad pre-HOT / multi-minute ticker structure as the primary setup view;
- evaluate +5/+10/+20 enrichment out of fold.

### Entry controller
Only after a candidate is judged crack-worthy:
- construct event-driven active-second states rather than fixed 5s timestamps;
- evaluate on each newly observed active second;
- use current price, sub-minute pressure, local high/low, activity intensity
  and elapsed setup state;
- learn ENTER now versus WAIT for the next active price update;
- allow repeated WAIT decisions so the controller searches for a favorable low
  without hindsight-selecting a fixed pullback percentage or fixed wait length.

The first 2% pullback should remain a comparison baseline, not a final rule.

Do not tune the exit policy yet.  First prove (a) crack enrichment and (b)
entry-price timing on the correct hierarchical state.
