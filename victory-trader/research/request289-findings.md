# Request289 — explosive-upside path audit

## Result

Request289 completed successfully after reproducing Request286's exact stage-1
cross-fit seeds and shortlist.

This is a hindsight diagnostic only.  Future minute opens/highs/lows are labels
and were never available to the causal selector or the first-2%-pullback entry.

## Population

May5-8:

- all causal 2% pullback entries: 330
- exact Request286 stage-1-shortlisted pullback entries: 54
- shortlist frozen-policy resolved trades: 36
- shortlist entries with complete 60-minute path: 51

The corrected shortlist thresholds match Request286:
- May5: 0.0531571
- May6: 0.0569283
- May7: 0.0601931
- May8: 0.0517036

## How much upside actually existed?

Across all 330 pullback entries, future minute-high MFE was:

- 10m mean: +2.31%
- 20m mean: +2.93%
- 30m mean: +3.37%
- 60m mean: +4.29%

At 60m:
- >= +5% potential: 27.64% (89/322)
- >= +10% potential: 7.76% (25/322)
- >= +20% potential: 1.55% (5/322)

For the exact stage-1 shortlist, future minute-high MFE was:

- 10m mean: +2.06%
- 20m mean: +2.67%
- 30m mean: +2.95%
- 60m mean: +4.24%

At 60m:
- >= +5% potential: 25.49% (13/51)
- >= +10% potential: 5.88% (3/51)
- >= +20% potential: 1.96% (1/51)

The stricter minute-open MFE for the shortlist still averaged +3.85% at 60m,
so the result is not only an intraminute-high artifact.

## Selection bottleneck

The current stage-1 selector is not an explosive-upside selector.

At 60m, compared with all pullback entries:
- +5% potential rate uplift: -2.15pp
- +10% potential rate uplift: -1.88pp
- +20% potential rate uplift: +0.41pp, but this is only one shortlisted row

The +10% rate ratio is 0.758.  In other words, the current shortlist is
actually less concentrated in +10% opportunities than the full pullback-entry
population on these development days.

This is consistent with the selector's training objective: it was built around
generic downstream value, not around explosive future excursion.

## Exit/hold bottleneck

The frozen exit policy also destroys much of the upside when a crack is present.

Among 34 resolved shortlisted trades with complete 60m paths:

- frozen realized trade mean: -0.72%
- mean 60m minute-high MFE: +5.04%
- mean gap between later 60m high and realized exit: +5.76 percentage points

Exit reasons:
- hard stop: 12
- 10m max hold: 13
- 2% trailing stop: 9

### Trades that later had >= +5% potential

11 resolved shortlisted trades:

- frozen realized mean: +1.12%
- mean 60m high MFE: +10.91%
- median time to 60m peak: 52 minutes
- 72.7% exited before the first +5% hit
- only 9.1% actually realized >= +5%

### Trades that later had >= +10% potential

3 resolved shortlisted trades:

- frozen realized mean: +0.13%
- mean 60m high MFE: +20.80%
- median time to peak: 52 minutes
- median adverse excursion before the later peak: -1.92%
- 100% exited before the first +10% hit
- 0% realized >= +10%

### The >= +20% example

One shortlisted resolved trade later reached a +40.58% minute-high excursion.

- frozen policy realized: +4.52%
- time to 60m peak: 41 minutes
- current policy exited before +20%

## Interpretation

Two separate bottlenecks are now proven on the development window.

1. Selection:
   the existing downstream-value selector does not enrich +10% crack
   opportunities.  It is the wrong objective for the newly clarified strategy.

2. Monetization:
   when the shortlist does contain a large mover, the fixed 10-minute /
   2%-trailing policy usually exits far too early.  The strongest three +10%
   opportunities were all exited before +10%, with average later MFE above
   +20%.

Therefore the project should no longer optimize generic short-horizon trade
return as the central target.

## Next boundary

Request290 should test whether explosive upside itself is learnable from the
causal state.

Use all causal pullback entries, not the old shortlist, and make conservative
60-minute minute-open MFE the primary continuous label.  Compare:

- pre-HOT representation,
- pullback-state representation,
- pullback + completed-second representation.

Evaluate OOF ranking correlation plus enrichment of +5% and +10% future
excursions in a train-derived top fraction.

Do not make +10% a hard training class; it is too rare.  Learn the continuous
upside potential and use +5/+10 thresholds only as diagnostics.

Only after a dedicated crack-opportunity selector shows enrichment should the
next controller learn continuation/hold value so it can stay in the rare
large winners without blindly widening risk on every trade.

Do not open May11-20 or June15-19 yet.
