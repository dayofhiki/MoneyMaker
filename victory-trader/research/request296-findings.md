# Request296 — untouched fresh validation

## Result

Request296 completed successfully on five previously unopened trading days:
2026-06-15, 2026-06-16, 2026-06-17, 2026-06-18 and 2026-06-22.

The architecture and action thresholds were frozen from May5-8 development.
No fresh-date tuning was performed.

The preregistered validation gate passed.

## Crack selection

Fresh population:
- complete 60m rows: 1,271
- selected rows: 119
- selected rate: 9.36%
- score Spearman: 0.1521
- population mean 60m MFE: +2.722%
- selected mean 60m MFE: +3.276%

Tail enrichment:
- +5%: 16.76% population -> 26.05% selected = 1.55x
- +10%: 4.48% -> 5.04% = 1.12x
- +20%: 0.63% -> 0% selected

So the crack selector generalized clearly for +5% opportunities and only weakly
for +10%. It failed to capture the very sparse +20% tail in this fresh block.

By day, selected mean MFE exceeded population on Jun15, Jun16 and Jun17, was
slightly higher on Jun18, and was worse on Jun22. +10% enrichment was positive
on Jun15-17 and absent on Jun18/Jun22.

## Entry validation

Complete selected entry episodes: 129
- entered: 112
- entry rate: 86.82%
- mean delay: 32.55s
- median delay: 6.5s
- mean price improvement versus immediate: +0.243%
- MFE gain versus immediate on the same entered episodes: +0.180pp
- +5% opportunity: 27.68% chosen vs 27.68% immediate
- +10% opportunity: 7.14% chosen vs 6.25% immediate

Cash-adjusted fixed-horizon entry utility:
- frozen entry policy: -0.918%
- immediate entry: -1.014%
- improvement: +0.0955pp

The timing policy beat immediate entry on 3/5 fresh days:
- Jun15: +0.385pp
- Jun16: -0.972pp
- Jun17: +0.846pp
- Jun18: -0.382pp
- Jun22: +0.372pp

This is a real out-of-sample replication of the relative entry-timing edge:
the policy again bought cheaper, retained slightly more MFE, preserved +5%
opportunities, improved +10% opportunity rate, and beat immediate entry on a
majority of days.

## Critical limitation

Absolute fresh utility remained negative.

The selected setups were not, as a group, strong enough in this fresh block to
make the current entry-quality utility positive. The policy improved a bad
immediate baseline, but did not turn the front half into a profitable trading
system.

Therefore Request296 validates:
1. a modest crack-selection signal, especially around +5% opportunity;
2. a reproducible local-turn entry-timing edge relative to immediate entry.

It does NOT validate:
- realized profitability;
- a strong +10/+20 crack selector;
- HOLD/EXIT;
- position sizing;
- transaction-cost survival.

## Interpretation

The entry controller now has stronger evidence than the selector.

The entry edge survived frozen fresh validation:
- cheaper price,
- more remaining MFE,
- non-lower +10% opportunity,
- 3/5 daily wins versus immediate.

The selector generalized, but more weakly than on development:
+5 enrichment is meaningful, +10 is mild, +20 is absent. The next major
bottleneck is therefore not to retune entry on the fresh dates. It is to
monetize the selected opportunity while preserving the frozen entry behavior,
and later improve crack ranking on separate development data without contaminating
this fresh block.

## Next boundary

Freeze Request296 setup+entry architecture and do not tune it on June15-22.

Proceed to causal second-resolution HOLD/EXIT research using development data.
The hold controller should answer whether the explosive thesis is still alive,
allow large winners to continue, and exit when the path deteriorates.

Evaluation must compare:
- frozen old 10m / 2% trailing policy;
- immediate exit/short holds;
- dynamic HOLD/EXIT;
- capture fraction of available +5/+10/+20 upside;
- adverse excursion and realized return;
- later left-on-table after exit.

Fresh June15-22 should remain sealed for HOLD/EXIT final validation until the
hold policy is frozen.
