# R316: frozen event audit does not justify event-conditioned routing, but exposes a structural-event coverage gap

Isolated official branch run [37454081321](https://github.com/dayofhiki/MoneyMaker/actions/runs/37454081321)
completed successfully from source `5ceeafc8dc28b60eedfc91a90badf75a195cd77f`.
The uploaded artifact `moneymaker-research-request-316-isolated` has archive
digest `sha256:26b7abb638588eb72dbe4f08cf41f23a205407f93a6b6ad0b046f6a5839ffc93`.
The exact `request316.json` inside that artifact hashes to
`a148a3e06d46b7a1ba67e9144842c15b0485db1c218ebaea611a2709febb5377`.

R316 did not touch main, refit a model, tune a threshold, change risk/costs,
or open June HOLD / July-August. It reused the exact frozen R311 828-case
May11-20 cohort, including 650 complete labels and 38 BASE-net>=5 positives.

## Acquisition integrity

All three declared sources completed without failure or retry.

- 570 observed tickers
- 581 provider requests, 0 retries
- 973 normalized event rows
- 807 classified 8-K rows
- 152 ticker-news rows
- 14 split rows
- 38 same-day 8-K records were retained only as timing audits
- 0 same-day reverse-split records occurred in the frozen cohort

The zero same-day reverse-split count is important: the existing MoneyMaker
protocol excludes same-day split events from the original momentum universe.
R316 therefore cannot test the full reverse-split-day / "병합빔" phenomenon.

## Frozen cohort baseline

Across the unchanged 650 complete R311 labels:

| Arm | pooled AUROC | AP | Brier | same-day AUROC |
|---|---:|---:|---:|---:|
| N | .6509 | .0899 | .05633 | .6386 |
| Q | .6272 | .0815 | .05718 | .6143 |
| M | .6425 | .0992 | .05629 | .6406 |

Weighted BASE-net>=5 prevalence is 5.94%.

## Primary event families

### Fresh ticker news in the preceding 24 hours

Only 23 complete cases carried a qualifying news item and none was BASE-net>=5
positive. The complement has 627 complete cases and all 38 positives. The
prevalence contrast is therefore large descriptively, but this group fails the
preregistered support floor and cannot identify rank heterogeneity.

This result does not establish that fresh news is harmful. The current sample
contains too few event positives for a rank comparison and represents only
Massive ticker-tagged news within the frozen R311 HOT universe.

### Prior classified 8-K within 7 calendar days

This is the only primary family with adequate preregistered support:

- event: 120 complete cases, 8 positives, 5 positive days
- complement: 530 complete cases, 30 positives, 8 positive days
- weighted prevalence: 6.25% vs 5.68%, ratio 1.249
- M same-day AUROC: .6240 vs .6492
- M-Q same-day increment differs by only +.0353

Ticker-day bootstrap 95% intervals all cross the null:

- prevalence ratio [.446, 2.343]
- M same-day AUROC difference [-.227, .290]
- M-Q increment difference [-.153, .214]

Therefore prior 8-K context does not materially explain the current momentum
transport problem under the fixed R316 gate.

### Reverse split in the preceding 20 calendar days

Only 12 complete cases qualify, with 2 positives on 2 days, so the family is
formally underpowered.

The point estimates are nevertheless unusual and worth preserving as a
hypothesis for a separately constructed structural-event cohort:

- weighted BASE-net>=5 prevalence 14.29% vs 5.72% in the complement
- point prevalence ratio 3.02
- frozen M pooled AUROC .2222
- frozen M same-day AUROC 0.0 on the tiny qualifying support
- complement M same-day AUROC .6583

The ticker-day bootstrap prevalence interval is extremely wide and includes
the null. The rank-difference interval is strongly negative, but only 661/1000
draws even contain sufficient within-day class structure because the event
support is tiny. R316 therefore does not promote this signal.

It does, however, make a concrete architectural warning visible: a small
recent-reverse-split subgroup can behave unlike the ordinary HOT population,
while the existing protocol entirely omits same-day split events. A dedicated
event cohort is required before concluding whether this is real or sampling
noise.

## Preregistered interpretation

R316 status is `not_supported_by_fixed_gate`.

No primary family passes the material-heterogeneity gate. Only `prior_8k_7d`
has adequate support, and it does not show a robust difference. Therefore R316
does **not** justify adding event-conditioned specialists or a mixture-of-experts
router to the current MoneyMaker pipeline.

Equally, R316 does **not** falsify structural-event trading. The strongest
candidate discussed before the audit, reverse-split behavior, is specifically
underrepresented by the inherited universe and same-day reverse splits are
absent by construction.

## Research decision

Keep the current Work/momentum track independent. Do not modify its model from
R316.

For the event track, the next justified experiment is not another feature added
to R311. It is a separately sampled **structural-event cohort**, beginning with
reverse splits, where event membership is fixed from corporate-action data
before price outcomes are loaded. That cohort should include the actual split
execution day plus preregistered pre/post windows, preserve point-in-time
knowledge, model executable costs/halts, and compare explosive-upside rates
against matched non-event controls.

Only if that dedicated cohort establishes a repeatable event-specific
distribution should event-context interactions be tested, followed later by a
specialist/router if the simple conditioning model is insufficient.
