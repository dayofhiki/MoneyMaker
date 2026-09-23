# Actionable active transport v1.0 — request 147

## Purpose

Request 146 removed future-bar-availability filtering from market inference.
The corrected Focus layer remained strong, but the Focus -> active observation
handoff retained only 93.60% of Focus-captured exact-prior runner crossings,
below the frozen 95% floor. Temporarily unscoreable active subscriptions rose
to 6.35% and mean scoreable occupancy fell to 18.73 / 20.

This request improves that bottleneck without changing subscription capacity,
HOT capacity, or the five-addition transport limit.

No new market date is opened.

## Causal design

The existing market-hazard model estimates next-minute runner hazard on rows
whose exact next-minute target is identifiable. Request 146 then scores every
current causal market row with that model.

Request 147 adds a second causal model:

    P(exact next-minute bar exists | current causal state)

It is fitted only on historical FIT rows using the frozen BASELINE_FEATURES.
At inference it sees no future bar or future label.

The active-observation priority is fixed as:

    active_priority =
        market_hazard_probability
        * next_minute_observability_probability

This is a factorized estimate of an actionable next-step opportunity. No
weight, blend coefficient, threshold, or transport budget is tuned.

Focus-60 remains ranked by the original market hazard probability. Only the
Focus -> active observation utility changes.

## Stage-2 target

For the exact event "runner crossing in the next wall-clock minute", a missing
next-minute aggregate implies that the event did not occur in that minute.
Therefore stage-2 HOT models use target 0 when no exact next-minute aggregate
exists. This is supervised future outcome information only and is never an
input feature.

## Frozen dates

- market-hazard / observability fit: through 2026-01-16;
- stage-2 fit: 2026-01-20 through 2026-02-09;
- revalidation: 2026-05-07, 05-08, 05-11, 05-12, 05-13.

May 21 and later remain sealed.

## Frozen resource constraints

- Focus <= 60;
- active subscriptions <= 20;
- max post-initial active additions <= 5 per decision;
- HOT <= 10;
- same one-second feature set and stage-2 model family.

Increasing capacity or churn is not an allowed way to pass.

## Request-146 comparator

- active retention given Focus: 0.9360146252;
- temporarily unscoreable subscription rate: 0.0634871795;
- mean scoreable active occupancy: 18.7302564103;
- HOT within 1m: 0.4953751285;
- HOT within 2m: 0.5858170606;
- HOT within 5m: 0.6536485098.

## Frozen gate

Request 147 passes only if all of the following hold:

1. request-146 causal handoff gate conditions pass;
2. active retention given Focus >= 95%;
3. temporarily unscoreable subscription rate is below request 146;
4. mean scoreable active occupancy is above request 146;
5. HOT within 1m is nonlower than request 146;
6. HOT within 2m is nonlower than request 146;
7. HOT within 5m is nonlower than request 146;
8. one-second coverage >= 99%;
9. max subscriptions <= 20;
10. max HOT occupancy <= 10;
11. mean post-initial additions <= 5 and maximum <= 5;
12. subscription/selection mismatch = 0.

This is a same-opened-block engineering revalidation, not fresh evidence of
profitability.

## Consequence

If the gate passes, this actionable transport becomes the corrected attention
foundation. The next research step is to regenerate first-HOT economic
opportunity state on the corrected population and revalidate opportunity
selection / executability before building the end-to-end BUY/WAIT/ABSTAIN ->
HOLD/EXIT replay.

If it fails, diagnose the Focus -> active loss without changing capacity or
opening May 21+.
