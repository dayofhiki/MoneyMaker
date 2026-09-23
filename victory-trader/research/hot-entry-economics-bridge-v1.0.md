# First-HOT entry economics bridge v1.0 — conditional pre-registration

## Status

Prepared while request 138 is still running. This diagnostic may execute only
if request 138 passes its full attention/observation promotion gate.

It reuses exactly the request-138 evaluation sessions:
2026-05-07, 2026-05-08, 2026-05-11, 2026-05-12 and 2026-05-13.

It opens no later date. May 14 and later remain sealed.

## Question

Now that the system can find and maintain HOT candidates, do first-HOT episodes
actually contain enough post-signal movement to support a realistic entry
policy after modeled trading friction?

This is an economic bridge, not an executable entry strategy.

## Frozen episode definition

- one episode = one ticker-day;
- use the first chronological HOT state in that ticker-day;
- entry reference = the next exact minute bar open after the HOT decision;
- missing/non-positive entry reference is unevaluable and never shifted;
- no second HOT or later state in the same ticker-day may create another entry.

The next-minute open is a causal historical reference. It is not a claim about
a live fill.

## Frozen fixed-horizon diagnostics

For every valid first-HOT entry reference, report gross and LIGHT/BASE/STRESS
round-trip returns using exact future opens at:

- 1 minute;
- 2 minutes;
- 5 minutes;
- 10 minutes;
- 15 minutes;
- 30 minutes.

Use the existing execution-cost scenarios unchanged.

No fixed horizon is selected as a policy from these outcomes.

## Frozen hindsight opportunity ceiling

For each valid entry, inspect every exact observed future open from minute 1
through minute 30 and record the highest BASE round-trip return and its minute.

This is explicitly non-executable. It only answers whether enough economic
movement exists after first HOT for better entry/exit learning to be worthwhile.

Report:

- first-HOT episodes and valid-entry coverage;
- fixed-horizon gross/BASE/STRESS means;
- fixed-horizon BASE median, positive rate and p05;
- 30-minute oracle best BASE mean/median/positive rate/p05;
- median oracle-best minute;
- the same oracle diagnostics by trading day.

## Interpretation rule

This diagnostic does not promote a trading policy.

If the 30-minute oracle opportunity ceiling is weak or broadly negative, the
current HOT definition still does not provide an economically useful entry
population and ENTRY/ABSTAIN research should first improve candidate value.

If the oracle ceiling is clearly positive while fixed-horizon policies remain
weak, the primary problem is selective entry and adaptive exit timing. The next
research branch should learn:

1. whether this HOT episode is worth trading at all;
2. conditional on an eligible episode, when to enter;
3. after entry, repeatedly HOLD or EXIT using evolving state.

The already-known historical lessons from v3.3-v4.2 may be reused as design
evidence, but request-139 outcomes may not be used to claim an executable
profit edge.

May 14 and later remain sealed for a separately preregistered ENTRY/ABSTAIN
policy.
