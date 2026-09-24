# Request 167 — BUY-gated exact path-transition HOLD diagnostic

## Purpose

Request 166 showed that the unchanged Request-145 HOLD classifier collapses on
the stricter Request-165 BUY population:

- HOLD AUC 0.4959;
- only one semantic P(HOLD)>0.5 row;
- only 1/5 days with all observability signs positive.

At the same time residual multi-minute value remained directionally observable.
The next intervention therefore changes causal state representation, not the
action threshold or holding horizon.

## Prior evidence

Expanded-history v3.5-v3.7 previously established that a position's current
path level alone is weaker than also observing how the position-relative path
is changing over exact prior minutes. v3.7's exact 1/2/3/5/8/13-minute
transition family was the first historical continuation branch to pass its
development bridge in all three months.

Request 167 ports that supported information family to the current
Request-165 BUY population.

## Data

No new market date is opened.

Reuse Request-166 frozen path artifacts:
- fit: Apr 30 through May 8;
- calibration: May 11 through May 20;
- evaluation diagnostic: June 8-12.

## Only intervention

Keep the Request-166 labels, HGB classifier capacity, chronological Platt
calibration and strict semantic HOLD boundary P>0.5.

Add the supported causal path state:

1. current entry-relative return;
2. running maximum favorable excursion;
3. running maximum adverse excursion;
4. drawdown from running high;
5. rebound from running low;
6. path range;
7. minutes since running high;
8. minutes since running low;
9. observed-minute fraction since entry;
10. path efficiency.

For each of those ten variables add exact timestamp deltas at:
- 1 minute;
- 2 minutes;
- 3 minutes;
- 5 minutes;
- 8 minutes;
- 13 minutes.

A missing timestamp is never compressed to a previous observed row. The delta
is missing unless the literal t-k-minute state exists.

No future price, next execution reference, outcome, fixed horizon, phase subset,
lag search or threshold search is allowed.

## Remaining-option signal

Keep the Request-166 remaining-option regressor unchanged. This experiment asks
only whether richer transition information repairs the short-horizon HOLD
controller.

## Controller signal gate

Reuse the within-path Request-145 conditions through
`evaluate_observability`:

- executable decision-state coverage >=90%;
- HOLD AUC >0.50;
- semantic-HOLD arithmetic and day-balanced one-minute advantage >0;
- global remaining-value Spearman >0;
- median same-held-minute Spearman >0;
- predicted-excess-positive day-balanced realized excess >0;
- all signs positive on at least 4/5 evaluation days.

The June 8-12 anchor-to-path coverage remains a separate Request-166 execution
diagnostic. Therefore report both:

- `controller_signal_pass`;
- `full_position_bridge_pass`, which additionally requires anchor-to-path
  coverage >=90%.

If the controller signal passes but the full bridge fails only on anchor
coverage, the next no-new-date check should move the frozen controller to the
Request-165 June15/16/17/18/22 block where entry observability already passed.
If the controller signal itself fails, do not tune P>0.5 or the lag grid.
