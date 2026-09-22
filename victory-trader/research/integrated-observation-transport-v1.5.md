# Integrated observation transport v1.5 — pre-registration

## Status

Frozen after the attention/transport bridge and fault-injection tests passed,
before inspecting 2026-04-01, 2026-04-02, 2026-04-06, 2026-04-07 or
2026-04-08. Good Friday is not a trading session. April 9 and later remain
sealed.

## Frozen hierarchy

1. Fit market-wide focus only through 2026-01-16.
2. Fit the downstream minute and one-second rerankers only on 2026-01-20
   through 2026-02-09.
3. Select learned focus-60 continuously when scores update.
4. Feed an event-driven rank-40 incumbent selector into shortlist-20.
5. Pass the exact selected membership to bounded observation transport; the
   transport may report failure but may not rerank or substitute names.
6. Use the existing one-second reranker and allocate at most HOT-10.

No model feature, fit range, capacity or threshold is selected from April.

## Promotion gate

Retain the request-126 performance requirements:

- every day has at least 10 first +10% crossings;
- immediate HOT capture exceeds the frozen hierarchy, is at least 30%, and is
  nonlower on at least four of five sessions;
- focus exact-prior capture is at least 50%;
- shortlist retains at least 95% of focus captures;
- HOT retains at least 90% of shortlist captures;
- one-second row coverage is at least 99%;
- HOT occupancy never exceeds 10.

Add transport requirements:

- selected and active membership have zero mismatched rows;
- concurrent high-resolution subscriptions never exceed 20;
- mean subscription additions are at most 5 per decision.

If any condition fails, do not tune on these dates. Attribute the failure to
focus, shortlist, transport, second-data or HOT allocation. ENTRY/ABSTAIN is
unblocked only if the full gate passes.
