# Request274 — causal exit integrity audit

Reference: Request273 commit ebd04ac14364f430b211cbb9a495899724d2b958,
https://github.com/dayofhiki/MoneyMaker/actions/runs/36395013793.

Request273 completed but all 36 fit rules failed. Best fit mean (diagnostic,
not selected) was -0.1138445% per resolved candidate including cash and
-0.3919504% per resolved trade. 25.71% of resolved trades lost at least 2%.
244 candidates, 73 entries, 70 resolved trades, 3 unresolved entries.
These are not portfolio returns.

Code review found fallback settlement at the last observed price BEFORE a
missing deadline. A later bar could also trigger max-hold after an absent
exact deadline. Both now remain unresolved, never silently cash/zero.
Observed earlier stops and exact deadlines retain the existing minute-open
proxy; quote/fill validation is still outstanding.

Request274 freezes the same 36 rules, entries, costs and numerical stack,
replaying May5-7 only. It records every rule's outcomes and input hashes.
No new dates or promotion. Nine targeted tests passed for missing deadlines,
exact caps, earlier stops and independence from future observations.

Next model hypothesis: distinguish recoverable pullbacks from deterioration
at entry, using realized causal downstream-policy labels rather than future
best exits. Do not tune on May11-20 or open June15-19 to rescue failed rules.
