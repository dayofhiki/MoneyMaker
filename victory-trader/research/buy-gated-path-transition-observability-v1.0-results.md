# Request 167 — path-transition HOLD results

Authoritative run: `36028968382`  
Artifact: `10819853854`

No new dates were opened.

## Result

**FAIL**

Adding the previously supported exact 1/2/3/5/8/13-minute path-transition
family did not restore the one-minute HOLD classifier on the Request-165 BUY
population.

Baseline Request-166:
- HOLD AUC: 0.495862
- semantic HOLD rows: 1

Transition model:
- HOLD AUC: **0.496467**
- semantic HOLD rows: **0**
- all-sign-positive days: **0/5**

Daily HOLD AUC:
- Jun 8: 0.4893
- Jun 9: 0.4976
- Jun10: 0.5208
- Jun11: 0.4752
- Jun12: 0.4970

Transition lag availability was substantial (1m 88.4%, 2m 84.4%, 3m 81.1%,
5m 74.5%, 8m 64.9%, 13m 49.1%), so the failure is not explained by total
absence of transition history.

## Diagnosis

For the stricter BUY population, the binary question "is exactly one more
minute better than exiting now?" is not a stable controller target. This remains
true even after adding causal path dynamics.

The remaining multi-minute option-value signal is still positive:
- global Spearman +0.0900
- same-minute median Spearman +0.0533
- 93.1% of eligible minute groups positive
- predicted-excess-positive day-balanced realized excess +0.5782%

Next: apply the supported path-transition information to the multi-minute
remaining-option regression itself, rather than continuing to tune or enrich
the failed one-minute classifier.
