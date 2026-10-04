# R304 findings — common expiry does not repair HOLD learning

Official run: https://github.com/dayofhiki/MoneyMaker/actions/runs/36883479409
Source44954ce79ae2a4a41b66302b1f3a1ac1f094eb15; artifact11172631677.
All86 positions resolve and exact R303 replay passes. Economic gate FAILED.

A R303 mean BASE -1.397021%; B common-cap teacher -1.631033%. B has4/86
positive trades,59 model exits,26 hard stops and1 cap exit. All four days lose:
May5 -2.099714%, May6 -1.435363%, May7 -1.951184%, May8 -1.285430%.
Mean gross -0.089823%; mean cost drag1.541210pp. B median hold21.5 seconds.
Reachable gross +5/+10/+20 opportunities14/5/2; realized net capture1/0/0.
These counts are hindsight diagnostics, not causal opportunity identification.

Removing the teacher's entry+10min expiry increased nonzero target support but
did not teach economically useful continuation. Do not promote or tune the
failed arm. The next R305 diagnosis separates fixed-entry feasible net upside
from exit-model mistakes and tests causal observability, rather than adding
features or another policy iteration. Existing86 entries, hard stop, execution,
cap and costs remain fixed; May5–8 only and June sealed.
