# R313 — waiting for two-print evidence does not improve frozen discrimination

Local diagnostic completed after preregistration commit `6ba5ee0`. No official GitHub run or remote publication; upload awaits explicit authorization after automatic approval rejection. Same original828 May identities, original costs/stop/cap and frozen R311 M model. No outcome fits, model changes, acquisition, window/count search, promotion or sealed dates. **85 targeted tests**, critical Ruff and compile pass.

Two-print evidence means two recorded one-second aggregate bars in10s, not two individual trades. The count is distinct traded seconds. A single observed bar may contain many transactions; no full tick tape or quotes are available.

The first completed print with at least two completed prints in the last10s is available for489/828 cases, with469 complete labels.339 never meet the fixed evidence condition in the original five-minute observation window;20 have incomplete entry/terminal fills.174 evidence clocks equal the original first clock and315 are later. All174 same-clock features, labels and scores replay within1e-9. Original M score maximum replay error1.11e-16. Additional wait median4s (includes same-clock cases),90th percentile118.6s.

## Paired identities, different entry-time labels

Both complete observations exist for469 original identities. Original first observation on this SAME subset has31 net5-positive cases. Evidence arrival has25. It is inappropriate to compare evidence AUROC against the full650 original cohort or call a Brier reduction calibration improvement when label prevalence changed.

| Observation on paired469 | Same-day AUROC | Pooled AUROC | AP | Brier | Constant original training-prior Brier |
|---|---:|---:|---:|---:|---:|
| Original first | .632889 | .632173 | .097861 | .062597 | .062495 |
| Two-print evidence arrival | .594062 | .603932 | .080307 | .053625 | .053487 |

Evidence-minus-original same-day rank **-.038826**, ticker-day95% CI[-.130246,.055556]; day CI[-.140713,.046216]. Pooled difference-.028241, ticker-day CI[-.109819,.052761]. No beneficial rank effect is established. Evidence day AUC improves on2/8 dates. Both Brier point estimates remain worse than the unchanged training-prior baseline on their own outcomes. Brier differences reflect target/prevalence changes and cannot isolate calibration. No promotion gate is invented.

| Original net5 possibility | Evidence net5 possibility | Paired cases |
|---|---|---:|
| No | No |434|
| No | Yes |4|
| Yes | No |10|
| Yes | Yes |21|

Of all38 original positive cases,21 retain a labeled net5 opportunity;10 have a resolved later label below5;7 never reach the evidence condition. Thus only21/38=55.3% remain demonstrated at evidence arrival. The seven unavailable cases are excluded by this hypothetical evidence requirement, not proof every dynamic controller must lose them. Four new opportunities appear at the later observation. None is realized trading profit; these are hindsight reachable ceilings under the unchanged risk rules.

A separately marked post-result description: the ten paired lost opportunities wait a median46s and entry price rises a median2.446%; the four gains wait a median134.5s and entry price falls a median2.267%. Original-to-later entry-fill gap median33.5→6.5s for losses,82.5→7s for gains. Tiny outcome-defined groups, no inference or fitted rule. This is consistent with the intended distinction between paying up during acceleration and benefiting from a pullback, while failing to establish a causal predictive discriminator. A blanket evidence requirement handles both situations the same way.

## Research consequence

Do not promote R312 elapsed inputs or R313 blanket evidence waiting. The known timestamp ambiguity and sparse observations remain limitations, but these specific changes did not repair discrimination. Preserve early scoring and continuous observation; no new fixed waiting delay is justified. The next separately registered development audit should decompose observable price movement, decision-to-fill repricing, execution costs and stop-limited continuation, before designing a conditional WAIT/ENTER intervention. It must keep missing paths and distinguish hindsight decomposition from deployable signals. Current results do not support picking a new feature, threshold or sell policy from these outcomes.
