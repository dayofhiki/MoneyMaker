# Request295 — matched local-turn ablation

## Result

Request295 passed its development gate.

It reproduced Request294 exactly and separated the source of the first positive
entry-timing edge.

### A. Request293-style base direct entry

- utility Spearman: 0.2247
- mean decision utility: +1.347%
- immediate benchmark: +1.732%
- gain vs immediate: -0.385pp
- beat immediate on 2/4 held-out days
- price improvement: +0.103%
- MFE gain: +0.013pp

### B. Turn-enriched direct entry, no separate turn gate

- utility Spearman: 0.2614
- mean decision utility: +1.442%
- gain vs immediate: -0.290pp
- beat immediate on 2/4 held-out days
- price improvement: +0.145%
- MFE gain: -0.029pp

The 16 local sequence features improve current-entry value estimation, but that
alone does not create a positive stopping edge.

### C. Full value + local-turn gate

- mean decision utility: +2.019%
- immediate benchmark: +1.732%
- gain vs immediate: +0.286pp
- beat immediate on 3/4 held-out days
- price improvement: +0.319%
- MFE gain: +0.197pp
- +10% opportunity rate: 13.25% vs 12.05% on the same entered episodes

Held-out day gains versus immediate:
- May 5: +0.840pp
- May 6: +0.820pp
- May 7: +0.606pp
- May 8: -0.798pp

Therefore Request294's improvement is not explained by feature enrichment
alone. The action gate that requires both acceptable long-horizon entry value
and a short-horizon turn signal contributes material incremental value.

## Freeze decision

Do not tune May5-8 further.

Freeze the current hierarchy for untouched validation:

1. broad first-HOT crack setup score;
2. active-second observation;
3. turn-enriched current-entry value model;
4. local-turn gate;
5. strong-value override remains optional but is not central.

For a single fresh policy, aggregate the four development fold action
thresholds without fresh tuning:

- utility threshold: median of the four fold thresholds;
- turn threshold: median of the four fold thresholds;
- strong-value override: disabled because it was selected in only one of four
  folds and accounted for only three Request294 entries.

Fresh dates must not be used to alter any model, threshold, feature or policy.

## Request296 boundary

Open June 15-19, 2026 exactly once.

Reconstruct the same validated attention/first-HOT population using the frozen
Request163 upstream architecture and existing Request163 stage2 training
artifact. Train crack and entry models only on the existing May5-8 development
data, then score June15-19 once.

Primary fresh checks:
- crack selector preserves +5/+10 enrichment;
- selected-entry policy beats immediate entry in mean fixed-horizon utility;
- policy beats immediate on at least 3/5 fresh days;
- entry price improvement remains positive;
- MFE and +10% opportunity are not degraded on entered episodes.

This validates selection and entry only. It is not yet a realized trading-P&L
claim because HOLD/EXIT remains unfrozen.
