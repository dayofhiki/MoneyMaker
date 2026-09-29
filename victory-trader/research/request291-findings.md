# Request291 — hierarchical crack setup and event-driven entry

## Result

Request291 completed successfully through the consolidated
`MoneyMaker Research Request` workflow. The overall research gate failed, but
the failure is highly diagnostic: setup selection and event-driven observation
improved materially, while the direct WAIT policy was biased toward excessive
delay by its target construction.

## 1. Crack setup selection is now meaningfully better than Request290

Population:
- 980 full first-HOT candidates with complete 60-minute minute-open labels
- population mean 60m MFE: +2.93%
- +5% opportunity rate: 16.33%
- +10% opportunity rate: 4.69%
- +20% opportunity rate: 1.12%

38-feature baseline:
- Spearman: 0.1433
- selected rate: 20.61%
- selected mean MFE: +4.22%
- +5% enrichment: 1.36x
- +10% enrichment: 1.58x
- +20% enrichment: 2.21x

98-feature multi-minute representation:
- Spearman: 0.1660
- held-out selected rows: 105 / 980 = 10.71%
- selected mean MFE: +4.79%
- +5% rate: 27.62% vs 16.33% population = 1.69x
- +10% rate: 9.52% vs 4.69% population = 2.03x
- +20% rate: 2.86% vs 1.12% population = 2.55x

This is the first clear evidence in the new crack-oriented research that broad
HOT-time setup information can concentrate large future moves.

Important caution:
- the rich representation selected only 10.7% while the baseline selected
  20.6%, so its higher tail enrichment is not a perfectly matched-cardinality
  comparison;
- rich Spearman improved, but +10/+20 AUC did not improve versus baseline
  (roughly 0.598/0.628 rich versus 0.600/0.651 baseline).

Therefore the hierarchy is supported, but the incremental value of the extra
multi-minute features should be verified at matched selection rates before
claiming that all 98 features are superior.

The setup support check failed only because 980 rows were below the pre-set
1,000-row floor. This is a support rule, not a performance failure.

## 2. Event-driven seconds solved Request290's sparse-clock problem

Request290:
- 885 sequential states
- 320 labeled adjacent WAIT comparisons

Request291:
- 4,526 event states
- 4,429 labeled WAIT-option rows
- 97 complete selected episodes
- 100% second-feature coverage
- mean gap to next active second: 4.34s
- p95 gap: 21s

So updating when the market actually produces an active one-second aggregate
increased usable state support dramatically. This strongly favors event-driven
observation over fixed wall-clock 5-second checkpoints.

The 5,000-state support gate was missed narrowly (4,526); again this is not a
signal-performance failure.

## 3. The WAIT signal became genuinely learnable

Request290 second-enriched WAIT signal:
- Spearman: -0.0328
- sign AUC: 0.5134

Request291 event-driven WAIT-option signal:
- Spearman: 0.2789
- sign AUC: 0.6089

This is a major improvement. Active-second state contains information about
whether delaying entry still has option value.

However:
- hindsight WAIT-beneficial rate: 85.32%
- model predicted WAIT on 94.90% of labeled states
- even states predicted ENTER still had +0.62pp average hindsight future-best
  advantage
- predicted-WAIT states had +1.26pp average hindsight future-best advantage

That distribution shows the teacher target itself strongly favors waiting.

## 4. The direct recurrent policy failed economically

The OOF policy:
- waited in 100% of the 97 episodes
- mean delay: 193s
- median delay: 211s
- mean entry-price improvement vs immediate: -0.062%
- mean MFE change vs immediate: -0.531pp

It did not improve +10/+20 opportunity rates.

So the model learned a real WAIT ranking signal, but the zero-threshold action
rule turned that signal into chronic procrastination.

## 5. Two target-design problems explain the procrastination

### A. Oracle future-best target

The training target was:

future best entry MFE among every later active state in the five-minute window
minus current entry MFE.

Because the teacher is allowed to search all later states with hindsight, it
almost always finds some later state that looks better. This makes WAIT positive
85% of the time and does not teach a proper stopping boundary.

This target can be useful as an option-value diagnostic, but should not be used
directly as the live WAIT action value.

### B. Moving 60-minute endpoint

Each possible delayed entry received its own fresh 60-minute forward horizon:
`entry_t + 60m`.

Therefore a later entry is allowed to see later market time than an earlier
entry. This creates an artificial advantage for waiting.

All candidate entry times within one setup must instead be evaluated against
the same fixed terminal horizon, e.g. HOT_t + 60m.

## 6. The old 2% pullback still teaches something useful

Among the 105 crack-selected candidates:
- only 53 (50.48%) ever produced the fixed 2% pullback entry;
- when it existed, that entry price was on average 1.27% better than immediate
  active-second entry;
- median improvement was 1.49%.

So the old 2% rule is too rigid because it misses roughly half of selected
setups, but its economic intuition was not nonsense: waiting for weakness often
does buy meaningfully cheaper.

The real controller needs to learn which setup can safely wait for a deeper
pullback and which setup is about to leave without one.

## Interpretation

Request291 supports the hierarchical architecture:

1. broad setup context for crack selection;
2. event-driven second-resolution observation for entry timing;
3. later, second-resolution continuation for HOLD/EXIT.

The setup stage now has useful crack enrichment, and the entry state now has
learnable timing information. The failed policy is primarily a stopping-target
and action-calibration problem rather than evidence against second-resolution
entry timing.

## Request292 boundary

Keep the hierarchy and fix only the entry decision.

1. Evaluate every candidate entry in an episode against the same terminal
   horizon anchored at HOT_t + 60m.
2. Replace the hindsight best-future WAIT target with a one-step / recursively
   fitted stopping value:
   - ENTER value at the next executable active-second price;
   - WAIT value from the next active state under the learned continuation
     policy;
   - no direct search for the best future entry at action time.
3. Add SKIP as a valid terminal action. Do not force an entry by the end of the
   observation window.
4. Derive the ENTER/WAIT margin from training folds rather than using an exact
   zero threshold, so small noisy predicted advantages do not cause endless
   waiting.
5. Compare OOF policy against:
   - immediate entry;
   - fixed 2% pullback;
   - no-trade cash.
6. Require positive entry-price improvement while preserving or improving
   fixed-horizon crack opportunity/capture.
7. Before promoting the 98-feature setup representation, run a matched-rate
   setup comparison against the 38-feature baseline.

Do not tune HOLD/EXIT yet. First make the crack selector and entry stopping
policy jointly coherent.
