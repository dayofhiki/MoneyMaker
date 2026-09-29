# Request282 — two-stage pullback risk veto findings

## Result

Request282 completed successfully but failed the two-stage economic gate.

The architecture was:

1. stage 1: Request280 downstream-value model with exact pre-HOT ticker history,
   selecting the train-derived top 20%;
2. stage 2: at the first causal 2% pullback, estimate severe-loss risk using the
   Request277-style pullback prefix plus strictly completed prior-minute context;
3. reject the highest 50% risk according to each outer fold's training-only
   threshold;
4. accepted trades use the unchanged -2.5% hard stop, 2% trailing drawdown and
   10-minute exact max hold.

May11-20 and June15-19 remained unopened.

## Signal quality

- pullback severe-loss OOF AUC across the full pullback population: 0.6701
- this clears the preregistered 0.65 signal floor

The signal therefore exists, but its economic usefulness inside the stage-1
shortlist is weak.

## Economics

Stage-1 pre-HOT top-20 only:
- 1,602 total candidate episodes
- 354 shortlisted candidates
- 54 shortlisted pullback triggers
- 36 resolved trades, 18 unresolved
- candidate mean: -0.01473%
- day-balanced candidate mean: -0.01468%
- trade mean: -0.64794%
- trade positive rate: 30.56% (11/36)
- severe-loss rate: 36.11% (13/36)
- positive days: 0/4

Two-stage risk veto:
- 354 shortlisted candidates
- 54 pullback triggers
- 16 triggers rejected
- 38 admitted
- 26 resolved trades, 12 unresolved
- candidate mean: -0.01098%
- day-balanced candidate mean: -0.01100%
- trade mean: -0.67134%
- trade positive rate: 26.92% (7/26)
- severe-loss rate: 34.62% (9/26)
- positive days: 0/4

Candidate-mean improvement versus stage 1 alone:
- +0.00375 percentage points

This is real harm reduction, but far too small and the selected trade subset
remains negative.

## What the veto actually removed

Relative to stage 1 alone, the veto removed 16 triggered episodes. Six of those
were previously unresolved, so 10 removed episodes had resolved trade outcomes.

From the resolved trade counts:
- removed positive trades: 11 - 7 = 4
- removed severe-loss trades: 13 - 9 = 4
- removed other non-positive, non-severe trades: 2

The ten resolved rejected trades therefore consisted of approximately:
- 4 winners
- 4 severe losers
- 2 milder losers

Their aggregate return was about -5.87 percentage points, or roughly -0.59%
per rejected resolved trade. The accepted resolved trades averaged -0.67%.

Interpretation: the veto removes negative expected value in aggregate, but it
does not rank the stage-1 shortlist sharply enough. It throws away many winners
along with severe losers and leaves a slightly worse average trade population.

## Important diagnostic mismatch

The reported severe-loss AUC of 0.6701 is measured across all causal pullback
states, not specifically inside the stage-1 top-20 shortlist where the veto is
actually used.

Request282 therefore proves that a globally useful risk signal does not
automatically remain useful after conditioning on the candidate selector.
The next diagnostic must evaluate risk discrimination *inside the shortlist*.

## Next boundary

Before changing stops or pullback percentages:

1. measure OOF severe-risk AUC/AP/calibration only on stage-1-shortlisted
   pullback states;
2. audit false vetoes versus true vetoes using the shortlist-conditioned
   population;
3. replace the binary ENTER-versus-CASH veto with a causal WAIT/recheck action
   if high first-pullback risk is often temporary;
4. during WAIT, update only newly completed causal path/minute context and
   reconsider ENTER / WAIT / REJECT continuously.

This better matches the intended trader behavior: shortlist promising stocks,
wait for a pullback, and if the first pullback still looks dangerous, keep
watching rather than permanently throwing the stock away.

Do not open May11-20 or June15-19 until the shortlist-conditioned stage-2
decision produces positive development economics.
