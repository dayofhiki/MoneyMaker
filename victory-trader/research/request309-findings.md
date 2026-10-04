# R309 — linear pipeline improves observability, but momentum increment is unproven

Official run **37217714691**, artifact **11308719271**, source **f897f81c120f715600d1bb9635cd61df33e51efc** (PR75). Original R306 input: 97 May5–8 candidate episodes, 4,526 states, 99 causal inputs, reachable BASE-net >=5% labels, nine first-snapshot positives. No data acquisition, June HOLD access, entry/exit/risk/cost policy changes or threshold selection. Original 86 entries remain frozen. All five models' predictions and full result JSON exactly match local execution.

## Primary first-decision results

Day/episode weighted evaluation. Higher AUROC/AP is better; lower Brier is better. First-snapshot training-prior Brier is **0.084378**; observed weighted prevalence is **0.089182**.

| Arm | Input/model/training | AUROC | AP | Brier | Predicted mean |
|---|---|---:|---:|---:|---:|
| U | 99-input original tree / all states | 0.597127 | 0.127366 | 0.103026 | 0.047866 |
| V | fixed 10-input original tree / all states | 0.553405 | 0.160062 | 0.100015 | 0.113649 |
| W | 99-input linear pipeline / all states | 0.734104 | 0.175890 | 0.080544 | 0.080220 |
| X PRIMARY | fixed 10-input linear pipeline / all states | 0.720376 | 0.233835 | 0.078507 | 0.093950 |
| F secondary | fixed 10-input linear pipeline / first only | 0.481361 | 0.093195 | 0.084379 | 0.090124 |

Primary X improves AUROC/AP/Brier versus U and beats the training-only first-snapshot prior Brier, but **fails the preregistered research gate**: only one evaluable day improves over U and the paired ticker-day AUROC gain CI includes zero. W is a predeclared mechanism control, not a replacement primary arm to choose after X fails. Its stronger development metrics justify a separate fixed investigation, not promotion.

## What changed, and what the comparison identifies

R308 increased first-state mass fivefold without improving primary discrimination. R309 kept all-state day/episode weights for its 2x2 comparison and changed feature package and the linear preprocessing/model package independently. U->V compression alone did not improve first AUROC. U->W and V->X both improve first AUROC with a simple L2 logistic pipeline. X does not beat W AUROC, so there is no evidence that the fixed 10 inputs are sufficient to preserve all useful information in the original 99.

The pipeline includes training-only weighted clipping, imputation, standardization, missing indicators and episode-count-based regularization mass. Thus this comparison supports the PACKAGE as a promising intervention; it does not isolate tree capacity, clipping, scaling or regularization as the unique cause. Continuous + missing dimensions are 198 for W and 20 for X/F. C=.1 is fixed, without search or balancing. Original tree parameters remain unchanged.

First-only F trained on 76/68/80/67 observations with 9/4/7/7 positives respectively. It does not preserve initial ranking. This is consistent with limited early-example coverage or loss of useful state diversity, but cannot distinguish those explanations or prove that later observations are required. All-state fitting has 26/20/23/21 episodes with a positive state, still correlated trajectories; it is not thousands of independent successes.

## Day stability and uncertainty

| Held-out day | First positives / episodes | U AUROC | V AUROC | W AUROC | X AUROC | F AUROC |
|---|---:|---:|---:|---:|---:|---:|
| May5 | 0/21 | undefined | undefined | undefined | undefined | undefined |
| May6 | 5/29 | 0.658333 | 0.691667 | 0.766667 | 0.791667 | 0.666667 |
| May7 | 2/17 | 0.866667 | 0.133333 | 0.800000 | 0.833333 | 0.733333 |
| May8 | 2/30 | 0.607143 | 0.696429 | 0.714286 | 0.500000 | 0.535714 |

W improves two of three evaluable days; X improves only May6. Paired ticker-day AUROC gain CI: W-U **[0.021909,0.294445]**, X-U **[-0.097558,0.351145]**. Paired Brier difference CI: W-U **[-0.040923,-0.006957]**, X-U **[-0.043631,-0.007913]**. W's day-cluster AUROC CI **[-0.031250,0.226235]** includes zero. With four reused days, nine first positives and candidate selection conditioned on upstream OOF models that is not nested, none of these intervals establish independent transport or validate deployment.

## Additional descriptive pair decomposition

Added AFTER fitting, without any new fit, threshold or gate changes. See diagnostics/request309-pair-decomposition.json. Partition every positive-negative pair by whether both episodes come from the same day. Preserve original equal-day episode weights and .5 credit for tied probabilities. The partition reconstructs reported pooled AUROC exactly; within-day pairs contribute 23.75% of total pair weight.

| Arm | Within-day pair AUROC | Between-day pair AUROC |
|---|---:|---:|
| U | 0.718068 | 0.559449 |
| V | 0.504886 | 0.568521 |
| W | 0.767318 | 0.723757 |
| X | 0.746891 | 0.712115 |
| F | 0.662690 | 0.424871 |

Of W's pooled AUROC gain 0.136977, only 0.011699 comes from within-day pair improvement and 0.125279 from between-day pair improvement (91.46% of gain). X similarly gets 94.44% of its pooled gain from between-day pairs. These percentages are algebraic contributions, not causal explanations. Between-day comparison is part of probability transport, but the full gain must not be described as a corresponding gain in recognizing early momentum among today's candidates. Within-day discrimination also improves modestly; that is the relevant lead to investigate further.

## Signal versus background remains unresolved

W's standardized continuous coefficients are consistent in sign across all folds for price, known costs, time-of-day, previous-close return, 60-observation sign-flip rate, 30-second signed efficiency and 120-second signed-volume proxy. Coefficients are saved in the official JSON. These inputs are correlated and regularized: signs and magnitudes are descriptive associations, not causal feature attribution, true aggressor order flow or a rule to select entries. They show why it is necessary to separate the short-clock increment from price/cost/time/background effects before claiming momentum capture.

## Secondary fixed snapshots

| Snapshot | U AUROC | V AUROC | W AUROC | X AUROC | F AUROC | Episodes |
|---|---:|---:|---:|---:|---:|---:|
| +20s | 0.650516 | 0.576691 | 0.646905 | 0.559171 | 0.528976 | 92 |
| +60s | 0.586023 | 0.558959 | 0.709452 | 0.583255 | 0.463602 | 88 |
| +120s | 0.578327 | 0.674384 | 0.629675 | 0.649956 | 0.494993 | 82 |

All-state AUROC U/V/W/X/F: 0.638654/0.600706/0.698112/0.654220/0.623427. Anchors start at the first saved cash decision and retain original availability/gaps. They are secondary descriptions, not optimal delays or forced holds.

## Next fixed investigation

The next comparison must keep this pipeline fixed and compare background controls (current price, known transaction drag, clock and previous-close return) against the same controls plus predeclared clock momentum. The full 99-input W pipeline can be an exact replay reference, not a selected deployment strategy. Primary emphasis should be within-day positive-negative ranking as well as first-state AUROC/AP/Brier, because the large pooled gain chiefly reflects between-day comparison. Specify feature packages and criteria before fitting; do not search among features whose coefficients happened to look attractive here. Additional independent coverage remains necessary for validation. June stays sealed.

## Verification

58 targeted tests passed and critical Ruff checks passed locally and in official CI. New tests verify training-only preprocessing, preserving missing indicators, all-missing dimensions, unchanged regularization mass/predictions under repeated within-episode rows, held-day label isolation, exact first-only row counts, future-day rejection and saved-score mismatch abort. Every R306 C score replayed exactly. All linear fits converged with 12–28 iterations; no parameter adjustment was necessary. Official five-arm scores and JSON match local outputs with zero difference. Results persist in results/request309.json; official artifact includes scored states and source/log provenance. Research gate FAILED; no executable profit claim or model promotion.
