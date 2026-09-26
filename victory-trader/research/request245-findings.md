# Request245 findings — action-value calibration FAIL

Authoritative run: `36261424680`  
Artifact: `10912557316`

No new dates were opened. No threshold, policy action or model capacity was tuned from the May11-20 calibration result.

## Verdict

Request245 **fails all five preregistered adequacy checks**.

The Request243 sequential controller is not merely unprofitable; its independently trained action-value heads are badly miscalibrated for direct action comparison.

## Student target support

The clearest structural defect is WAIT support.

| Target | Valid rows | Mean | Std | Positive | Exactly zero |
|---|---:|---:|---:|---:|---:|
| ENTER | 1,146 | -0.5648% | 2.0187% | 31.41% | 0.09% |
| WAIT | 1,380 | +0.0049% | 0.1864% | 0.58% | **99.13%** |
| HOLD advantage | 5,866 | +0.2292% | 1.6046% | 51.94% | 5.59% |

WAIT target standard deviation is only **9.24%** of ENTER target standard deviation.

A regression head trained on a target that is almost always exactly zero is then compared directly against a wide-scale ENTER regressor and against a literal zero action boundary. Tiny positive regression noise can therefore become a WAIT action even when the teacher supplied essentially no positive continuation support.

## Calibration on decision-reachable May states

### ENTER

- resolved comparable rows: **647**
- Spearman(predicted, realized): **0.0578**
- mean prediction: **-0.6109%**
- mean realized: **-1.5379%**
- optimism bias: **+0.9270 percentage points**
- predicted-positive sign precision: **22.0%**

The model barely ranks ENTER value and materially overstates it.

### WAIT

- comparable rows: **740**
- Spearman: **-0.0362**
- mean prediction: **+0.00164%**
- mean realized continuation: **-0.19685%**
- predictions > 0: **63.51%**
- realized continuation > 0: **3.38%**
- positive-prediction precision: **4.04%**

This is the strongest diagnosis. The WAIT head predicts tiny positive values very often despite almost no positive realized support.

### HOLD

- reachable position rows: **1,016**
- Spearman: **0.0350**
- mean prediction: **+0.3965%**
- mean realized HOLD-minus-EXIT advantage: **-0.2222%**
- optimism bias: **+0.6187 percentage points**
- predictions > 0: **92.62%**
- realized positive advantage: **42.13%**

The controller therefore over-HOLDs.

## Direct action comparisons

For 631 flat states where ENTER and WAIT outcomes are both identified:

- predicted ENTER-minus-WAIT vs realized Spearman: **0.1482**
- chosen ENTER rows: **93**
- chosen ENTER actually beat both WAIT and cash: **18.28%**
- chosen ENTER mean advantage over best alternative: **-1.7770%**
- chosen WAIT rows: **342**
- chosen WAIT actually beat both ENTER and cash: **1.75%**
- chosen WAIT mean advantage over best alternative: **-0.6950%**
- realized best-action accuracy: **25.67%**

For HOLD/EXIT:

- chosen HOLD rows: **941**
- chosen HOLD had positive realized advantage: **42.30%**
- chosen HOLD mean advantage: **-0.2221%**
- chosen EXIT correctness: **60.0%**

Every preregistered gate check fails.

## Interpretation

Request243 successfully created the *mechanics* of a recurrent trader but used the wrong numerical interface between learning and action.

Three separately trained regressors produce values with very different target distributions and calibration errors, yet the controller treats them as directly comparable dollar-like quantities. In particular:

1. WAIT is almost unsupported by the weak teacher, but tiny positive regression noise triggers WAIT.
2. ENTER is optimistic by nearly one percentage point and has almost no ranking power.
3. HOLD is positive almost everywhere in prediction, so positions are held too long despite negative realized incremental value.

This explains why adding WAIT/HOLD mechanics in Request243 did not improve economics.

## Next research boundary

Request246 should test a **direct action-advantage controller**, not another absolute-value regressor.

On the same chronological training/calibration blocks:

- classify whether ENTER beats both WAIT and cash using realized same-teacher counterfactual values;
- separately classify whether WAIT beats cash when ENTER is not preferred;
- classify HOLD vs EXIT directly from HOLD advantage;
- use the natural 0.5 classifier boundary only, with no May-tuned threshold;
- preserve causal features, costs, chronology and unresolved-value exclusions;
- report class support explicitly, because the weak teacher may provide too few genuinely positive WAIT examples.

If WAIT-positive support is too sparse to learn, do not manufacture it with class balancing or a tuned threshold. That failure would mean the next intervention must improve the chronological teacher/bootstrapping process rather than the classifier.
