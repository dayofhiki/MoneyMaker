# Request246 findings — direct action advantage improves control but FAILS

Authoritative run: `36261741736`  
Artifact: `10913180476`  
Repository tests and Flat Files checks passed.

No new dates were opened. No class balancing or May-tuned threshold was used.

## Verdict

Request246 **fails the full gate**, but it validates the core Request245 diagnosis.

Replacing cross-scale absolute-value regression with direct action-preference classification materially improves the controller:

| Metric | Request243 refit | Request246 |
|---|---:|---:|
| Entries | 173 | **114** |
| Resolved trades | 100 | **75** |
| Unresolved trades | 73 | **39** |
| Resolved BASE mean | -1.6464% | **-0.8747%** |
| STRESS mean | -3.6270% | **-2.9492%** |
| Severe loss <= -2% | 48.0% | **34.67%** |
| Initial WAIT actions | 442 | **1** |
| HOLD actions | 1,437 | **511** |

Resolved means condition on different subsets and are not paired portfolio returns, but the direction of the control change is substantial.

## Training support

- ENTER-better-than-WAIT/cash:
  - 1,146 rows
  - 358 positive, 788 negative
  - positive rate **31.24%**
- WAIT-positive conditional on ENTER not being best:
  - 788 rows
  - **3 positive**, 785 negative
  - positive rate **0.38%**
- HOLD-better-than-EXIT:
  - 5,866 rows
  - 3,047 positive, 2,819 negative
  - positive rate **51.94%**

The WAIT bottleneck is therefore not a classifier-threshold problem. The chronological teacher supplies essentially no positive WAIT continuation examples.

## Decision quality

On comparable reachable flat states:

- realized best-action accuracy: **67.77%**
- chosen ENTER rows: **74**
- chosen ENTER truly beats WAIT and cash: **36.49%**
- chosen ENTER mean advantage over best alternative: **-1.0000%**
- chosen WAIT rows: **1**
- chosen WAIT truly advantageous: **0%**
- ABSTAIN rows: **257**

The direct classifier is much better at refusing marginal opportunities, but its chosen entries still do not have adequate economic discrimination.

On reachable position states:

- chosen HOLD rows: **363**
- chosen HOLD positive realized advantage: **60.33%**
- chosen HOLD mean advantage: **+0.2909%**

This is the first clear evidence in this branch that directly learning an action comparison can improve a recurrent decision component. The HOLD intervention passes its preregistered precision requirement.

## Interpretation

Two conclusions are now separated.

1. **Action interface problem confirmed.**  
   Directly learning HOLD-vs-EXIT works materially better than comparing a noisy absolute HOLD regressor to zero. Direct flat classification also reduces overtrading and severe losses.

2. **WAIT teacher problem confirmed.**  
   Request246 cannot become a patient entry controller because the Request243 teacher almost never supplies a positive WAIT continuation. Only 3 training rows say that WAIT is beneficial after ENTER has been rejected. No honest classifier can learn robust waiting from that support.

The remaining negative BASE mean is therefore not evidence that direct action learning is useless. It says the flat-state teacher target still describes the wrong redecision process.

## Next research boundary

Request247 should replace the sparse learned-teacher WAIT continuation with a fixed **one-observed-event redecision bootstrap**.

For each training state:

- ENTER-now value remains the forced ENTER value under the same frozen position policy;
- WAIT-one-event value is the forced ENTER value at the **next actually observed eligible state**, not the learned flat teacher's later choice;
- require the next state to remain inside the causal entry window;
- compare ENTER now, WAIT one observed event, and cash;
- train the same direct action classifiers without class balancing or tuned thresholds;
- at inference, re-evaluate after every WAIT, allowing consecutive waits through receding one-step decisions;
- retain Request246's direct HOLD-vs-EXIT classifier.

This is not a best-future-price target. The WAIT horizon is fixed in advance to exactly the next observed eligible decision state. It tests whether the model can learn the trader behavior the project actually wants: reject the current price, observe one more state, then reassess.
