# R312 — local elapsed-evidence intervention fails to improve discrimination

Local execution completed against official R306/R310/R311 artifacts with frozen numerical versions. Preregistration commit `cd31983` precedes fits. **No official GitHub run, remote branch, merge or policy promotion has occurred:** automatic approval review rejected publishing pending explicit user authorization. Python3.12 local execution needs official Python3.11 reproduction after approval.

Completed-print counts/gaps refer to observed one-second aggregate bars/distinct traded seconds, not individual tick-level transactions; each bar can contain many trades. `n` is the separate transaction-count input.

All828 original identities retained;679 available observations;650 complete labels;38 net5-positive cases. All original4,526 training clock inputs and available evaluation inputs replay. Canonical first observations and fixed risk/cost labels replay. Frozen M probabilities match R311 to maximum absolute error1.11e-16. Eight new integrity/causality tests plus inherited tests total **80 passed**, critical Ruff and compile checks pass.

| Training support | Package | Same-day AUROC | Pooled AUROC | AP | Brier (lower is better) |
|---|---|---:|---:|---:|---:|
| Original all-state frozen | M existing | .640588 | .642512 | .099204 | .056287 |
| Original all-state frozen | K temporal support | .612924 | .618464 | .081100 | .057382 |
| Original all-state frozen | T elapsed direction | .626583 | .625925 | .088133 | .056958 |
| Expanding first observations | M existing | .584681 | .577752 | .076858 | .059606 |
| Expanding first observations | K temporal support | .598062 | .572296 | .070776 | .060966 |
| Expanding first observations | T elapsed direction | .561273 | .544168 | .070660 | .063826 |

Frozen T-M same-day rank change **-.014005**, ticker-day95% CI[-.073668,.037972]. Expanding T-M **-.023408**, CI[-.080741,.028600]. Expanding T-K **-.036788**, CI[-.126827,.054912]. None establishes a beneficial or definitively harmful ranking effect. Expanding T increases Brier by .004220 versus M, ticker-day CI[.001172,.007375]; day CI[.001300,.007071] also positive. These intervals condition on scores, not retrained histories, so do not capture every training dependency. The probability-error deterioration is a clearer adverse indication than rank uncertainty.

Only **2/9 preregistered checks pass** (positive counts/days). Coverage remains78.50%; no coverage gate relaxation. Frozen T beats M on3/8 dates; expanding T beats its matched M on3/8 dates. No post-result arm, feature, threshold or seed selection. Models keep C=.1 and equal-day/episode independent mass. Expanding training grows97→653 first episodes and9→43 positive training episodes before May20; scoring each date uses only strictly earlier dates. More matching samples did not demonstrate improvement.

## What this result does and does not establish

The timestamp ambiguity identified in R311 is real, but adding explicit support and elapsed rates to this fixed linear pipeline did not solve it. This rejects the tested representation as a sufficient intervention. It does not disprove all time-aware representations, and does not prove that observation density or momentum is irrelevant. The support control cannot distinguish directional causality by itself; the full directional contrast includes missingness and nominal signals.

Between frozen and expanding tracks, both trajectory position/weighting and training population/sample size change. Their difference cannot identify a pure population effect. Historical97 first observations also retain label-eligible shortlist conditioning. Eight previously explored May dates,38 first winners, incomplete fills and censored identities restrict inference. None of these results represents a realized trading return.

## Next registered question

Freeze R312 and stop adding features based on these outcomes. R313 should use the unchanged R311 M model to test an outcome-independent information-arrival observation: first completed print within the original five-minute window with at least two completed prints in the last10s. It may coincide with the original first observation. Select using timestamps alone before labeling; retain absent/unresolved identities; keep cap/cost/absorbing stop fixed.

Compare original and evidence-arrival observations on the SAME paired labeled identities; separately report opportunity retention/loss/gain and time-to-observation. The entry-time labels differ, so any rank gain is not a pure same-target model effect or evidence for a trading rule. This is a bounded diagnostic of information arrival versus waiting cost, not a forced wait policy. No delay/threshold search, new fit, market acquisition or sealed dates.
