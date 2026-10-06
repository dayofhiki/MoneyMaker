# R312 — elapsed-time evidence and first-observation support

Registered before new feature fits. R311 remains frozen: its 828 identity-hash sampled May11–20 HOT cases, 679 available observations and 650 complete labels are reused without acquisition, replacement, later-observation substitution or outcome-based filtering. June HOLD and July–August stay sealed. This is previously explored May development, not pristine validation.

Terminology: a completed "print" here is an observed one-second aggregate bar, not an individual tick/trade. Counts measure distinct traded seconds; `n` separately records aggregate transaction count. Inter-print gaps are gaps between aggregate observation timestamps. No tick-level tape or quote spread is reconstructed.

## Fixed inputs and pipeline

Three packages, unchanged R309 L2 logistic C=.1, seed20264705, training-only weighted 1%/99% clipping, median imputation, scaling and missing indicators. Original equal-day/episode weights sum to independent training episode count. No parameter, feature, date, threshold or seed search.

* M: unchanged R310/R311 28 clock/background/coverage inputs.
* K: original Q nine inputs plus ten temporal-support inputs: preceding completed-print inter-print gap; for each 10/30/120s window, nominal return baseline age measured from decision, observed internal span (zero for one print, missing for no print), and maximum inter-print gap from the baseline through the current print (missing without baseline). Original activity fraction already supplies observation count.
* T: K plus the unchanged 19 nominal clock-signal inputs and seven elapsed-direction inputs. For each window, 100*log(current/baseline) divided by actual completed-print elapsed seconds, and 100*log(current/first internal print) divided by internal span, missing for fewer than two internal prints. Acceleration is the difference of the 10s and 30s baseline log rates. Units are percent-log per second, not nominal-window percent returns.

Only second bars ending at/before decision enter features. No forward fill, interpolation, trading-density filter or stale-price cutoff. Missing historical evidence stays missing and is transformed using training rows only. T-K tests the whole directional/missingness package beyond explicit temporal support; it cannot prove a pure causal momentum effect. T-M tests the full representation intervention.

## Two fixed support tracks

Frozen track: fit M/K/T once on all original R310 May5–8 states (97 episodes / 4,526 rows), score every available R311 observation; M must replay official R311 probabilities within 1e-9. This isolates the input intervention at unchanged historical training support.

Primary expanding-first track: at each evaluation day fit M/K/T using the original 97 first saved observations plus every complete first R311 observation on STRICTLY earlier evaluation dates. Score all available cases on the current date, including unresolved labels. Held/later-day labels and preprocessing never enter the fit; retain per-fold fit dates, positive counts, independent mass and transform audits. At May11 only the original97 first observations are available. No missing case is replaced or fabricated. Training support changes both trajectory position and population/sample size relative to the frozen track; between-track differences cannot attribute a pure population effect. The historical97 first observations retain their inherited label-eligible selection limitation. This chronological diagnostic does not reopen any gate-dependent HOLD evaluation or implement a live retraining policy.

## Fixed evaluation and uncertainty

Same complete 650 R311 first cases for both tracks. Report every828 identity and missing reason; all679 available scores saved. Primary endpoint: expanding-first T-M same-day weighted positive-negative AUROC. Secondary: T-K directional increment, K-M mechanism control and the same contrasts in the frozen track. Pooled AUROC/AP/Brier, per-day metrics and per-fold training prior Brier accompany ranks. No entry threshold or realized portfolio PnL.

1,000 paired bootstrap draws per day and ticker-day clustering, seed20264750 + cluster index, original evaluation weights retained and invalid rank draws counted. Intervals condition on fitted predictions and do not refit training histories, so they do not capture all model-training dependence in the expanding track. Both contrasts are fixed before fits; no best-arm selection or multiple-comparison confirmation claim.

Gate requires available/complete identity coverage >=90%, >=20 positives on >=4 days, expanding T same-day AUROC and AP above M and K, T Brier below M/K and fold training prior, T-M per-day AUROC improving on a majority of evaluable dates, and ticker-day same-day T-M and T-K gain lower bounds >0. The inherited coverage deficit will keep this gate closed unless a separately registered data study resolves it; no gate relaxation. Point improvements remain diagnostic even if coverage fails. Model/policy promotion stays false regardless.

## Integrity and outputs

Use official R310 run37218791436, R306 raw run37211254516 and R311 run37223064559 artifacts. Rebuild all original and available evaluation clock inputs; replay first-observation and absorbing-stop net5 labels; check exact cohort identities, dates and saved packages. Keep original costs, -2.5% BASE stop, next-print pending fills and HOT+60min/session cap. No network market requests. Hash declared input files; checkpoint expanded training features, all828 scored identities, JSON metrics/fit audits and source SHA. One consolidated official request312 on merge; findings-only commits do not dispatch.
