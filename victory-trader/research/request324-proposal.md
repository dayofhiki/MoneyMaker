# Proposed R324: separate initial opportunity from state transition inference

This is an unexecuted proposal, not a completed preregistration. Pin exact input,
feature/transform contracts, weighting, support/gates and initial-score source
before fitting or evaluating a next model. Do not tune alpha after R323's failure,
select V as fallback, or pretend a new absolute head was already improved.

R321's nearby timing survives its elapsed control; R322's chronological bridge
fails absolute admission and R323's fixed joint score trades admission for timing.
That particular shared linear score is not evidence that all multitask models
fail. A different mechanism should explicitly propagate an estimated opportunity
state from the past rather than rank every current point with one shared logit.

Let latent y_t mean the inherited reachable BASE net>=5 opportunity label. It is
a hindsight research target, not an observable online market state or realized
return. Train separate conditional transition heads q01(x_t)=P(y_t=1|y_previous=0,
x_t) and q11(x_t)=P(y_t=1|y_previous=1,x_t) on adjacent ORIGINAL observed states
with both labels complete. Do not bridge over censored endpoints in training;
retain all observations for history and evaluation. Training previous labels may
condition the target head, but NEVER enter live/held inference or reset its state.

Initialize p_first from a separately frozen causal initial absolute classifier.
At later observed completed clocks, propagate only predictions:
p_t=(1-p_previous)*q01(x_t)+p_previous*q11(x_t).
Use current causal price/activity/history and actual preceding observation gap,
not a future endpoint or target. Gap may be large; include/audit it explicitly,
do not treat all observations as equally spaced or forward-fill market features.
Keep missing/censored observations in the inference chain, while their labels
remain unused. All initial and transition fits at held dates must use only past
training dates. No teacher-forced true previous label at evaluation, retrospective
reset, peak/phase selection, future episode mean or future-best state.

Preregister one rich trajectory head, a time/gap-only transition control, a Q/history
transition control and constant-transition control. SAME frozen initial score
for all controls isolates the learned update. Compare against unchanged current-
state absolute classifier and own first-only scores on ALL655 complete episodes,
plus the inherited conditional whole/local timing diagnostics. Fit training-only
transforms and equal independent episode/day weights separately by previous-label
support; do not use synthetic independent pair counts or calibrate on evaluation.

The training-only feasibility audit finds8445 adjacent complete transitions from
286 episodes;0→1 has88 transitions across38 episodes and1→0 has99 across48,
both on all4 training dates. These are not independent event counts. Freeze
minimum independent onset/decay support and explicit no-support fallbacks before
fitting. Recursive inference can propagate prediction errors, especially across
long gaps; support diagnostics, preceding-day folds and future-mutation/no-label
inference tests are necessary. The approximate transition contract is not a
claim of a fully identified Markov process or optimal Bayesian filtering.

Require independent cluster evidence for BOTH absolute admission/calibration
and conditional timing against matched controls, preserve all failed gates and
frozen W comparisons. No feature/window/solver/initial-model search, threshold
policy, profit claim, promotion, acquisition or sealed June HOLD/July-August.
This is a hypothesis to test, not the established explanation of R323's failure.
