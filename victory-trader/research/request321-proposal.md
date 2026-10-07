# Proposed R321: elapsed drift versus nearby state evolution

This is a next-study proposal, not an executed experiment or a completed
preregistration. Freeze precise inputs, support, metrics and checks separately
before new fits/evaluation. R320's conditional timing gain is promising, while
the size of elapsed-time/observation-count coefficients leaves an unresolved
distinction: earlier-versus-later opportunity decay, versus discrimination of
nearby evolving states. The coefficients do not establish that time drift caused
the improvement. Keep R320 frozen and do not choose V over W as a fallback.

Reuse only the pinned official R319/R320 trajectories and labels; no acquisition,
new window search, policy/cost/stop change or sealed periods. Fit ONE elapsed-only
pair learner on the SAME50 mixed episodes/all47,825 training pairs, using the
exact same weight mass, parameters and training-only transform contract as R320.
Its feature allowlist should be limited to the three predeclared R319 history
support variables. Compare against fixed R320 U/V/W and matched state Z scores.
This control estimates how far observation age/count alone can go; it does not
provide a trading policy or redefine which evaluation cases are scored.

Evaluate two separately reported questions. Preserve original whole-episode
timing rank as an anchor. The priority diagnostic restricts positive-negative
evaluation PAIRS within the same episode to clocks at most30s apart, a fixed
proximity inherited from the existing lag package, not a searched entry delay.
Retain every original observation/episode and report pair/episode availability;
episodes without both labels in such a pair are explicitly nonevaluable for
that conditional diagnostic. No selecting the best pair, phase or future score.
Give each eligible episode equal mass within day and each eligible pair equal
mass within episode, with whole-day/ticker-day resampling and original weights.

Predeclare interval comparisons W versus elapsed-only and W versus U, require
both-scheme positive intervals and adequate independent local-pair support,
and report all failed gates. Compare V only as the already fixed secondary
ablation; do not select it as a new model. An elapsed-only control plus fixed
proximity cannot perfectly remove every time/quality confound, but it tests a
specific alternative explanation without another static-interaction search.

If conditional price/activity timing survives this check, the next distinct
question is its value for absolute remaining-opportunity admission. A potential
two-stage study must generate training timing features only from preceding-day
fits, normalize scores using training data alone, compare matched controls on all
broad episodes, and preserve out-of-date evaluation. No own-day fitted scores may
be fed into a meta-classifier. Sequential WAIT/ENTER/HOLD/EXIT profitability
remains a separate contract after those signal questions; no intermediate rank
result justifies opening the sealed periods or claiming profits.
