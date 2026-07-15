# The SER evidence contract (feature_log_marginal / null_log_marginal / BF)

## The problem this fixes

`BaseSERState.feature_log_evidence` was underspecified: it was "the per-feature
logBF up to a shared, feature-independent baseline that cancels in alpha." That
baseline was a per-kernel choice, so the same field held different things across
families -- the glm stack stored the RELATIVE BF (M_j - M_0, with
null_log_likelihood = 0), while linear / cox / the legacy JJ modules stored the
ABSOLUTE marginal (M_j) with a real null. Both yielded a correct `ser_log_bf`
(because each set its own null consistently), but `feature_log_evidence` itself
was not comparable across kernels, and nothing in the contract forced it to be.

## The pinned contract

Every kernel stores, on ITS OWN scale (its own approximation / working
likelihood):

- `feature_log_marginal[j]` -- the ABSOLUTE per-feature log marginal likelihood
  of the single-effect model with the effect on feature j (b_j integrated under
  the prior). A length-p array.
- `null_log_marginal` -- the SER's b=0 null log marginal, on the SAME scale. It
  is feature-independent, so it is ONE scalar per SER (per `single_effects[l]`),
  and it is the b=0 null at THAT effect's leave-one-out offset (hence per-effect
  across the L components, and conditional on the other effects).

Derived (properties, one definition, comparable by construction):

- `BaseSERState.feature_log_bf = feature_log_marginal - null_log_marginal`
- `BaseSERState.ser_log_bf   = marginal_log_likelihood - null_log_marginal`
  where `marginal_log_likelihood = logsumexp(feature_log_marginal) - log p`.
- `GIBSSState.ser_log_bf` aggregates over effects; `ser_log_bayes_factor` is a
  historical alias.

## Why it is comparable (the one invariant)

`feature_log_marginal` and `null_log_marginal` must be computed by the SAME
functional -- one at the fitted b_j, one at b=0. Then any eta-free base-measure
constant (log f0 for two-group, -1/2 log(2 pi sigma^2) for Gaussian, ...) appears
in BOTH and CANCELS in `feature_log_bf`. So the BF is comparable across kernels
up to genuine approximation error, never up to an arbitrary bookkeeping constant.
The failure mode to avoid: computing the marginal one way and the null another
(e.g. an ELBO marginal minus an exact null -- scale mismatch). What legitimately
differs between kernels is which null MODEL they use: shared/quad use b=0 at the
offset; the profiled kernels use the profiled null (b0 maximized at b=0); the
legacy JJ modules use their profiled JJ null. The stored `null_log_marginal`
makes that choice explicit and inspectable rather than a hidden convention.

## Where it is enforced

`response_ser.build_ser_state(mu, var, feature_log_bf, coefficient_kl,
prior_variance, null_log_marginal)` is the choke point for the glm stack: it
takes the kernel's native relative BF plus the null and stores
`feature_log_marginal = feature_log_bf + null_log_marginal`. The null is supplied
by `glm._null_log_marginal(fs, aux, offset)`, which recomputes the b=0 baseline on
the kernel's scale (one terms pass; a scalar Newton `_profile_null` when profiled;
the fixed-tilt bound at xi0 for jj) -- matching exactly what each kernel subtracts
internally, so no kernel signature had to change. cox_poisson threads its PL(0)
null; linear / cox / legacy already stored the absolute marginal + real null and
only needed the field rename.

## Caveat

The reference is b=0 at the current (leave-one-out) offset, so `ser_log_bf` is
same-data comparability (conditional on the other effects), NOT an absolute model
evidence -- there is no absolute log p(y | null) in a coordinate-ascent SER.
