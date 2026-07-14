# Two-group rework: one first-class family, explicit approximations

## What exists today (and why it is being reworked)

Three modules implement two-group enrichment:

- `twogroup.py`: a WRAPPER that nests an arbitrary inner SuSiE family
  (`state.family_state.inner_family_state`), swaps `data.y` for a derived
  response through `SimpleNamespace`, resolves inner steps by module
  introspection (`import_module` + `TWOGROUP_RESPONSE` module attributes +
  step-name conventions), and re-wraps every schedule step. Two injection
  modes (Ez regression / llr pass-through). The intercept-step name
  (`update_intercept_step` vs `estimate_intercept_step`) is load-bearing:
  the marginal module names its step differently *so the wrapper cannot
  find it*.
- `twogroup_marginal.py`: the good idea -- z marginalized analytically
  (`response.TwoGroupMarginal`), each per-feature fit an exact-marginal SER
  via `glm_ser` -- but still driven THROUGH the wrapper for f0/f1 EM and
  data prep, so it inherits the nesting and the naming trick.
- `twogrouplocaljj.py`: the old fused-EM localjj base (soft Ez labels,
  JJ bound), superseded by the marginal (milestone 25: retire).

The rework: ONE module, the marginal method, built directly on the glm
machinery -- the same pattern as `cox_poisson`: family-specific coupling
(here f0/f1 and the llr they induce) quarantined into engine-refreshed
per-row state, everything the kernels see per-observation. No inner/outer
state nesting, no introspection, no response injection, no name-based
dispatch.

## The model

Per observation i: summary statistic `bhat_i` with standard error `se_i`,
latent membership `z_i in {0,1}`,

    z_i ~ Bernoulli(sigmoid(eta_i)),   eta_i = b0 + x_i' Sum_l b_l gamma_l
    bhat_i | z_i = 0 ~ (f0 * N(0, se_i^2))   (normal-means convolution)
    bhat_i | z_i = 1 ~ (f1 * N(0, se_i^2))

with a SuSiE prior on the effects and empirical-Bayes f0/f1. Define
`llr_i = log f1(bhat_i; se_i) - log f0(bhat_i; se_i)`.

## The estimation scheme, step by step, with every approximation named

1. **z is integrated out analytically -- EXACT.** The per-observation
   marginal is `loglik_i(eta) = softplus(eta + llr_i) - softplus(eta)`
   (+ `log f0`, eta-free). This is `response.TwoGroupMarginal`; there is no
   E-step over z anywhere in the effect fits. (The old EM path regressed on
   a soft label Ez and paid for it with diffuse PIPs.)

2. **Per-feature b integration -- Gauss-Hermite on the exact marginal.**
   `glm_ser`: Newton with the MM majorizer weight `w(eta) = mu(1-mu)`
   (the true curvature `w(eta) - w(eta+llr)` is indefinite; the majorizer
   makes the ascent monotone), then an `order`-node GH rule centered at the
   stationary point with the majorizer-Laplace width. Approximations:
   (a) quadrature error, vanishing in `order` for integrands covered by
   the proposal; (b) the marginal in b is non-log-concave, so the Newton
   fixed point is a local mode -- the GH tail integrates the TRUE loglik so
   mild multimodality inside the proposal width is captured, mass far
   outside it is not; (c) the proposal width uses the majorizer curvature
   (>= true curvature), i.e. it is conservative (wider), which is the safe
   direction for quadrature coverage.

3. **Across effects -- gIBSS message passing.** Each effect update sees the
   leave-one-out predictor. Base case: the LOO posterior MEAN only (the
   plug-in that defines gIBSS). Optional: `Smoothed(TwoGroupMarginal(),
   GH(k))` treats the LOO message as a random offset `o ~ N(mean, var)` and
   integrates it with a nested k-node GH -- the only smoother valid for
   this family (Taylor/TaylorFixed require an exact cumulant curvature,
   which the majorizer is not; JJ bounds are Bernoulli-specific). The
   factorization across effects remains variational either way.

4. **f0/f1 -- generalized EM with a plug-in E-step.** After each sweep, one
   M-step each: `f0 <- update_nm(bhat, se, 1-Ez)`, `f1 <- update_nm(bhat,
   se, Ez)` with `Ez_i = sigmoid(eta_i + llr_i)` evaluated at the current
   predictor. Approximations: (a) eta is plugged in at its posterior mean
   (under a `Smoothed` response the E-step instead GH-averages over
   `o ~ N(mean, var)`, matching the fit); (b) one M-step per sweep
   (generalized EM), interleaved with the effect updates rather than run to
   convergence; (c) the M-steps themselves maximize the expected
   complete-data loglik given that plug-in q(z) -- exact EM structure, only
   the E-step input is approximate.

5. **The intercept -- one EM coordinate update per sweep, AFTER the
   effects.** The marginal loglik saturates (`-> llr_i` as `eta -> +inf`),
   so at weakly structured eta the b0-alone objective is maximized on the
   boundary `b0 -> +inf` ("everything enriched"); the interior stationary
   point is only a local max. A direct Newton runs off; iterated EM would
   also climb there eventually (each EM step monotonically increases the
   marginal). The scheme: freeze `Ez` at the current b0 (single E-step),
   run the CONCAVE logistic M-step in b0 to convergence (always finite),
   once per sweep, ordered after the effect updates so eta is already
   structured. This is a deliberate LOCAL ascent that stays at the interior
   fixed point; it is initialization- and ordering-dependent by design.
   `intercept="profiled"`/`"null"` are refused: the per-feature profile and
   the b=0 null intercept hit the same boundary mode.

6. **Initialization -- covariate-free two-group EB.** Before the sweeps,
   alternate the classic no-covariate two-group EM: `Ez = sigmoid(b0 +
   llr)`, f0/f1 M-steps, `b0 <- logit(mean(Ez))` (the exact intercept
   M-step when there are no effects: prior enrichment = mixing weight).
   This replaces both the wrapper's `estimate_f_step` and the b0 = 0 start:
   the sweeps begin from the standard two-group fit and add covariate
   moderation. Same boundary caveat as any two-group EB when f0 and f1 are
   both fully free; anchoring f0 (e.g. `PointMass`/fixed null) is the
   standard practice and the default.

7. **Prior variance per effect** -- the existing EB step
   (`linear.estimate_prior_variance`), maximizing the SER evidence in the
   prior variance. Evidence is RELATIVE to the b=0 baseline at the current
   offset (glm convention); the baseline is pv-free so the argmax is
   unaffected.

8. **intercept_var** (fed to the smoothed E-step/aux as part of ov) is
   recorded from the M-step curvature `1/sum mu(1-mu)` -- complete-data
   information >= observed information, so this UNDERSTATES the intercept's
   variance. It only enters as an O(1/n) additive term in ov.

## Architecture

`TwoGroupFamilyState(glm.GLMFamilyState)` -- a frozen subclass adding
`f0, f1, llr, update_f0, update_f1, init_em_iters`. The kernels are reached
through `glm._fit_effect` with a family-built aux (`llr` in the slot where
glm puts `data.y`, plus ov when Smoothed): the kernel dispatch, profiled
machinery, and Smoothed plumbing are reused verbatim; nothing in glm.py
changes. `TwoGroupData(LinearData)` carries `bhat`/`se` next to the design
(op/pre-centering reused; `data.y = llr0` at prep time is a placeholder the
family never reads back).

Schedule (per sweep):

    snapshot
    [subtract message, update effect, prior variance, add message] x L
    update_intercept_step        # EM-interior, after effects (see 5)
    update_mixture_step          # f0/f1 M-steps + llr refresh (see 4)
    convergence (alpha SKL)

`fit(X, bhat, se, f0=..., f1=..., L=...)` is the one-call entry point.

Kernel/intercept axes: `kernel="quad"` (default) and `"vi"` (with a
Smoothed response); `intercept="shared"` only (see 5); `"linear"`/`"jj"`
refused by the inherited GLMFamilyState validation (response not
quadratic / not Bernoulli).

## What is deleted / migrated

- `twogroup.py` -- REPLACED by the new family (wrapper machinery deleted:
  response injection, schedule wrapping, introspection, Ez_override).
- `twogroup_marginal.py` -- absorbed; `fit` moves here.
- `twogrouplocaljj.py` -- retired (milestone 25); the marginal subsumes it.
- Thresholding modes (`hard_threshold_Ez_step`/`lfdr_threshold_Ez_step`)
  are not two-group methods -- they binarize and run logistic SuSiE. Users
  who want them: threshold outside and call `methods.fit_glm_susie`.
- `gseasusie.fit.fit_gsea_susie_twogroup` -- rebuilt on `twogroup.fit`
  (the marginal method); the base_method/variant config jungle goes away.
- Tests: `test_twogroup_marginal.py` migrates to the new API (and joins the
  default suite -- it was accidentally caught by the `test_twogroup_*`
  quarantine glob); `test_gibss_twogroup.py`/`test_gibss_twogrouplocaljj.py`
  are replaced by the new module's tests; `test_gsea_twogroup.py` updates
  with the gseasusie migration.
