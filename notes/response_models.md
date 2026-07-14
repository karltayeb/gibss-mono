# Response-model SER: plan & core structure

> The full response x kernel menu (validity grid, Bernoulli/Poisson/Gaussian
> specializations, coincidences, old-module mapping) lives in
> `notes/response_kernel_tables.md`.

## Goal
Marginalized two-group SER **without EM** — integrate the discrete latent `z`
analytically, fit the resulting per-observation marginal likelihood with the
existing per-column SER machinery. Generalize the "what likelihood" into a
**ResponseModel** so logistic / two-group / Poisson / ... are one kernel, different
response.

## Core structure (the underlying invariant)
Every per-column SER fit is: maximize/integrate over `b`
```
  sum_i loglik_i(offset_i + x_ij b)   +   log prior(b)
```
The ONLY thing that changes between families is the per-observation
`loglik_i(eta)`. Expose it as three functions of the linear predictor:
```
  ResponseModel.terms(eta, aux) -> (loglik, grad, weight)
    loglik : per-obs log-likelihood at eta
    grad   : d loglik / d eta            (the "working residual", used in the gradient)
    weight : curvature for Newton/Laplace (>0; Fisher/majorizing), used in the Hessian
```
`aux` = per-observation auxiliary data (y for logistic/Poisson, llr for two-group).

The kernel is then response-agnostic:
```
  grad_b  = local_moment(1, grad) - b/pv
  curv_b  = local_moment(2, weight) + 1/pv          # Newton / Laplace precision
  dll     = local_moment(0, loglik(eta) - loglik(offset))   # GH-node evidence
```

## Two-group marginal (the key derivation)
z_i ~ Bernoulli(sigma(eta_i)) prior; data ~ f1 if z=1 else f0; llr = log f1/f0.
Marginalize z (discrete -> closed form, NO EM):
```
  loglik_i(eta) = log[ sigma(eta) e^{llr} + (1 - sigma(eta)) ]      (+ const log f0)
  grad_i(eta)   = sigma(eta + llr) - sigma(eta)  =  Ez - mu         # Ez = sigma(eta+llr)
  hess_obs      = w(eta) - w(eta+llr)   (non-definite -> mixture non-log-concave)
  weight        = w(eta)  (Fisher / EM curvature; positive, robust)  # majorizer
```
So the marginal *score* is the logistic score with `y := Ez` (why EM's M-step is
logistic-on-Ez). Direct fit uses `grad = Ez - mu`, `weight = w(eta)` (Fisher
scoring; recomputes Ez each Newton step -> no outer EM loop), then GH-over-b gives
the exact per-feature marginal (the part EM never did).

Bernoulli:  loglik = y*eta - softplus, grad = y - mu, weight = w.
Poisson:    loglik = y*eta - exp(eta), grad = y - exp(eta), weight = exp(eta).
Gaussian:   loglik = -(y-eta)^2/2v, grad = (y-eta)/v, weight = 1/v. The per-effect
            integrand is exactly Gaussian so GH is exact -> linear SuSiE = GLM(Gaussian),
            and glm_ser reproduces the closed-form single-effect BF/mu to ~1e-13. This
            is the whole point: every family (Gaussian/logistic/Poisson/two-group) is
            one kernel, one ResponseModel.

## Paths considered
- (A) Standalone twogroup marginal kernel — fast to validate, not general. (stepping stone)
- (B) **Response seam in ser_ops per-column kernels** — the real structure; logistic
      is just `Bernoulli`. Chosen.
- Cox: partial likelihood is NOT per-observation separable (risk sets couple obs);
  does not fit `terms(eta, aux)` directly — but see milestone 14: the Cox-Poisson
  (Breslow) reduction quarantines the coupling into a per-row engine-tuned quantity
  and the rest IS the seam.

## Milestones
1. [done] `response.py`: ResponseModel + Bernoulli/TwoGroupMarginal/Poisson; grad ==
   d loglik/deta (finite diff), loglik vs brute marginal. (`test_response.py`)
2. [done] Generic per-column kernel `response_ser.glm_ser` consuming a response;
   Bernoulli reproduces `quadrature_ser` exactly (atol 1e-12).
3. [done] Two-group SER = kernel + TwoGroupMarginal; matches brute per-feature logBF.
4. [done] `twogroup_marginal.py` wires it into the full engine loop. Recovers the
   enrichment feature at PIP ~1.0, sharper than the EM path (`test_twogroup_marginal`).
5. [done] `glm.py`: generic GLM-SER engine family parameterized by a ResponseModel.
   Logistic = GLM(Bernoulli), Poisson = GLM(Poisson), one code path (`test_glm.py`).
   The glm_ser -> BaseSERState adapter is shared (`response_ser.build_ser_state`).
6. [done] Profiling (per-feature intercept) is response-generic: `glm_profile_map`
   (MAP) + `glm_profile_ser` (GH tail), reproducing `local_irls_centered` /
   `profile_ser` to ~1e-14 across dense/sparse x exact/chebyshev x linear/newton.
   Wired into the `glm` family (`profile`/`background`/`node_intercept`); matches
   `logistic_localtaylor` profile mode end-to-end. See "Profiling" below.
7. [done] Offset/variance integration: `response.Smoothed(base, method, order)` -- a
   FAMILY, not a kernel mode. For a canonical-link exp family the zero-mean random
   offset only changes the cumulant (A -> Ahat = E_o A(eta+o), still convex), so the
   smoothed model is a first-class ResponseModel; the kernels never know. aux carries
   the per-row variance as `(y, ov)` (the mean stays the ordinary fixed offset in
   eta). method="gh" (nested GH on true terms; any base) or "taylor"
   (Ahat = A + 1/2 A'' ov, closed form via `ExponentialFamily.cumulant_derivs`;
   refuses majorizing bases at construction -- TwoGroupMarginal's weight != A'').
   Kernels (glm_ser, glm_profile_map/ser) took ZERO new arguments: aux is any pytree,
   broadcast leaf-wise. The `glm` family detects Smoothed and feeds the leave-one-out
   message variance. Reproduces quadrature_ser / profile_ser's offset-integrated path
   to ~1e-10 across dense/sparse x exact/cheb; zero variance collapses to mean-only.
8. [done] JJ smoothers as Smoothed methods + response-specific smoother architecture.
   Smoothers are resolved on the FAMILY hierarchy, each at the level whose guarantee
   it needs: `ResponseModel.smoothed_terms` implements "gh" (needs only terms);
   `ExponentialFamily` adds "taylor" (needs cumulant_derivs); `Bernoulli` adds
   "jj_local"/"jj_fixed" (needs the PG bound on softplus). `smoothing_methods()`
   declares support; `Smoothed.__post_init__` validates (unknown -> ValueError,
   unsupported-for-base -> TypeError). No central if-ladder; a new response-specific
   smoother = one override. The jj methods give Ahat >= Atilde -> a TRUE ELBO (the
   desideratum gh/taylor miss; see notes/ser random offset.md). JJEnvelope (nee jj_local): pointwise-
   optimal tilt xi^2 = eta^2 + ov, closed form Ahat = eta/2 - xi/2 + softplus(xi),
   envelope-theorem grad, weight = 2 lambda(xi) (MM majorizer, like TwoGroupMarginal)
   -- LOCALITY IS FREE: terms is evaluated on entry-shaped eta (per row/feature/node),
   so re-tuning xi per evaluation needs no state and no kernel changes. jj_fixed:
   fixed per-row tilt in aux = (y, ov, xi), tuned by `glm.update_xi_step` from the
   full-predictor second moment (globaljj-style; held on GLMFamilyState like the
   shared intercept); Ahat quadratic in eta -> GH tail over b exact. Pattern: local
   parameter = pointwise function of eta inside terms; global parameter = per-row
   state riding the aux pytree, refreshed by a schedule step.
9. [done] `glm_jj_ser`: the classic localjj recovered exactly in the generic stack.
   The old per-entry tilt is NOT free state -- xi_ij^2 = E_q[eta_ij^2] = (offset +
   x m)^2 + x^2 v + ov, a function of the per-feature posterior -- so the tuner is a
   KERNEL (where (m, v) live), while evaluation goes through the jj_fixed fixed-tilt
   terms with kernel-built entry-shaped xi in aux. Conjugate MM fixed point (exact
   Gaussian update of the quadratic bound; no Newton, no GH), closed-form JJ ELBO
   evidence = certified lower bound, support-only O(nnz). Matches
   `ser_ops.localjj_ser(variational=True)` to 1e-9+ (dense/BCOO x fixed/random
   offset); evidence verified <= the quadrature evidence. Three kernels now:
   glm_ser (quadrature), glm_profile_ser (profiled intercept), glm_jj_ser
   (conjugate bound). JJEnvelope through glm_ser remains the TIGHTER evidence
   (envelope bound per GH node); glm_jj_ser is the certified/cheapest one.
10. [done] Ahat schemes reified as `Smoother` VALUES (supersedes the method-string /
   hierarchy-dispatch of milestone 8). `Smoothed(base, smoother)` with GH(order) /
   Taylor() / JJEnvelope() / JJFixed() frozen dataclasses; each scheme owns its
   implementation, its aux contract, its base requirements (`validate(base)`,
   raising at construction: Taylor needs ExponentialFamily, JJ needs Bernoulli),
   and the note's desiderata as queryable flags: `certified` (Ahat >= Atilde ->
   evidence is a true ELBO; JJ yes, GH/Taylor no) and `convex` (Taylor no). The
   deciding argument: Atilde is LINEAR in the offset law, so a future mixture
   working model is one generic `MixtureOf(component)` scheme -- Ahat_mix =
   sum_j pi_j Ahat(eta + mu_j; var_j) -- with certified/convex provably surviving
   the mixing; impossible to express as method strings without combinatorial names.
   Responses shrank back to pure families (terms + cumulant_derivs capability).
   Engine/kernels: isinstance(smoother, JJFixed) replaces the method-string checks.
11. [done] jj kernel wired into the glm family. `GLMFamilyState.profile: bool` became
   `kernel: "quad" | "profile" | "jj"`; kernel="jj" routes to glm_jj_ser (needs
   Smoothed(..., JJFixed())), where the KERNEL builds the per-entry tilt xi^2 =
   E_q[eta^2] + ov = (offset + x m)^2 + x^2 v + ov -- the tilt sees the b-posterior
   variance Veta = x^2 v. That is deliberately NOT a tilt option inside glm_ser's GH
   tail: at a quadrature node b is FIXED, so Var(eta | b) = 0 and adding x^2 v would
   double-count the b-uncertainty and loosen the bound; Veta belongs to the estimator
   that takes E_q[bound] instead of integrating b (the conjugate kernel). aux stays
   (y, ov) under kernel="jj" (update_xi_step no-ops); the shared-intercept Newton
   re-tunes xi per iterate (JJ-MM). NAMING: the envelope scheme is
   `JJEnvelope` (renamed from JJLocal, which wrongly suggested classic localjj);
   classic localjj = kernel="jj" + JJFixed.
12. [done] `TaylorFixed`: the global-expansion working model (IRLS + score modes).
   The Taylor twin of JJFixed -- same scheme/tuner split: Ahat = quadratic expansion
   of the delta-smoothed cumulant At = A + 1/2 A'' ov at a SUPPLIED per-row anchor
   zhat (aux = (y, ov, zhat)); quadratic in eta -> conditionally Gaussian (Newton
   one-shot, GH tail exact). Generic over ExponentialFamily (Poisson IRLS free),
   unlike the old logistic-only `irls` module. Tuners via the generalized
   `Smoother.row_param(effect_mean, effect_var, intercept)` hook +
   `glm.update_row_param_step` (replaces update_xi_step; GLMFamilyState.xi ->
   row_param): anchor="update" = IRLS (re-expand at the current full predictor each
   sweep; tops-parity with the old irls module verified), anchor="null" = score mode
   (expansion pinned at the intercept-only null; one-step/score-test flavor;
   recovers). NOT byte-identical to old irls: it convolved working data at the FULL
   predictor variance, the new scheme smooths with the LOO variance (consistent with
   every other scheme). Scheme grid now symmetric: local/pointwise {Taylor,
   JJEnvelope} vs fixed/row-tuned {TaylorFixed, JJFixed}, x tuner {engine step,
   conjugate kernel (JJ only), score-null}.
13. [done] `glm.initialize_state_mean_message`: mean-only (ov = 0) for any scheme,
   via the repo's message-type convention (MeanMessage's add/subtract drop variance)
   rather than a flag. Smoothed(Bernoulli(), TaylorFixed()) under it = the pure
   fixed-offset IRLS working model (gIBSS-style global expansion); GH/Taylor collapse
   to the unwrapped base exactly (engine-level alpha parity at 1e-8, tested).
14. [done] `cox_poisson.py`: Cox PH in the framework via the Breslow/Poisson
   reduction. Given Lambda0(t_i) (Breslow cumulative hazard of the FULL predictor),
   the likelihood is per-obs Poisson with y = delta and base offset log Lambda0 --
   profiling Lambda0 back out recovers the Breslow partial likelihood, so the fixed
   points agree. The coupling is the third instance of the per-row engine-tuned
   pattern (after JJFixed xi, TaylorFixed zhat): `update_breslow_step` refreshes
   `GLMFamilyState.glm_offset` (new field: fixed/family-refreshed per-row BASE
   offset -- also usable for Poisson exposure) via O(n log n) suffix sums reusing
   cox.py's FixedCoxContext. `row_param` hook signature generalized: third arg is
   now `base` = intercept + glm_offset. No shared intercept (Lambda0 absorbs it).
   VERIFIED: score identity grad(PL) == delta - exp(eta + logLambda0) vs autodiff
   O(n^2) partial likelihood at 1e-10 (with ties); tops parity + effect size within
   5% vs the dedicated cox.py stack; Smoothed(Poisson(), GH(5)) gives
   offset-INTEGRATED Cox (cox.py is mean-message only -- new capability). Evidence
   semantics: profile-likelihood (conditional on current Lambda0), not the exact
   partial-likelihood BF.
15. [done] `glm_vi_ser` (kernel="vi"): Gaussian-restricted variational SER,
   q(b|gamma=j) = N(m_j, v_j) by coordinate ascent on the per-feature ELBO. THE
   structural point: E_{b~N(m,v)}[ll(off + x b)] is a Gaussian convolution of the
   cumulant with per-entry variance x^2 v (+ ov; independent Gaussians add), so the
   Ahat SCHEMES double as the variational expectation operator -- the kernel feeds
   the response entry-shaped variance, as glm_jj_ser feeds an entry-shaped tilt.
   Stationarity: smoothed score for m; 1/v = 1/pv + sum x^2 weight (Price). VERIFIED
   identities: (a) Gaussian base -> VI is exact, == glm_ser closed form (m, v, AND
   elbo == logBF) at 1e-8; (b) with JJEnvelope the pointwise tilt becomes xi^2 =
   eta^2 + x^2 v + ov = classic localjj's variational tilt -> glm_vi_ser(JJEnvelope)
   == glm_jj_ser exactly (1e-7, with/without ov) -- the conjugate kernel is the
   quadratic-bound special case of Gaussian VI; (c) per-feature ELBO <= free-form
   (glm_ser) evidence, as the variational principle demands. Four kernels now:
   quad (free-form q via GH tail) / profile / jj (conjugate bound) / vi (Gaussian q).
16. [done] Intercept variance tracked: the shared intercept is now an explicit
   flat-prior Gaussian factor q(b0) = N(intercept, intercept_var), whose coordinate
   update IS estimate_intercept's Newton plus v0 = 1/sum_i w_i (the vi stationarity
   for an all-ones column). v0 flows into the smoothers' ov as a constant per row
   (O(1/n)), alongside the message variance, and into row_param tuning; mean-field
   self-exclusion when updating b0 itself; dropped under MeanMessage (mean-only) so
   the collapse identities survive; kernel="profile" unaffected (b0 profiled
   in-kernel, the more complete treatment). Verified: v0 == inverse observed
   curvature by autodiff; O(1/n) magnitude engine-level; recovery unchanged.
   Expected effect (per the earlier offset-integration finding): a shared-constant
   ov shifts features and null together -> mostly evidence calibration, little PIP.
17. [done] `glm_vi_profile_ser` (kernel="vi_profile"): profiled Gaussian VI --
   q(b|gamma) = N(m, v), b0_j maximized out POINTWISE per feature, never modeled
   (the partial-likelihood move; offset-shift-invariant evidence). Flavor (a) chosen
   deliberately over joint Gaussian q(b0, b). Structure: glm_profile_map's 2-D
   Newton with SMOOTHED weights (the background split survives unchanged: x = 0
   entries have entry variance = ov), v by Price + envelope theorem (1/v = 1/pv +
   sum x^2 w -- CONDITIONAL variance, matching localjj_centered_ser's "profiled
   mean, conditional variance" parameterization; slightly overconfident when
   columns are uncentered, same caveat as the Cox reduction). No GH tail. VERIFIED:
   (a) JJEnvelope -> == ser_ops.localjj_centered_ser at 1e-7 (dense/BCOO x
   fixed/random offset) -- BOTH previously-missing table cells (jj-profile and
   vi-profile) are now this one kernel; (b) Gaussian + pre-centered -> exact ==
   glm_profile_ser; (c) offset-shift invariance. Six kernels: quad / profile / jj /
   vi / vi_profile (+ the intercept-variance-tracking shared path).
18. [done] Closed-form `linear`/`linear_profile` kernels replace the quadratic-order
   clamp (which was briefly milestone-adjacent and then rejected as hiding a real
   compute-path difference). Quadratic responses (Gaussian base or TaylorFixed/
   JJFixed schemes; `response.quadratic` composes base OR scheme) are REFUSED by
   quad/profile with a redirecting error and get dedicated kernels: glm_linear_ser /
   glm_linear_profile_ser -- ONE terms pass (row O(n) + support O(nnz)), weighted
   linear regression in closed form: no Newton, no GH tail, no background machinery
   (the intercept background of a quadratic is the closed-form quadratic
   L0 + G0 c - W c^2/2, so exact/chebyshev is moot). Parity with quad/profile at
   1e-7..1e-8 on all outputs (dense/BCOO; Cox-Reid is exact for quadratics so even
   coefficient_kl matches). Named "linear" because Gaussian(v) + kernel="linear" IS
   linear SuSiE and the solve is weighted linear regression ("quadratic" would
   collide with "quad" = quadrature). Kernel functions stay permissive (tests use
   them for parity); the ENGINE enforces compatibility both directions.
19. [done] cox_poisson partial-likelihood read-out (default): the working-Poisson
   fixed point is the PL MAP (score identity, envelope theorem), but its per-feature
   variance AND evidence curve are conditional on the shared Breslow baseline
   (I_cond = sum_k d_k S2_k/S0_k >= I_PL = sum_k d_k Var_Rk(x): the Schur/profiling
   correction = risk-set mean-centering of x -- the per-event-time analog of the
   H0b^2/H00 intercept correction). Fix: keep mu, REPLACE (var, log_bf, ckl) with
   the PL Laplace read-out at the mode via cox.py's own sorted per-column kernel
   (one vmapped call per effect update; dense only -- default_schedule(
   pl_curvature=False) for BCOO). Curvature-only correction was tried first and
   moved nothing: the volume factor is ~constant across features and cancels in
   alpha; the evidence CURVE had to be replaced. DIAGNOSIS of the earlier ~0.18
   dPIP vs cox.py: ~0.17 was a CONFIG mismatch (cox.py defaults
   estimate_prior_variance=False; glm defaults True -- flatter null-effect alpha
   under shrinking pv), only ~0.007 was estimator semantics; with configs matched,
   corrected gap < 5e-4; ADDING a 3-step ridge-Newton PL POLISH to the read-out
   (the working mode is the PL mode only for the feature anchoring the shared
   baseline; other features sit slightly off) makes each effect update compute
   IDENTICAL per-feature quantities to cox.py's -- engine gap ~1e-5 PIP, residual
   is sweep dynamics only. Variances widen as theory demands, mode polish-adjusted.
20. [done] Baseline treatment reframed as the estimator axis it is (user's
   observation: the PL read-out as a default 'correction' flag broke the parallel
   with the intercept-treatment axis). `default_schedule(baseline="profiled" |
   "shared")` replaces pl_curvature: shared baseline == shared intercept
   (conditional/diagonal curvature, sparse-capable, profiled kernels meaningful);
   profiled baseline == profiled intercept (Schur curvature via the PL read-out,
   == cox.py, dense-only, REFUSES kernel="profile"/"vi_profile" -- the per-feature
   baseline subsumes any per-feature intercept, PL invariant to it). Neither is a
   correction of the other. Three-rung parallel documented in the tables note;
   middle rung (shared baseline + tracked variance, glm_offset_var from
   Var(b0k) ~= 1/d_k) still open -- the Cox analog of intercept_var.
21. [done, negative result] Baseline-uncertainty middle rung for Cox is VACUOUS
   (user's derivation; see notes/cox poisson baseline uncertainty.md). Increments
   have exact Gamma(d_k, S0_k) posteriors (Breslow = posterior mean); e^{o_j} =
   Lambda0(t_j) is linear in them, so E_q[e^{o_j}] = plug-in EXACTLY and the
   mean-field-integrated b-update equals the point-estimate one (Poisson cumulant
   is within-family under offset convolution -- unique such cumulant). The
   information reversal (PL: late events zero info; collapsed Poisson: longest
   survivors max curvature ~log n) is caused by the q(o)q(b) FACTORIZATION, not
   point estimation; I_PL is not subject-diagonal, so no ov-channel fix exists --
   the profiled rung (risk-set space) is the only faithful one. glm_offset_var
   will NOT be built; intercept_var stays logistic-specific. Also simplifies the
   LogNormal opportunity: Smoothed(Poisson, Gaussian) == offset += ov/2, one line.
22. [done] baseline="null": third point on the Cox baseline axis -- the baseline
   frozen at the b=0 Nelson-Aalen estimate (set once by set_null_baseline_step in
   before_fit; no Breslow refresh), i.e. the SCORE analysis. Identity tested: the
   per-feature Poisson score at 0 against the frozen offset equals the textbook
   log-rank observed-minus-expected (binary covariate, risk-set definition,
   1e-10). Engine test pins that glm_offset never moves and selection still
   recovers. Two null-anchor knobs compose: baseline="null" (hazard) and
   TaylorFixed("null") + kernel="linear" (cumulant expansion) = the fully
   classical one-pass score test.
23. [done] Intercept treatment de-compounded (user: the profiled/shared/null
   options were scattered). GLMFamilyState.kernel is now the b-integration axis
   only ("quad"|"linear"|"vi"|"jj") and `intercept` ("shared"|"profiled"|"null")
   is the nuisance-treatment axis, mirroring cox_poisson's baseline= exactly; the
   float moved to intercept_value. Old compound names refused with a redirecting
   error (repo convention: no shims). intercept="null" = fit once at the b = 0
   null in initialize_state then freeze (score-analysis intercept; == logit base
   rate for Bernoulli, tested frozen through a full fit). Why not a
   default_schedule() arg like cox: glm's treatments select different JIT-STATIC
   KERNELS (family state), while cox's baseline treatments differ in SCHEDULE
   STEPS -- each axis lives where its mechanism lives; both now read
   axis="shared"|"profiled"|"null". jj x profiled refused (no conjugate profiled
   kernel). Suite green post-rename.
24. [done] intercept="null" hardened for row-tuned schemes: the init-time null fit
   now ALTERNATES (retune row_param at the current null predictor <-> intercept
   Newton), with mean-field self-exclusion of intercept_var from the tilt/anchor
   during the fit -- so the fixed point is the EXACT null MLE for every scheme
   (Taylor anchors: Newton-fast; JJFixed: MM, tilt tight at b0; a frozen zero
   anchor would give only the one-step estimator, and an included intercept_var
   would shift the JJ fixed point by O(v0)). Composition test: null intercept x
   {TaylorFixed(null/update), JJFixed} x kernel="linear" all hit logit(ybar) at
   1e-6 and stay frozen through the fit.
25. [partly done] Retire the EM two-group path: DONE. The marginal is now the
   only two-group method -- reworked into a first-class glm family (`twogroup.py`,
   no wrapper/introspection/inner-state nesting), and `twogroup_marginal.py` +
   `twogrouplocaljj.py` are deleted (see notes/twogroup rework.md). Still open:
   fold `logistic_localtaylor` onto the generic kernels and retire the EM
   `localjj` path; `ser_ops`'s logistic-specific quadrature_ser/profile_ser become
   deletable (Bernoulli parity holds at 1e-9..1e-12), and the dedicated
   localjj/globaljj/`irls` modules are functionally subsumed (glm_jj_ser
   reproduces localjj_ser exactly; Smoothed JJEnvelope/JJFixed cover the
   variational elaborations). glm_jj_ser is already wired into the glm family as
   kernel="jj".

## Profiling = background + correction (the sparse-centering story)
A profiled per-column fit (`offset + b0_j + x_ij b_j`, b0 profiled out -> offset-shift
invariant) splits every reduction into:
  - **background (l_d)**: the x^0 intercept terms `Sum_i grad`, `Sum_i weight`,
    `Sum_i loglik` at the b=0 config `eta0 = offset + b0`. Depends only on the scalar
    b0, shared across features/nodes. `_background`: "exact" O(n*p) or "chebyshev"
    O(n*D + D*p) surrogate of `c -> Sum_i f(offset+c)`.
  - **correction (l_s)**: everything with an x factor (gb, H0b, Hbb, the loglik
    support diff). Off-support rows cancel -> pure support -> **O(nnz)** on BCOO.
"exact vs chebyshev" is purely the background's treatment; the correction is always
the direct support reduction. This is why sparse profiling exploits sparsity: only the
scalar-intercept background sees all rows, and even that is O(nD) under chebyshev.

Centering is the fused-exact special case: `CenteredOperator.moment`'s binomial folds
background (the k=0/all-rows term) + correction into one call (dense; on BCOO the
per-ENTRY interface is guarded off since centering densifies -- use the background
split instead).

## The intercept degeneracy (two-group marginal) -- the one real subtlety
The marginal loglik `softplus(eta+llr) - softplus(eta) -> llr` as `eta -> +inf`, so at
`b = 0` the intercept-only objective is *maximized at b0 -> +inf* whenever `sum llr > 0`
(the "everything enriched" solution; f1 absorbs all obs). The interior value is only a
local max. So a direct marginal Newton on b0 alone runs off.
Resolution is pure ordering, not a penalty: update b0 *after* the SER effects each
sweep (`default_schedule` puts it in `after_sweep`), so `b` has already structured
`eta` and pins b0 to its interior fixed point. The step is `update_intercept_step`
(not `estimate_intercept_step`) so the `twogroup` wrapper -- which drives an inner
`estimate_intercept_step` *before* the effects -- leaves it alone. The EM path never
hit this because its intercept is a logistic MLE on a soft label Ez in (0,1), which is
finite by construction; the price EM pays is a diffuse Ez and much softer PIPs.

## Definition of done
Simple, human-readable: one per-column kernel (`glm_ser`), a tiny ResponseModel per
family, the two-group SER with no EM matching/beating the EM twogroup on PIP, generic
GLM family for logistic/Poisson, suite green. [reached for milestones 1-5]
