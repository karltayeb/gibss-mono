# Response-model SER: plan & core structure

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
  does not fit `terms(eta, aux)` cleanly -> out of scope, note it.

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
7. [open] Offset/variance integration in glm_ser (GH-over-o), then fold
   `logistic_localtaylor` onto it. Retire the EM `twogrouplocaljj`/localjj path once
   the marginal is the default. Not done -- localtaylor still carries the per-row-var
   message that glm_ser doesn't yet.

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
