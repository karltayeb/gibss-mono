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

## Paths considered
- (A) Standalone twogroup marginal kernel — fast to validate, not general. (stepping stone)
- (B) **Response seam in ser_ops per-column kernels** — the real structure; logistic
      is just `Bernoulli`. Chosen.
- Cox: partial likelihood is NOT per-observation separable (risk sets couple obs);
  does not fit `terms(eta, aux)` cleanly -> out of scope, note it.

## Milestones
1. `response.py`: ResponseModel + Bernoulli/TwoGroupMarginal/Poisson; validate grad/
   weight vs finite-diff, loglik vs brute marginal.
2. Generic per-column kernel (`glm_ser`) consuming a response; Bernoulli reproduces
   `local_irls`/`quadrature_ser` exactly.
3. Two-group SER = kernel + TwoGroupMarginal; validate vs current twogroup fixture.
4. Wire the twogroup module onto it; retire the EM kernels.
5. Poisson SER as a generality check.
6. Simplify + fold logistic_localtaylor onto the generic kernel if clean.

## Definition of done
Simple, human-readable: one per-column kernel, a tiny ResponseModel per family, the
two-group SER with no EM, matching the current twogroup on alpha/PIP, suite green.
