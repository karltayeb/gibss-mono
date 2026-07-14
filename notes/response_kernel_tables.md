# Response x kernel tables (the glm family menu)

Reference for `glm.GLMFamilyState(response=..., kernel=...)`. A cell is an estimator;
refusals are construction/call-time TypeErrors (tested). Everything here composes
with the orthogonal knobs at the bottom. See milestones 7-17 in
`response_models.md` for how each piece was built and verified, and
`notes/ser random offset.md` for the Ahat framework the schemes implement.

## Axes (2026-07-13 refactor)

`GLMFamilyState` now has TWO orthogonal axes (the old compound kernel names
profile/vi_profile/linear_profile were kernel x intercept pairs and are refused
with a redirecting error):

- kernel = "quad" | "linear" | "vi" | "jj"   (how b is integrated)
- intercept = "shared" | "profiled" | "null" (the nuisance-treatment axis --
  the logistic instance of cox_poisson's `baseline`):
    "shared"   refit each sweep at the total predictor (+ tracked intercept_var)
    "profiled" per-feature b0_j inside the kernel (Schur curvature)
    "null"     fit ONCE at the b = 0 null in initialize_state, then frozen
               (the score-analysis intercept; logit of the base rate for logistic)
  The float moved to `intercept_value`. jj x profiled is refused (no conjugate
  profiled kernel; vi x profiled + JJEnvelope IS centered localjj).

Column headers below use the compound names as ESTIMATOR labels; in code they are
now (kernel, intercept) pairs: profile = (quad, profiled), vi_profile =
(vi, profiled), linear_profile = (linear, profiled).

## Validity rules (all bases)

- `kernel="jj"` (conjugate) requires `Smoothed(Bernoulli(), JJFixed())`: the exact
  Gaussian coordinate update needs a bound QUADRATIC in eta.
- `kernel="vi"` / `"vi_profile"` require `Smoothed(base, pointwise scheme)`
  (GH/Taylor/JJEnvelope): the scheme doubles as the Gaussian expectation operator
  E_q and receives the kernel-owned per-entry variance x^2 v + ov; row-tuned
  schemes (TaylorFixed/JJFixed) conflict with that and are refused; unwrapped bases
  have no expectation operator.
- Scheme x base: GH composes with ANY base (incl. TwoGroupMarginal);
  Taylor/TaylorFixed need ExponentialFamily (exact cumulant derivs);
  JJEnvelope/JJFixed need Bernoulli (the PG bound is on softplus).

- Quadratic responses (`response.quadratic`, composing as base.quadratic OR
  smoother.quadratic: Gaussian base, TaylorFixed/JJFixed schemes) are REFUSED by
  quad/profile and REQUIRED by the closed-form kernels `"linear"` /
  `"linear_profile"` (one terms pass, weighted linear regression: no Newton, no GH
  tail, no background machinery -- the intercept background of a quadratic is
  itself the closed-form quadratic L0 + G0 c - W c^2/2). Explicit compute paths
  instead of silently integrating a Gaussian numerically; parity with quad/profile
  verified at 1e-7..1e-8 on all outputs, dense and BCOO. Kernel-level functions
  (glm_ser etc.) stay permissive for expert use and tests; the ENGINE enforces.

Flags: [q] quadratic -> lives in the linear kernels; q(b|gamma) automatically
Gaussian, quad == vi == linear at the fixed point. [c] certified -> the evidence is
a true ELBO (exact for the quadratic/conjugate cells; up to b-quadrature for
JJEnvelope under quad/profile).

## Bernoulli (logistic)

| response \ kernel        | quad (free-form, GH tail) | profile (free-form, profiled b0) | linear (closed form) | linear_profile | jj (conjugate) | vi (Gaussian q) | vi_profile |
|--------------------------|---------------------------|----------------------------------|----------------------|----------------|----------------|-----------------|------------|
| Bernoulli()              | logistic SuSiE (quadrature_ser) | profiled (profile_ser)     | x                    | x              | x              | x               | x          |
| Smoothed(B, GH(k))       | offset-integrated (oi=k)  | profiled + offset-int.           | x                    | x              | x              | Gaussian VI, GH E_q (new) | profiled Gaussian VI (new) |
| Smoothed(B, Taylor())    | delta-smoothed (oi="taylor") | profiled delta                | x                    | x              | x              | Gaussian VI, delta (new) | profiled, delta (new) |
| Smoothed(B, TaylorFixed("update")) [q] | x           | x                                | IRLS (irls module)   | profiled IRLS  | x              | x               | x          |
| Smoothed(B, TaylorFixed("null")) [q]   | x           | x                                | score mode (new)     | profiled score (new) | x        | x               | x          |
| Smoothed(B, JJEnvelope()) [c] | envelope + GH tail (new, tightest JJ) | profiled envelope (new) | x        | x              | x              | == classic localjj | == centered localjj |
| Smoothed(B, JJFixed()) [q][c] | x                     | x                                | globaljj (globaljj module) | profiled globaljj | classic localjj | x     | x          |

21/49 valid. Coincidences (parity-tested at <= 1e-7): vi+JJEnvelope == jj+JJFixed
== localjj_ser(variational=True); vi_profile+JJEnvelope == localjj_centered_ser;
linear kernels == quad/profile kernels on quadratic responses (the parity tests
call the permissive kernel functions directly). Every old logistic module has a cell (quadrature_ser,
profile_ser, irls, globaljj, localjj_ser, localjj_centered_ser); the genuinely new
cells are score mode, envelope-through-quadrature, and the four GH/Taylor Gaussian-VI
cells.

## Poisson

| response \ kernel        | quad              | profile               | linear        | linear_profile | jj | vi                  | vi_profile           |
|--------------------------|-------------------|-----------------------|---------------|----------------|----|---------------------|----------------------|
| Poisson()                | Poisson SuSiE     | profiled Poisson      | x             | x              | x  | x                   | x                    |
| Smoothed(P, GH(k))       | offset-integrated | profiled + offset-int.| x             | x              | x  | Gaussian VI, GH E_q | profiled Gaussian VI |
| Smoothed(P, Taylor())    | delta-smoothed    | profiled delta        | x             | x              | x  | Gaussian VI, delta  | profiled, delta      |
| Smoothed(P, TaylorFixed("update")) [q] | x   | x                     | Poisson IRLS  | profiled IRLS  | x  | x                   | x                    |
| Smoothed(P, TaylorFixed("null")) [q]   | x   | x                     | Poisson score | profiled score | x  | x                   | x                    |

14/25 valid. Notes:
- The jj column is STRUCTURALLY empty: A = exp admits no quadratic upper bound
  (outgrows every quadratic), so Poisson has no certified cell at all; a certified
  Poisson ELBO would need a different bound family, not a tilt.
- Nearly the whole table is new capability: the old stack was logistic-only.
  Poisson IRLS / score mode fell out of validating against ExponentialFamily.
- The table doubles as the COX table: cox_poisson = these rows + the Breslow
  glm_offset refresh, no intercept (offset-integrated Cox, Cox-IRLS, Cox-score,
  profiled-Gaussian-VI Cox are all one family_state_kwargs away).
- The exact smoothed Poisson cumulant is not just closed form -- it is
  WITHIN-FAMILY: E_o[exp(eta+o)] = exp(eta + ov/2), so Smoothed(Poisson, Gaussian)
  == Poisson with offset += ov/2 (one line, not a scheme; GH(k) on Poisson
  numerically approximates an offset shift). Consequence: Poisson offset
  integration only matters insofar as ov/2 varies across rows; with the exact
  baseline posterior it is a strict no-op for Cox (see notes/cox poisson baseline
  uncertainty.md). Softplus remains the only cumulant in the stack that genuinely
  needs numerical smoothing (Gaussian: Taylor exact; Poisson: offset shift exact).

## Gaussian

| response \ kernel        | linear               | linear_profile        | vi              | vi_profile        | quad/profile/jj |
|--------------------------|----------------------|-----------------------|-----------------|-------------------|-----------------|
| Gaussian(v)              | linear SuSiE (exact) | profiled linear SuSiE | x (not Smoothed)| x                 | x               |
| Smoothed(G, GH/Taylor)   | == linear SuSiE      | == profiled [cen]     | == linear SuSiE (allowed, redundant) | == profiled [cen] (allowed, redundant) | x |
| Smoothed(G, TaylorFixed) | == linear SuSiE      | == profiled           | x (row-tuned)   | x                 | x               |

(Every Gaussian response is quadratic -- base quadraticity survives any scheme -- so
quad/profile/jj refuse and the family lives in the linear kernels; kernel="linear"
with Gaussian(v) IS linear SuSiE, which is what named the kernel. The vi cells for
pointwise schemes are permitted but redundant: they iterate to what linear computes
in closed form.)

14/25 valid but only TWO distinct estimators (linear SuSiE, profiled linear SuSiE):
the maximally degenerate corner, and deliberately so -- the collapses ARE the
framework's exactness anchors:
- schemes collapse: quadratic cumulant -> Taylor exact, TaylorFixed exact at ANY
  anchor, GH integrates a quadratic exactly;
- kernels collapse: quadratic likelihood -> every per-feature posterior is exactly
  Gaussian -> free-form == Gaussian q, GH tail exact, ELBO == log marginal (the one
  table where "certified" is trivially true everywhere);
- offset integration collapses hardest: E_o[ll(eta+o)] = ll(eta) - ov/(2 sigma^2),
  a per-row CONSTANT -- grad/weight untouched, cancels in every ll - ll0, so it
  changes nothing (alpha, moments, relative evidence). This is E_q[log lik]
  semantics, deliberately; the MARGINAL N(y; eta, sigma^2 + ov) would inflate the
  weight, but that is a different estimator than the variational one computed here.
- [cen]: vi_profile == profile needs centered columns (H0b = 0 -> conditional ==
  Schur variance); prep_data centers dense X by default.
Practical: for real Gaussian work use gibss.linear (estimates residual variance,
supports obs_variance); glm's Gaussian(variance=v) holds sigma^2 FIXED and exists
mainly as the exactness anchor for tests. Any future change that makes two Gaussian
cells disagree has introduced a bug in a seam.

## Cox (cox_poisson: the Poisson table x a BASELINE axis)

`cox_poisson` = the Poisson rows + `update_breslow_step` (glm_offset =
log Lambda0(t_i) from the total predictor, each sweep) + no intercept (the baseline
absorbs it) + a `baseline` axis on `default_schedule` -- the Cox instance of the
intercept-treatment axis. The parallel is exact (same Schur algebra):

| nuisance treatment      | logistic intercept          | Cox baseline                       |
|-------------------------|-----------------------------|------------------------------------|
| shared point estimate   | quad kernel (pre-var-track) | baseline="shared"                  |
| shared + tracked var    | quad + intercept_var        | VACUOUS: E[Lambda0] = plug-in (see notes/cox poisson baseline uncertainty.md) |
| profiled per feature    | profile kernel (Schur)      | baseline="profiled" (default)      |

baseline="null": the baseline frozen at the b=0 Nelson-Aalen estimate
(inc = d_k/|R_k|; harmonic exposures H_n - H_{j-1} under no censoring) -- the
SCORE analysis: per-feature Poisson score at 0 == PL score at 0 == the log-rank
observed-minus-expected (binary x; identity tested). Cheapest (no refresh);
conditional information (the log-rank variance is the hypergeometric = PL/Schur
one, so the shared-baseline caveat applies at the null). Pairs with
TaylorFixed("null") + kernel="linear" for the fully classical one-pass score test.
baseline="shared": one baseline anchored at the total predictor; conditional
curvature I_cond = sum_k d_k S2_k/S0_k (the joint Hessian's diagonal); O(n) nuisance
work per effect, SPARSE-capable; overconfident where covariates are not risk-set-
centered. baseline="profiled": the baseline is re-profiled per feature and per
coefficient value, analytically, via the PL read-out (mu kept -- envelope theorem --
polished; var/evidence replaced by the PL Laplace formula); Schur curvature
I_PL = sum_k d_k Var_{R_k}(x) <= I_cond, gap = risk-set mean-centering (the
many-intercepts H0b^2/H00); matches cox.py < 1e-3 PIP (configs matched); dense X
only for now. Neither is a "correction" of the other: same estimator axis as
quad-vs-profile.

Kernel table under baseline="shared" (all cells meaningful):

| response \ kernel        | quad                  | profile               | linear         | linear_profile     | vi                | vi_profile         |
|--------------------------|-----------------------|-----------------------|----------------|--------------------|-------------------|--------------------|
| Poisson()                | Cox SuSiE (Breslow)   | profiled-b0 Cox       | x              | x                  | x                 | x                  |
| Smoothed(P, GH(k))       | offset-integrated Cox | profiled + offset-int.| x              | x                  | Gaussian-VI Cox   | profiled GVI Cox   |
| Smoothed(P, Taylor())    | delta-smoothed Cox    | profiled delta        | x              | x                  | GVI Cox, delta    | profiled, delta    |
| Smoothed(P, TaylorFixed("update")) [q] | x       | x                     | Cox IRLS       | profiled Cox IRLS  | x                 | x                  |
| Smoothed(P, TaylorFixed("null")) [q]   | x       | x                     | Cox score mode | profiled Cox score | x                 | x                  |

Under baseline="profiled" the same table applies EXCEPT the profile/vi_profile/
linear_profile columns are REFUSED (ValueError): per-feature baseline profiling
subsumes any per-feature intercept (the PL is invariant to it), so those kernels
would be redundant by construction. (jj omitted -- structurally empty for Poisson.)

Cox-specific notes:
- Smoothed responses: the profiled-baseline read-out is the MEAN-predictor PL
  (smoothing shapes the fit, not the read-out) -- approximation, documented.
- The LogNormal() exact-smoothing opportunity applies verbatim (Poisson cumulant).
- Ties: Breslow convention throughout (hazard step and read-out).

## Orthogonal knobs (compose with any valid cell)

- message type: `initialize_state` (ov = leave-one-out message variance +
  intercept variance) vs `initialize_state_mean_message` (ov = 0, mean-only;
  intercept variance also dropped).
- profile kernels: `background="exact"|"chebyshev"`; `node_intercept=
  "linear"|"newton"` (quad-profile only; vi_profile has no nodes).
- `quadrature_order`: quad kernel only (the [q] rows live in the linear kernels, which have no quadrature).
- `glm_offset`: fixed per-row base offset (Poisson exposure; refreshed per sweep by
  cox_poisson for the Breslow hazard).
- shared-intercept paths (quad/jj/vi) track q(b0) = N(intercept, intercept_var);
  profiled paths handle b0 in-kernel instead.
