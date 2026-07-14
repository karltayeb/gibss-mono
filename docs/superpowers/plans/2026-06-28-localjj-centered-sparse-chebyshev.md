# Sparse intercept-decoupling for the vi/jj (localjj) kernels

There are **two distinct** ways to decouple the intercept from the features for a
localjj-style fit, and they have very different sparse stories. Keep them separate.

## Case 1 — PROFILED intercept (per-feature `b0_j`): SOLVED on sparse

Each feature gets its own intercept, profiled out per feature. At a zero entry
`x_ij = 0` the predictor is `offset_i + b0_j` and the entry variance is exactly `ov_i`
(no effect-variance term), so the off-support fill-in is a sum over rows of a smooth
function of the **single scalar** `b0_j` — a 1-D background, exactly the
`logistic_profile` pattern.

Status: **works on sparse today.** `glm_vi_profile_ser` (kernel `vi`,
`intercept="profiled"`) and `glm_profile_ser` consume it through the 1-D `_background`
(`response_ser._background`, exact or chebyshev). Verified: `fit_glm_susie(Xs, y,
method="localjj", intercept="profiled")` runs and recovers on BCOO designs. This is
what the deferred note here originally described; it is no longer deferred.

## Case 2 — SHARED intercept + FIXED centering (`center=True`): the remaining gap

Center the columns by a fixed offset `c_j` and keep a single shared intercept
(in `offset`). Per feature the model is `eta_i = offset_i + (x_ij - c_j) b_j`. At a
zero entry the centered value is `-c_j`, so:

- predictor shift `s_j = -c_j m_j`  (a location shift — 1-D, fine)
- entry variance `ov_i + c_j^2 v_j`  (a **second** per-feature parameter `d_j = c_j^2 v_j`)

The off-support reduction is therefore `Sum_i f(offset_i + s_j, ov_i + d_j)` — a
function of **two** per-feature scalars. The 1-D `_background` cannot express it. (The
quad and linear kernels don't hit this: quad integrates the effect through the GH tail
so there is no per-entry effect variance, and linear's weights are constant. Only the
vi/jj kernels carry a per-entry effect variance/tilt.)

Status: **refused** (`glm._fit_effect_raw` raises for kernel in {vi, jj} + sparse
`column_center`; the front door raises a guided error pointing to `intercept="profiled"`).

### Options if we ever want it

1. **2-D Chebyshev surrogate** over `(s_j, d_j)`: tensor panel, build `O(n*Ds*Dd)`,
   eval `O(Ds*Dd*p)`, per VI iteration (the surrogate must be refit each step because
   `(m,v)` move). Exact-in-the-limit; needs an accuracy study over the `d` range
   (`c_j^2 v_j` can be O(1) for null features).
2. **Variance-Taylor**: expand the smoothed terms in the variance addon `d_j` around
   `ov`; the 0th order and the loglik's 1st order reuse existing `_background` outputs
   (`∂_ve Âhat = ½ Âhat''= -½ w`), but the Newton **step** corrections need a 3rd
   `eta`-derivative that `terms()` does not expose. Clean only for the `taylor2`
   smoother, where the variance dependence is exactly linear.

Both materialize an `(n, p)`-shaped transient for the exact reference, i.e. dense
memory — so "just densify and use the dense centered vi" is the honest O(n*p) fallback,
and a surrogate is the only sub-dense route.

### Recommendation: use `intercept="profiled"` instead — don't build Case 2 yet

Empirically, shared-centered and profiled localjj give the **same inference**. On
strongly off-center dense designs (where the winner-take-all matters), across 6 seeds,
centered-shared and profiled localjj select identical top features and identical causal
PIP (1.00); they differ only on the diffuse *null* second effect. And
`glm_vi_profile_ser`'s own docstring notes that centering over profiling fixes only a
"slight overconfidence" in `v` where a column correlates with the intercept.

So Case 2 buys a marginal variance correction over a path (Case 1) that already works on
sparse and gives equivalent selection. Given that it requires an *approximation inside an
iterative inference loop* (silent-corruption risk), it is not worth landing until there
is a concrete need. The front door guides `center=True` + sparse vi/jj users to
`intercept="profiled"`.

Prior validation (dense, superseded impl): univariate kernel fixed point == brute-force
joint `(b0, beta, xi)` JJ optimum (1e-3); see `tests/test_gibss_localjj_centering.py`.
