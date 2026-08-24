"""Q2-only ELBO variants: `compute_elbo_gaussian` (the CF integrator, an alternative to
`compute_elbo`'s Compress canonicalization) and `compute_elbo_jj` (the Jaakkola-Jordan lower
bound with the per-row tilts fit optimally on the fly).

Guarantees pinned here:
  * compute_elbo_gaussian computes the SAME exact Q2 ELBO as compute_elbo -- two integrators
    (CF product vs Compress peel) of one object -- for the logistic base, and the same
    analytic MGF for the Poisson base;
  * compute_elbo_jj is a LOWER bound to the exact ELBO of the same q, and its optimal-tilt
    closed form matches an independent per-row numeric maximization of the JJ bound;
  * each objective is won by the method that optimizes it: cf_cavi maximizes the exact ELBO,
    globaljj maximizes the JJ ELBO;
  * both variants score a globaljj (kernel='linear') state directly, where compute_elbo is
    blocked by the kernel/response validation;
  * both reject a free-form Q1 state (no closed-form transform).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from gibss.methods import fit_glm_susie
from gibss.glm import prep_data
from gibss.elbo import compute_elbo, compute_elbo_gaussian, compute_elbo_jj


def _sim(seed=0, n=500, p=25, causal=4, beta=1.5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.5 + beta * X[:, causal])))).astype(float)
    return X, y


@pytest.mark.parametrize("method", ["cf_cavi", "gibss_gaussian"])
def test_cf_gaussian_matches_compress(method):
    # CF and Compress are two builders of the same offset-integrated cumulant, so the CF ELBO
    # equals compute_elbo's Compress ELBO up to quadrature.
    X, y = _sim(seed=1)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=3, method=method, max_iter=60)
    assert compute_elbo_gaussian(d, st) == pytest.approx(compute_elbo(d, st), abs=1e-6)


def test_cf_gaussian_matches_compute_elbo_poisson():
    # Poisson: both use the analytic log-normal MGF, so they are identical (not just close).
    rng = np.random.default_rng(2)
    n, p, causal = 500, 25, 4
    X = rng.standard_normal((n, p))
    y = rng.poisson(np.exp(-0.3 + 1.0 * X[:, causal])).astype(float)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, family="poisson", L=3, method="cf_cavi", max_iter=40)
    assert compute_elbo_gaussian(d, st) == pytest.approx(compute_elbo(d, st), abs=1e-9)


@pytest.mark.parametrize("method", ["cf_cavi", "gibss_gaussian", "globaljj"])
def test_jj_is_a_lower_bound(method):
    # the JJ bound is <= the exact log-likelihood, so the JJ ELBO is <= the exact ELBO of q.
    X, y = _sim(seed=3)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=3, method=method, max_iter=60)
    assert compute_elbo_jj(d, st) <= compute_elbo_gaussian(d, st) + 1e-6


def test_jj_optimal_tilt_matches_numeric_maximization():
    # independent oracle: maximize the FULL JJ bound over the per-row tilt xi on a grid and
    # sum -- must equal the closed form at xi^2 = E[eta^2] (which our code uses). The closed
    # form is the true max, so it is >= the grid max and within grid resolution of it.
    X, y = _sim(seed=4)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=3, method="cf_cavi", max_iter=60)
    fs = st.family_state
    m = np.asarray(st.total_message.mean) + float(fs.intercept_value)
    v = np.asarray(st.total_message.var) + float(fs.intercept_var)
    Eeta2 = m**2 + v
    xis = np.linspace(1e-4, 40.0, 20001)
    lam = np.tanh(xis / 2.0) / (4.0 * xis)
    jj = (y[:, None] * m[:, None] - np.logaddexp(0.0, xis)[None, :]
          - 0.5 * (m[:, None] - xis[None, :])
          - lam[None, :] * (Eeta2[:, None] - xis[None, :] ** 2))
    grid_ll = jj.max(axis=1).sum()
    kl = sum(float(e.kl) for e in st.single_effects) + float(fs.intercept_kl)
    grid_elbo = grid_ll - kl
    e_jj = compute_elbo_jj(d, st)
    assert e_jj >= grid_elbo - 1e-9            # closed form is the true per-row maximum
    assert e_jj == pytest.approx(grid_elbo, abs=5e-3)  # grid reaches it up to resolution


def test_each_method_wins_its_own_objective():
    # cf_cavi maximizes the exact Q2 ELBO; globaljj maximizes the JJ ELBO. So on the same
    # data, cf_cavi has the highest exact ELBO and globaljj the highest JJ ELBO.
    X, y = _sim(seed=5)
    d = prep_data(X, y)
    sc = fit_glm_susie(X, y, L=3, method="cf_cavi", max_iter=80)
    sj = fit_glm_susie(X, y, L=3, method="globaljj", max_iter=80)
    assert compute_elbo_gaussian(d, sc) >= compute_elbo_gaussian(d, sj) - 1e-6
    assert compute_elbo_jj(d, sj) >= compute_elbo_jj(d, sc) - 1e-6


def test_variants_score_globaljj_where_compute_elbo_is_blocked():
    # compute_elbo swaps in a Compress response but keeps kernel='linear' and trips
    # __post_init__; both variants avoid that (cf retags kernel='vi_gh'; jj builds no table).
    X, y = _sim(seed=6)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=3, method="globaljj", max_iter=60)
    with pytest.raises(ValueError, match="quadratic"):
        compute_elbo(d, st)
    assert np.isfinite(compute_elbo_gaussian(d, st))
    assert np.isfinite(compute_elbo_jj(d, st))


def test_q1_state_rejected():
    # a free-form Q1 fit (logistic) has no closed-form transform for either variant.
    X, y = _sim(seed=7)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=2, method="logistic", max_iter=20)
    assert st.single_effects[0].b_nodes is not None  # confirm it is Q1-shaped
    for fn in (compute_elbo_gaussian, compute_elbo_jj):
        with pytest.raises(ValueError, match="Q2-compatible"):
            fn(d, st)


def test_variants_with_random_intercept():
    # the shared and random intercept fold into both variants; cf still matches compute_elbo,
    # jj is still a lower bound.
    rng = np.random.default_rng(8)
    n, p, causal = 500, 25, 4
    X = rng.standard_normal((n, p))
    b0i = rng.standard_normal(n)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.5 + 1.5 * X[:, causal] + b0i)))).astype(float)
    d = prep_data(X, y)
    st = fit_glm_susie(X, y, L=3, method="cf_cavi", random_intercept=True,
                       random_intercept_variance=0.5, max_iter=50)
    assert compute_elbo_gaussian(d, st) == pytest.approx(compute_elbo(d, st), abs=1e-6)
    assert compute_elbo_jj(d, st) <= compute_elbo_gaussian(d, st) + 1e-6
