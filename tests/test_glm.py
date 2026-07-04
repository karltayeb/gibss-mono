"""Generic GLM-SER engine family: one kernel, any ResponseModel.

`glm.GLMFamilyState(response=...)` runs a full SuSiE on any per-observation
likelihood. Logistic SuSiE = GLM(Bernoulli), Poisson SuSiE = GLM(Poisson), same code
path. The kernel-level equivalences (glm_ser == quadrature_ser for Bernoulli, Poisson
vs brute) live in test_response; here we guard the engine wiring and recovery.
"""

import numpy as np
import pytest

from gibss import glm
from gibss import logistic_localtaylor as LT
from gibss.engine import fit_ibss
from gibss.response import Bernoulli, Poisson


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _tops(state):
    return {int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_glm_bernoulli_recovers_and_matches_localtaylor(seed):
    rng = np.random.default_rng(seed)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)

    d = glm.prep_data(X, y)
    g = fit_ibss(d, glm.initialize_state(d, L=2, response=Bernoulli()), glm.default_schedule(), max_iter=50)
    dl = LT.prep_data(X, y)
    lt = fit_ibss(dl, LT.initialize_state(dl, L=2), LT.default_schedule(), max_iter=50)

    assert _feat_pip(g, causal) > 0.9
    assert causal in _tops(g)
    # two logistic approximations (quadrature-over-b vs local-Taylor) agree on the top
    assert _tops(g) == _tops(lt)
    assert np.isfinite(g.family_state.intercept)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_glm_poisson_recovers(seed):
    rng = np.random.default_rng(seed)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p)) * 0.6
    y = rng.poisson(np.exp(0.2 + 0.8 * X[:, causal])).astype(float)

    d = glm.prep_data(X, y)
    g = fit_ibss(d, glm.initialize_state(d, L=2, response=Poisson()), glm.default_schedule(), max_iter=50)

    assert _feat_pip(g, causal) > 0.9
    assert causal in _tops(g)
    assert np.isfinite(g.family_state.intercept)


def test_glm_profile_recovers_and_matches_localtaylor():
    # GLM(Bernoulli, profile=True) full SuSiE agrees with logistic_localtaylor's
    # profile mode on the selected features (per-feature profiled intercept).
    from gibss import logistic_localtaylor as LT
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    g = fit_ibss(d, glm.initialize_state(d, L=2, response=Bernoulli(),
                 family_state_kwargs={"profile": True}), glm.default_schedule(), max_iter=50)
    dl = LT.prep_data(X, y)
    lt = fit_ibss(dl, LT.initialize_state(dl, L=2, family_state_kwargs={"profile": True}),
                  LT.default_schedule(), max_iter=50)
    assert _feat_pip(g, causal) > 0.9
    assert _tops(g) == _tops(lt)


def test_glm_profile_dense_sparse_cheb_parity():
    # profiled SuSiE: sparse (chebyshev background) reproduces dense (exact) -- the
    # sparsity-exploiting profiled path.
    import jax
    from jax.experimental import sparse
    rng = np.random.default_rng(1)
    n, p, causal = 400, 10, 3
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, causal]))), n).astype(float)

    def run(Xin, bg):
        d = glm.prep_data(Xin, y)
        return fit_ibss(d, glm.initialize_state(d, L=2, response=Bernoulli(),
                        family_state_kwargs={"profile": True, "background": bg}),
                        glm.default_schedule(), max_iter=50)

    dense = run(X, "exact")
    sparse_cheb = run(sparse.BCOO.fromdense(jax.numpy.asarray(X)), "chebyshev")
    np.testing.assert_allclose(np.asarray(dense.alpha), np.asarray(sparse_cheb.alpha), atol=1e-6)


def test_glm_dense_sparse_parity():
    # full engine loop agrees on dense vs BCOO input (intercept-Newton accumulation
    # over sweeps keeps this slightly looser than the kernel-level 1e-9 parity)
    import jax
    from jax.experimental import sparse

    rng = np.random.default_rng(0)
    n, p, causal = 400, 10, 4
    Xd = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4)
    y = rng.binomial(1, 1 / (1 + np.exp(-(Xd[:, causal]))), n).astype(float)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(Xd))

    def run(X):
        d = glm.prep_data(X, y)
        return fit_ibss(d, glm.initialize_state(d, L=2, response=Bernoulli()), glm.default_schedule(), max_iter=40)

    np.testing.assert_allclose(np.asarray(run(Xd).alpha), np.asarray(run(Xs).alpha), atol=2e-3)


def test_glm_response_is_static_across_families():
    # the same module/schedule serves different families purely via the response arg
    rng = np.random.default_rng(0)
    n, p = 300, 8
    X = rng.normal(size=(n, p))
    yb = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * X[:, 0]))), n).astype(float)
    yp = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
    for resp, y in [(Bernoulli(), yb), (Poisson(), yp)]:
        d = glm.prep_data(X, y)
        s = fit_ibss(d, glm.initialize_state(d, L=1, response=resp), glm.default_schedule(), max_iter=30)
        assert np.isfinite(s.family_state.intercept)
        assert s.family_state.response is resp
