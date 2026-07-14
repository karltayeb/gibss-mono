"""Generic GLM-SER engine family: one kernel, any ResponseModel.

`glm.GLMFamilyState(response=...)` runs a full SuSiE on any per-observation
likelihood. Logistic SuSiE = GLM(Bernoulli), Poisson SuSiE = GLM(Poisson), same code
path. The kernel-level equivalences (glm_ser == quadrature_ser for Bernoulli, Poisson
vs brute) live in test_response; here we guard the engine wiring and recovery.
"""

import numpy as np
import pytest

from gibss import glm
from gibss.legacy import logistic_localtaylor as LT
from gibss.engine import fit_ibss
from gibss.response import GH, Bernoulli, Gaussian, JJEnvelope, JJFixed, Poisson, Smoothed, Taylor, TaylorFixed


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
    assert np.isfinite(g.family_state.intercept_value)


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
    assert np.isfinite(g.family_state.intercept_value)


def test_glm_profile_recovers_and_matches_localtaylor():
    # GLM(Bernoulli, profile=True) full SuSiE agrees with logistic_localtaylor's
    # profile mode on the selected features (per-feature profiled intercept).
    from gibss.legacy import logistic_localtaylor as LT
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    g = fit_ibss(d, glm.initialize_state(d, L=2, response=Bernoulli(),
                 family_state_kwargs={"kernel": "quad", "intercept": "profiled"}), glm.default_schedule(), max_iter=50)
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
                        family_state_kwargs={"kernel": "quad", "intercept": "profiled", "background": bg}),
                        glm.default_schedule(), max_iter=50)

    dense = run(X, "exact")
    sparse_cheb = run(sparse.BCOO.fromdense(jax.numpy.asarray(X)), "chebyshev")
    np.testing.assert_allclose(np.asarray(dense.alpha), np.asarray(sparse_cheb.alpha), atol=1e-6)


@pytest.mark.parametrize("smoother,kw", [
    (GH(), {}),
    (GH(), {"kernel": "quad", "intercept": "profiled"}),
    (Taylor(), {}),
    (Taylor(), {"kernel": "quad", "intercept": "profiled"}),
    (JJEnvelope(), {}),
    (JJEnvelope(), {"kernel": "quad", "intercept": "profiled"}),
    (TaylorFixed(), {"kernel": "linear"}),
    (TaylorFixed(), {"kernel": "linear", "intercept": "profiled"}),
    (JJFixed(), {"kernel": "linear"}),
    (JJFixed(), {"kernel": "linear", "intercept": "profiled"}),
])
def test_glm_smoothed_response_runs_and_recovers(smoother, kw):
    # a Smoothed response consumes the message variance (o ~ N(mean, var)) through the
    # ordinary aux seam and still recovers the signal, shared-intercept and profiled.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2, response=Smoothed(Bernoulli(), smoother),
                  family_state_kwargs=kw), glm.default_schedule(), max_iter=50)
    assert _feat_pip(st, causal) > 0.9


def test_glm_jj_kernel_recovers():
    # classic localjj end-to-end: the conjugate kernel tunes the per-entry tilt
    # xi^2 = E_q[eta^2] + ov -- (offset + x m)^2 + x^2 v + ov, so the tilt SEES the
    # b-posterior variance -- while the JJFixed response supplies the fixed-tilt
    # bound. Certified ELBO, no GH anywhere.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2, response=Smoothed(Bernoulli(), JJFixed()),
                  family_state_kwargs={"kernel": "jj"}), glm.default_schedule(), max_iter=50)
    assert _feat_pip(st, causal) > 0.9
    assert causal in _tops(st)
    assert np.isfinite(st.family_state.intercept_value)


def test_glm_taylor_fixed_score_mode_recovers():
    # score mode: the expansion stays anchored at the intercept-only null (weights
    # frozen at the null fit) -- the one-step / score-test flavor. Still identifies
    # the causal feature.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2,
                  response=Smoothed(Bernoulli(), TaylorFixed(anchor="null")),
                  family_state_kwargs={"kernel": "linear"}),
                  glm.default_schedule(), max_iter=50)
    assert _feat_pip(st, causal) > 0.9
    assert causal in _tops(st)


def test_glm_taylor_fixed_matches_irls_tops():
    # the IRLS working model (TaylorFixed anchor="update") agrees with the old
    # logistic-only irls module on the selected features. (Not byte-identical: the
    # old module convolves the working data at the FULL predictor variance, the new
    # scheme smooths with the leave-one-out variance.)
    from gibss.legacy import irls
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    g = fit_ibss(d, glm.initialize_state(d, L=2,
                 response=Smoothed(Bernoulli(), TaylorFixed()),
                 family_state_kwargs={"kernel": "linear"}),
                 glm.default_schedule(), max_iter=50)
    di = irls.prep_data(X, y)
    ir = fit_ibss(di, irls.initialize_state(di, L=2), irls.default_schedule(), max_iter=50)
    assert _feat_pip(g, causal) > 0.9
    assert causal in _tops(g) and causal in _tops(ir)
    assert _tops(g) == _tops(ir)


def test_glm_mean_message_zeroes_offset_variance():
    # initialize_state_mean_message drops the message variance, so every Smoothed
    # scheme sees ov = 0: GH collapses to the unwrapped base family exactly, and
    # TaylorFixed becomes the pure (fixed-offset) IRLS working model and recovers.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    base = fit_ibss(d, glm.initialize_state_mean_message(d, L=2, response=Bernoulli()),
                    glm.default_schedule(), max_iter=50)
    gh0 = fit_ibss(d, glm.initialize_state_mean_message(d, L=2,
                   response=Smoothed(Bernoulli(), GH(5))),
                   glm.default_schedule(), max_iter=50)
    np.testing.assert_allclose(np.asarray(base.alpha), np.asarray(gh0.alpha), atol=1e-8)
    irls0 = fit_ibss(d, glm.initialize_state_mean_message(d, L=2,
                     response=Smoothed(Bernoulli(), TaylorFixed()),
                     family_state_kwargs={"kernel": "linear"}),
                     glm.default_schedule(), max_iter=50)
    assert _feat_pip(irls0, causal) > 0.9
    assert causal in _tops(irls0)


def test_glm_vi_kernel_recovers():
    # Gaussian-restricted variational SER end-to-end (q(b|gamma) = N(m, v)).
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2, response=Smoothed(Bernoulli(), GH(5)),
                  family_state_kwargs={"kernel": "vi"}), glm.default_schedule(), max_iter=50)
    assert _feat_pip(st, causal) > 0.9
    assert causal in _tops(st)


def test_glm_intercept_variance_is_curvature():
    # estimate_intercept returns (b0, v0) with v0 = 1/sum w = the inverse observed
    # curvature of the loglik in b0 (exact for an unwrapped ExponentialFamily);
    # checked against autodiff of the summed Bernoulli loglik.
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    n = 400
    off = rng.normal(size=n) * 0.5
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.7 + off))), n).astype(float)

    class D:
        pass
    d = D()
    d.y = y

    class M:
        mean = jnp.asarray(off)
        var = jnp.zeros(n)

    class S:
        family_state = glm.GLMFamilyState(response=Bernoulli())
        total_message = M()

    b0, v0 = glm.estimate_intercept(d, S())

    def ll(b):
        eta = jnp.asarray(off) + b
        return jnp.sum(jnp.asarray(y) * eta - jax.nn.softplus(eta))

    assert abs(float(jax.grad(ll)(jnp.asarray(b0)))) < 1e-6         # at the MAP
    curv = -float(jax.grad(jax.grad(ll))(jnp.asarray(b0)))
    np.testing.assert_allclose(v0, 1.0 / curv, rtol=1e-8)


def test_glm_intercept_variance_flows_into_offset_variance():
    # after a fit with a Smoothed response, intercept_var is tracked (> 0, ~1/(n w))
    # and consumed through ov; under MeanMessage (mean-only) it is excluded, which
    # the mean-message collapse test guards. Recovery is unaffected.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2, response=Smoothed(Bernoulli(), GH(5))),
                  glm.default_schedule(), max_iter=50)
    assert 0.0 < st.family_state.intercept_var < 0.1  # O(1/n)
    assert _feat_pip(st, causal) > 0.9


def test_glm_vi_profile_kernel_recovers():
    # profiled Gaussian VI end-to-end: no shared intercept step, offset-shift
    # invariant per-feature evidence, q(b|gamma) Gaussian.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st = fit_ibss(d, glm.initialize_state(d, L=2, response=Smoothed(Bernoulli(), GH(5)),
                  family_state_kwargs={"kernel": "vi", "intercept": "profiled"}),
                  glm.default_schedule(), max_iter=50)
    assert _feat_pip(st, causal) > 0.9
    assert causal in _tops(st)
    assert st.family_state.intercept_value == 0.0  # never modeled


def test_glm_kernel_response_compatibility_enforced():
    # quadratic responses are refused by the quadrature kernels (with a redirect to
    # the closed-form linear kernels), and vice versa -- explicit compute paths, no
    # silent clamping.
    with pytest.raises(ValueError, match="linear"):
        glm.GLMFamilyState(response=Smoothed(Bernoulli(), JJFixed()), kernel="quad")
    with pytest.raises(ValueError, match="linear"):
        glm.GLMFamilyState(response=Gaussian(), kernel="quad", intercept="profiled")
    with pytest.raises(ValueError, match="quadratic"):
        glm.GLMFamilyState(response=Smoothed(Bernoulli(), GH(5)), kernel="linear")
    with pytest.raises(ValueError, match="split into orthogonal"):
        glm.GLMFamilyState(response=Bernoulli(), kernel="linear_profile")
    with pytest.raises(ValueError, match="intercept"):
        glm.GLMFamilyState(intercept="global")
    with pytest.raises(ValueError, match="centered localjj"):
        glm.GLMFamilyState(response=Smoothed(Bernoulli(), JJFixed()),
                           kernel="jj", intercept="profiled")


def test_glm_null_intercept_fits_once_and_freezes():
    # intercept="null": fit at the b = 0 null model in initialize_state (= logit of
    # the base rate for Bernoulli), then frozen through the whole fit -- the
    # score-analysis intercept, the logistic analog of cox baseline="null".
    rng = np.random.default_rng(0)
    n, p, causal = 500, 12, 4
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    st0 = glm.initialize_state(d, L=2, family_state_kwargs={"intercept": "null"})
    b0_null = st0.family_state.intercept_value
    np.testing.assert_allclose(b0_null, np.log(y.mean() / (1 - y.mean())), atol=1e-6)
    fit = fit_ibss(d, st0, glm.default_schedule(), max_iter=30)
    assert fit.family_state.intercept_value == b0_null  # frozen
    assert _feat_pip(fit, causal) > 0.9


def test_glm_null_intercept_composes_with_row_tuned_schemes():
    # the fully classical one-pass score test: null intercept + null expansion
    # anchor + closed-form kernel. The init-time null fit must tune row_param first.
    rng = np.random.default_rng(1)
    n, p, causal = 400, 8, 2
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.3 * X[:, causal]))), n).astype(float)
    d = glm.prep_data(X, y)
    for smoother in (TaylorFixed("null"), TaylorFixed("update"), JJFixed()):
        st = glm.initialize_state(d, L=1, response=Smoothed(Bernoulli(), smoother),
                                  family_state_kwargs={"kernel": "linear", "intercept": "null"})
        np.testing.assert_allclose(st.family_state.intercept_value,
                                   np.log(y.mean() / (1 - y.mean())), atol=1e-6)
        fit = fit_ibss(d, st, glm.default_schedule(), max_iter=20)
        assert fit.family_state.intercept_value == st.family_state.intercept_value
        assert int(np.argmax(np.asarray(fit.single_effects[0].alpha))) == causal


def test_glm_unknown_kernel_refused():
    with pytest.raises(ValueError, match="kernel"):
        glm.GLMFamilyState(kernel="newton")


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
        assert np.isfinite(s.family_state.intercept_value)
        assert s.family_state.response is resp
