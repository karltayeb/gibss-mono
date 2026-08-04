"""The fit_glm_susie front door: presets, axis resolution, and override rules.

Presets are partial configs over the axes, so `method=` must be pure sugar --
byte-identical to spelling the same axes explicitly -- and explicit kwargs must
beat preset values. The translation layer only rejects what doesn't translate;
everything else defers to GLMFamilyState/Smoother.validate (checked here only
through the error types that surface).
"""

import numpy as np
import pytest

from gibss.engine import MeanMessage
from gibss.methods import PRESETS, fit_glm_susie
from gibss.response import GH, Bernoulli, JJEnvelope, JJFixed, Smoothed, TaylorFixed


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _tops(state):
    return {int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects}


CAUSAL = 4


def _data(family):
    rng = np.random.default_rng(0)
    n, p = 500, 12
    if family == "poisson":
        X = rng.normal(size=(n, p)) * 0.6
        y = rng.poisson(np.exp(0.2 + 0.8 * X[:, CAUSAL])).astype(float)
    elif family == "gaussian":
        X = rng.normal(size=(n, p))
        y = 0.5 + 1.0 * X[:, CAUSAL] + rng.normal(size=n)
    else:
        X = rng.normal(size=(n, p))
        y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, CAUSAL]))), n).astype(float)
    return X, y


@pytest.mark.parametrize("method", sorted(PRESETS))
def test_every_preset_recovers(method):
    X, y = _data(PRESETS[method].get("family", "logistic"))
    st = fit_glm_susie(X, y, L=2, method=method, max_iter=50)
    assert _feat_pip(st, CAUSAL) > 0.9
    assert CAUSAL in _tops(st)


def _valid_posterior(state):
    for e in state.single_effects:
        a, mu, var = np.asarray(e.alpha), np.asarray(e.mu), np.asarray(e.var)
        assert np.isfinite(a).all() and (a >= -1e-9).all() and abs(a.sum() - 1) < 1e-4
        assert np.isfinite(mu).all() and np.isfinite(var).all() and (var > 0).all()


@pytest.mark.parametrize("method", ["logistic", "poisson", "globaljj", "irls", "score"])
def test_sparse_center_valid_and_recovers(method):
    # the center=True + sparse path must produce a VALID posterior (finite, alpha a
    # simplex, var>0) and recover a strong causal -- guards against silent corruption
    # from the centered kernels (glm_center_ser / CenteredOperator linear).
    import jax
    from jax.experimental import sparse
    rng = np.random.default_rng(0)
    n, p, causal = 500, 20, 3
    X = (rng.normal(size=(n, p)) + 2.0) * (rng.random((n, p)) < 0.12)  # off-center, sparse
    Xd = X.copy()
    if method == "poisson":
        xb = Xd[:, causal] * 0.8; xb -= xb.mean()
        y = rng.poisson(np.exp(0.3 + xb)).astype(float)
    else:
        xb = Xd[:, causal] * 3.0; xb -= xb.mean()
        y = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + xb))), n).astype(float)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))
    st = fit_glm_susie(Xs, y, L=2, method=method, center=True, max_iter=60)
    assert st.family_state.background == "chebyshev"
    _valid_posterior(st)
    assert causal in _tops(st)


def test_preset_is_sugar_for_explicit_kwargs():
    X, y = _data("logistic")
    a = fit_glm_susie(X, y, L=2, method="localjj", max_iter=50)
    b = fit_glm_susie(
        X, y, L=2, variational_family="gaussian", offset_integration="jj", max_iter=50
    )
    np.testing.assert_array_equal(np.asarray(a.alpha), np.asarray(b.alpha))


def test_explicit_kwarg_beats_preset():
    # localjj dispatches to the conjugate jj kernel only with a shared intercept;
    # overriding intercept must also flip the dispatch to vi + JJEnvelope
    # (centered localjj), not error on jj + profiled.
    X, y = _data("logistic")
    shared = fit_glm_susie(X, y, L=2, method="localjj", max_iter=50)
    assert shared.family_state.kernel == "jj"
    assert isinstance(shared.family_state.response.smoother, JJFixed)
    prof = fit_glm_susie(X, y, L=2, method="localjj", intercept="profiled", max_iter=50)
    assert prof.family_state.kernel == "vi"
    assert prof.family_state.intercept == "profiled"
    assert isinstance(prof.family_state.response.smoother, JJEnvelope)
    assert _feat_pip(prof, CAUSAL) > 0.9


def test_score_preset_config():
    X, y = _data("logistic")
    st = fit_glm_susie(X, y, L=2, method="score", max_iter=50)
    fs = st.family_state
    assert fs.kernel == "linear"
    assert fs.intercept == "null"
    assert isinstance(fs.response.smoother, TaylorFixed)
    assert fs.response.smoother.anchor == "null"
    assert _feat_pip(st, CAUSAL) > 0.9


def test_irls_preset_is_mean_message():
    X, y = _data("logistic")
    st = fit_glm_susie(X, y, L=2, method="irls", max_iter=50)
    assert isinstance(st.total_message, MeanMessage)
    assert st.family_state.kernel == "linear"


def test_family_object_passthrough():
    X, y = _data("logistic")
    a = fit_glm_susie(X, y, L=2, family=Smoothed(Bernoulli(), GH(5)), max_iter=50)
    b = fit_glm_susie(
        X, y, L=2, offset_integration="gh", offset_quadrature_points=5, max_iter=50
    )
    np.testing.assert_array_equal(np.asarray(a.alpha), np.asarray(b.alpha))


def _feature_log_marginal(state):
    # per-feature log evidence, stacked over effects (L, p): the pre-softmax quantity
    # PIP saturates away, so a far more sensitive equivalence check than PIP.
    return np.stack([np.asarray(e.feature_log_marginal) for e in state.single_effects])


@pytest.mark.parametrize("intercept", ["shared", "profiled"])
def test_compress_offset_matches_gh(intercept):
    # M1: offset_integration="compress" wires Compress through the quad kernel. With the
    # aggregate offset as a single Gaussian (K=1), it must reproduce the GH smoother's
    # per-feature log evidence to Chebyshev-interpolation error -- the whole
    # _resolve/_aux/terms path.
    X, y = _data("logistic")
    kw = dict(L=3, offset_quadrature_points=15, intercept=intercept, max_iter=60)
    gh = fit_glm_susie(X, y, offset_integration="gh", **kw)
    cp = fit_glm_susie(X, y, offset_integration="compress", **kw)
    np.testing.assert_allclose(_feature_log_marginal(cp), _feature_log_marginal(gh), atol=1e-4)


def test_compress_offset_matches_gh_sparse():
    import jax
    from jax.experimental import sparse

    rng = np.random.default_rng(1)
    n, p = 400, 20
    Xd = (rng.random((n, p)) < 0.25) * rng.normal(1.0, 0.3, (n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 1.5 * Xd[:, 0]))), n).astype(float)
    X = sparse.BCOO.fromdense(jax.numpy.asarray(Xd))
    kw = dict(L=3, offset_quadrature_points=15, max_iter=60)
    gh = fit_glm_susie(X, y, offset_integration="gh", **kw)
    cp = fit_glm_susie(X, y, offset_integration="compress", **kw)
    np.testing.assert_allclose(_feature_log_marginal(cp), _feature_log_marginal(gh), atol=1e-4)


def test_compress_offset_rejects_vi_kernel():
    # Compress is a fixed-per-row-offset scheme (quad kernel only); the vi kernel folds
    # the effect's own variance into the offset, which its tables can't express.
    X, y = _data("logistic")
    with pytest.raises(ValueError, match="unconstrained"):
        fit_glm_susie(X, y, L=2, offset_integration="compress",
                      variational_family="gaussian", max_iter=5)


def test_prior_variance_threads_through():
    X, y = _data("logistic")
    st = fit_glm_susie(
        X, y, L=2, prior_variance=0.5, estimate_prior_variance=False, max_iter=5
    )
    assert all(e.prior_variance == 0.5 for e in st.single_effects)


@pytest.mark.parametrize("method", sorted(PRESETS))
def test_every_preset_runs_on_sparse(method):
    # the center=None default must be layout- AND kernel-aware: on a BCOO design only
    # the quad kernel consumes centering, so every OTHER preset must fall back to
    # uncentered rather than crash (guards the center=True front-door regression).
    import jax
    from jax.experimental import sparse
    fam = PRESETS[method].get("family", "logistic")
    X, y = _data(fam)
    X = X * (np.random.default_rng(1).random(X.shape) < 0.4)  # genuinely sparse
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))
    st = fit_glm_susie(Xs, y, L=2, method=method, max_iter=30)
    assert st.family_state.background == "chebyshev"


def test_linear_kernel_sparse_centered_matches_dense():
    # globaljj (kernel='linear') centers a sparse design exactly and row-wise, so the
    # BCOO fit matches the dense (eager-centered) fit to machine precision -- unlike the
    # quad kernel's ~1e-5 chebyshev tolerance, this path has no approximation.
    import jax
    from jax.experimental import sparse
    rng = np.random.default_rng(4)
    n, p, causal = 400, 10, 3
    X = (rng.normal(size=(n, p)) + 2.0) * (rng.random((n, p)) < 0.4)  # off-center, sparse
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3 + 2 * X[:, causal]))), n).astype(float)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))
    dense = fit_glm_susie(X, y, L=2, method="globaljj", max_iter=50)
    sp = fit_glm_susie(Xs, y, L=2, method="globaljj", max_iter=50)
    assert dense.family_state.kernel == "linear"
    assert causal in _tops(dense)
    np.testing.assert_allclose(np.asarray(dense.alpha), np.asarray(sp.alpha), atol=1e-9)


def test_explicit_center_true_unsupported_sparse_raises():
    # an EXPLICIT center=True on a sparse + non-quad kernel is a clear front-door error,
    # not a deep NotImplementedError from the dispatch.
    import jax
    from jax.experimental import sparse
    X, y = _data("logistic")
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))
    with pytest.raises(ValueError, match="center=True is not supported"):
        fit_glm_susie(Xs, y, method="localjj", center=True, max_iter=5)


def test_center_and_adaptive_background_front_door():
    # the front door defaults center=True and resolves background by layout: exact for
    # dense, chebyshev for sparse. center=True makes dense and BCOO the SAME model, so
    # the two fits agree (previously sparse silently skipped centering).
    import jax
    from jax.experimental import sparse
    rng = np.random.default_rng(3)
    n, p, causal = 400, 10, 3
    X = (rng.normal(size=(n, p)) + 2.0) * (rng.random((n, p)) < 0.4)  # off-center, sparse
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3 + 2 * X[:, causal]))), n).astype(float)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))

    dense = fit_glm_susie(X, y, L=2, max_iter=50)
    sp = fit_glm_susie(Xs, y, L=2, max_iter=50)
    assert dense.family_state.background == "exact"
    assert sp.family_state.background == "chebyshev"
    assert causal in _tops(dense)
    np.testing.assert_allclose(np.asarray(dense.alpha), np.asarray(sp.alpha), atol=1e-5)


def test_explicit_background_beats_adaptive():
    # an explicit background overrides the layout-adaptive default.
    import jax
    from jax.experimental import sparse
    X, y = _data("logistic")
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(X))
    st = fit_glm_susie(Xs, y, L=1, intercept="profiled", background="exact", max_iter=10)
    assert st.family_state.background == "exact"


def test_invalid_configurations_raise():
    X, y = _data("logistic")
    with pytest.raises(ValueError, match="unknown method"):
        fit_glm_susie(X, y, method="probit-ish")
    with pytest.raises(ValueError, match="unknown family"):
        fit_glm_susie(X, y, family="probit")
    with pytest.raises(ValueError, match="unknown offset_integration"):
        fit_glm_susie(X, y, offset_integration="pg")
    with pytest.raises(ValueError, match="unknown variational_family"):
        fit_glm_susie(X, y, variational_family="cauchy")
    # JJ bound is Bernoulli-specific: rejected downstream by Smoother.validate
    with pytest.raises(TypeError, match="JJ"):
        fit_glm_susie(X, y, family="poisson", offset_integration="jj")
    # Gaussian cumulant is quadratic: offset integration is exact and free
    with pytest.raises(ValueError, match="quadratic cumulant"):
        fit_glm_susie(X, y, family="gaussian", offset_integration="gh")
    # Gaussian q needs a scheme (it IS the E_q operator)
    with pytest.raises(ValueError, match="offset-integration scheme"):
        fit_glm_susie(X, y, variational_family="gaussian")
    # unknown intercept: rejected downstream by GLMFamilyState
    with pytest.raises(ValueError, match="intercept"):
        fit_glm_susie(X, y, intercept="wat")
