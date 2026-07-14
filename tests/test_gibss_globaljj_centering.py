import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.legacy.globaljj as gjj
from gibss.engine import fit_ibss


def _sim(rng, n, p, feat, signal=2.0, base=-0.5, sparsity=None):
    X = rng.normal(size=(n, p))
    if sparsity is not None:
        X = X * rng.binomial(1, sparsity, size=(n, p))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(base + signal * X[:, feat])))).astype(float)
    return X, y


def _run(Xmat, y, profile, L=1, max_iter=50):
    data = gjj.prep_data(Xmat, y)
    st = gjj.initialize_state(data, L=L, family_state_kwargs={"profile": profile})
    return fit_ibss(data, st, gjj.default_schedule(), max_iter=max_iter)


def _assert_monotone(history, label):
    h = np.asarray([e for e in history if np.isfinite(e)])
    diffs = np.diff(h)
    # variational coordinate ascent: ELBO non-decreasing up to float error
    worst = float(diffs.min()) if len(diffs) else 0.0
    assert worst > -1e-6, f"{label}: ELBO decreased by {worst:.2e}; history={h}"


def test_globaljj_elbo_monotone_uncentered_and_centered():
    # THE key check: the ELBO must be monotone with and without centering.
    rng = np.random.default_rng(0)
    n, p = 400, 15
    X, y = _sim(rng, n, p, 6, signal=2.0)
    for Xmat in (jnp.asarray(X), sparse.BCOO.fromdense(jnp.asarray(X))):
        for profile in (False, True):
            st = _run(Xmat, y, profile=profile, L=3, max_iter=40)
            label = f"{'sparse' if isinstance(Xmat, sparse.BCOO) else 'dense'}/profile={profile}"
            _assert_monotone(st.family_state.elbo_history, label)


def test_globaljj_centering_recovers_signal():
    rng = np.random.default_rng(1)
    n, p = 500, 20
    X, y = _sim(rng, n, p, 11, signal=2.0)
    for Xmat in (jnp.asarray(X), sparse.BCOO.fromdense(jnp.asarray(X))):
        st = _run(Xmat, y, profile=True, L=1)
        assert int(np.argmax(np.asarray(st.single_effects[0].alpha))) == 11


def test_globaljj_centered_sparse_matches_dense():
    rng = np.random.default_rng(2)
    n, p = 400, 15
    X, y = _sim(rng, n, p, 3, signal=2.0, sparsity=0.3)
    d = _run(jnp.asarray(X), y, profile=True, L=1)
    s = _run(sparse.BCOO.fromdense(jnp.asarray(X)), y, profile=True, L=1)
    np.testing.assert_allclose(np.asarray(d.single_effects[0].mu),
                               np.asarray(s.single_effects[0].mu), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(d.single_effects[0].alpha),
                               np.asarray(s.single_effects[0].alpha), rtol=1e-5, atol=1e-6)


def test_globaljj_centered_intercept_decoupled():
    # with centering, intercept = sum(y-0.5)/sum(tau), independent of effects
    rng = np.random.default_rng(3)
    n, p = 400, 12
    X, y = _sim(rng, n, p, 5, signal=2.0)
    # tighten convergence so the centering (and thus the decoupling) settles
    data = gjj.prep_data(jnp.asarray(X), y)
    from dataclasses import replace
    st = gjj.initialize_state(data, L=1, family_state_kwargs={"profile": True})
    st = replace(st, family_state=replace(st.family_state, elbo_tolerance=1e-10))
    st = fit_ibss(data, st, gjj.default_schedule(), max_iter=200)
    fs = st.family_state
    tau = 2.0 * np.asarray(gjj._lambda_xi(jnp.asarray(fs.xi)))
    decoupled = float(np.sum(y - 0.5) / np.sum(tau))
    # decoupling is exact at the fixed point; ~1e-5 at this convergence level
    np.testing.assert_allclose(float(fs.intercept), decoupled, atol=1e-4)


def test_globaljj_centered_uncentered_recover_same_signal():
    rng = np.random.default_rng(4)
    n, p = 500, 18
    X, y = _sim(rng, n, p, 9, signal=2.0)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    c = _run(Xs, y, profile=True, L=1)
    u = _run(Xs, y, profile=False, L=1)
    assert int(np.argmax(np.asarray(c.single_effects[0].alpha))) == 9
    assert int(np.argmax(np.asarray(u.single_effects[0].alpha))) == 9
    np.testing.assert_allclose(float(c.single_effects[0].mu[9]),
                               float(u.single_effects[0].mu[9]), rtol=2e-2)
