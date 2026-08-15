"""End-to-end exact-CAVI (self-normalized offset) logistic SuSiE via
`offset_integration='compress_selfnorm'`: the IBSS sweep folds each other effect AND the
shared intercept against its TRUE, non-Gaussian quadrature posterior (not Gaussian
moments). Validates that it runs on dense and sparse designs, recovers signals, agrees
with the Gaussian-compress fit on an easy problem, and populates the intercept nodes."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss.methods import fit_glm_susie

jax.config.update("jax_enable_x64", True)


def _sim(seed, n=250, p=30, signals=((3, 1.8), (17, -1.6)), b0=0.4, sparse_x=False):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < 0.15).astype(float) if sparse_x else rng.normal(size=(n, p))
    b = np.zeros(p)
    for j, v in signals:
        b[j] = v
    logit = X @ b + b0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return jnp.asarray(X), jnp.asarray(y), [j for j, _ in signals]


def test_selfnorm_dense_recovers_signals_and_matches_gaussian():
    X, y, truth = _sim(0)
    s = fit_glm_susie(X, y, L=3, offset_integration="compress_selfnorm")
    s0 = fit_glm_susie(X, y, L=3, offset_integration="compress")
    top = {int(j) for j in np.argsort(-np.asarray(s.pip))[: len(truth)]}
    assert set(truth) <= top
    assert s.converged and s.n_iter < 100
    # both are approximations of the same posterior -> close ser_log_bf on an easy problem
    assert np.allclose(np.asarray(s.ser_log_bf), np.asarray(s0.ser_log_bf), atol=0.5)


def test_selfnorm_sparse_recovers_signals():
    X, y, truth = _sim(1, n=350, p=40, signals=((5, 2.2), (22, -2.0)), b0=-1.0, sparse_x=True)
    Xsp = sparse.BCOO.fromdense(X)
    s = fit_glm_susie(Xsp, y, L=3, offset_integration="compress_selfnorm")
    top = {int(j) for j in np.argsort(-np.asarray(s.pip))[: len(truth)]}
    assert set(truth) <= top
    assert s.converged


def test_selfnorm_populates_intercept_quadrature():
    X, y, _ = _sim(2)
    s = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm")
    fs = s.family_state
    # the shared intercept was fit by quadrature: nodes present, value ~ posterior mean
    assert fs.intercept_b_nodes is not None and fs.intercept_log_node_weight is not None
    W = jax.nn.softmax(jnp.asarray(fs.intercept_log_node_weight))
    m0 = float(jnp.sum(W * jnp.asarray(fs.intercept_b_nodes)))
    assert abs(m0 - float(fs.intercept_value)) < 1e-6


def test_selfnorm_effects_carry_nodes():
    X, y, _ = _sim(3)
    s = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm")
    for e in s.single_effects:
        assert e.b_nodes is not None  # quad kernel populated the raw posterior
        assert e.b_nodes.shape[1] == X.shape[1]
