"""Self-normalized (`offset_integration='compress_selfnorm'`) offset fold on the
PROFILED and DENSE-CENTERED quad kernels.

Both fold each other effect against its TRUE, non-Gaussian quadrature posterior
`(b_nodes, logW)`. Profiled kernels now expose those nodes (`glm_profile_ser_nodes`),
so the fold works under `intercept='profiled'` -- with `o = X*b` (the per-feature
profiled intercept is excluded from the message, and there is no shared intercept to
fold). Dense designs are centered eagerly, so a `center=True` fit runs `glm_ser` on the
already-centered X and the fold reads that same centered X. A defensive guard fails
loudly if a FITTED effect reaches the fold without nodes (which would silently fold a
zero offset)."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gibss import glm
from gibss.methods import fit_glm_susie

jax.config.update("jax_enable_x64", True)


def _sim(seed, n=250, p=30, signals=((3, 1.8), (17, -1.6)), b0=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    for j, v in signals:
        b[j] = v
    logit = X @ b + b0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return jnp.asarray(X), jnp.asarray(y), [j for j, _ in signals]


def _flm(state):
    """Per-effect per-feature log marginal (the evidence contract; NOT the pip)."""
    return np.stack([np.asarray(e.feature_log_marginal) for e in state.single_effects])


def test_profiled_selfnorm_runs_matches_gaussian_and_carries_nodes():
    X, y, truth = _sim(0)
    s = fit_glm_susie(
        X, y, L=3, offset_integration="compress_selfnorm", intercept="profiled"
    )
    s0 = fit_glm_susie(X, y, L=3, offset_integration="compress", intercept="profiled")
    # runs, converges, recovers both signals
    assert s.converged and s.n_iter < 100
    top = {int(j) for j in np.argsort(-np.asarray(s.pip))[: len(truth)]}
    assert set(truth) <= top
    # the profiled quad kernel now exposes its nodes to the fold
    for e in s.single_effects:
        assert e.b_nodes is not None
        assert e.b_nodes.shape[1] == X.shape[1]
    # self-normalized vs Gaussian-compress agree on the per-feature log evidence
    assert np.max(np.abs(_flm(s) - _flm(s0))) < 0.5


def test_center_dense_selfnorm_runs_and_matches_manual_centering():
    X, y, truth = _sim(1)
    s = fit_glm_susie(X, y, L=3, offset_integration="compress_selfnorm", center=True)
    # equivalent fit on a manually pre-centered design with center=False
    Xc = jnp.asarray(np.asarray(X) - np.asarray(X).mean(0)[None, :])
    sm = fit_glm_susie(Xc, y, L=3, offset_integration="compress_selfnorm", center=False)
    assert s.converged and s.n_iter < 100
    top = {int(j) for j in np.argsort(-np.asarray(s.pip))[: len(truth)]}
    assert set(truth) <= top
    # eager centering == manual centering: the fold reads the same centered X
    assert np.max(np.abs(_flm(s) - _flm(sm))) < 1e-6


def test_selfnorm_fold_guard_trips_on_fitted_effect_without_nodes():
    """A fitted effect (finite kl) that reaches the self-normalized fold with
    b_nodes=None would be silently dropped -> a zero offset -> a wrong fit. The guard
    must raise instead. (An unfit/empty effect has kl=inf and is legitimately skipped.)"""
    X, y, _ = _sim(2, n=120, p=12, signals=((2, 1.8), (7, -1.6)))
    s = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm")
    # manufacture a fitted effect (finite kl) with its quad nodes stripped
    e0 = s.single_effects[0]
    assert np.isfinite(float(e0.kl))
    bad = replace(e0, b_nodes=None, log_node_weight=None)
    state = replace(s, single_effects=[bad, s.single_effects[1]])
    data = glm.prep_data(X, y, center=False)
    # folding the OTHER effects for target l=1 now sees the fitted-but-nodeless effect
    try:
        glm._compress_fold_aux(data, state, 1)
    except ValueError as ex:
        assert "quad nodes" in str(ex)
    else:
        raise AssertionError("guard did not trip on a fitted effect with no nodes")
