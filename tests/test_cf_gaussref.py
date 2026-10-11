"""Q2 CAVI through the Gaussian offset reference plus a CF tilt (`CharFnGaussRef`).

The likelihood is the K-component probit mixture of the logistic law, so the ground truth
is exact: the leave-one-out offset is a product of Gaussian mixtures, every component of
the enumerated product is a Gaussian `N(m, v)`, and the probit mixture integrates each
component in closed form (`ProbitMixture.terms` at variance `v`). The tilt engine must
match that enumeration without ever forming it.
"""

from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss.cf_offset import (
    _PROBIT_MIXTURES,
    CharFnGaussRef,
    _mixture_tail_Tmax,
    smoothed_nodes,
    smoothed_nodes_sparse,
)
from gibss.response import Bernoulli, ProbitMixture
from gibss.response_ser import _gh_rule, _stein_sums

K = 8


def _mix(z, v):
    """Probit-mixture `(A, A', A'')` at nodes `z` (n, M) under offset variance `v` (n,)."""
    ll, g, w = ProbitMixture(K).terms(
        Bernoulli(), jnp.asarray(z), (jnp.zeros_like(z), jnp.broadcast_to(v[:, None], z.shape))
    )
    return -np.asarray(ll), -np.asarray(g), np.asarray(w)


def _exact_atilde(effects, znodes):
    """Exact zero-mean `E_o (A, A', A'')(z + o)` by enumerating the product mixture."""
    effects = [tuple(np.asarray(t) for t in e) for e in effects]
    n = effects[0][0].shape[0]
    s = sum(x @ (a * m) for x, a, m, _ in effects)
    out = [np.zeros_like(znodes) for _ in range(3)]
    for combo in itertools.product(*[range(e[1].shape[0]) for e in effects]):
        w, m_i, v_i = 1.0, np.zeros(n), np.zeros(n)
        for (x, a, mu, var), c in zip(effects, combo):
            w *= a[c]
            m_i += x[:, c] * mu[c]
            v_i += x[:, c] ** 2 * var[c]
        for acc, t in zip(out, _mix(znodes + (m_i - s)[:, None], jnp.asarray(v_i))):
            acc += w * t
    return out


def _rand_effects(rng, n, L, C, mu_scale=1.0, var_scale=0.3):
    effects = []
    for _ in range(L):
        a = np.exp(rng.normal(size=C))
        a /= a.sum()
        effects.append((
            jnp.asarray(rng.normal(size=(n, C))),
            jnp.asarray(a),
            jnp.asarray(rng.normal(size=C) * mu_scale),
            jnp.asarray(np.exp(rng.normal(size=C) * 0.5) * var_scale),
        ))
    return effects


@pytest.mark.parametrize("L", [1, 2, 3])
@pytest.mark.parametrize("var_scale", [0.01, 0.3, 2.0])
def test_tilt_matches_enumerated_mixture(L, var_scale):
    rng = np.random.default_rng(10 * L + int(100 * var_scale))
    effects = _rand_effects(rng, 10, L, 3, var_scale=var_scale)
    z, A0, A1, A2, *_ = smoothed_nodes(effects, M=48, reference=K)
    e0, e1, e2 = _exact_atilde(effects, np.asarray(z))
    for got, want in [(A0, e0), (A1, e1), (A2, e2)]:
        assert np.max(np.abs(np.asarray(got) - want)) < 1e-9


def test_tilt_vanishes_for_gaussian_offset():
    # every other effect has its alpha on one feature, so the offset is exactly Gaussian
    # per row and the reference is already the answer
    rng = np.random.default_rng(3)
    effects = []
    for _ in range(3):
        a = np.zeros(4)
        a[rng.integers(4)] = 1.0
        effects.append((jnp.asarray(rng.normal(size=(12, 4))), jnp.asarray(a),
                        jnp.asarray(rng.normal(size=4)), jnp.asarray(rng.uniform(.1, 1, 4))))
    _, D0, D1, D2, *_ = smoothed_nodes(effects, M=32, reference=K, residual=True)
    assert max(float(jnp.max(jnp.abs(D))) for D in (D0, D1, D2)) < 1e-12


def test_tilt_is_smaller_than_plugin_residual():
    # six diffuse effects of moderate size: the tilt off the moment-matched Gaussian is
    # 6-15x smaller than the plug-in residual here (sup over the nodes, per channel)
    rng = np.random.default_rng(5)
    effects = _rand_effects(rng, 20, 6, 5, mu_scale=0.3, var_scale=0.1)
    _, G0, G1, G2, *_ = smoothed_nodes(effects, M=48, reference=K, residual=True)
    _, P0, P1, P2, *_ = smoothed_nodes(effects, M=48, residual=True)
    for G, P in [(G0, P0), (G1, P1), (G2, P2)]:
        assert float(jnp.max(jnp.abs(G))) < 0.25 * float(jnp.max(jnp.abs(P)))


def test_plugin_residual_mode_is_consistent():
    # residual=True under the plug-in reference is (A - At, A' - At', At'' - A'')
    rng = np.random.default_rng(6)
    effects = _rand_effects(rng, 8, 2, 3)
    z, A0, A1, A2, *_ = smoothed_nodes(effects, M=32)
    _, D0, D1, D2, *_ = smoothed_nodes(effects, M=32, residual=True)
    sig = jax.nn.sigmoid(z)
    np.testing.assert_allclose(D0, jax.nn.softplus(z) - A0, atol=1e-13)
    np.testing.assert_allclose(D1, sig - A1, atol=1e-13)
    np.testing.assert_allclose(D2, A2 - sig * (1 - sig), atol=1e-13)


def test_terms_reproduce_exact_cumulant_off_nodes():
    # the table presentation (closed-form reference + Clenshaw tilt) at arbitrary eta
    # inside the fit interval, against the enumerated mixture. Off the nodes the error
    # is the Chebyshev interpolation error, about 1e-4 at the default M=48 for this
    # offset (the plug-in CharFnOffset table is 2-3x worse at the same M); M=128 checks
    # the presentation itself.
    rng = np.random.default_rng(7)
    n = 10
    effects = _rand_effects(rng, n, 2, 3)
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    sm = CharFnGaussRef(M=128, order=K)
    aux = sm.build_aux(Bernoulli(), y, effects)
    hw = float(aux[3][0])
    eta = jnp.asarray(rng.uniform(-0.9 * hw, 0.9 * hw, size=(n, 5)))
    e0, e1, e2 = _exact_atilde(effects, np.asarray(eta))
    for k in range(eta.shape[1]):
        ll, g, w = sm.terms(Bernoulli(), eta[:, k], aux)
        np.testing.assert_allclose(ll, np.asarray(y * eta[:, k]) - e0[:, k], atol=1e-8)
        np.testing.assert_allclose(g, np.asarray(y) - e1[:, k], atol=1e-8)
        np.testing.assert_allclose(w, e2[:, k], atol=1e-8)


@pytest.mark.parametrize("centered", [False, True])
def test_sparse_build_matches_dense(centered):
    from jax.experimental import sparse

    rng = np.random.default_rng(40 + centered)
    n, C = 30, 8
    Xd = jnp.asarray((rng.random((n, C)) < 0.35) * rng.standard_normal((n, C)))
    colmean = Xd.mean(axis=0) if centered else None
    Xdense = Xd - colmean[None, :] if centered else Xd
    dense_eff, sparse_eff = [], []
    for _ in range(3):
        a = np.exp(rng.standard_normal(C))
        a /= a.sum()
        law = (jnp.asarray(a), jnp.asarray(rng.standard_normal(C)),
               jnp.asarray(0.3 * np.exp(0.3 * rng.standard_normal(C))))
        dense_eff.append((Xdense,) + law)
        sparse_eff.append(law)
    d = smoothed_nodes(dense_eff, M=48, reference=K, residual=True, offset_var=0.2)
    s = smoothed_nodes_sparse(sparse.BCOO.fromdense(Xd), sparse_eff, M=48, reference=K,
                              residual=True, offset_var=0.2, colmean=colmean)
    for a, b in zip(d, s):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_mixture_tail_rule_meets_tolerance():
    s, pi, _ = _PROBIT_MIXTURES[K]
    for tol in (1e-6, 1e-10, 1e-14):
        T = _mixture_tail_Tmax(K, tol)
        tail = sum((2 / math.pi) * p * sk * math.sqrt(math.pi / 2)
                   * math.erfc(T / (sk * math.sqrt(2))) for sk, p in zip(s, pi))
        assert tail <= tol and tail > 0.5 * tol


def test_integrated_terms_match_gauss_hermite():
    # E_b of the reference terms and the two Stein sums, closed form vs 60-node GH
    rng = np.random.default_rng(8)
    n = 50
    eta = jnp.asarray(rng.normal(0, 3, n))
    y = jnp.asarray((rng.random(n) < 0.5) * 1.0)
    ov, ve = jnp.asarray(rng.uniform(0, 2, n)), jnp.asarray(rng.uniform(0, 3, n))
    pm = ProbitMixture(K)
    closed = pm.integrated_terms(Bernoulli(), eta, (y, ov), ve)
    u, lw = _gh_rule(60)
    u, wts = jnp.asarray(u), jnp.exp(jnp.asarray(lw)) / jnp.sqrt(jnp.pi)
    ll, g, w = pm.terms(Bernoulli(), eta[None] + jnp.sqrt(2 * ve)[None] * u[:, None],
                        (y[None], ov[None]))
    gh = (jnp.sum(wts[:, None] * ll, 0), jnp.sum(wts[:, None] * g, 0)) + _stein_sums(wts, u, w)
    for a, b in zip(closed, gh):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_fit_matches_cf_cavi():
    # the two routes to the CAVI-in-Q2 fixed point agree to the table's error, at a fixed
    # sweep count from the same start
    from gibss.methods import fit_glm_susie

    rng = np.random.default_rng(9)
    n, p = 300, 20
    X = rng.normal(size=(n, p))
    eta = -0.5 + X[:, 3] - 0.8 * X[:, 11]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    kw = {"L": 3, "max_iter": 8, "tol": 0.0}
    ref = fit_glm_susie(X, y, method="cf_cavi", **kw)
    new = fit_glm_susie(X, y, method="cf_cavi_gaussref", **kw)
    np.testing.assert_allclose(new.alpha, ref.alpha, atol=1e-5)
    np.testing.assert_allclose(new.ser_log_bf, ref.ser_log_bf, atol=1e-4)


def test_front_door_rejects_unsupported_axes():
    from gibss.methods import fit_glm_susie

    X, y = np.zeros((4, 2)), np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="cf_gaussref"):
        fit_glm_susie(X, y, offset_integration="cf_gaussref")
    with pytest.raises(ValueError, match="cf_gaussref"):
        fit_glm_susie(X, y, family="poisson", variational_family="gaussian",
                      offset_integration="cf_gaussref")
