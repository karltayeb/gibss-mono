"""The response-generic Gaussian-VI-with-GH-over-b kernel `glm_vi_gh_ser`.

Covers: (1) it reduces to the trusted `glm_vi_ser` on a plain base (no offset);
(2) an empty CharFnOffset table == the plain base (table plumbing / vector aux leaves);
(3) a real offset table runs, converges, and moves the answer off the plug-in; and
(4) the converged (m, v) satisfies the EXACT CAVI stationarity, computed from a
brute-force offset-integrated cumulant -- i.e. it really is the Q2 CAVI fixed point.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss.cf_offset import CharFnOffset
from gibss.operators import DenseOperator
from gibss.response import Bernoulli, Smoothed, GH
from gibss.response_ser import glm_vi_ser, glm_vi_gh_ser


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rand_effects(rng, n, L, C, var_scale=0.3):
    eff = []
    for _ in range(L):
        x = rng.standard_normal((n, C))
        a = np.exp(rng.standard_normal(C))
        a /= a.sum()
        mu = rng.standard_normal(C)
        var = var_scale * np.exp(rng.standard_normal(C) * 0.3)
        eff.append((jnp.asarray(x), jnp.asarray(a), jnp.asarray(mu), jnp.asarray(var)))
    return eff


def _problem(rng, n, p):
    X = rng.standard_normal((n, p))
    y = (rng.uniform(size=n) < _sigmoid(X @ rng.standard_normal(p))).astype(float)
    offset = rng.standard_normal(n) * 0.5
    return DenseOperator(jnp.asarray(X)), jnp.asarray(y), jnp.asarray(offset)


@pytest.mark.parametrize("order", [15, 25])
def test_reduces_to_glm_vi_ser_plain_base(order):
    """Plain base, no offset table: GH-over-b (glm_vi_gh_ser) == the convolution-seam
    VI (glm_vi_ser with ov=0) to tight tol -- both integrate b~N(m,v) over the same
    base cumulant with the same GH nodes."""
    rng = np.random.default_rng(0)
    op, y, offset = _problem(rng, n=200, p=8)
    pv = 1.0
    base = Bernoulli()
    m1, v1, bf1, kl1 = glm_vi_ser(
        op, (y, 0.0), offset, pv, response=Smoothed(base, GH(order))
    )
    m2, v2, bf2, kl2 = glm_vi_gh_ser(op, y, offset, pv, response=base, order=order)
    assert np.allclose(np.asarray(m1), np.asarray(m2), atol=1e-6)
    assert np.allclose(np.asarray(v1), np.asarray(v2), atol=1e-6)
    assert np.allclose(np.asarray(bf1), np.asarray(bf2), atol=1e-6)
    assert np.allclose(np.asarray(kl1), np.asarray(kl2), atol=1e-6)


def test_empty_table_matches_plain_base():
    """An empty CharFnOffset table (no other effects -> Atilde = A) run through the
    table path must equal the plain-base run: validates the vector-aux (coef (n,M+1))
    broadcasting through the GH-over-b."""
    rng = np.random.default_rng(1)
    op, y, offset = _problem(rng, n=150, p=6)
    pv = 0.5
    base = Bernoulli()
    order = 20
    m_plain, v_plain, bf_plain, kl_plain = glm_vi_gh_ser(
        op, y, offset, pv, response=base, order=order
    )
    sm = CharFnOffset(M=48)
    aux = sm.build_aux(base, y, [])  # empty offset
    m_t, v_t, bf_t, kl_t = glm_vi_gh_ser(
        op, aux, offset, pv, response=Smoothed(base, sm), order=order
    )
    assert np.allclose(np.asarray(m_plain), np.asarray(m_t), atol=1e-6)
    assert np.allclose(np.asarray(v_plain), np.asarray(v_t), atol=1e-6)
    assert np.allclose(np.asarray(bf_plain), np.asarray(bf_t), atol=1e-6)
    assert np.allclose(np.asarray(kl_plain), np.asarray(kl_t), atol=1e-6)


def test_real_table_runs_and_moves_off_plugin():
    """A real offset table runs, converges (finite, v>0), and the offset integration
    changes the answer vs the b=0-plug-in (empty-table) fit."""
    rng = np.random.default_rng(2)
    op, y, offset = _problem(rng, n=200, p=5)
    pv = 1.0
    base = Bernoulli()
    sm = CharFnOffset(M=48)
    effects = _rand_effects(rng, n=200, L=3, C=5, var_scale=0.8)  # 3 other effects
    aux = sm.build_aux(base, y, effects)
    m, v, bf, kl = glm_vi_gh_ser(
        op, aux, offset, pv, response=Smoothed(base, sm), order=20
    )
    assert np.all(np.isfinite(np.asarray(m)))
    assert np.all(np.asarray(v) > 0)
    assert np.all(np.isfinite(np.asarray(bf)))
    # differs from the empty-table (no offset integration) fit
    aux0 = sm.build_aux(base, y, [])
    m0, *_ = glm_vi_gh_ser(op, aux0, offset, pv, response=Smoothed(base, sm), order=20)
    assert np.max(np.abs(np.asarray(m) - np.asarray(m0))) > 1e-4


def _brute_atilde_deriv(effects, eta, which, gh_order=200):
    """Exact Atilde-derivative (which in {0:A, 1:A', 2:A''}) of the ZERO-MEAN offset
    formed by `effects`, at arbitrary points `eta` (any shape). Enumerate the product
    mixture, GH-integrate each Gaussian component."""
    effects = [(np.asarray(x), np.asarray(a), np.asarray(m), np.asarray(v))
               for x, a, m, v in effects]
    n = effects[0][0].shape[0]
    Cs = [e[1].shape[0] for e in effects]
    s = np.zeros(n)
    for x, a, m, _ in effects:
        s += x @ (a * m)
    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)

    def f(z):
        sig = _sigmoid(z)
        return {0: np.logaddexp(0.0, z), 1: sig, 2: sig * (1.0 - sig)}[which]

    out = np.zeros_like(eta)
    # eta has leading axis n (rows); component means/vars are per-row (n,)
    for combo in itertools.product(*[range(c) for c in Cs]):
        w = 1.0
        cm = np.zeros(n)
        cv = np.zeros(n)
        for (x, a, mu, var), c in zip(effects, combo):
            w *= a[c]
            cm += x[:, c] * mu[c]
            cv += x[:, c] ** 2 * var[c]
        # zero-mean offset component: N(cm - s, cv); broadcast over eta's trailing axes
        mean = (cm - s).reshape((n,) + (1,) * (eta.ndim - 1))
        sd = np.sqrt(cv).reshape((n,) + (1,) * (eta.ndim - 1))
        for gk, wk in zip(nodes, wts):
            out = out + w * wk * f(eta + mean + np.sqrt(2.0) * sd * gk)
    return out


def test_fixed_point_satisfies_exact_cavi_stationarity():
    """At the kernel's converged (m, v), the EXACT CAVI gradient -- computed from a
    brute-force offset-integrated cumulant (not the table) -- vanishes, and 1/v matches
    the exact curvature. This ties the kernel to the true Q2 CAVI fixed point."""
    rng = np.random.default_rng(3)
    n, p = 8, 3
    op, y, offset = _problem(rng, n=n, p=p)
    X = np.asarray(op.X)
    yv = np.asarray(y)
    off = np.asarray(offset)
    pv = 1.0
    base = Bernoulli()
    sm = CharFnOffset(M=48)
    effects = _rand_effects(rng, n=n, L=2, C=3, var_scale=0.5)  # enumerable (3^2)
    aux = sm.build_aux(base, y, effects)
    order = 40
    m, v, bf, kl = glm_vi_gh_ser(
        op, aux, offset, pv, response=Smoothed(base, sm), order=order
    )
    m, v = np.asarray(m), np.asarray(v)

    # GH over b ~ N(m_j, v_j): eta points (n, p, order)
    gnodes, gwts = np.polynomial.hermite.hermgauss(order)
    gwts = gwts / np.sqrt(np.pi)
    b_pts = m[None, :] + np.sqrt(2.0 * v)[None, :] * gnodes[:, None]  # (order, p)
    eta = off[:, None, None] + X[:, :, None] * b_pts.T[None, :, :]  # (n, p, order)

    A1 = _brute_atilde_deriv(effects, eta, which=1)  # Atilde' at the points
    A2 = _brute_atilde_deriv(effects, eta, which=2)  # Atilde''
    Eb_g = np.einsum("k,npk->np", gwts, yv[:, None, None] - A1)  # E_b[y - Atilde']
    Eb_w = np.einsum("k,npk->np", gwts, A2)  # E_b[Atilde'']

    grad_m = np.sum(X * Eb_g, axis=0) - m / pv  # should be ~0 at the CAVI stationary pt
    prec = 1.0 / pv + np.sum(X**2 * Eb_w, axis=0)  # exact curvature
    assert np.max(np.abs(grad_m)) < 1e-4, f"grad_m={grad_m}"
    assert np.allclose(1.0 / v, prec, rtol=1e-3, atol=1e-4)


def test_glm_vi_gh_ser_separated_column_large_pv_converges():
    """Regression: on a (near-)separated column at large prior variance the undamped
    Newton m-step used to overshoot to +-inf, giving m ~ -200 and a confidently-wrong
    large-NEGATIVE log_bf (which flips the PIP of the true signal). The +-4 damping must
    keep it at the finite optimum. Checked against a direct brute-force ELBO maximum."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    n = 120
    xc = rng.standard_normal(n)
    y = (xc > 0).astype(float)  # perfectly separated by column c...
    flip = rng.choice(n, 3, replace=False)
    y[flip] = 1.0 - y[flip]  # ...with 3 flips -> near-separation (finite but huge MLE)
    base = Bernoulli()
    sm = CharFnOffset(M=48)
    resp = Smoothed(base, sm)
    aux = sm.build_aux(base, jnp.asarray(y), [])  # empty offset -> base cumulant
    op = DenseOperator(jnp.asarray(xc[:, None]))
    off = jnp.zeros(n)
    gn, gw = np.polynomial.hermite.hermgauss(160)
    gw = gw / np.sqrt(np.pi)

    ll0 = -n * np.log(2.0)  # b=0 null at offset=0: sum_i -softplus(0)

    def neg_elbo(z, pv):
        m, v = z[0], np.exp(z[1])
        b = m + np.sqrt(2 * v) * gn
        eta = xc[:, None] * b[None, :]
        ll = (gw * (y[:, None] * eta - np.logaddexp(0, eta))).sum(1).sum()
        return -((ll - ll0) - 0.5 * (np.log(pv / v) + (v + m * m) / pv - 1))

    for pv in (20.0, 100.0):
        m, v, bf, kl = glm_vi_gh_ser(op, aux, off, pv, resp, order=20, n_iter=500)
        assert np.isfinite(float(m[0])) and abs(float(m[0])) < 20  # not the +-inf overshoot
        assert float(bf[0]) > 0  # the true signal has POSITIVE evidence, not -1e4
        bfB = -minimize(neg_elbo, [1.0, 0.0], args=(pv,)).fun
        assert abs(float(bf[0]) - bfB) < 1e-4
