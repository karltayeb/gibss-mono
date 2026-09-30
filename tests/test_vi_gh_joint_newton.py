"""The joint (m, v) Newton of the vi_gh kernels: its Stein-identity node sums are the
true third/fourth cumulant expectations, and it lands on the same fixed point as the
classic m-Newton / Price-v alternation, in fewer iterations, on the columns where the
alternation crawls (weakly supported: v ~ prior variance)."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss._numerics import _gh_rule
from gibss.operators import BCOOOperator, DenseOperator
from gibss.response import Bernoulli
from gibss.response_ser import _joint_newton_step, _stein_sums, glm_vi_gh_ser


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_stein_sums_are_cumulant_expectations():
    """sqrt(2/v) T3 == sum_k W_k x A'''(node_k) and T4 / v == sum_k W_k x^2 A''''(node_k)
    (Stein's identities are exact for a Gaussian, and GH integrates the polynomial
    weights exactly), against the closed-form Bernoulli cumulant derivatives."""
    rng = np.random.default_rng(0)
    order = 60  # both sides are GH approximations; a high order makes them agree tightly
    nodes_np, logw_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    wts = jnp.exp(jnp.asarray(logw_np)) / jnp.sqrt(jnp.pi)
    eta = jnp.asarray(rng.standard_normal((7, 5)))
    x = jnp.asarray(rng.standard_normal((7, 5)))
    m = jnp.asarray(rng.standard_normal(5) * 0.3)
    v = jnp.asarray(np.exp(rng.standard_normal(5) * 0.5) * 0.1)
    sd = jnp.sqrt(2.0 * x**2 * v[None, :])
    shft = eta[None] + x[None] * m[None, None, :] + sd[None] * nodes[:, None, None]
    base = Bernoulli()
    _, _, w = base.terms(shft, 0.0)
    Ew, T3, T4 = _stein_sums(wts, nodes, w)
    A3, A4 = base.cumulant_derivs(shft)
    W = wts[:, None, None]
    assert np.allclose(np.asarray(Ew), np.asarray(jnp.sum(W * w, 0)), atol=1e-12)
    # the analytic v-derivatives (Price) of E[g] and E[w], vs the Stein node sums; the
    # GH order is high enough for both sides to agree to ~1e-9 on the smooth logistic
    # cumulant.
    # the nodes sit at |x| sqrt(2v) u, so b - m = sign(x) sqrt(2v) u (odd sum gets the sign)
    lhs3 = jnp.sign(x) * jnp.sqrt(2.0 / v)[None, :] * T3
    rhs3 = jnp.sum(W * x[None] * A3, 0)
    lhs4 = T4 / v[None, :]
    rhs4 = jnp.sum(W * x[None] ** 2 * A4, 0)
    assert np.max(np.abs(np.asarray(lhs3 - rhs3))) < 1e-8
    assert np.max(np.abs(np.asarray(lhs4 - rhs4))) < 1e-8


def _alternation(op, y, offset, pv, order, n_iter=400, tol=1e-12):
    """The classic alternation (Newton on m at fixed v, then v <- 1/(1/pv + W)), run to a
    tight tolerance as the reference fixed point."""
    nodes_np, logw_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    wts = jnp.exp(jnp.asarray(logw_np)) / jnp.sqrt(jnp.pi)
    x = op.entry_x
    off_e = op.broadcast_rows(offset)
    y_e = op.broadcast_rows(y)
    base = Bernoulli()
    m = jnp.zeros(op.p)
    v = jnp.full(op.p, pv)
    for it in range(n_iter):
        sd = jnp.sqrt(2.0 * x**2 * op.broadcast_cols(v))
        shft = off_e[None] + (x * op.broadcast_cols(m))[None] + sd[None] * nodes.reshape((-1,) + (1,) * x.ndim)
        _, g, w = base.terms(shft, y_e[None])
        W = wts.reshape((-1,) + (1,) * x.ndim)
        G = op.local_moment(1, jnp.sum(W * g, 0))
        Wm = op.local_moment(2, jnp.sum(W * w, 0))
        prec = 1.0 / pv + Wm
        step = jnp.clip((G - m / pv) / prec, -4.0, 4.0)
        m, v = m + step, 1.0 / prec
        if float(jnp.max(jnp.abs(step))) < tol:
            break
    return m, v, it + 1


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_joint_newton_same_fixed_point_fewer_iterations(layout):
    """On a sparse 0/1 design with weakly supported columns, the kernel (joint Newton)
    reaches the alternation's fixed point to ~1e-9 and needs fewer iterations than the
    alternation does to reach its own tolerance."""
    rng = np.random.default_rng(1)
    n, p = 400, 30
    Xd = (rng.random((n, p)) < 0.06).astype(float)  # ~24 support rows per column
    beta = np.zeros(p)
    beta[[3, 11]] = [2.0, -2.0]
    y = (rng.random(n) < _sigmoid(Xd @ beta - 1.0)).astype(float)
    offset = jnp.asarray(rng.standard_normal(n) * 0.3 - 1.0)
    y = jnp.asarray(y)
    op = DenseOperator(jnp.asarray(Xd)) if layout == "dense" else BCOOOperator(
        sparse.BCOO.fromdense(jnp.asarray(Xd))
    )
    pv, order = 1.0, 15
    m_ref, v_ref, k_ref = _alternation(op, y, offset, pv, order)
    m, v, bf, kl = glm_vi_gh_ser(op, y, offset, pv, response=Bernoulli(), order=order)
    assert np.max(np.abs(np.asarray(m) - np.asarray(m_ref))) < 1e-8
    assert np.max(np.abs(np.asarray(v) - np.asarray(v_ref))) < 1e-8

    # iteration count of the joint step itself (replaying the kernel's update rule)
    x = op.entry_x
    off_e = op.broadcast_rows(offset)
    y_e = op.broadcast_rows(y)
    nodes_np, logw_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    wts = jnp.exp(jnp.asarray(logw_np)) / jnp.sqrt(jnp.pi)
    mm, vv = jnp.zeros(op.p), jnp.full(op.p, pv)
    for k in range(1, 101):
        sd = jnp.sqrt(2.0 * x**2 * op.broadcast_cols(vv))
        shft = off_e[None] + (x * op.broadcast_cols(mm))[None] + sd[None] * nodes.reshape((-1,) + (1,) * x.ndim)
        _, g, w = Bernoulli().terms(shft, y_e[None])
        W = wts.reshape((-1,) + (1,) * x.ndim)
        Ew, T3, T4 = _stein_sums(wts, nodes, w)
        vb = op.broadcast_cols(vv)
        mm, vv, resid = _joint_newton_step(
            mm, vv, op.local_moment(1, jnp.sum(W * g, 0)), op.local_moment(2, Ew),
            op.local_moment(2, jnp.sign(x) * jnp.sqrt(2.0 / vb) * T3),
            op.local_moment(2, T4 / vb),
            1.0 / pv,
        )
        if float(resid) <= 1e-8:
            break
    assert k < k_ref, (k, k_ref)
    assert k <= 12


def test_residual_split_matches_unsplit_kernel():
    """A table smoother with `residual_order` integrates its plug-in base on the kernel's
    order and its Chebyshev residual on fewer nodes; the fit must match the unsplit
    kernel (residual_order=None) to ~1e-8 -- the residual is small and smooth."""
    from gibss.cf_offset import CharFnOffset
    from gibss.response import Smoothed

    rng = np.random.default_rng(3)
    n, p = 300, 6
    X = jnp.asarray(rng.standard_normal((n, p)) * 0.5)
    y = jnp.asarray((rng.random(n) < _sigmoid(np.asarray(X) @ (rng.standard_normal(p)))).astype(float))
    offset = jnp.asarray(rng.standard_normal(n) * 0.3)
    others = []
    for _ in range(2):
        a = np.exp(rng.standard_normal(4)); a /= a.sum()
        others.append((jnp.asarray(rng.standard_normal((n, 4))), jnp.asarray(a),
                       jnp.asarray(rng.standard_normal(4) * 0.5),
                       jnp.asarray(0.2 * np.exp(rng.standard_normal(4) * 0.3))))
    op = DenseOperator(X)
    out = {}
    for ro in (None, "auto", 5):
        sm = CharFnOffset(M=48, residual_order=ro)
        aux = sm.build_aux(Bernoulli(), y, others)
        out[ro] = glm_vi_gh_ser(op, aux, offset, 1.0, response=Smoothed(Bernoulli(), sm), order=15)
    for ro in ("auto", 5):
        for k, tol in zip(range(3), (1e-8, 1e-8, 1e-7)):
            assert np.max(np.abs(np.asarray(out[ro][k]) - np.asarray(out[None][k]))) < tol, (ro, k)
    # the split is real: a high explicit residual order reproduces the unsplit numbers
    sm = CharFnOffset(M=48, residual_order=15)
    aux = sm.build_aux(Bernoulli(), y, others)
    same = glm_vi_gh_ser(op, aux, offset, 1.0, response=Smoothed(Bernoulli(), sm), order=15)
    for k in range(3):
        assert np.max(np.abs(np.asarray(same[k]) - np.asarray(out[None][k]))) < 1e-12
