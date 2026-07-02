"""Local (per-feature-weight) SER via the Vandermonde moment recombination:
   curvature_j = sum_r (m_j^r/r!) * moment(2+r, g^{(r)}(offset))
Exact for polynomial weight g; convergent (in order) for smooth g.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gibss.operators import DenseOperator
from gibss.ser_ops import local_gaussian_ser


def _recover_x2(var, pv):
    return 1.0 / np.asarray(var) - 1.0 / pv


def _direct_local_curvature(Xd, offset, m, g):
    # sum_i g(offset_i + x_ij m_j) x_ij^2, per feature j
    eta = offset[:, None] + Xd * m[None, :]
    return (g(eta) * Xd**2).sum(0)


def test_vandermonde_exact_for_polynomial_weight():
    # g(eta) = eta^2  => Taylor terminates at r=2 => recombination is EXACT
    rng = np.random.default_rng(0)
    n, p = 50, 10
    Xd = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.4
    m = rng.normal(size=p) * 0.8
    pv = 1.0

    def weight_fn(order, off):
        # g=eta^2: derivatives [eta^2, 2 eta, 2, 0, ...] evaluated at eta=off
        derivs = [off**2, 2.0 * off, 2.0 * jnp.ones_like(off)]
        derivs += [jnp.zeros_like(off)] * (order + 1 - len(derivs))
        return derivs[: order + 1]

    op = DenseOperator(jnp.asarray(Xd))
    _, var, _ = local_gaussian_ser(
        op, weight_fn, jnp.asarray(offset), jnp.asarray(m), jnp.zeros(n), pv, order=2
    )
    x2 = _recover_x2(var, pv)
    ref = _direct_local_curvature(Xd, offset, m, lambda e: e**2)
    np.testing.assert_allclose(x2, ref, atol=1e-9)


def test_vandermonde_converges_for_smooth_weight():
    # g(eta) = sigmoid'(eta) = s(1-s): smooth, needs increasing order
    rng = np.random.default_rng(1)
    n, p = 60, 8
    Xd = rng.normal(size=(n, p)) * 0.6
    offset = rng.normal(size=n) * 0.3
    m = rng.normal(size=p) * 0.5
    pv = 1.0

    def g_scalar(e):  # elementwise sigmoid'(e)
        s = jax.nn.sigmoid(e)
        return s * (1 - s)

    def g(e):
        return g_scalar(e)

    def weight_fn(order, off):
        # g^{(0..order)}(off), each elementwise via nested grad + vmap
        ds = []
        dk = g_scalar
        for _ in range(order + 1):
            ds.append(jax.vmap(dk)(off))
            dk = jax.grad(dk)
        return ds

    op = DenseOperator(jnp.asarray(Xd))
    ref = _direct_local_curvature(Xd, offset, m, lambda e: g(e))
    errs = []
    for order in (2, 4, 6):
        _, var, _ = local_gaussian_ser(
            op, weight_fn, jnp.asarray(offset), jnp.asarray(m), jnp.zeros(n), pv, order=order
        )
        errs.append(np.max(np.abs(_recover_x2(var, pv) - ref)))
    # error decreases with order
    assert errs[1] < 0.3 * errs[0]  # clear geometric decrease with order
    assert errs[2] < 0.3 * errs[1]
    assert errs[2] < 5e-3
