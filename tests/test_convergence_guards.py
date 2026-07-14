"""Convergence handling: early-exit (tol) + non-finite guard on the mode-finders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gibss.operators import DenseOperator
from gibss.legacy.ser_ops import (
    local_irls,
    local_irls_centered,
    localjj_centered_ser,
    profile_ser,
    quadrature_ser,
)


def _stress():
    # extreme imbalance (offset saturates mu->1) + strong effect: the regime that
    # NaN'd undamped Newton.
    rng = np.random.default_rng(0)
    n, p = 600, 40
    Xd = rng.normal(size=(n, p))
    offset = np.full(n, 4.5) + rng.normal(size=n) * 0.2  # mu ~ 0.99
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 3.0 * Xd[:, 0]))), size=n).astype(float)
    return DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset)


@pytest.mark.parametrize(
    "fn",
    [
        lambda op, y, o: local_irls(op, y, o, 1.0),
        lambda op, y, o: local_irls_centered(op, y, o, 1.0),
        lambda op, y, o: quadrature_ser(op, y, o, 1.0, order=11),
        lambda op, y, o: profile_ser(op, y, o, 1.0, order=11),
        lambda op, y, o: localjj_centered_ser(op, y, o, 1.0),
    ],
)
def test_all_outputs_finite_under_saturation(fn):
    op, y, offset = _stress()
    out = fn(op, y, offset)
    for a in out:
        assert np.all(np.isfinite(np.asarray(a))), "non-finite output under saturation"


def test_early_exit_matches_fixed_iterations():
    # early-exit (tol) must give the same result as running the full cap
    rng = np.random.default_rng(1)
    n, p = 400, 20
    Xd = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 2]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    a = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0, n_iter=200, tol=1e-10)
    b = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0, n_iter=200, tol=0.0)  # no early exit
    for x, z in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(z), atol=1e-8)


def test_nonfinite_feature_nullified():
    # a degenerate column (all zeros) can't blow up but exercises the guard path;
    # force a NaN-prone setup and check the output stays finite + that column -> b=0
    rng = np.random.default_rng(2)
    n, p = 300, 10
    Xd = rng.normal(size=(n, p))
    Xd[:, 5] = 0.0  # degenerate feature
    offset = np.full(n, 5.0)  # saturated
    y = rng.binomial(1, 0.99, size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    b, var, lbf = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    assert np.all(np.isfinite(np.asarray(b)))
    np.testing.assert_allclose(float(np.asarray(b)[5]), 0.0, atol=1e-10)  # degenerate -> 0
