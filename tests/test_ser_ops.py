import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.linear as lin
from gibss._jj import lambda_xi
from gibss.operators import BCOOOperator, DenseOperator, LowRankOperator
from gibss.ser_ops import global_gaussian_ser


def _ops(Xd):
    """Dense, BCOO, and an exact low-rank operator for the SAME dense matrix."""
    n, p = Xd.shape
    U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
    lr = LowRankOperator(jnp.asarray(U * s), jnp.asarray(Vt))
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
        "lowrank": lr,
    }


@pytest.mark.parametrize("kind", ["dense", "bcoo", "lowrank"])
def test_global_ser_matches_linear_kernel(kind):
    rng = np.random.default_rng(0)
    n, p = 60, 15
    Xd = rng.normal(size=(n, p))
    y = jnp.asarray(rng.normal(size=n))
    offset = jnp.asarray(rng.normal(size=n) * 0.3)
    pv = 1.7
    tau = jnp.asarray(np.abs(rng.normal(size=n)) + 0.5)
    op = _ops(Xd)[kind]
    mu, var, _ = global_gaussian_ser(op, tau, tau * (y - offset), pv)

    # linear kernel uses tau via obs_variance; build data with matching tau
    data = lin.prep_data(jnp.asarray(Xd), y, center=False)
    mu_ref, var_ref, _ = lin.fit_univariate_linear_regression(data, tau, offset, pv)
    np.testing.assert_allclose(mu, mu_ref, atol=1e-9)
    np.testing.assert_allclose(var, var_ref, atol=1e-9)


@pytest.mark.parametrize("kind", ["dense", "bcoo", "lowrank"])
def test_global_ser_matches_globaljj_kernel(kind):
    rng = np.random.default_rng(1)
    n, p = 60, 15
    Xd = rng.normal(size=(n, p))
    y = jnp.asarray(rng.binomial(1, 0.4, size=n).astype(float))
    offset = jnp.asarray(rng.normal(size=n) * 0.3)
    xi = jnp.asarray(np.abs(rng.normal(size=n)) + 0.5)
    pv = 1.3
    tau = 2.0 * lambda_xi(xi)
    op = _ops(Xd)[kind]
    mu, var, _ = global_gaussian_ser(op, tau, y - 0.5 - tau * offset, pv)

    # independent reference: the global-JJ per-feature curvature/gradient
    Xn, tn = np.asarray(Xd), np.asarray(tau)
    r = np.asarray(y - 0.5) - tn * np.asarray(offset)
    var_ref = 1.0 / (1.0 / pv + (tn[:, None] * Xn**2).sum(0))
    mu_ref = var_ref * (r[:, None] * Xn).sum(0)
    np.testing.assert_allclose(mu, mu_ref, atol=1e-9)
    np.testing.assert_allclose(var, var_ref, atol=1e-9)


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_global_ser_precentering_matches_manual(kind):
    rng = np.random.default_rng(2)
    n, p = 50, 12
    Xd = rng.normal(size=(n, p)) + 0.7  # nonzero column means
    y = jnp.asarray(rng.normal(size=n))
    offset = jnp.zeros(n)
    pv = 1.0
    tau = jnp.asarray(np.abs(rng.normal(size=n)) + 0.5)
    cbar = jnp.asarray(np.asarray(Xd).mean(0))
    from gibss.operators import CenteredOperator
    op = CenteredOperator.from_offsets(_ops(Xd)[kind], cbar)  # centering via the operator
    mu, var, _ = global_gaussian_ser(op, tau, tau * (y - offset), pv)

    # manual: fit on the explicitly centered design
    Xc = np.asarray(Xd) - np.asarray(cbar)
    tn = np.asarray(tau)
    x2 = (tn[:, None] * Xc**2).sum(0)
    num = (tn * np.asarray(y - offset))[:, None] * Xc
    num = num.sum(0)
    var_ref = 1.0 / (1.0 / pv + x2)
    mu_ref = var_ref * num
    np.testing.assert_allclose(mu, mu_ref, atol=1e-10)
    np.testing.assert_allclose(var, var_ref, atol=1e-10)
