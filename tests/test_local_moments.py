"""`DesignOperator.local_moments`: several per-entry reductions in one call. Dense fuses
them into ONE variadic reduce (so a producer the weights share is evaluated once); BCOO keeps
its separate segment sums. Both must equal the per-reduction `local_moment`."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.operators import BCOOOperator, CenteredOperator, DenseOperator


def _case(seed=0, n=120, p=9):
    rng = np.random.default_rng(seed)
    Xd = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4)
    return Xd, rng.normal(size=(n, p)), rng.random((n, p))


@pytest.mark.parametrize("kind", ["dense", "bcoo", "centered"])
def test_local_moments_equals_local_moment(kind):
    Xd, Wa, Wb = _case()
    if kind == "dense":
        op = DenseOperator(jnp.asarray(Xd))
        specs = ((1, jnp.asarray(Wa)), (2, jnp.asarray(Wb)), (0, jnp.asarray(Wa * Wb)))
    elif kind == "centered":
        op = CenteredOperator.from_weights(DenseOperator(jnp.asarray(Xd)), jnp.ones(Xd.shape[0]))
        specs = ((1, jnp.asarray(Wa)), (2, jnp.asarray(Wb)))
    else:
        X = sparse.BCOO.fromdense(jnp.asarray(Xd))
        op = BCOOOperator(X)
        rows, cols = X.indices[:, 0], X.indices[:, 1]
        specs = ((1, jnp.asarray(Wa)[rows, cols]), (2, jnp.asarray(Wb)[rows, cols]))
    got = op.local_moments(specs)
    assert len(got) == len(specs)
    for (k, W), g in zip(specs, got):
        np.testing.assert_allclose(np.asarray(g), np.asarray(op.local_moment(k, W)), rtol=1e-12)


def test_dense_local_moments_under_jit_matches_separate_sums():
    Xd, Wa, Wb = _case(1)
    op = DenseOperator(jnp.asarray(Xd))
    f = jax.jit(lambda a, b: op.local_moments(((1, a), (2, b))))
    M1, M2 = f(jnp.asarray(Wa), jnp.asarray(Wb))
    np.testing.assert_allclose(np.asarray(M1), (Xd * Wa).sum(0), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(M2), (Xd**2 * Wb).sum(0), rtol=1e-12)
