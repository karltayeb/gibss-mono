import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, CenteredOperator, DenseOperator
from gibss.ser_ops import global_gaussian_ser


def _bases(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_centered_ops_match_explicit_centered_design(kind):
    rng = np.random.default_rng(0)
    n, p = 40, 10
    Xd = rng.normal(size=(n, p)) + 0.6
    c = rng.normal(size=p)  # arbitrary offsets
    Xc = Xd - c  # explicit centered design
    u = jnp.asarray(rng.normal(size=n))
    v = jnp.asarray(rng.normal(size=p))
    w = jnp.asarray(np.abs(rng.normal(size=n)))
    op = CenteredOperator.from_offsets(_bases(Xd)[kind], jnp.asarray(c))
    np.testing.assert_allclose(op.matvec(v), Xc @ np.asarray(v), atol=1e-10)
    np.testing.assert_allclose(op.rmatvec(u), Xc.T @ np.asarray(u), atol=1e-10)
    for k in (0, 1, 2, 3):
        np.testing.assert_allclose(
            op.moment(k, w), (Xc**k * np.asarray(w)[:, None]).sum(0), atol=1e-9
        )
    np.testing.assert_allclose(op.gram_matvec(v), Xc.T @ (Xc @ np.asarray(v)), atol=1e-9)


def test_from_weights_is_the_weighted_mean_and_schur():
    rng = np.random.default_rng(1)
    n, p = 50, 12
    Xd = rng.normal(size=(n, p)) + 0.4
    tau = jnp.asarray(np.abs(rng.normal(size=n)) + 0.3)
    base = DenseOperator(jnp.asarray(Xd))
    op = CenteredOperator.from_weights(base, tau)
    # cached c == tau-weighted column mean
    cbar = (np.asarray(tau) @ Xd) / np.asarray(tau).sum()
    np.testing.assert_allclose(op.c, cbar, atol=1e-10)
    # at the weighted mean, moment(2, tau) == Schur complement S2 - S1^2/W
    S2 = (np.asarray(tau)[:, None] * Xd**2).sum(0)
    S1 = np.asarray(tau) @ Xd
    W = np.asarray(tau).sum()
    np.testing.assert_allclose(op.moment(2, tau), S2 - S1**2 / W, atol=1e-9)


def test_recenter_updates_offsets_base_shared():
    rng = np.random.default_rng(2)
    Xd = rng.normal(size=(30, 6))
    base = DenseOperator(jnp.asarray(Xd))
    op1 = CenteredOperator.from_weights(base, jnp.ones(30))  # unweighted mean
    np.testing.assert_allclose(op1.c, Xd.mean(0), atol=1e-10)
    w2 = jnp.asarray(np.abs(rng.normal(size=30)) + 0.2)
    op2 = op1.recenter(w2)
    np.testing.assert_allclose(op2.c, (np.asarray(w2) @ Xd) / np.asarray(w2).sum(), atol=1e-10)
    assert op2.base is base  # base reused, not rebuilt


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_centered_operator_matches_manual_centering(kind):
    # global_gaussian_ser(CenteredOperator(base, c), ...) == the same on a manually
    # centered dense X -- the operator carries the centering (no cbar arg anymore).
    rng = np.random.default_rng(3)
    n, p = 45, 11
    Xd = rng.normal(size=(n, p)) + 0.5
    c = jnp.asarray(Xd.mean(0))
    tau = jnp.asarray(np.abs(rng.normal(size=n)) + 0.4)
    r = jnp.asarray(rng.normal(size=n))
    pv = 1.3
    base = _bases(Xd)[kind]
    a = global_gaussian_ser(CenteredOperator.from_offsets(base, c), tau, r, pv)
    b = global_gaussian_ser(DenseOperator(jnp.asarray(np.asarray(Xd) - np.asarray(c))), tau, r, pv)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-10)


def test_centered_operator_is_pytree_jittable():
    rng = np.random.default_rng(4)
    Xd = jnp.asarray(rng.normal(size=(20, 5)))
    op = CenteredOperator.from_offsets(DenseOperator(Xd), jnp.asarray(np.asarray(Xd).mean(0)))
    w = jnp.asarray(np.abs(rng.normal(size=20)))
    f = jax.jit(lambda o, w: o.moment(2, w))
    ref = ((np.asarray(Xd) - np.asarray(Xd).mean(0)) ** 2 * np.asarray(w)[:, None]).sum(0)
    np.testing.assert_allclose(f(op, w), ref, atol=1e-10)
