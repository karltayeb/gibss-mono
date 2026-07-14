"""Offset integration: Gaussian-convolve the logistic cumulant over the random
leave-one-out offset (mean-only vs o ~ N(mean, var)).

Covers the convolved-cumulant primitive, kernel-level backward compat (ov=0 is the
identity), correctness vs a brute-force double integral, convexity/robustness, and
the engine-level integrate_offset wiring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gibss.operators import DenseOperator, as_operator
from gibss.ser_ops import (
    _smooth_cumulant,
    local_irls,
    local_irls_centered,
    quadrature_ser,
    profile_ser,
)


# --------------------------------------------------------------------------- #
# convolved-cumulant primitive
# --------------------------------------------------------------------------- #
def _brute_cumulant(eta, ov):
    z = np.linspace(-9, 9, 6001)
    dens = np.exp(-(z**2) / (2 * ov)) / np.sqrt(2 * np.pi * ov) * (z[1] - z[0])
    p = 1.0 / (1.0 + np.exp(-(eta + z)))
    return (
        np.sum(np.logaddexp(0, eta + z) * dens),
        np.sum(p * dens),
        np.sum(p * (1 - p) * dens),
    )


@pytest.mark.parametrize("eta", [-3.0, -1.0, 0.0, 0.7, 2.5])
@pytest.mark.parametrize("ov", [0.25, 1.0])
def test_smooth_cumulant_matches_brute(eta, ov):
    Ab, sb, wb = _brute_cumulant(eta, ov)
    # gh is accurate; taylor is only asymptotic in ov -> loose tol for it
    for integ, tol in [(15, 1e-6), ("taylor", 5e-2)]:
        A, s, w = [float(x) for x in _smooth_cumulant(jnp.array(eta), jnp.array(ov), integ)]
        assert abs(A - Ab) < tol and abs(s - sb) < tol and abs(w - wb) < tol


@pytest.mark.parametrize("meth", ["taylor", 3, 9])
def test_no_var_forces_none_no_wasted_gh(meth):
    # offset_var=None (e.g. a MeanMessage) must short-circuit to the identity for
    # ANY offset_integration -- no GH-over-offset run on zeros.
    op, y, offset = _data()
    a = quadrature_ser(op, y, offset, 1.0, order=11, offset_var=None, offset_integration=meth)
    b = quadrature_ser(op, y, offset, 1.0, order=11, offset_var=None, offset_integration="none")
    for x, z in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(z), atol=1e-12)


@pytest.mark.parametrize("offset_integration", ["none", "taylor", 9])
def test_smooth_cumulant_ov0_is_exact(offset_integration):
    eta = jnp.linspace(-4, 4, 21)
    A, s, w = _smooth_cumulant(eta, jnp.zeros_like(eta), offset_integration)
    p = jax.nn.sigmoid(eta)
    np.testing.assert_allclose(np.asarray(A), np.asarray(jax.nn.softplus(eta)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(s), np.asarray(p), atol=1e-12)
    np.testing.assert_allclose(np.asarray(w), np.asarray(p * (1 - p)), atol=1e-12)


def test_taylor2_internally_consistent():
    # s~ = dA~/deta, w~ = ds~/deta (finite difference) -> Newton on A~ is coherent
    eta = jnp.linspace(-3, 3, 25)
    ov = jnp.full_like(eta, 0.7)
    h = 1e-5
    Ap = _smooth_cumulant(eta + h, ov, "taylor")[0]
    Am = _smooth_cumulant(eta - h, ov, "taylor")[0]
    A, s, w = _smooth_cumulant(eta, ov, "taylor")
    np.testing.assert_allclose(np.asarray((Ap - Am) / (2 * h)), np.asarray(s), atol=1e-6)
    sp = _smooth_cumulant(eta + h, ov, "taylor")[1]
    sm = _smooth_cumulant(eta - h, ov, "taylor")[1]
    np.testing.assert_allclose(np.asarray((sp - sm) / (2 * h)), np.asarray(w), atol=1e-6)


# --------------------------------------------------------------------------- #
# kernel backward compat: ov=None == ov=0
# --------------------------------------------------------------------------- #
def _data(seed=0, n=200, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.4
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.1 * X[:, 0]))), n).astype(float)
    return DenseOperator(jnp.asarray(X)), jnp.asarray(y), jnp.asarray(offset)


@pytest.mark.parametrize(
    "fn",
    [
        lambda op, y, o, ov: local_irls(op, y, o, 1.0, offset_var=ov),
        lambda op, y, o, ov: local_irls_centered(op, y, o, 1.0, offset_var=ov),
        lambda op, y, o, ov: quadrature_ser(op, y, o, 1.0, order=11, offset_var=ov),
        lambda op, y, o, ov: profile_ser(op, y, o, 1.0, order=11, offset_var=ov),
    ],
)
def test_ov_zero_matches_none(fn):
    op, y, offset = _data()
    a = fn(op, y, offset, None)
    b = fn(op, y, offset, jnp.zeros(op.n))
    for x, z in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(z), atol=1e-10)


# --------------------------------------------------------------------------- #
# correctness: offset-integrated quadrature vs brute double integral
# --------------------------------------------------------------------------- #
def _brute_quad_logbf(X, y, offset, ov, j, pv):
    zb = np.linspace(-9, 9, 201)
    dzb = zb[1] - zb[0]
    bpri = np.exp(-(zb**2) / (2 * pv)) / np.sqrt(2 * np.pi * pv)
    zo = np.linspace(-6, 6, 201)
    odens = np.exp(-(zo**2) / 2) / np.sqrt(2 * np.pi) * (zo[1] - zo[0])
    sd = np.sqrt(ov)
    eta = offset[:, None, None] + X[:, j][:, None, None] * zb[None, :, None]
    A = np.sum(np.logaddexp(0, eta + sd[:, None, None] * zo[None, None, :]) * odens[None, None, :], 2)
    ll = (y[:, None] * (offset[:, None] + X[:, j][:, None] * zb[None, :]) - A).sum(0)
    A0 = np.sum(np.logaddexp(0, offset[:, None] + sd[:, None] * zo[None, :]) * odens[None, :], 1)
    ll0 = (y * offset - A0).sum()
    return np.log(np.sum(np.exp(ll - ll0) * bpri) * dzb)


def test_quadrature_ser_offset_integration_matches_brute():
    rng = np.random.default_rng(3)
    n, p = 180, 4
    X = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.5
    ov = rng.uniform(0.1, 1.2, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.0 * X[:, 0]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    _, _, lbf, _ = quadrature_ser(
        op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=25,
        offset_var=jnp.asarray(ov), offset_integration=15,
    )
    brute = np.array([_brute_quad_logbf(X, y, offset, ov, j, 1.0) for j in range(p)])
    np.testing.assert_allclose(np.asarray(lbf), brute, atol=1e-4)


def test_offset_integration_changes_logbf():
    # sanity: with real offset variance the evidence is not the mean-only evidence
    op, y, offset = _data()
    ov = jnp.full(op.n, 0.8)
    _, _, lbf0, _ = quadrature_ser(op, y, offset, 1.0, order=15)
    _, _, lbf1, _ = quadrature_ser(op, y, offset, 1.0, order=15, offset_var=ov, offset_integration=11)
    assert np.max(np.abs(np.asarray(lbf1) - np.asarray(lbf0))) > 1e-3


def test_taylor2_close_to_gh_small_ov():
    op, y, offset = _data()
    ov = jnp.full(op.n, 0.3)
    _, _, lg, _ = quadrature_ser(op, y, offset, 1.0, order=15, offset_var=ov, offset_integration=15)
    _, _, lt, _ = quadrature_ser(op, y, offset, 1.0, order=15, offset_var=ov, offset_integration="taylor")
    # 2nd-order agrees to <1% rel; abs error grows with signal strength (summed
    # cumulant error), so allow ~0.15 nat on the strong feature.
    np.testing.assert_allclose(np.asarray(lt), np.asarray(lg), atol=0.15)


# --------------------------------------------------------------------------- #
# convexity / robustness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset_integration", ["taylor", 9])
def test_finite_under_saturation_with_offset_var(offset_integration):
    rng = np.random.default_rng(0)
    n, p = 400, 20
    X = rng.normal(size=(n, p))
    offset = np.full(n, 4.5) + rng.normal(size=n) * 0.2
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 3.0 * X[:, 0]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    ov = jnp.full(n, 1.0)
    for fn in (
        lambda: local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0, offset_var=ov, offset_integration=offset_integration),
        lambda: quadrature_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=11, offset_var=ov, offset_integration=offset_integration),
        lambda: profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=11, offset_var=ov, offset_integration=offset_integration),
    ):
        for a in fn():
            assert np.all(np.isfinite(np.asarray(a)))


# --------------------------------------------------------------------------- #
# engine wiring
# --------------------------------------------------------------------------- #
def test_engine_integrate_offset_runs_and_differs():
    import gibss.legacy.logistic_localtaylor as Q
    from gibss.engine import fit_ibss

    rng = np.random.default_rng(1)
    n, p = 500, 40
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 2.0 * X[:, 3] + 1.5 * X[:, 10]))), n).astype(float)
    data = Q.prep_data(X, y)

    def run(init, **fs):
        st = fit_ibss(
            data,
            init(data, L=3, family_state_kwargs={"estimate_prior_variance": False, **fs}),
            Q.default_schedule(), max_iter=30,
        )
        return st

    # message type drives it: MeanMessage init = fixed offset, Message init = integrated
    mean_only = run(Q.initialize_state_mean_message)
    integ = run(Q.initialize_state, offset_integration=7)
    integ_t2 = run(Q.initialize_state, offset_integration="taylor")

    # both recover the signals
    for st in (mean_only, integ, integ_t2):
        top = {int(np.argmax(e.alpha)) for e in st.single_effects}
        assert {3, 10} <= top
    # offset integration is not a no-op at L>1 (some alpha shifts)
    a0 = np.concatenate([np.asarray(e.alpha) for e in mean_only.single_effects])
    a1 = np.concatenate([np.asarray(e.alpha) for e in integ.single_effects])
    assert np.max(np.abs(a0 - a1)) > 1e-4
    # taylor2 and gh agree closely on the posterior
    at = np.concatenate([np.asarray(e.alpha) for e in integ_t2.single_effects])
    assert np.max(np.abs(a1 - at)) < 5e-2
