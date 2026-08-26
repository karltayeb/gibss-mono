"""Sparse (BCOO) pre-centering for the conjugate JJ kernel (localjj), via the 2-D background.

Centering makes a zero design entry a point mass at the fill `-c_j`, and the JJ per-entry tilt
`xi^2 = (offset - c_j m)^2 + c_j^2 v + ov` puts TWO per-feature params off-support (the mean
shift `c_j m` and the variance `c_j^2 v`). `glm_jj_center_ser` sums that all-rows off-support
background with a 2-D Chebyshev surrogate in `(c m, c^2 v)`, corrected on the O(nnz) support
entries -- the exact centered model, sparse. These pin it against the dense eagerly-centered
fit, which fits the identical `eta = offset + (x_ij - c_j) b`.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss.methods import fit_glm_susie
from gibss.glm import prep_data
from gibss.elbo import compute_elbo_jj
from gibss.operators import as_operator
from gibss.response import Bernoulli, JJFixed, Smoothed
from gibss.response_ser import glm_jj_ser, glm_jj_center_ser


@pytest.mark.parametrize("background", ["exact", "chebyshev"])
def test_kernel_matches_eager_centered_dense(background):
    # glm_jj_center_ser on a BCOO design + column means c must equal glm_jj_ser on the
    # eagerly-centered dense design X - c (both fit eta = offset + (x-c) b).
    rng = np.random.default_rng(0)
    n, p = 300, 15
    X = (rng.uniform(size=(n, p)) < 0.3).astype(float)
    c = X.mean(0)
    y = (rng.uniform(size=n) < 0.4).astype(float)
    offset = rng.standard_normal(n) * 0.3
    pv = 1.5
    resp = Smoothed(Bernoulli(), JJFixed())

    op_d = as_operator(jnp.asarray(X - c))
    m_ref, v_ref, lbf_ref, kl_ref = glm_jj_ser(
        op_d, (jnp.asarray(y), 0.0), jnp.asarray(offset), pv, resp
    )
    op_s = as_operator(jsparse.BCOO.fromdense(jnp.asarray(X)))
    m, v, lbf, kl = glm_jj_center_ser(
        op_s, (jnp.asarray(y), 0.0), jnp.asarray(offset), jnp.asarray(c), pv, resp,
        background=background,
    )
    np.testing.assert_allclose(m, m_ref, atol=1e-9)
    np.testing.assert_allclose(v, v_ref, atol=1e-9)
    np.testing.assert_allclose(lbf, lbf_ref, atol=1e-8)
    np.testing.assert_allclose(kl, kl_ref, atol=1e-9)


def test_kernel_handles_offset_variance():
    # the tilt folds a per-row offset variance ov; centered sparse must still match dense.
    rng = np.random.default_rng(2)
    n, p = 250, 12
    X = (rng.uniform(size=(n, p)) < 0.35).astype(float)
    c = X.mean(0)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    offset = rng.standard_normal(n) * 0.2
    ov = np.abs(rng.standard_normal(n)) * 0.1  # per-row offset variance
    resp = Smoothed(Bernoulli(), JJFixed())
    op_d = as_operator(jnp.asarray(X - c))
    ref = glm_jj_ser(op_d, (jnp.asarray(y), jnp.asarray(ov)), jnp.asarray(offset), 1.0, resp)
    op_s = as_operator(jsparse.BCOO.fromdense(jnp.asarray(X)))
    out = glm_jj_center_ser(
        op_s, (jnp.asarray(y), jnp.asarray(ov)), jnp.asarray(offset), jnp.asarray(c), 1.0, resp,
    )
    for a, b in zip(out[:3], ref[:3]):
        np.testing.assert_allclose(a, b, atol=1e-8)


def test_full_fit_bcoo_center_matches_dense():
    # end-to-end through the front door: BCOO + center=True + localjj (2-D background) equals
    # the dense eagerly-centered fit -- same PIPs, evidence, intercept, and JJ ELBO.
    rng = np.random.default_rng(1)
    n, p, causal = 400, 25, 6
    Xd = (rng.uniform(size=(n, p)) < 0.25).astype(float)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.4 + 2.0 * Xd[:, causal])))).astype(float)
    Xs = jsparse.BCOO.fromdense(Xd)
    sd = fit_glm_susie(Xd, y, L=3, method="localjj", center=True, max_iter=60)
    ss = fit_glm_susie(Xs, y, L=3, method="localjj", center=True, max_iter=60)
    assert causal in ss.get_credible_sets()[0]
    np.testing.assert_allclose(np.asarray(sd.alpha), np.asarray(ss.alpha), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(sd.ser_log_bf), np.asarray(ss.ser_log_bf), atol=1e-9
    )
    assert float(sd.family_state.intercept_value) == pytest.approx(
        float(ss.family_state.intercept_value), abs=1e-7
    )
    ej_d = compute_elbo_jj(prep_data(Xd, y, center=True), sd)
    ej_s = compute_elbo_jj(prep_data(Xs, y, center=True), ss)
    assert ej_d == pytest.approx(ej_s, abs=1e-8)


def test_front_door_no_longer_raises():
    # the combination that used to raise now fits.
    rng = np.random.default_rng(3)
    n, p = 200, 15
    Xs = jsparse.BCOO.fromdense((rng.uniform(size=(n, p)) < 0.3).astype(float))
    y = (rng.uniform(size=n) < 0.5).astype(float)
    st = fit_glm_susie(Xs, y, L=2, method="localjj", center=True, max_iter=20)
    assert st.family_state.kernel == "jj"
    assert np.all(np.isfinite(np.asarray(st.alpha)))
