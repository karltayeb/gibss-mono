"""Sparse (BCOO) pre-centering for the Gaussian-VI kernel (`vi`: Gaussian q + a pointwise
offset scheme), via the 2-D background.

`glm_vi_ser` folds the effect's own variance into the scheme's per-entry variance `x^2 v +
ov`; under centering a zero design entry is the fill `-c_j`, which puts TWO per-feature
params off-support (the mean shift `c_j m` and the variance add `c_j^2 v`).
`glm_vi_center_ser` sums that all-rows off-support background with a 2-D Chebyshev
surrogate in `(c m, c^2 v)`, corrected on the O(nnz) support entries -- the exact centered
model, sparse. These pin it against the dense eagerly-centered `glm_vi_ser`, which fits the
identical `eta = offset + (x_ij - c_j) b`, for every pointwise scheme.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss.methods import fit_glm_susie
from gibss.operators import as_operator
from gibss.response import (
    GH,
    Bernoulli,
    JJEnvelope,
    MacKay,
    ProbitMixture,
    Smoothed,
    Taylor,
)
from gibss.response_ser import glm_vi_center_ser, glm_vi_ser

SCHEMES = {
    "gh": GH(15),
    "probit": ProbitMixture(5),
    "taylor": Taylor(),
    "jj_envelope": JJEnvelope(),
    "mackay": MacKay(),
}


@pytest.mark.parametrize("background", ["exact", "chebyshev"])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_kernel_matches_eager_centered_dense(scheme, background):
    rng = np.random.default_rng(0)
    n, p = 300, 15
    X = (rng.uniform(size=(n, p)) < 0.3).astype(float)
    c = X.mean(0)
    y = (rng.uniform(size=n) < 0.4).astype(float)
    offset = rng.standard_normal(n) * 0.3
    ov = np.abs(rng.standard_normal(n)) * 0.2  # per-row offset variance
    pv = 1.5
    resp = Smoothed(Bernoulli(), SCHEMES[scheme])
    aux = (jnp.asarray(y), jnp.asarray(ov))

    ref = glm_vi_ser(as_operator(jnp.asarray(X - c)), aux, jnp.asarray(offset), pv, resp)
    out = glm_vi_center_ser(
        as_operator(jsparse.BCOO.fromdense(jnp.asarray(X))), aux, jnp.asarray(offset),
        jnp.asarray(c), pv, resp, background=background,
    )
    tol = 1e-8 if background == "exact" else 1e-7
    for a, b in zip(out, ref):
        np.testing.assert_allclose(a, b, atol=tol)


def test_surrogate_rounds_match_exact_when_iterates_leave_the_first_box():
    # dense-ish columns with a strong effect push a = c m past the first box (fit at m = 0),
    # so the per-round refit runs; the answer must still match the exact (n, p) background
    # and the dense kernel
    rng = np.random.default_rng(5)
    n, p = 400, 8
    X = (rng.uniform(size=(n, p)) < 0.5).astype(float)
    c = X.mean(0)
    eta = -1.0 + 4.0 * (X[:, 2] - c[2]) - 3.0 * (X[:, 5] - c[5])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    offset = rng.standard_normal(n) * 0.2
    pv = 4.0
    resp = Smoothed(Bernoulli(), GH(15))
    args = (
        as_operator(jsparse.BCOO.fromdense(jnp.asarray(X))), (jnp.asarray(y), 0.0),
        jnp.asarray(offset), jnp.asarray(c), pv, resp,
    )
    out_c = glm_vi_center_ser(*args, background="chebyshev")
    out_x = glm_vi_center_ser(*args, background="exact")
    ref = glm_vi_ser(
        as_operator(jnp.asarray(X - c)), (jnp.asarray(y), 0.0), jnp.asarray(offset), pv, resp
    )
    assert np.max(np.abs(np.asarray(c * out_c[0]))) > 0.5  # left the m = 0 box
    for a, b in zip(out_c, out_x):
        np.testing.assert_allclose(a, b, atol=1e-7)
    for a, b in zip(out_c, ref):
        np.testing.assert_allclose(a, b, atol=1e-7)


@pytest.mark.parametrize("integ", ["gh", "probit_mixture", "mackay"])
def test_full_fit_bcoo_center_matches_dense(integ):
    # through the front door: BCOO + gaussian q + a pointwise scheme now centers by default
    # (it used to fall back to uncentered, and raise on an explicit center=True) and equals
    # the dense eagerly-centered fit
    rng = np.random.default_rng(1)
    n, p, causal = 400, 25, 6
    Xd = (rng.uniform(size=(n, p)) < 0.25).astype(float)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.4 + 2.0 * Xd[:, causal])))).astype(float)
    Xs = jsparse.BCOO.fromdense(Xd)
    kw = {"L": 3, "variational_family": "gaussian", "offset_integration": integ, "max_iter": 40}
    sd = fit_glm_susie(Xd, y, **kw)
    ss = fit_glm_susie(Xs, y, **kw)
    assert sd.family_state.kernel == "vi" and ss.family_state.kernel == "vi"
    assert causal in ss.get_credible_sets()[0]
    np.testing.assert_allclose(np.asarray(sd.alpha), np.asarray(ss.alpha), atol=1e-7)
    np.testing.assert_allclose(np.asarray(sd.ser_log_bf), np.asarray(ss.ser_log_bf), atol=1e-7)
    assert float(sd.family_state.intercept_value) == pytest.approx(
        float(ss.family_state.intercept_value), abs=1e-7
    )
    # and it IS the centered model: the uncentered sparse fit differs
    su = fit_glm_susie(Xs, y, center=False, **kw)
    assert np.max(np.abs(np.asarray(su.ser_log_bf) - np.asarray(ss.ser_log_bf))) > 1e-3
