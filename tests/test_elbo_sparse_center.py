"""The ELBO evaluators are layout independent on a pre-centered design.

A centered BCOO design keeps its column means in `data.column_center` and the off-support
fill is -c_j; a dense design is centered eagerly. Scoring the SAME state on both layouts
must agree. The Poisson path once called the sparse log-kappa builders without the column
means, scoring off-support entries at 0, a near-constant 0.3-0.5 nat gap per fit on 025.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from gibss import glm
from gibss.elbo import compute_elbo, compute_elbo_gaussian
from gibss.methods import fit_glm_susie

jax.config.update("jax_enable_x64", True)


def _design(family):
    rng = np.random.default_rng(5)
    n, p = 400, 24
    Xd = (rng.uniform(size=(n, p)) < 0.1).astype(float)
    eta = -1.0 + 1.0 * Xd[:, 3] - 0.8 * Xd[:, 17]
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return Xd, BCOO.fromdense(jnp.asarray(Xd)), y


@pytest.mark.parametrize("family", ["poisson", "logistic"])
@pytest.mark.parametrize(
    "kw",
    [
        dict(variational_family="gaussian", offset_integration="none"),  # gIBSS in Q2
        dict(method="cf_cavi"),  # exact CAVI in Q2
    ],
)
def test_q2_elbo_is_layout_independent(family, kw):
    Xd, Xs, y = _design(family)
    st = fit_glm_susie(Xs, y, L=2, family=family, center=True, **kw)
    dd, ds = glm.prep_data(Xd, y, center=True), glm.prep_data(Xs, y, center=True)
    for score in (
        lambda d: compute_elbo_gaussian(d, st, score_intercept="shared"),
        lambda d: compute_elbo(d, st, order=16, M=64),
    ):
        a, b = float(score(dd)), float(score(ds))
        assert abs(a - b) < 1e-6, (a, b)


@pytest.mark.parametrize("family", ["poisson", "logistic"])
def test_q1_elbo_is_layout_independent(family):
    Xd, Xs, y = _design(family)
    st = fit_glm_susie(Xs, y, L=2, family=family, center=True, offset_integration="gh")
    dd, ds = glm.prep_data(Xd, y, center=True), glm.prep_data(Xs, y, center=True)
    a = float(compute_elbo(dd, st, order=16, M=64))
    b = float(compute_elbo(ds, st, order=16, M=64))
    assert abs(a - b) < 1e-6, (a, b)
