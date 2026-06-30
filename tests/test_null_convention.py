"""All logistic SER families must score the null (b=0) at its OWN profiled
intercept, not the shared full-model intercept -- otherwise the BF inflates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

jax.config.update("jax_enable_x64", True)

import gibss.globaljj as G
import gibss.localjj as L
import gibss.logistic_profile as P
import gibss.logistic_quadrature as Q
from gibss.engine import fit_ibss


def _data(seed=1, n=1000, p=50):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 2.0 * X[:, 3]))), size=n).astype(float)
    return jnp.asarray(X), jnp.asarray(y)


def _profiled_null(y):
    y = np.asarray(y)
    return -minimize_scalar(
        lambda b0: -np.sum(y * b0 - np.logaddexp(0, b0)),
        bounds=(-10, 10), method="bounded",
    ).fun


def _fit(mod, X, y, **kw):
    data = mod.prep_data(X, y)
    st = fit_ibss(
        data,
        mod.initialize_state(data, L=1, family_state_kwargs={"estimate_prior_variance": False, **kw}),
        mod.default_schedule(), max_iter=40,
    )
    e = st.single_effects[0]
    return float(e.null_log_likelihood), float(np.asarray(st.ser_log_bayes_factor)[0])


@pytest.mark.parametrize(
    "mod,kw",
    [(Q, {}), (G, {}), (G, {"center": True}), (L, {}), (L, {"center": True}),
     (P, {"background_mode": "exact"})],
)
def test_null_is_profiled(mod, kw):
    X, y = _data()
    true_null = _profiled_null(y)
    null_ll, _ = _fit(mod, X, y, **kw)
    # JJ nulls are a (tight, null-tuned) lower bound -> allow a small slack below
    assert null_ll <= true_null + 1e-6
    assert null_ll >= true_null - 1.0, f"null {null_ll:.2f} not re-profiled (true {true_null:.2f})"


def test_ser_logbf_on_same_scale_as_profile():
    # after the null fix every method's BF is <= the exact profiled BF (+slack)
    # and clusters near it (no ~3-nat shared-intercept inflation).
    X, y = _data()
    _, ref = _fit(P, X, y, background_mode="exact")
    for mod, kw in [(Q, {}), (G, {}), (L, {})]:
        _, bf = _fit(mod, X, y, **kw)
        assert bf <= ref + 0.5, f"{mod.__name__} BF {bf:.2f} inflated vs profile {ref:.2f}"
        assert abs(bf - ref) < 2.0
