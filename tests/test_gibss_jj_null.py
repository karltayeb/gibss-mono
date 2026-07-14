import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import gibss.legacy.globaljj as gj
import gibss.legacy.localjj as lj
import gibss.legacy.logistic_localtaylor as q
from gibss._jj import jj_null_log_likelihood
from gibss.engine import fit_ibss


def test_jj_null_tuned_is_tight_vs_exact():
    # at offset_var=0 the null-tuned JJ null is the tight bound ~ exact logistic null
    rng = np.random.default_rng(0)
    n = 800
    offset = rng.normal(size=n) * 0.6
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-offset))).astype(float)
    jj = float(jj_null_log_likelihood(jnp.asarray(y), jnp.asarray(offset)))
    exact = float(np.sum(y * offset - np.logaddexp(0.0, offset)))
    # tight: within a few nats on n=800 (JJ null bound at xi=|eta| is a lower bound)
    assert jj <= exact + 1e-9
    assert exact - jj < 0.05 * n  # not loose


def test_globaljj_and_localjj_null_coincide():
    # the BF denominator is method-independent: same offset/offset_var -> same null
    rng = np.random.default_rng(1)
    n = 500
    offset = rng.normal(size=n) * 0.5
    offset_var = np.abs(rng.normal(size=n)) * 0.2
    y = rng.binomial(1, 0.4, size=n).astype(float)
    g = float(jj_null_log_likelihood(jnp.asarray(y), jnp.asarray(offset), jnp.asarray(offset_var)))
    # localjj scores its null the same way (xi_null = sqrt(offset^2+offset_var))
    xi_null = np.sqrt(offset**2 + offset_var)
    l = float(lj._jj_bound_null_log_likelihood(
        jnp.asarray(y), jnp.asarray(offset), jnp.asarray(xi_null), jnp.asarray(offset_var)))
    np.testing.assert_allclose(g, l, rtol=1e-12)


def test_globaljj_bf_not_inflated():
    # after the null fix, globaljj BF is a valid lower bound (<= exact) and ~ localjj
    rng = np.random.default_rng(2)
    n, p = 1000, 50
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.5 + 1.5 * X[:, 7])))).astype(float)
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    def bf(mod, **kw):
        data = mod.prep_data(Xj, yj)
        st = fit_ibss(data, mod.initialize_state(data, L=1, **kw), mod.default_schedule(), max_iter=50)
        return float(np.asarray(st.ser_log_bayes_factor)[0])

    exact = bf(q, quadrature_order=15)
    g = bf(gj)
    loc = bf(lj)
    assert g <= exact + 0.5  # variational <= exact (no longer inflated)
    np.testing.assert_allclose(g, exact, atol=1.0)
    np.testing.assert_allclose(g, loc, atol=1.0)
