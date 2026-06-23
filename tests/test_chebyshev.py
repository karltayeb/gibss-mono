import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import gibss.chebyshev as cb


def _logistic_null(y, offset):
    """f(c), f'(c), f''(c) for the intercept-only logistic loglik."""
    y = np.asarray(y)
    offset = np.asarray(offset)

    def f(c):
        c = np.asarray(c)
        eta = offset[:, None] + c[None, :] if c.ndim else offset + c
        if c.ndim:
            return np.sum(y[:, None] * eta - np.logaddexp(0.0, eta), axis=0)
        return np.sum(y * eta - np.logaddexp(0.0, eta))

    def g(c):  # f'
        return np.sum(y - 1.0 / (1.0 + np.exp(-(offset + c))))

    def h(c):  # f''
        s = 1.0 / (1.0 + np.exp(-(offset + c)))
        return -np.sum(s * (1.0 - s))

    return f, g, h


def test_value_grad_hess_accuracy():
    rng = np.random.default_rng(0)
    n = 4000
    offset = rng.normal(size=n) * 0.7
    y = rng.binomial(1, 0.3, size=n).astype(float)
    f, g, h = _logistic_null(y, offset)

    # null intercept ~ logit(mean) region; center a few panels there
    origin = float(np.log(y.mean() / (1 - y.mean())))
    W = 1.0
    panels = cb.cheb_init(f, origin, W, N=12, K_max=8,
                          seed_points=np.array([origin - 1.5, origin + 1.5]))

    xs = np.linspace(origin - 1.4, origin + 1.4, 50)
    v = np.asarray(cb.cheb_val(panels, jnp.asarray(xs)))
    gg = np.asarray(cb.cheb_grad(panels, jnp.asarray(xs)))
    hh = np.asarray(cb.cheb_hess(panels, jnp.asarray(xs)))

    v_true = np.array([f(x) for x in xs])
    g_true = np.array([g(x) for x in xs])
    h_true = np.array([h(x) for x in xs])

    np.testing.assert_allclose(v, v_true, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(gg, g_true, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hh, h_true, rtol=1e-5, atol=1e-5)


def test_logistic_null_pointwise_convergence():
    # l_d (logistic null vs intercept) is smooth -> Chebyshev nails value+derivs at low N
    rng = np.random.default_rng(9)
    n = 6000
    offset = rng.normal(size=n) * 0.6
    y = rng.binomial(1, 0.35, size=n).astype(float)
    f, g, h = _logistic_null(y, offset)
    origin = float(np.log(y.mean() / (1 - y.mean())))
    # W scaled to the null SE, as in the real code (W = panel_width / sqrt(-l_d''))
    W = 2.0 / np.sqrt(-h(origin))
    cs = np.linspace(origin - 0.45 * W, origin + 0.45 * W, 200)  # inside one cell
    v_true = np.array([f(c) for c in cs])
    g_true = np.array([g(c) for c in cs])
    h_true = np.array([h(c) for c in cs])
    for N in [6, 8]:
        p = cb.cheb_init(f, origin, W, N, K_max=4)
        # relative error: l_d magnitude ~ O(n); the realistic narrow interval is easy
        assert np.max(np.abs(np.asarray(cb.cheb_val(p, jnp.asarray(cs))) - v_true)) < 1e-6
        assert np.max(np.abs(np.asarray(cb.cheb_grad(p, jnp.asarray(cs))) - g_true)) < 1e-6
        assert np.max(np.abs(np.asarray(cb.cheb_hess(p, jnp.asarray(cs))) - h_true)) < 1e-5


def test_clenshaw_matches_numpy():
    from numpy.polynomial import chebyshev as C
    rng = np.random.default_rng(1)
    coef = rng.normal(size=9)
    t = np.linspace(-1, 1, 20)
    got = np.asarray(cb._clenshaw(jnp.asarray(coef[None, :] * np.ones((20, 1))), jnp.asarray(t)))
    want = C.chebval(t, coef)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_ensure_extends_and_reuses():
    rng = np.random.default_rng(2)
    n = 2000
    offset = rng.normal(size=n) * 0.5
    y = rng.binomial(1, 0.5, size=n).astype(float)
    f, g, h = _logistic_null(y, offset)
    origin, W = 0.0, 1.0
    panels = cb.cheb_init(f, origin, W, N=10, K_max=8)
    assert int(panels.n_active) == 1

    # request a far point -> band extends
    panels2 = cb.cheb_ensure(panels, f, np.array([3.2]))
    assert int(panels2.n_active) >= 4
    # already covered -> identity (no rebuild)
    panels3 = cb.cheb_ensure(panels2, f, np.array([2.0]))
    assert panels3 is panels2

    # accuracy holds across the extended band (constant per-panel degree)
    xs = np.linspace(-0.4, 3.0, 40)
    v = np.asarray(cb.cheb_val(panels2, jnp.asarray(xs)))
    v_true = np.array([f(x) for x in xs])
    np.testing.assert_allclose(v, v_true, rtol=1e-6, atol=1e-5)


def test_miss_mask():
    rng = np.random.default_rng(3)
    f, _, _ = _logistic_null(rng.binomial(1, 0.4, 500).astype(float), rng.normal(size=500))
    panels = cb.cheb_init(f, 0.0, 1.0, N=8, K_max=8)  # single cell [-0.5, 0.5]
    x = jnp.asarray([0.0, 0.4, 0.6, -0.6, 2.0])
    miss = np.asarray(cb.cheb_miss(panels, x))
    np.testing.assert_array_equal(miss, [False, False, True, True, True])
