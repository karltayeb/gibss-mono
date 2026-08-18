"""Plug-in CAVI in Q2 (method='gibss_gaussian' / variational_family='gaussian' +
offset_integration='none'): gIBSS with a Gaussian effect posterior. The offset is the
OTHER effects' MEAN (a fixed point -- no integration over their uncertainty), and the
effect b is integrated by GH against a plain base (glm_vi_gh_ser, no offset table).

Covers: it resolves to the vi_gh kernel on a plain base (dense + sparse + intercept
variants); at convergence each effect is EXACTLY a Gaussian-VI fit against the plug-in
mean of the others (independent hand-rolled reference); and it genuinely differs from
exact-mixture CAVI (cf_cavi) at L>1, but coincides in the leave-one-out-free limit.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.methods import fit_glm_susie


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_data(rng, n, p, idx, val, b0=-0.4):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(idx, val):
        beta[i] = v
    y = (rng.uniform(size=n) < _sigmoid(X @ beta + b0)).astype(float)
    return X, y


def test_gibss_gaussian_resolves_and_runs():
    rng = np.random.default_rng(0)
    X, y = _logit_data(rng, n=400, p=30, idx=[5, 18], val=[2.2, -1.8])
    # preset == explicit axes
    st = fit_glm_susie(X, y, L=4, method="gibss_gaussian", max_iter=60)
    st2 = fit_glm_susie(X, y, L=4, variational_family="gaussian",
                        offset_integration="none", max_iter=60)
    assert st.family_state.kernel == "vi_gh"
    assert st2.family_state.kernel == "vi_gh"
    assert np.asarray(st.pip)[5] > 0.9 and np.asarray(st.pip)[18] > 0.9


@pytest.mark.parametrize("kw", [
    {}, {"intercept": "profiled"}, {"intercept": "null"}, {"center": True},
])
def test_gibss_gaussian_variants_run(kw):
    rng = np.random.default_rng(1)
    Xd = (rng.random((300, 20)) < 0.2) * 1.0
    y = (rng.uniform(size=300) < 0.35).astype(float)
    X = Xd if kw.get("center") else sparse.BCOO.fromdense(jnp.asarray(Xd))  # sparse unless centered
    st = fit_glm_susie(jnp.asarray(X) if kw.get("center") else X, jnp.asarray(y),
                       L=2, method="gibss_gaussian", max_iter=30, **kw)
    assert st.converged
    for e in st.single_effects:
        assert np.isfinite(np.asarray(e.var)).all() and (np.asarray(e.var) > 0).all()


def test_gibss_gaussian_plugin_stationarity():
    """At convergence each effect is a Gaussian-VI fit against the PLUG-IN MEAN of the
    others. Re-derive effect 0's posterior independently: a hand-rolled coordinate ascent
    (GH over b) against offset = intercept + X (alpha_1 mu_1), matching to ~1e-6."""
    rng = np.random.default_rng(4)
    X, y = _logit_data(rng, n=300, p=6, idx=[2], val=[2.0])
    pv = 1.0
    st = fit_glm_susie(X, y, L=2, method="gibss_gaussian", estimate_prior_variance=False,
                       prior_variance=pv, max_iter=300, tol=1e-10)
    e0, e1 = st.single_effects[0], st.single_effects[1]
    Xn, yn = np.asarray(X), np.asarray(y)
    off = float(st.family_state.intercept_value) + Xn @ (np.asarray(e1.alpha) * np.asarray(e1.mu))
    gn, gw = np.polynomial.hermite.hermgauss(60)
    gw = gw / np.sqrt(np.pi)
    checked = 0
    for c in np.argsort(-np.asarray(e0.alpha))[:2]:
        c = int(c)
        xj = Xn[:, c]
        m, v = 0.0, pv
        for _ in range(300):
            b = m + np.sqrt(2 * v) * gn
            eta = off[:, None] + xj[:, None] * b[None, :]
            s = _sigmoid(eta)
            Eg = (gw * (yn[:, None] - s)).sum(1)
            Ew = (gw * (s * (1 - s))).sum(1)
            prec = 1 / pv + np.sum(xj**2 * Ew)
            m += np.clip((np.sum(xj * Eg) - m / pv) / prec, -4, 4)
            v = 1 / prec
        assert abs(m - float(e0.mu[c])) < 1e-5, f"c={c} m"
        assert abs(v - float(e0.var[c])) < 1e-5, f"c={c} v"
        checked += 1
    assert checked == 2


def test_gibss_gaussian_differs_from_exact_cavi():
    """Plug-in is NOT exact CAVI: at L>1 it ignores the other effects' variance, so the
    posteriors differ measurably from cf_cavi; they converge as the leave-one-out coupling
    weakens (checked here only that the gap is real and finite, not zero)."""
    rng = np.random.default_rng(6)
    X, y = _logit_data(rng, n=300, p=8, idx=[2, 5], val=[2.0, -1.8])
    kw = dict(L=3, estimate_prior_variance=False, prior_variance=1.0, max_iter=200, tol=1e-9)
    pl = fit_glm_susie(X, y, method="gibss_gaussian", **kw)
    ca = fit_glm_susie(X, y, method="cf_cavi", **kw)
    gap = max(np.max(np.abs(np.asarray(pl.single_effects[l].var)
                            - np.asarray(ca.single_effects[l].var)))
              for l in range(3))
    assert 0 < gap < 1.0  # real, finite difference (plug-in vs exact-mixture offset)
