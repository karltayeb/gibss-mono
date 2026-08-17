"""CAVI in Q2 via the quadrature-peel builder (method='compress_cavi' /
offset_integration='compress').

Compress and CharFnOffset are two builders of the SAME offset-integrated cumulant table,
consumed by the SAME kernel (glm_vi_gh_ser). So a fit with 'compress_cavi' must match one
with 'cf_cavi' to their combined quadrature tolerance -- the definitive check that the
ported builder is wired correctly and targets the same Q2 CAVI fixed point. Covers dense
+ sparse (BCOO) agreement, that the front door resolves to the vi_gh kernel with a
Compress smoother, and the gaussian-only guard.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.response import Compress, Smoothed
from gibss.methods import fit_glm_susie


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_data(rng, n, p, signal_idx, signal_val):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(signal_idx, signal_val):
        beta[i] = v
    y = (rng.uniform(size=n) < _sigmoid(X @ beta - 0.4)).astype(float)
    return X, y


def _flogmarg(st):
    return np.stack([np.asarray(e.feature_log_marginal) for e in st.single_effects])


def _rel_flogmarg_gap(a, b):
    """Max |Δ feature_log_marginal| relative to its scale. The log-evidence grows with n
    (it is a sum over observations), so a RELATIVE bound is the meaningful builder-
    agreement check; at the default degree M=48 the two builders agree to ~1e-6 here."""
    fa, fb = _flogmarg(a), _flogmarg(b)
    return np.max(np.abs(fa - fb)) / max(np.max(np.abs(fa)), 1.0)


def _fit(X, y, method, **kw):
    return fit_glm_susie(
        X, y, L=4, method=method, estimate_prior_variance=False,
        prior_variance=1.0, max_iter=50, tol=1e-7, **kw
    )


def test_compress_cavi_resolves_to_vi_gh():
    rng = np.random.default_rng(0)
    X, y = _logit_data(rng, n=200, p=20, signal_idx=[3], signal_val=[2.0])
    st = _fit(X, y, "compress_cavi")
    fs = st.family_state
    assert fs.kernel == "vi_gh"
    assert isinstance(fs.response, Smoothed)
    assert isinstance(fs.response.smoother, Compress)


def test_compress_matches_cf_dense():
    rng = np.random.default_rng(1)
    X, y = _logit_data(rng, n=350, p=40, signal_idx=[5, 22], signal_val=[2.5, -2.2])
    a = _fit(X, y, "cf_cavi")
    b = _fit(X, y, "compress_cavi")
    # feature_log_marginal is the fit-equivalence quantity (per [[compare-log-marginal]])
    assert _rel_flogmarg_gap(a, b) < 1e-4
    assert np.max(np.abs(np.asarray(a.alpha) - np.asarray(b.alpha))) < 1e-3
    assert np.corrcoef(np.asarray(a.pip), np.asarray(b.pip))[0, 1] > 0.9999


def test_compress_matches_cf_sparse():
    rng = np.random.default_rng(2)
    n, p = 500, 60
    Xd = (rng.random((n, p)) < 0.15).astype(float)
    # a real logistic signal on two sparse columns
    beta = np.zeros(p); beta[[7, 31]] = [1.8, -1.6]
    y = (rng.uniform(size=n) < _sigmoid(Xd @ beta - 0.8)).astype(float)
    X = sparse.BCOO.fromdense(jnp.asarray(Xd))
    a = _fit(X, y, "cf_cavi")
    b = _fit(X, y, "compress_cavi")
    assert _rel_flogmarg_gap(a, b) < 1e-4
    assert np.corrcoef(np.asarray(a.pip), np.asarray(b.pip))[0, 1] > 0.9999


def test_cf_requires_gaussian_vfam():
    # cf is Q2-only (no closed-form CF for a free-form Q1 posterior); it must reject
    # variational_family='unconstrained'. (compress, by contrast, is valid for BOTH
    # Q1/unconstrained -> quad and Q2/gaussian -> vi_gh; see test_compress_both_families.)
    rng = np.random.default_rng(3)
    X, y = _logit_data(rng, n=120, p=12, signal_idx=[2], signal_val=[1.5])
    with pytest.raises(ValueError, match="variational_family="):
        fit_glm_susie(X, y, L=2, offset_integration="cf",
                      variational_family="unconstrained")


def test_compress_both_families():
    # compress resolves to vi_gh (Q2) with gaussian q and to quad (Q1) with unconstrained.
    rng = np.random.default_rng(4)
    X, y = _logit_data(rng, n=150, p=15, signal_idx=[3], signal_val=[1.8])
    q2 = fit_glm_susie(X, y, L=2, offset_integration="compress",
                       variational_family="gaussian", max_iter=20)
    q1 = fit_glm_susie(X, y, L=2, offset_integration="compress",
                       variational_family="unconstrained", max_iter=20)
    assert q2.family_state.kernel == "vi_gh"
    assert q1.family_state.kernel == "quad"
