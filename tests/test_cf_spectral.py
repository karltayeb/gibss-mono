"""Spectral CAVI-in-Q2 effect update (`cf_spectral`): collapsed on two-level designs,
per entry on general dense designs.

Ground truth is independent of every CF code path: the leave-one-out offset is a product
of small Gaussian mixtures, so enumerating its components and adding the effect's own
Gaussian gives each row's shift law as an explicit Gaussian mixture, and a high-order GH
rule integrates the logistic cumulant against each component. The kernel's objective,
its (m, v) derivatives, its fixed point and the null must all match that enumeration.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss.cf_offset import CharFnOffset, _effect_cf
from gibss.cf_spectral import (
    CharFnSpectral,
    ChebDesign,
    DenseDesign,
    _prepare,
    _spectral_terms,
    cheb_design,
    cheb_error,
    cheb_factor,
    cheb_factor_tail,
    dense_design,
    glm_spectral_ser,
    spectral_null_loglik,
    two_level_design,
)
from gibss.methods import fit_glm_susie
from gibss.response import Bernoulli, Binomial, Smoothed
from gibss.response_ser import _gh_rule

GH_NODES, GH_LOGW = _gh_rule(120)
GH_W = np.exp(GH_LOGW) / np.sqrt(np.pi)
OV = 0.1  # homogeneous zero-mean Gaussian offset (the shared intercept's variance)


def _gauss_softplus(mean, var):
    """E softplus(N(mean, var)) by 120-node GH (exact to ~1e-13 for these scales)."""
    z = mean[..., None] + np.sqrt(2.0 * var)[..., None] * GH_NODES
    return np.logaddexp(0.0, z) @ GH_W


def _problem(rng, layout, n=14, p=4, L=2):
    B = (rng.uniform(size=(n, p)) < 0.4).astype(float)
    B[0, :] = 1.0  # no empty column
    if layout in ("dense_continuous", "dense_cheb"):  # every entry distinct: no collapse
        B = rng.normal(size=(n, p)) * np.exp(rng.normal(size=p) * 0.5)
    c = B.mean(0)
    others = []
    for _ in range(L):
        a = np.exp(rng.normal(size=p))
        others.append((a / a.sum(), rng.normal(size=p) * 1.2, np.exp(rng.normal(size=p)) * 0.3))
    centered = layout.endswith("centered")
    Xt = B - c if centered else B  # the design as the model sees it
    if layout.startswith("bcoo"):
        X = jsparse.BCOO.fromdense(jnp.asarray(B))
        colmean = jnp.asarray(c) if centered else None
    else:
        X, colmean = jnp.asarray(Xt), None
    eta = rng.normal(size=n) * 1.5 - 0.5
    return X, colmean, Xt, others, eta


def _brute(Xt, others, eta, s, m, v):
    """sum_i s_i E A(eta_i - mean_i + o'_i + xt_ij b) per column j, b ~ N(m_j, v_j), with
    o' the enumerated product mixture (+ N(0, OV)); eta carries the offset mean."""
    n, p = Xt.shape
    mean = sum(Xt @ (a * mu) for a, mu, _ in others)
    out = np.zeros(p)
    for combo in itertools.product(*[range(p) for _ in others]):
        w, cm, cv = 1.0, np.zeros(n), np.full(n, OV)
        for (a, mu, var), cidx in zip(others, combo):
            w *= a[cidx]
            cm = cm + Xt[:, cidx] * mu[cidx]
            cv = cv + Xt[:, cidx] ** 2 * var[cidx]
        mu_tot = (eta - mean + cm)[:, None] + Xt * m[None, :]
        var_tot = cv[:, None] + Xt**2 * v[None, :]
        out += w * (s[:, None] * _gauss_softplus(mu_tot, var_tot)).sum(0)
    return out


def _setup(layout, family, seed=0):
    rng = np.random.default_rng(seed)
    X, colmean, Xt, others, eta = _problem(rng, layout)
    n, p = Xt.shape
    if family == "binomial":
        ntr = rng.integers(1, 5, size=n).astype(float)
        y = np.minimum(rng.poisson(1.0, size=n), ntr).astype(float)
        base, yleaf, s = Binomial(), (jnp.asarray(y), jnp.asarray(ntr)), ntr
    else:
        y = (rng.uniform(size=n) < 0.4).astype(float)
        base, yleaf, s = Bernoulli(), jnp.asarray(y), np.ones(n)
    sm = CharFnSpectral()
    design = sm.design(X, colmean)
    if layout == "dense_cheb":  # the Chebyshev-in-x collapse, converged
        design = cheb_design(design, 64)
    law = (jnp.full(p, 3.0), jnp.full(p, 2.0))  # generous reach for the grid
    aux = sm.build_spectral_aux(
        base, yleaf, X, [tuple(jnp.asarray(t) for t in e) for e in others],
        jnp.asarray(eta), design, law, offset_var=OV, colmean=colmean,
    )
    return {"Xt": Xt, "others": others, "eta": eta, "y": y, "s": s, "base": base, "design": design, "aux": aux}


LAYOUTS = ["dense01", "dense_centered", "bcoo", "bcoo_centered", "dense_continuous",
           "dense_cheb"]


@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_terms_match_enumerated_mixture(layout, family):
    P = _setup(layout, family)
    rng = np.random.default_rng(1)
    p = P["Xt"].shape[1]
    m, v = rng.normal(size=p) * 1.5, np.exp(rng.normal(size=p)) * 0.5
    pre, _ = _prepare(P["design"], P["aux"], jnp.asarray(P["eta"]), P["base"])
    T = np.asarray(_spectral_terms(P["design"], P["aux"], pre, jnp.asarray(m), jnp.asarray(v)))
    F = lambda mm, vv: _brute(P["Xt"], P["others"], P["eta"], P["s"], mm, vv)
    base0 = F(np.zeros(p), np.zeros(p))
    np.testing.assert_allclose(T[0], F(m, v) - base0, atol=1e-9)
    h = 1e-4
    d_m = (F(m + h, v) - F(m - h, v)) / (2 * h)
    d_v = (F(m, v + h) - F(m, v - h)) / (2 * h)
    d_mv = (F(m + h, v + h) - F(m + h, v - h) - F(m - h, v + h) + F(m - h, v - h)) / (4 * h * h)
    d_vv = (F(m, v + h) - 2 * F(m, v) + F(m, v - h)) / (h * h)
    np.testing.assert_allclose(T[1], d_m, atol=1e-7)
    np.testing.assert_allclose(T[2], d_v, atol=1e-7)
    np.testing.assert_allclose(T[3], d_mv, atol=1e-5)
    np.testing.assert_allclose(T[4], d_vv, atol=1e-5)


@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_is_the_q2_cavi_optimum(layout, family):
    """The solved (m, v) is stationary for the brute-force ELBO, and elbo_rel equals it."""
    P = _setup(layout, family, seed=2)
    pv = 1.7
    m, v, elbo_rel, _kl = glm_spectral_ser(P["design"], P["aux"], jnp.asarray(P["eta"]), pv,
                                          P["base"])
    m, v = np.asarray(m), np.asarray(v)
    ytil = P["y"] @ P["Xt"]
    p = m.shape[0]
    base0 = _brute(P["Xt"], P["others"], P["eta"], P["s"], np.zeros(p), np.zeros(p))

    def elbo(mm, vv):
        kl_ = 0.5 * (np.log(pv / vv) + (vv + mm**2) / pv - 1.0)
        return mm * ytil - (_brute(P["Xt"], P["others"], P["eta"], P["s"], mm, vv) - base0) - kl_

    np.testing.assert_allclose(np.asarray(elbo_rel), elbo(m, v), atol=1e-9)
    h = 1e-5
    np.testing.assert_allclose((elbo(m + h, v) - elbo(m - h, v)) / (2 * h), 0.0, atol=1e-6)
    np.testing.assert_allclose((elbo(m, v + h) - elbo(m, v - h)) / (2 * h), 0.0, atol=1e-6)


@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_null_matches_enumerated_mixture(layout, family):
    P = _setup(layout, family, seed=3)
    p = P["Xt"].shape[1]
    ea = _brute(P["Xt"], P["others"], P["eta"], P["s"], np.zeros(p), np.zeros(p))[0]  # b = 0
    want = P["y"] @ P["eta"] - ea
    got = float(spectral_null_loglik(P["aux"], jnp.asarray(P["eta"]), P["base"]))
    assert abs(got - want) < 1e-9


def _brute_gaussian(Xt, others, eta, s, m, v):
    """The Gaussian-offset counterpart of `_brute`: each row's offset is N(0, V_i), V_i the
    product mixture's variance (+ OV)."""
    V = np.full(Xt.shape[0], OV)
    for a, mu, var in others:
        mean = Xt @ (a * mu)
        V = V + (Xt**2) @ (a * (mu**2 + var)) - mean**2
    mu_tot = eta[:, None] + Xt * m[None, :]
    return (s[:, None] * _gauss_softplus(mu_tot, V[:, None] + Xt**2 * v[None, :])).sum(0)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_gaussian_offset_law_matches_moment_matched_gaussian(layout):
    """offset_law="gaussian": the kernel optimizes the ELBO with each row's offset replaced
    by its moment-matched Gaussian, and its null is that Gaussian's."""
    rng = np.random.default_rng(8)
    X, colmean, Xt, others, eta = _problem(rng, layout)
    n, p = Xt.shape
    y = (rng.uniform(size=n) < 0.4).astype(float)
    sm = CharFnSpectral(offset_law="gaussian")
    design = sm.design(X, colmean)
    if layout == "dense_cheb":
        design = cheb_design(design, 64)
    aux = sm.build_spectral_aux(
        Bernoulli(), jnp.asarray(y), X, [tuple(jnp.asarray(t) for t in e) for e in others],
        jnp.asarray(eta), design, (jnp.full(p, 3.0), jnp.full(p, 2.0)), offset_var=OV,
        colmean=colmean,
    )
    pv = 1.7
    m, v, elbo_rel, _ = glm_spectral_ser(design, aux, jnp.asarray(eta), pv)
    m, v = np.asarray(m), np.asarray(v)
    s = np.ones(n)
    F = lambda mm, vv: _brute_gaussian(Xt, others, eta, s, mm, vv)
    base0 = F(np.zeros(p), np.zeros(p))

    def elbo(mm, vv):
        kl = 0.5 * (np.log(pv / vv) + (vv + mm**2) / pv - 1.0)
        return mm * (y @ Xt) - (F(mm, vv) - base0) - kl

    np.testing.assert_allclose(np.asarray(elbo_rel), elbo(m, v), atol=1e-9)
    h = 1e-5
    np.testing.assert_allclose((elbo(m + h, v) - elbo(m - h, v)) / (2 * h), 0.0, atol=1e-6)
    null = float(spectral_null_loglik(aux, jnp.asarray(eta)))
    assert abs(null - (y @ eta - base0[0])) < 1e-9


def test_two_level_design_detection():
    rng = np.random.default_rng(0)
    B = (rng.uniform(size=(20, 5)) < 0.3).astype(float)
    d = two_level_design(jnp.asarray(B - B.mean(0)))
    np.testing.assert_allclose(np.asarray(d.B), B * (B.std(0) > 0))
    assert two_level_design(jnp.asarray(rng.normal(size=(20, 5)))) is None
    assert two_level_design(jsparse.BCOO.fromdense(jnp.asarray(2.0 * B))) is None
    sm = CharFnSpectral()
    assert isinstance(sm.design(jnp.asarray(rng.normal(size=(20, 5)))), DenseDesign)
    with pytest.raises(ValueError, match="0/1 set membership"):
        sm.design(jsparse.BCOO.fromdense(jnp.asarray(2.0 * B)))


def test_dense_blocking_is_exact():
    """Column blocks (with zero padding in the last) give the same sums as one block."""
    P = _setup("dense_continuous", "binomial", seed=9)
    Xt = jnp.asarray(P["Xt"])
    n, p = Xt.shape
    rng = np.random.default_rng(2)
    m, v = jnp.asarray(rng.normal(size=p)), jnp.asarray(np.exp(rng.normal(size=p)))
    eta = jnp.asarray(P["eta"])
    outs = []
    for d in (dense_design(Xt), dense_design(Xt, block_elems=3 * n)):
        pre, ytil = _prepare(d, P["aux"], eta, P["base"])
        outs.append((np.asarray(_spectral_terms(d, P["aux"], pre, m, v)), np.asarray(ytil)))
    assert dense_design(Xt, block_elems=3 * n).Xb.shape == (2, n, 2)
    np.testing.assert_allclose(outs[1][0], outs[0][0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outs[1][1], outs[0][1], rtol=1e-12, atol=1e-12)


def _continuous_problem(seed, n=300, p=8):
    """A dense continuous design with the other effects' offset CF and a spread of
    effect sizes, for the Chebyshev collapse."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.standard_t(5, size=(n, p)))
    y = jnp.asarray((rng.uniform(size=n) < 0.3).astype(float))
    eta = jnp.asarray(rng.normal(size=n) * 0.7 - 0.8)
    others = [(jnp.full(p, 1.0 / p), jnp.zeros(p), jnp.full(p, 0.05))]
    sm = CharFnSpectral()
    design = sm.design(X)
    m = jnp.asarray(np.linspace(-1.5, 1.5, p))
    v = jnp.full(p, 2e-3)
    aux = sm.build_spectral_aux(Bernoulli(), y, X, others, eta, design, (m, v))
    return sm, design, aux, eta, m, v


def test_cheb_collapse_converges_and_the_estimate_bounds_its_error():
    """The N-node collapse converges to the per-entry sums, and `cheb_error` bounds the
    actual error of the value, score and Price channels at every N."""
    _sm, design, aux, eta, m, v = _continuous_problem(10)
    pre, ytil = _prepare(design, aux, eta, Bernoulli())
    exact = np.asarray(_spectral_terms(design, aux, pre, m, v))
    wabs = jnp.sum(jnp.abs(aux.phi), axis=0)
    errs = []
    for N in (8, 16, 32, 64, 96):
        cd = cheb_design(design, N)
        prc, ytc = _prepare(cd, aux, eta, Bernoulli())
        got = np.asarray(_spectral_terms(cd, aux, prc, m, v))
        actual = np.abs(got - exact)[:3]
        est = np.asarray(cheb_error(cd, aux, wabs, m, v))[:3]
        assert np.all(actual <= est + 1e-11), N
        np.testing.assert_allclose(np.asarray(ytc), np.asarray(ytil), rtol=1e-13)  # exact
        errs.append(actual[0].max())
    assert errs[0] > 1e-4 and errs[-1] < 1e-10


def test_cheb_factor_converges_and_the_tail_bounds_its_error():
    """An effect's offset CF factor by the Chebyshev collapse over the features matches the
    exact feature scan, within the tail bound at every N (per row, per frequency), also
    for a near-zero-alpha feature whose phase the series does not resolve."""
    _sm, design, aux, _eta, m, _v = _continuous_problem(12)
    rng = np.random.default_rng(3)
    p = m.shape[0]
    a = np.exp(rng.normal(size=p) * 2.0)
    alpha, mu, var = jnp.asarray(a / a.sum()), jnp.asarray(rng.normal(size=p)), jnp.full(p, 0.02)
    X = design.Xb.transpose(1, 0, 2).reshape(design.Xb.shape[1], -1)[:, :p]
    exact = np.asarray(_effect_cf((X, alpha, mu, var), aux.tau))
    errs = []
    for N in (8, 16, 32, 64):
        cd = cheb_design(design, N)
        err = np.abs(np.asarray(cheb_factor(cd, alpha, mu, var, aux.tau)) - exact).max(0)
        tail = np.asarray(cheb_factor_tail(cd, alpha, mu, var, aux.tau))
        assert np.all(err <= tail + 1e-13)
        errs.append(err.max())
    # feature 1 (alpha ~ 1e-4, mu 3.3 on a wide column) needs ~220 nodes at Tmax, so it
    # still leaves alpha_1 * O(1) at N = 64; the bound above reports it
    assert errs[0] > 1e-1 and 1e-6 < errs[-1] < 1e-3


def test_kernel_design_ladder():
    """N grows with the effect size, never shrinks below the floor, and past the ladder
    (or with dense_kernel="exact") the per-entry kernel runs."""
    sm, design, aux, _eta, m, v = _continuous_problem(11)
    p = m.shape[0]
    small = sm.kernel_design(design, aux, Bernoulli(), (0.05 * m, v))
    big = sm.kernel_design(design, aux, Bernoulli(), (m, v))
    assert isinstance(small, ChebDesign) and isinstance(big, ChebDesign)
    assert small.theta.shape[0] < big.theta.shape[0]
    again = sm.kernel_design(design, aux, Bernoulli(), (0.05 * m, v), floor=big)
    assert again is big  # the same design back: the solve stands
    huge = sm.kernel_design(design, aux, Bernoulli(), (jnp.full(p, 60.0), v))
    assert isinstance(huge, DenseDesign)
    exact = CharFnSpectral(dense_kernel="exact")
    assert exact.kernel_design(design, aux, Bernoulli(), (m, v)) is design
    with pytest.raises(ValueError, match="dense_kernel"):
        CharFnSpectral(dense_kernel="cheb")


def _separated(rng, n=300, p=6):
    B = (rng.uniform(size=(n, p)) < 0.2).astype(float)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    y[B[:, 0] > 0] = 0.0  # column 0 is separated: the data push m_0 toward -inf
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    eta = jnp.asarray(rng.normal(size=n) - 1.0)
    others = [(jnp.full(p, 1.0 / p), jnp.zeros(p), jnp.full(p, 0.1))]
    return X, y, eta, others


def _solve(X, y, eta, others, law, sm, pv):
    design = sm.design(X)
    aux = sm.build_spectral_aux(Bernoulli(), jnp.asarray(y), X, others, eta, design, law)
    return aux, design, glm_spectral_ser(design, aux, eta, pv)


def test_separated_column_reaches_the_stationary_point():
    """At a large prior variance the plain joint Newton 2-cycles on a separated column
    (W collapses as |m| grows, v jumps toward pv, W recovers); the ELBO line search must
    land on the true stationary point, the same one on any adequate grid."""
    rng = np.random.default_rng(4)
    X, y, eta, others = _separated(rng)
    pv = 400.0
    p = X.shape[1]
    sols = []
    for law_m in (12.0, 200.0):
        sm = CharFnSpectral(ntau_bucket=0)
        aux, design, (m, v, lbf, _) = _solve(
            X, y, eta, others, (jnp.full(p, law_m), jnp.ones(p)), sm, pv
        )
        pre, ytil = _prepare(design, aux, eta, Bernoulli())
        t = np.asarray(_spectral_terms(design, aux, pre, m, v))
        m, v = np.asarray(m), np.asarray(v)
        np.testing.assert_allclose(np.asarray(ytil) - t[1] - m / pv, 0.0, atol=1e-9)  # score
        np.testing.assert_allclose(1.0 - v / pv - 2.0 * t[2] * v, 0.0, atol=1e-9)  # Price
        sols.append((m, np.asarray(lbf)))
    assert sols[0][0][0] < -15.0
    np.testing.assert_allclose(sols[0][1], sols[1][1], atol=1e-7)


def test_grid_guard_refits_an_undersized_grid():
    """A grid sized for m = 0 is too short once the separated column's |m| is known; the
    guard flags it and the refit agrees with a grid sized generously up front."""
    rng = np.random.default_rng(4)
    X, y, eta, others = _separated(rng)
    pv = 400.0
    p = X.shape[1]
    sm = CharFnSpectral(ntau_bucket=0)
    law = (jnp.zeros(p), jnp.full(p, 1e-6))
    aux, design, (m, v, _, _) = _solve(X, y, eta, others, law, sm, pv)
    assert not sm.grid_ok(aux, eta, design, m, v)
    first = aux.tau.shape[0]
    for _ in range(sm.max_refits):  # the engine's refit loop (glm._spectral_solve)
        aux, design, (m2, v2, lbf2, _) = _solve(X, y, eta, others, (m, v), sm, pv)
        if sm.grid_ok(aux, eta, design, m2, v2):
            break
        m, v = m2, v2
    else:
        raise AssertionError("grid never covered the solved reach")
    assert aux.tau.shape[0] > first
    big = CharFnSpectral(ntau_bucket=0)
    _, _, (m3, _, lbf3, _) = _solve(X, y, eta, others, (jnp.full(p, 200.0), jnp.ones(p)), big, pv)
    np.testing.assert_allclose(np.asarray(lbf2), np.asarray(lbf3), atol=1e-7)
    np.testing.assert_allclose(np.asarray(m2), np.asarray(m3), atol=1e-6)


def _sparse_logit(rng, n, p, d=0.08):
    B = (rng.uniform(size=(n, p)) < d).astype(float)
    beta = np.zeros(p)
    beta[[3, 7]] = [2.5, -2.2]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(B @ beta - 1.5)))).astype(float)
    return B, y


def _continuous_logit(rng, n, p):
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[3, 7]] = [1.2, -1.0]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ beta - 0.5)))).astype(float)
    return X, y


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param("dense", marks=pytest.mark.slow),
        pytest.param("bcoo", marks=pytest.mark.slow),
        "bcoo_uncentered",
        "dense_continuous",
        pytest.param("dense_continuous_exact", marks=pytest.mark.slow),
    ],
)
def test_fit_matches_high_accuracy_cf_cavi(layout):
    """Same CAVI-in-Q2 fixed point as cf_cavi. At its defaults (M=48 table, GH15) cf_cavi
    is ~1e-3 nats off in the log marginal; refined (M=200, GH51) it is the reference the
    spectral fit must match, here to ~1e-7."""
    rng = np.random.default_rng(5)
    if layout.startswith("dense_continuous"):
        B, y = _continuous_logit(rng, 400, 40)
    else:
        B, y = _sparse_logit(rng, 400, 40)
    X = jsparse.BCOO.fromdense(jnp.asarray(B)) if layout.startswith("bcoo") else jnp.asarray(B)
    kw = {"L": 3, "max_iter": 60, "tol": 1e-7, "center": layout != "bcoo_uncentered"}
    ref = fit_glm_susie(
        X, y, family=Smoothed(Bernoulli(), CharFnOffset(M=200, residual_order=None)),
        variational_family="gaussian", effect_quadrature_points=51, **kw,
    )
    default = fit_glm_susie(X, y, method="cf_cavi", **kw)
    if layout == "dense_continuous_exact":
        spec = fit_glm_susie(
            X, y, family=Smoothed(Bernoulli(), CharFnSpectral(dense_kernel="exact")),
            variational_family="gaussian", **kw,
        )
    else:
        spec = fit_glm_susie(X, y, method="cf_cavi_spectral", **kw)
    flm = lambda st: np.stack([np.asarray(e.feature_log_marginal) for e in st.single_effects])
    np.testing.assert_allclose(flm(spec), flm(ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(spec.pip), np.asarray(ref.pip), atol=1e-7)
    np.testing.assert_allclose(
        float(spec.family_state.intercept_value), float(ref.family_state.intercept_value),
        atol=1e-7,
    )
    np.testing.assert_allclose(flm(default), flm(ref), atol=5e-3)  # the default's table error
    assert spec.pip[3] > 0.9 and spec.pip[7] > 0.9


@pytest.mark.slow
@pytest.mark.parametrize("extra", ["random_intercept", "covariates", "binomial"])
def test_fit_matches_reference_with_model_extras(extra):
    """The extras that change the offset or the cumulant: a random intercept (its per-row
    variance folds into the CF), fixed covariates (the shared factor keeps the table path),
    and Binomial trials (the per-row cumulant scale)."""
    rng = np.random.default_rng(7)
    B, y = _sparse_logit(rng, 400, 40)
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    kw = {"L": 3, "max_iter": 60, "tol": 1e-7}
    base = Bernoulli()
    if extra == "random_intercept":
        kw.update(random_intercept=True, random_intercept_variance=0.5)
    elif extra == "covariates":
        kw.update(covariates=rng.normal(size=(400, 2)))
    else:
        tr = rng.integers(1, 6, size=400).astype(float)
        y = rng.binomial(tr.astype(int), 0.2 + 0.6 * y).astype(float)
        kw.update(trials=tr)
        base = Binomial()
    ref = fit_glm_susie(
        X, y, family=Smoothed(base, CharFnOffset(M=160, residual_order=None)),
        variational_family="gaussian", effect_quadrature_points=41, **kw,
    )
    spec = fit_glm_susie(X, y, method="cf_cavi_spectral", **kw)
    flm = lambda st: np.stack([np.asarray(e.feature_log_marginal) for e in st.single_effects])
    # covariates: the intercept factor still runs through the M=48 table (~1e-5)
    np.testing.assert_allclose(flm(spec), flm(ref), atol=5e-5 if extra == "covariates" else 5e-6)
    np.testing.assert_allclose(np.asarray(spec.pip), np.asarray(ref.pip), atol=1e-6)


def test_front_door_guards():
    rng = np.random.default_rng(6)
    B, y = _sparse_logit(rng, 120, 10)
    with pytest.raises(ValueError, match="0/1 set membership"):
        fit_glm_susie(jsparse.BCOO.fromdense(jnp.asarray(2.0 * B)), y, L=2,
                      method="cf_cavi_spectral")
    with pytest.raises(ValueError, match="profiled"):
        fit_glm_susie(B, y, L=2, method="cf_cavi_spectral", intercept="profiled")
    with pytest.raises(ValueError, match="gaussian"):
        fit_glm_susie(B, y, L=2, method="cf_cavi_spectral", variational_family="unconstrained")
    with pytest.raises(ValueError, match="logistic"):
        fit_glm_susie(B, np.round(y * 3), L=2, method="cf_cavi_spectral", family="poisson")
