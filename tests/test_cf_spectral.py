"""Spectral CAVI-in-Q2 effect update (`cf_spectral`): Newton in (m, s) on the fixed-grid
Fourier surrogate, under the null / plug-in / Gaussian reference, collapsed on two-level
designs and the Chebyshev-in-x collapse, per entry on general dense designs.

Ground truth is independent of every CF code path: the leave-one-out offset is a product
of small Gaussian mixtures, so enumerating its components and adding the effect's own
Gaussian gives each row's shift law as an explicit Gaussian mixture, and a high-order GH
rule integrates the cumulant against each component (the logistic softplus, or the
probit mixture the Gaussian reference runs under). The kernel's objective, its (m, s)
derivatives, its fixed point and the null must all match that enumeration, for every
reference.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss._numerics import _ndtr
from gibss.cf_offset import CharFnOffset, _effect_cf
from gibss.cf_spectral import (
    CharFnSpectral,
    ChebDesign,
    DenseDesign,
    Reference,
    _prepare,
    _spectral_terms,
    cheb_design,
    cheb_error,
    cheb_factor,
    cheb_factor_tail,
    dense_design,
    glm_spectral_ser,
    ladder_ntau,
    ones_design,
    spectral_null_loglik,
    two_level_design,
)
from gibss.methods import fit_glm_susie
from gibss.response import _PROBIT_MIXTURES, Bernoulli, Binomial, Smoothed
from gibss.response_ser import _gh_rule

GH_NODES, GH_LOGW = _gh_rule(120)
GH_W = np.exp(GH_LOGW) / np.sqrt(np.pi)
OV = 0.1  # homogeneous zero-mean Gaussian offset (the shared intercept's variance)

REFS = [Reference("null"), Reference("plugin"), Reference("gaussian", 8)]
REF_IDS = [r.kind for r in REFS]


def _gauss_A(mean, var, order=None):
    """E A(N(mean, var)): softplus by 120-node GH (exact to ~1e-13 for these scales), or
    the order-K probit mixture's closed form."""
    if order is None:
        z = mean[..., None] + np.sqrt(2.0 * var)[..., None] * GH_NODES
        return np.logaddexp(0.0, z) @ GH_W
    s, pi, _ = _PROBIT_MIXTURES[order]
    out = 0.0
    for sk, pk in zip(s, pi):
        r = np.sqrt(1.0 + sk**2 * var)
        zh = sk * mean / r
        cdf = np.asarray(_ndtr(jnp.asarray(zh)))
        pdf = np.exp(-0.5 * zh * zh) / np.sqrt(2.0 * np.pi)
        out = out + pk * (r / sk) * (zh * cdf + pdf)
    return out


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


def _brute(Xt, others, eta, s, m, sd, order=None):
    """sum_i s_i E A(eta_i - mean_i + o'_i + xt_ij b) per column j, b ~ N(m_j, sd_j^2),
    with o' the enumerated product mixture (+ N(0, OV)); eta carries the offset mean."""
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
        var_tot = cv[:, None] + Xt**2 * sd[None, :] ** 2
        out += w * (s[:, None] * _gauss_A(mu_tot, var_tot, order)).sum(0)
    return out


def _order(ref):
    return None if ref.logistic else ref.order


def _setup(layout, family, ref, seed=0):
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
    sm = CharFnSpectral(reference=ref.kind, mixture_order=ref.order)
    design = sm.design(X, colmean)
    if layout == "dense_cheb":  # the Chebyshev-in-x collapse, converged
        design = cheb_design(design, 64)
    law = (jnp.full(p, 3.0), jnp.full(p, 2.0))  # generous reach for the grid
    aux = sm.build_spectral_aux(
        base, yleaf, X, [tuple(jnp.asarray(t) for t in e) for e in others],
        jnp.asarray(eta), design, law, offset_var=OV, colmean=colmean,
    )
    return {"Xt": Xt, "others": others, "eta": eta, "y": y, "s": s, "base": base,
            "design": design, "aux": aux, "sm": sm, "ref": ref}


LAYOUTS = ["dense01", "dense_centered", "bcoo", "bcoo_centered", "dense_continuous",
           "dense_cheb"]


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_terms_match_enumerated_mixture(layout, family, ref):
    """The surrogate (relative to its own b = 0 value) and all five (m, s)-derivative
    channels match the enumerated mixture, under every reference."""
    P = _setup(layout, family, ref)
    rng = np.random.default_rng(1)
    p = P["Xt"].shape[1]
    m, sd = rng.normal(size=p) * 1.5, np.exp(rng.normal(size=p) * 0.5) * 0.7
    eta = jnp.asarray(P["eta"])
    pre, _ = _prepare(ref, P["design"], P["aux"], eta, P["base"])
    T = np.asarray(_spectral_terms(ref, P["design"], P["aux"], pre, jnp.asarray(m), jnp.asarray(sd)))
    T00 = np.asarray(_spectral_terms(ref, P["design"], P["aux"], pre, jnp.zeros(p), jnp.zeros(p)))[0]
    F = lambda mm, ss: _brute(P["Xt"], P["others"], P["eta"], P["s"], mm, ss, _order(ref))
    base0 = F(np.zeros(p), np.zeros(p))
    np.testing.assert_allclose(T[0] - T00, F(m, sd) - base0, atol=1e-9)
    h = 1e-4
    d_m = (F(m + h, sd) - F(m - h, sd)) / (2 * h)
    d_s = (F(m, sd + h) - F(m, sd - h)) / (2 * h)
    d_mm = (F(m + h, sd) - 2 * F(m, sd) + F(m - h, sd)) / (h * h)
    d_ms = (F(m + h, sd + h) - F(m + h, sd - h) - F(m - h, sd + h) + F(m - h, sd - h)) / (4 * h * h)
    d_ss = (F(m, sd + h) - 2 * F(m, sd) + F(m, sd - h)) / (h * h)
    np.testing.assert_allclose(T[1], d_m, atol=1e-7)
    np.testing.assert_allclose(T[2], d_s, atol=1e-7)
    np.testing.assert_allclose(T[3], d_mm, atol=1e-5)
    np.testing.assert_allclose(T[4], d_ms, atol=1e-5)
    np.testing.assert_allclose(T[5], d_ss, atol=1e-5)


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_is_the_q2_cavi_optimum(layout, family, ref):
    """The solved (m, s) is stationary for the brute-force ELBO, elbo_rel equals it, and
    the null matches the enumeration."""
    P = _setup(layout, family, ref, seed=2)
    pv = 1.7
    eta = jnp.asarray(P["eta"])
    m, v, elbo_rel, _kl, err = glm_spectral_ser(P["design"], P["aux"], eta, pv, P["base"], ref)
    m, sd = np.asarray(m), np.sqrt(np.asarray(v))
    ytil = P["y"] @ P["Xt"]
    p = m.shape[0]
    F = lambda mm, ss: _brute(P["Xt"], P["others"], P["eta"], P["s"], mm, ss, _order(ref))
    base0 = F(np.zeros(p), np.zeros(p))

    def elbo(mm, ss):
        kl_ = 0.5 * (np.log(pv / ss**2) + (ss**2 + mm**2) / pv - 1.0)
        return mm * ytil - (F(mm, ss) - base0) - kl_

    np.testing.assert_allclose(np.asarray(elbo_rel), elbo(m, sd), atol=1e-9)
    h = 1e-5
    np.testing.assert_allclose((elbo(m + h, sd) - elbo(m - h, sd)) / (2 * h), 0.0, atol=1e-6)
    np.testing.assert_allclose((elbo(m, sd + h) - elbo(m, sd - h)) / (2 * h), 0.0, atol=1e-6)
    assert P["sm"].grid_ok(err, v)  # the generous grid is converged
    want = P["y"] @ P["eta"] - base0[0]
    got = float(spectral_null_loglik(P["aux"], eta, P["base"], ref))
    assert abs(got - want) < 1e-8  # the absolute null rides the same grid (not ladder-checked)


@pytest.mark.parametrize("layout", ["bcoo_centered", "dense_continuous"])
def test_references_share_the_fixed_point(layout):
    """Null and plug-in references are two accountings of the same logistic surrogate:
    same (m, v) and log BF. The Gaussian reference runs under the order-8 probit mixture,
    which differs from the logistic by its table error (~1e-7 on the mean), no more."""
    sols = {}
    for ref in REFS:
        P = _setup(layout, "bernoulli", ref, seed=5)
        m, v, lbf, _, _ = glm_spectral_ser(P["design"], P["aux"], jnp.asarray(P["eta"]), 1.7,
                                           P["base"], ref)
        sols[ref.kind] = (np.asarray(m), np.asarray(v), np.asarray(lbf))
    for k in range(3):
        np.testing.assert_allclose(sols["plugin"][k], sols["null"][k], atol=1e-9)
        np.testing.assert_allclose(sols["gaussian"][k], sols["null"][k], atol=2e-5)


def _brute_gaussian(Xt, others, eta, s, m, sd):
    """The Gaussian-offset counterpart of `_brute`: each row's offset is N(0, V_i), V_i the
    product mixture's variance (+ OV)."""
    V = np.full(Xt.shape[0], OV)
    for a, mu, var in others:
        mean = Xt @ (a * mu)
        V = V + (Xt**2) @ (a * (mu**2 + var)) - mean**2
    mu_tot = eta[:, None] + Xt * m[None, :]
    return (s[:, None] * _gauss_A(mu_tot, V[:, None] + Xt**2 * sd[None, :] ** 2)).sum(0)


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
    m, v, elbo_rel, _, _ = glm_spectral_ser(design, aux, jnp.asarray(eta), pv)
    m, sd = np.asarray(m), np.sqrt(np.asarray(v))
    s = np.ones(n)
    F = lambda mm, ss: _brute_gaussian(Xt, others, eta, s, mm, ss)
    base0 = F(np.zeros(p), np.zeros(p))

    def elbo(mm, ss):
        kl = 0.5 * (np.log(pv / ss**2) + (ss**2 + mm**2) / pv - 1.0)
        return mm * (y @ Xt) - (F(mm, ss) - base0) - kl

    np.testing.assert_allclose(np.asarray(elbo_rel), elbo(m, sd), atol=1e-9)
    h = 1e-5
    np.testing.assert_allclose((elbo(m + h, sd) - elbo(m - h, sd)) / (2 * h), 0.0, atol=1e-6)
    null = float(spectral_null_loglik(aux, jnp.asarray(eta)))
    assert abs(null - (y @ eta - base0[0])) < 1e-9


def test_two_level_design_detection():
    rng = np.random.default_rng(0)
    B = (rng.uniform(size=(20, 5)) < 0.3).astype(float)
    d = two_level_design(jnp.asarray(B - B.mean(0)))
    np.testing.assert_allclose(np.asarray(d.B), B * (B.std(0) > 0))
    assert d.rows.shape[0] == int((B * (B.std(0) > 0)).sum())
    assert two_level_design(jnp.asarray(rng.normal(size=(20, 5)))) is None
    assert two_level_design(jsparse.BCOO.fromdense(jnp.asarray(2.0 * B))) is None
    sm = CharFnSpectral()
    assert isinstance(sm.design(jnp.asarray(rng.normal(size=(20, 5)))), DenseDesign)
    with pytest.raises(ValueError, match="0/1 set membership"):
        sm.design(jsparse.BCOO.fromdense(jnp.asarray(2.0 * B)))
    one = ones_design(7)
    assert one.rows.shape == (7,) and float(one.lo[0]) == 0.0 and float(one.hi[0]) == 1.0
    assert ladder_ntau(33) == 33 and ladder_ntau(34) == 65 and ladder_ntau(5) == 33


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
def test_dense_blocking_is_exact(ref):
    """Column blocks (with zero padding in the last) give the same sums as one block."""
    P = _setup("dense_continuous", "binomial", ref, seed=9)
    Xt = jnp.asarray(P["Xt"])
    n, p = Xt.shape
    rng = np.random.default_rng(2)
    m, s = jnp.asarray(rng.normal(size=p)), jnp.asarray(np.exp(rng.normal(size=p) * 0.5))
    eta = jnp.asarray(P["eta"])
    outs = []
    for d in (dense_design(Xt), dense_design(Xt, block_elems=3 * n)):
        pre, ytil = _prepare(ref, d, P["aux"], eta, P["base"])
        outs.append((np.asarray(_spectral_terms(ref, d, P["aux"], pre, m, s)), np.asarray(ytil)))
    assert dense_design(Xt, block_elems=3 * n).Xb.shape == (2, n, 2)
    np.testing.assert_allclose(outs[1][0], outs[0][0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outs[1][1], outs[0][1], rtol=1e-12, atol=1e-12)


def _continuous_problem(seed, ref=Reference(), n=300, p=8):
    """A dense continuous design with the other effects' offset CF and a spread of
    effect sizes, for the Chebyshev collapse."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.standard_t(5, size=(n, p)))
    y = jnp.asarray((rng.uniform(size=n) < 0.3).astype(float))
    eta = jnp.asarray(rng.normal(size=n) * 0.7 - 0.8)
    others = [(jnp.full(p, 1.0 / p), jnp.zeros(p), jnp.full(p, 0.05))]
    sm = CharFnSpectral(reference=ref.kind, mixture_order=ref.order)
    design = sm.design(X)
    m = jnp.asarray(np.linspace(-1.5, 1.5, p))
    v = jnp.full(p, 2e-3)
    aux = sm.build_spectral_aux(Bernoulli(), y, X, others, eta, design, (m, v))
    return sm, design, aux, eta, m, v


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
def test_cheb_collapse_converges_and_the_estimate_bounds_its_error(ref):
    """The N-node collapse converges to the per-entry sums, and `cheb_error` bounds the
    actual error of the value, score and s-gradient channels at every N."""
    _sm, design, aux, eta, m, v = _continuous_problem(10, ref)
    s = jnp.sqrt(v)
    pre, ytil = _prepare(ref, design, aux, eta, Bernoulli())
    exact = np.asarray(_spectral_terms(ref, design, aux, pre, m, s))
    sw = jnp.ones(aux.phi.shape[0])
    if ref.kind == "null":
        zeta = jnp.zeros_like(aux.phi.real)
    elif ref.kind == "plugin":
        zeta = jnp.ones_like(aux.phi.real)
    else:
        zeta = jnp.exp(-0.5 * aux.V[:, None] * aux.tau[None, :] ** 2)
    wabs = (jnp.abs(aux.phi) + zeta).T @ sw  # sum_i s_i (|W_ik| + |Z_ik|)
    errs = []
    for N in (8, 16, 32, 64, 96):
        cd = cheb_design(design, N)
        prc, ytc = _prepare(ref, cd, aux, eta, Bernoulli())
        got = np.asarray(_spectral_terms(ref, cd, aux, prc, m, s))
        actual = np.abs(got - exact)[:3]
        est = np.asarray(cheb_error(cd, aux, wabs, m, s))[:3]
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
    with pytest.raises(ValueError, match="kind"):
        CharFnSpectral(reference="plug-in")


def _separated(rng, n=300, p=6):
    B = (rng.uniform(size=(n, p)) < 0.2).astype(float)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    y[B[:, 0] > 0] = 0.0  # column 0 is separated: the data push m_0 toward -inf
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    eta = jnp.asarray(rng.normal(size=n) - 1.0)
    others = [(jnp.full(p, 1.0 / p), jnp.zeros(p), jnp.full(p, 0.1))]
    return X, y, eta, others


def _solve(X, y, eta, others, law, sm, pv, ntau=None, init=None):
    design = sm.design(X)
    aux = sm.build_spectral_aux(Bernoulli(), jnp.asarray(y), X, others, eta, design, law, ntau=ntau)
    return aux, design, glm_spectral_ser(design, aux, eta, pv, Bernoulli(), sm.ref, init=init)


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
def test_separated_column_reaches_the_stationary_point(ref):
    """At a large prior variance the plain Newton map 2-cycles on a separated column (the
    curvature collapses as |m| grows, s jumps toward sqrt(pv), it recovers); the ELBO
    line search must land on the true stationary point, the same one on any adequate
    grid."""
    rng = np.random.default_rng(4)
    X, y, eta, others = _separated(rng)
    pv = 400.0
    p = X.shape[1]
    sols = []
    for law_m in (12.0, 200.0):
        sm = CharFnSpectral(reference=ref.kind, mixture_order=ref.order)
        aux, design, (m, v, lbf, _, _) = _solve(
            X, y, eta, others, (jnp.full(p, law_m), jnp.ones(p)), sm, pv
        )
        s = jnp.sqrt(v)
        pre, ytil = _prepare(ref, design, aux, eta, Bernoulli())
        t = np.asarray(_spectral_terms(ref, design, aux, pre, m, s))
        m, s = np.asarray(m), np.asarray(s)
        np.testing.assert_allclose(np.asarray(ytil) - t[1] - m / pv, 0.0, atol=1e-9)  # d/dm
        np.testing.assert_allclose(-t[2] - s / pv + 1.0 / s, 0.0, atol=1e-9)  # d/ds
        sols.append((m, np.asarray(lbf)))
    assert sols[0][0][0] < -15.0
    np.testing.assert_allclose(sols[0][1], sols[1][1], atol=1e-7)


@pytest.mark.parametrize("ref", REFS, ids=REF_IDS)
def test_grid_ladder_detects_and_cures_aliasing(ref):
    """A grid sized for m = 0 aliases once the separated column's |m| is known: the
    measured aliasing flags it, climbing the ladder (the engine's loop) cures it, and
    the result agrees with a grid sized generously up front. The estimate bounds the
    actual error on every rung."""
    rng = np.random.default_rng(4)
    X, y, eta, others = _separated(rng)
    pv = 400.0
    p = X.shape[1]
    sm = CharFnSpectral(reference=ref.kind, mixture_order=ref.order)
    big = CharFnSpectral(reference=ref.kind, mixture_order=ref.order)
    _, _, (m3, v3, lbf3, _, _) = _solve(X, y, eta, others, (jnp.full(p, 200.0), jnp.ones(p)), big, pv)
    law = (jnp.zeros(p), jnp.full(p, 1e-6))
    aux, design, (m, v, lbf, _, err) = _solve(X, y, eta, others, law, sm, pv)
    assert not sm.grid_ok(err, v)
    first = aux.tau.shape[0]
    ntau = first
    for _ in range(sm.max_refits):  # the engine's ladder (glm._spectral_solve)
        ntau = 2 * (ntau - 1) + 1
        aux, design, (m, v, lbf, _, err) = _solve(X, y, eta, others, (m, v), sm, pv, ntau=ntau, init=(m, v))
        actual = abs(float(lbf[0]) - float(lbf3[0]))
        assert actual <= max(float(jnp.max(err[0])), 1e-9) * 10.0 + 1e-9
        if sm.grid_ok(err, v):
            break
    else:
        raise AssertionError("the ladder never covered the solved reach")
    assert aux.tau.shape[0] > first
    np.testing.assert_allclose(np.asarray(lbf), np.asarray(lbf3), atol=1e-7)
    np.testing.assert_allclose(np.asarray(m), np.asarray(m3), atol=1e-6)


def test_non_null_references_alias_less():
    """The point of a non-null reference: on the same coarse grid the Fourier sum's
    aliasing scales with the residual it carries. At 33 nodes the plug-in residual
    (the gIBSS plug-in error) aliases less than the null residual (the whole evidence),
    and the Gaussian residual (the offset's non-Gaussian part, under a kernel whose
    Gaussian tail shortens Tmax) by orders of magnitude less. The measured estimate
    never under-reports the actual error on a certified rung, and no reference settles
    on a finer rung than the null one."""
    rng = np.random.default_rng(6)
    B, y = _sparse_logit(rng, 400, 30)
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    eta = jnp.asarray(rng.normal(size=400) * 2.5)
    p = B.shape[1]
    others = [(jnp.full(p, 1.0 / p), jnp.asarray(rng.normal(size=p)), jnp.full(p, 0.2))]
    law = (jnp.zeros(p), jnp.ones(p))
    coarse, settled = {}, {}
    for ref in REFS:
        fresh = lambda: CharFnSpectral(reference=ref.kind, mixture_order=ref.order)  # noqa: E731
        # (the ntau ratchet is per instance: a fixed-rung solve needs a fresh smoother)
        _, _, (_, _, lbf_fine, _, _) = _solve(X, y, eta, others, law, fresh(), 1.0, ntau=2049)
        _, _, (_, _, lbf33, _, _) = _solve(X, y, eta, others, law, fresh(), 1.0, ntau=33)
        coarse[ref.kind] = float(jnp.max(jnp.abs(lbf33 - lbf_fine)))
        sm = fresh()
        ntau = 33
        for _ in range(8):
            aux, _design, (m, v, lbf, _, err) = _solve(X, y, eta, others, law, sm, 1.0, ntau=ntau)
            if sm.grid_ok(err, v):
                break
            ntau = 2 * (ntau - 1) + 1
        settled[ref.kind] = aux.tau.shape[0]
        actual = float(jnp.max(jnp.abs(lbf - lbf_fine)))
        assert actual <= sm.alias_tol + 1e-10, (ref.kind, actual)
    assert coarse["null"] > 1e-6
    assert coarse["plugin"] < 0.5 * coarse["null"]
    assert coarse["gaussian"] < 1e-2 * coarse["null"]
    assert settled["plugin"] <= settled["null"] and settled["gaussian"] <= settled["null"]


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


@pytest.mark.parametrize("method", ["cf_cavi_spectral_plugin", "cf_cavi_spectral_gaussref"])
def test_fit_with_non_null_reference_matches_the_null_reference(method):
    """The reference presets reach the null-reference fit: the plug-in one to roundoff,
    the Gaussian one to the probit-mixture table error."""
    rng = np.random.default_rng(5)
    B, y = _sparse_logit(rng, 400, 40)
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    kw = {"L": 3, "max_iter": 60, "tol": 1e-7, "center": True}
    null = fit_glm_susie(X, y, method="cf_cavi_spectral", **kw)
    other = fit_glm_susie(X, y, method=method, **kw)
    flm = lambda st: np.stack([np.asarray(e.feature_log_marginal) for e in st.single_effects])
    tol = 1e-6 if method.endswith("plugin") else 5e-4
    np.testing.assert_allclose(flm(other), flm(null), atol=tol)
    np.testing.assert_allclose(np.asarray(other.pip), np.asarray(null.pip), atol=tol)


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


def test_vi_gh_kernel_matches_spectral_on_a_separated_column():
    """The table + GH kernel (`cf_cavi`) shares the spectral kernel's ELBO line search
    (`response_ser._monotone_newton`): on a separated column at a large fixed prior
    variance, where its plain joint Newton used to 2-cycle, both reach the same fixed
    point and the same log Bayes factors."""
    rng = np.random.default_rng(4)
    n, p = 300, 6
    B = (rng.uniform(size=(n, p)) < 0.2).astype(float)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    y[B[:, 0] > 0] = 0.0  # column 0 is separated
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    kw = {"L": 1, "max_iter": 30, "tol": 1e-8, "prior_variance": 400.0,
          "estimate_prior_variance": False, "center": False}
    table = fit_glm_susie(
        X, y, family=Smoothed(Bernoulli(), CharFnOffset(M=200, residual_order=None)),
        variational_family="gaussian", effect_quadrature_points=51, **kw,
    )
    spec = fit_glm_susie(X, y, method="cf_cavi_spectral", **kw)
    flm = lambda st: np.asarray(st.single_effects[0].feature_log_marginal)  # noqa: E731
    mu = lambda st: np.asarray(st.single_effects[0].mu)  # noqa: E731
    assert mu(spec)[0] < -10.0  # the separated column's effect runs far out
    np.testing.assert_allclose(flm(table)[0], flm(spec)[0], atol=1e-6)
    # the ELBO is flat in m out there (the column is separated): 1e-6 nats is ~1e-3 in m
    np.testing.assert_allclose(mu(table)[0], mu(spec)[0], atol=5e-3)
    # the other columns: at prior variance 400 the GH nodes over b reach past the table's
    # z-range, so the table kernel itself is ~5e-5 off there (its error, not the Newton's)
    np.testing.assert_allclose(flm(table), flm(spec), atol=1e-4)


def test_point_offset_law_is_gibss_q2():
    """offset_law="point" (preset spectral_point_offset) is gIBSS-Q2 in the spectral form:
    every component, the intercept included, enters the others at its mean only, and the
    fixed point is the plain-base Gaussian-VI kernel's (`gibss_gaussian`, GH over b, here
    at 51 nodes so its quadrature is not the limit), with no CF build. Both fit the
    intercept's own factor q(b0) by the Gaussian-VI update (the Q2 update of the b0
    component), so the estimated-intercept fits agree too."""
    rng = np.random.default_rng(5)
    B, y = _sparse_logit(rng, 400, 40)
    X = jsparse.BCOO.fromdense(jnp.asarray(B))
    flm = lambda st: np.stack([np.asarray(e.feature_log_marginal) for e in st.single_effects])  # noqa: E731
    for estimate in (False, True):
        kw = {"L": 3, "max_iter": 60, "tol": 1e-7, "center": True, "estimate_intercept": estimate}
        ref = fit_glm_susie(X, y, method="gibss_gaussian", effect_quadrature_points=51, **kw)
        spec = fit_glm_susie(X, y, method="spectral_point_offset", **kw)
        np.testing.assert_allclose(flm(spec), flm(ref), atol=1e-6)
        np.testing.assert_allclose(np.asarray(spec.pip), np.asarray(ref.pip), atol=1e-7)
        np.testing.assert_allclose(
            float(spec.family_state.intercept_value), float(ref.family_state.intercept_value),
            atol=1e-7,
        )
    with pytest.raises(ValueError, match="offset_law"):
        CharFnSpectral(offset_law="plugin")
