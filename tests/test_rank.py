"""Ranking SuSiE at k* = 1 topologies (forward / backward / alternating) -- rank.py.

The load-bearing identity: the ranking likelihood prod_nu prod_U lambda / e_k equals
the stratified Cox partial likelihood on the stacked [X; -X] design, so the
cox_poisson machinery (per-stratum Breslow offsets + the profiled read-out) fits
it exactly. Checked against a brute-force e_k, against fit_cox_susie (forward is
plain Cox), and against a Laplace approximation of the exact likelihood.
"""

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss import cox, fit_cox_susie, fit_susie_rank, rank
from gibss.cox_poisson import breslow_log_cumhaz, stratified_breslow_log_cumhaz

TOPOLOGIES = ("forward", "backward", "alternating")


def _brute_loglik(eta, order, nodes):
    """sum_nu [ sum_{U} eta - log e_k(lambda_S) ], e_k by enumerating subsets."""
    e = np.asarray(eta)[np.asarray(order)]
    total = 0.0
    for s, stop, k in nodes:
        lam = np.exp(e[s:stop])
        ek = sum(np.prod(c) for c in combinations(lam, k))
        total += e[s:s + k].sum() - np.log(ek)
    return total


def _simulate(rng, X, b, topology):
    """One ranking from the shortlist model: walk the path, peeling the top item
    (P ~ lambda) at winner nodes and the bottom item (P ~ 1/lambda) at loser nodes."""
    n = X.shape[0]
    eta = X @ b
    peels, _ = rank._peels(n, rank.topology_nodes(n, topology))
    alive = list(range(n))
    top, bottom = [], []
    for _, sign, _ in peels:
        g = sign * eta[alive] + rng.gumbel(size=len(alive))
        item = alive.pop(int(np.argmax(g)))
        (top if sign == 1 else bottom).append(item)
    return np.array(top + alive + bottom[::-1])


def _log_bf(effect):
    return np.asarray(effect.feature_log_marginal) - float(effect.null_log_marginal)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_log_likelihood_matches_brute_force_esp(topology):
    rng = np.random.default_rng(0)
    n = 8
    eta = rng.normal(size=n) * 1.5
    order = rng.permutation(n)
    nodes = rank.topology_nodes(n, topology)
    np.testing.assert_allclose(
        float(rank.log_likelihood(eta, order, topology)),
        _brute_loglik(eta, order, nodes), rtol=1e-12,
    )


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", [2, 3, 9, 10])
def test_stratified_cox_equals_ranking_likelihood(topology, n):
    # The stacked design's stratified partial likelihood IS the ranking likelihood
    # (value), and the Poisson working score at the per-stratum Breslow offsets IS
    # its gradient in b -- so the Breslow <-> Poisson alternation targets the exact
    # ranking MLE/MAP. Odd and even n (the alternating tail differs).
    rng = np.random.default_rng(n)
    p = 3
    X = rng.normal(size=(n, p))
    b = rng.normal(size=p)
    order = rng.permutation(n)
    d = rank.prep_data(X, order, topology)
    Xs = jnp.asarray(d.X)  # (dense) pre-centered: PL invariant to per-stratum shifts

    def pl(bb):
        eta = Xs @ bb
        total = 0.0
        for s in d.strata:
            t = np.asarray(s.fixed.time_sorted)[np.asarray(s.fixed.inverse_order)]
            ev = np.asarray(s.fixed.event_sorted)[np.asarray(s.fixed.inverse_order)]
            e = eta[s.rows]
            risk = t[None, :] >= t[:, None]
            lse = jax.scipy.special.logsumexp(jnp.where(risk, e[None, :], -jnp.inf), axis=1)
            total = total + jnp.sum(ev * (e - lse))
        return total

    exact = lambda bb: rank.log_likelihood(jnp.asarray(X) @ bb, order, topology)
    np.testing.assert_allclose(float(pl(jnp.asarray(b))), float(exact(jnp.asarray(b))),
                               rtol=1e-12, atol=1e-12)

    eta_s = Xs @ jnp.asarray(b)
    mu = jnp.exp(eta_s + stratified_breslow_log_cumhaz(d.strata, eta_s))
    score = Xs.T @ (d.y - mu)
    np.testing.assert_allclose(np.asarray(score), np.asarray(jax.grad(exact)(jnp.asarray(b))),
                               atol=1e-10)


def test_stratified_breslow_is_per_stratum():
    # each stratum's rows get the Breslow hazard of their own risk sets only
    rng = np.random.default_rng(1)
    n = 30
    eta = jnp.asarray(rng.normal(size=n))
    time = jnp.asarray(np.round(rng.exponential(size=n), 1))  # ties
    event = jnp.asarray((rng.random(n) < 0.7).astype(float))
    labels = rng.integers(0, 3, size=n)
    from gibss.cox_poisson import _build_strata

    strata = _build_strata(jnp.zeros((n, 1)), time, event, labels)
    got = stratified_breslow_log_cumhaz(strata, eta)
    for lab in range(3):
        rows = np.flatnonzero(labels == lab)
        fixed = cox.prepare_fixed_cox_context(time[rows], event[rows])
        np.testing.assert_allclose(np.asarray(got)[rows],
                                   np.asarray(breslow_log_cumhaz(fixed, eta[rows])), atol=1e-12)


def test_forward_is_cox_on_rank_times():
    # forward PL == Cox with time = rank, every item an event except the last:
    # same stacked rows, same model, same fit.
    rng = np.random.default_rng(2)
    n, p = 150, 8
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[3] = 0.8
    order = _simulate(rng, X, b, "forward")
    fr = fit_susie_rank(X, order, "forward", L=3, max_iter=60)
    event = np.ones(n)
    event[-1] = 0.0
    fc = fit_cox_susie(X[order], event_time=np.arange(n, dtype=float), event_type=event,
                       L=3, max_iter=60)
    np.testing.assert_allclose(np.asarray(fr.pip), np.asarray(fc.pip), atol=1e-10)
    for er, ec in zip(fr.single_effects, fc.single_effects, strict=True):
        np.testing.assert_allclose(np.asarray(er.mu), np.asarray(ec.mu), atol=1e-10)


def test_backward_is_forward_reversed_with_negated_design():
    # backward PL on `order` == forward PL on the reversed order with lambda -> 1/lambda,
    # i.e. -X; the coefficient b is the same.
    rng = np.random.default_rng(3)
    n, p = 150, 8
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[5] = -0.9
    order = _simulate(rng, X, b, "backward")
    fb = fit_susie_rank(X, order, "backward", L=3, max_iter=60)
    ff = fit_susie_rank(-X, order[::-1], "forward", L=3, max_iter=60)
    np.testing.assert_allclose(np.asarray(fb.pip), np.asarray(ff.pip), atol=1e-10)
    for eb, ef in zip(fb.single_effects, ff.single_effects, strict=True):
        np.testing.assert_allclose(np.asarray(eb.mu), np.asarray(ef.mu), atol=1e-10)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_ser_matches_laplace_of_exact_likelihood(topology):
    # L = 1, fixed prior variance: each feature's (mu, log BF) equal the ridge MAP and
    # Laplace evidence of the EXACT ranking likelihood of that feature alone.
    rng = np.random.default_rng(4)
    n, p, pv = 120, 6, 0.5
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[1] = 1.0
    order = _simulate(rng, X, b, topology)
    fit = fit_susie_rank(X, order, topology, L=1, prior_variance=pv,
                         estimate_prior_variance=False, max_iter=50)
    eff = fit.single_effects[0]

    def ll(bj, x):
        return rank.log_likelihood(x * bj, order, topology)

    ll_j = jax.jit(ll)
    g, h = jax.jit(jax.grad(ll)), jax.jit(jax.grad(jax.grad(ll)))
    Xc = jnp.asarray(X - X.mean(0))
    mus, lbfs = [], []
    for j in range(p):
        x, bj = Xc[:, j], 0.0
        for _ in range(30):
            bj = bj - (g(bj, x) - bj / pv) / (h(bj, x) - 1.0 / pv)
        prec = -h(bj, x) + 1.0 / pv
        mus.append(bj)
        lbfs.append(ll_j(bj, x) - ll_j(0.0, x) - 0.5 * bj**2 / pv - 0.5 * np.log(pv * prec))
    np.testing.assert_allclose(np.asarray(eff.mu), np.array(mus, dtype=float), atol=1e-8)
    np.testing.assert_allclose(_log_bf(eff), np.array(lbfs, dtype=float), atol=1e-8)


def test_alternating_recovers_causal_features():
    rng = np.random.default_rng(5)
    n, p = 500, 30
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[[4, 17]] = [0.7, -0.6]
    order = _simulate(rng, X, b, "alternating")
    fit = fit_susie_rank(X, order, "alternating", L=5, max_iter=100)
    pip = np.asarray(fit.pip)
    assert pip[4] > 0.95 and pip[17] > 0.95
    assert np.max(np.delete(pip, [4, 17])) < 0.2
    top = {int(np.argmax(e.alpha)): float(np.sum(np.asarray(e.alpha) * np.asarray(e.mu)))
           for e in fit.single_effects if float(np.max(e.alpha)) > 0.9}
    assert top[4] > 0.4 and top[17] < -0.4


def test_sparse_matches_dense():
    rng = np.random.default_rng(6)
    n, p = 200, 12
    X = (rng.random((n, p)) < 0.15).astype(float)
    b = np.zeros(p)
    b[2] = 1.2
    order = _simulate(rng, X, b, "alternating")
    kw = {"L": 3, "max_iter": 60, "center": False}
    fd = fit_susie_rank(X, order, "alternating", **kw)
    fs = fit_susie_rank(jsparse.BCOO.fromdense(jnp.asarray(X)), order, "alternating", **kw)
    np.testing.assert_allclose(np.asarray(fs.pip), np.asarray(fd.pip), atol=1e-8)
    for es, ed in zip(fs.single_effects, fd.single_effects, strict=True):
        np.testing.assert_allclose(_log_bf(es), _log_bf(ed), atol=1e-8)


def _breslow_binned_loglik(eta, order, topology, bins):
    """Direct Breslow bin model: in each peel group but the last, each sign's peels are
    tied events against the items alive at the group's start."""
    n = eta.shape[0]
    e = np.asarray(eta)[np.asarray(order)]
    peels, time = rank._peels(n, rank.topology_nodes(n, topology))
    group = time * bins // (n - 1)
    total = 0.0
    for g in range(bins - 1):
        alive = group >= g
        for sign in (1, -1):
            ev = [pos for t, sg, pos in peels if sg == sign and t * bins // (n - 1) == g]
            if ev:
                total += sign * e[ev].sum() - len(ev) * np.log(np.exp(sign * e[alive]).sum())
    return total


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("bins", [1, 3, 7, 29])
def test_binned_is_breslow_bin_model(topology, bins):
    # bins= codes the grouped peels as Breslow-tied events with the last group censored;
    # bins >= n - 1 is the full ranking; bins = 1 leaves nothing to fit.
    rng = np.random.default_rng(bins)
    n, p = 30, 2
    X = rng.normal(size=(n, p))
    b = rng.normal(size=p)
    order = rng.permutation(n)
    if bins == 1:
        with pytest.raises(ValueError, match="nothing to fit"):
            rank.prep_data(X, order, topology, bins=bins)
        return
    d = rank.prep_data(X, order, topology, bins=bins, center=False)
    eta_s = jnp.asarray(d.X) @ jnp.asarray(b)
    got = 0.0
    for s in d.strata:
        t = np.asarray(s.fixed.time_sorted)[np.asarray(s.fixed.inverse_order)]
        ev = np.asarray(s.fixed.event_sorted)[np.asarray(s.fixed.inverse_order)]
        e = np.asarray(eta_s)[np.asarray(s.rows)]
        risk = t[None, :] >= t[:, None]
        lse = np.log((np.exp(e)[None, :] * risk).sum(1))
        got += float((ev * (e - lse)).sum())
    eta = X @ b
    want = (float(rank.log_likelihood(eta, order, topology)) if bins == 29
            else _breslow_binned_loglik(eta, order, topology, bins))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_explicit_nodes_equal_preset():
    rng = np.random.default_rng(7)
    n, p = 40, 4
    X = rng.normal(size=(n, p))
    order = rng.permutation(n)
    nodes = list(rank.topology_nodes(n, "alternating"))[::-1]  # any order
    a = fit_susie_rank(X, order, nodes, L=2, max_iter=30)
    b = fit_susie_rank(X, order, "alternating", L=2, max_iter=30)
    np.testing.assert_allclose(np.asarray(a.pip), np.asarray(b.pip), atol=1e-12)


def test_input_validation():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError, match="permutation"):
        rank.prep_data(X, [0, 1, 2, 3, 3])
    with pytest.raises(NotImplementedError, match="k\\*"):
        rank.topology_nodes(8, "balanced")
    with pytest.raises(NotImplementedError, match="k\\*"):
        rank.topology_nodes(4, [(0, 4, 2), (0, 2, 1), (2, 4, 1)])
    with pytest.raises(ValueError, match="child block"):
        rank.topology_nodes(4, [(0, 4, 1), (1, 4, 1), (0, 2, 1)])
    with pytest.raises(ValueError, match="unknown topology"):
        rank.topology_nodes(4, "sideways")
