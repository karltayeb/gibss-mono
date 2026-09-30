"""The cost-only restructurings of the CF offset builder must be exact rewrites.

Covers: (1) the GEMM factor on a 0/1 (set-membership) design equals the per-feature
scan (dense) and the per-entry support sum (sparse, centered and not); (2) the per-fit
factor memo serves a factor only for the SAME law arrays on the SAME grid -- a changed
law or a changed grid recomputes -- so a memoized table equals a fresh one; (3) the
memo is bounded to one factor per slot.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.cf_offset import (
    CharFnOffset,
    _binary_matmul,
    _binary_terms,
    _effect_cf,
    _effect_factor_sparse,
    _is_binary_design,
    build_aux,
    build_aux_sparse,
    smoothed_nodes,
)
from gibss.response import Bernoulli


def _law(rng, p):
    a = np.exp(rng.standard_normal(p))
    a /= a.sum()
    mu = rng.standard_normal(p)
    var = 0.3 * np.exp(rng.standard_normal(p) * 0.3)
    return jnp.asarray(a), jnp.asarray(mu), jnp.asarray(var)


def test_binary_gemm_factor_matches_scan_dense():
    """0/1 dense design: `base + X @ D` == the feature scan `_effect_cf`, to 1e-13."""
    rng = np.random.default_rng(0)
    n, p = 40, 12
    X = jnp.asarray((rng.random((n, p)) < 0.3).astype(float))
    assert _is_binary_design(X)
    assert not _is_binary_design(X * 0.5)
    a, mu, var = _law(rng, p)
    tau = jnp.linspace(0.0, 7.0, 48)
    ref = _effect_cf((X, a, mu, var), tau)
    base, D = _binary_terms(a, mu, var, tau, jnp.zeros(p))
    got = _binary_matmul(X, base, D)
    assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < 1e-13


@pytest.mark.parametrize("centered", [False, True])
def test_binary_gemm_factor_matches_entry_sum_sparse(centered):
    """0/1 BCOO design: the SpMM factor equals the per-entry support sum (the general
    path), centered (baseline+support split) or not, to 1e-13."""
    rng = np.random.default_rng(1)
    n, p = 50, 9
    Xd = (rng.random((n, p)) < 0.25).astype(float)
    X = sparse.BCOO.fromdense(jnp.asarray(Xd))
    assert _is_binary_design(X)
    a, mu, var = _law(rng, p)
    tau = jnp.linspace(0.0, 6.0, 40)
    colmean = jnp.asarray(Xd.mean(0)) if centered else None
    ref = _effect_factor_sparse(X, a, mu, var, tau, colmean, False, 1 << 4)
    got = _effect_factor_sparse(X, a, mu, var, tau, colmean, True, 1 << 4)
    assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < 1e-13
    # and both equal the dense scan on the (eagerly centered) dense design
    Xc = jnp.asarray(Xd - (Xd.mean(0) if centered else 0.0))
    dense = _effect_cf((Xc, a, mu, var), tau)
    assert np.max(np.abs(np.asarray(got) - np.asarray(dense))) < 1e-12


def test_factor_memo_matches_fresh_and_invalidates():
    """Memoized tables equal fresh ones; a replaced law array or a new grid is never served
    stale; the memo holds one factor per slot."""
    rng = np.random.default_rng(2)
    n, p, L = 30, 6, 3
    X = jnp.asarray(rng.standard_normal((n, p)))
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    laws = [_law(rng, p) for _ in range(L)]
    effects = [(X,) + law for law in laws]
    keys = list(range(L))
    sm = CharFnOffset(M=32)
    base = Bernoulli()

    fresh = build_aux(base, y, effects, M=32, ntau=256, Tmax=8.0)
    first = sm.build_aux(base, y, effects, keys=keys, ntau=256, Tmax=8.0)
    again = sm.build_aux(base, y, effects, keys=keys, ntau=256, Tmax=8.0)  # all hits
    for f, a_, b_ in zip(fresh, first, again):
        assert np.max(np.abs(np.asarray(f) - np.asarray(a_))) < 1e-12
        assert np.max(np.abs(np.asarray(a_) - np.asarray(b_))) < 1e-14
    assert len(sm._cache.slots) == L and sm._cache.nbytes > 0

    # change effect 1's law (new arrays, as a SER update produces): the memoized table
    # must equal the fresh table on the NEW laws, not the old ones.
    laws[1] = _law(rng, p)
    effects = [(X,) + law for law in laws]
    fresh2 = build_aux(base, y, effects, M=32, ntau=256, Tmax=8.0)
    memo2 = sm.build_aux(base, y, effects, keys=keys, ntau=256, Tmax=8.0)
    for f, m_ in zip(fresh2, memo2):
        assert np.max(np.abs(np.asarray(f) - np.asarray(m_))) < 1e-12
    assert np.max(np.abs(np.asarray(memo2[4]) - np.asarray(first[4]))) > 1e-6
    assert len(sm._cache.slots) == L

    # a different grid (ntau) invalidates every memoized factor
    fresh3 = build_aux(base, y, effects, M=32, ntau=320, Tmax=8.0)
    memo3 = sm.build_aux(base, y, effects, keys=keys, ntau=320, Tmax=8.0)
    for f, m_ in zip(fresh3, memo3):
        assert np.max(np.abs(np.asarray(f) - np.asarray(m_))) < 1e-12

    # leave-one-out subsets reuse the same slots
    loo = sm.build_aux(base, y, effects[1:], keys=keys[1:], ntau=320, Tmax=8.0)
    ref = build_aux(base, y, effects[1:], M=32, ntau=320, Tmax=8.0)
    for f, m_ in zip(ref, loo):
        assert np.max(np.abs(np.asarray(f) - np.asarray(m_))) < 1e-12


def test_factor_memo_disabled_by_budget():
    """factor_cache_bytes=0 disables the memo (nothing stored, results unchanged)."""
    rng = np.random.default_rng(3)
    n, p = 20, 5
    X = jnp.asarray(rng.standard_normal((n, p)))
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    effects = [(X,) + _law(rng, p) for _ in range(2)]
    sm = CharFnOffset(M=24, factor_cache_bytes=0)
    out = sm.build_aux(Bernoulli(), y, effects, keys=[0, 1], ntau=256, Tmax=8.0)
    ref = build_aux(Bernoulli(), y, effects, M=24, ntau=256, Tmax=8.0)
    for f, m_ in zip(ref, out):
        assert np.max(np.abs(np.asarray(f) - np.asarray(m_))) < 1e-12
    assert sm._cache.slots == {}


def test_sparse_binary_memo_build_matches_dense():
    """The sparse binary (SpMM) memoized build equals the dense build end to end."""
    rng = np.random.default_rng(4)
    n, p, L = 40, 7, 3
    Xd = (rng.random((n, p)) < 0.3).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    laws = [_law(rng, p) for _ in range(L)]
    sm = CharFnOffset(M=32)
    d = sm.build_aux(Bernoulli(), y, [(jnp.asarray(Xd),) + l for l in laws],
                     keys=list(range(L)), offset_var=0.2)
    sm2 = CharFnOffset(M=32)
    s = sm2.build_aux_sparse(Bernoulli(), y, Xs, laws, keys=list(range(L)), offset_var=0.2)
    for f, m_ in zip(d, s):
        assert np.max(np.abs(np.asarray(f) - np.asarray(m_))) < 1e-10


def test_shared_halfwidth_is_max_over_rows():
    """`hw` is one shared value = T + kappa sqrt(max_i V_i), broadcast to (n,)."""
    rng = np.random.default_rng(5)
    n, p = 15, 4
    X = jnp.asarray(rng.standard_normal((n, p)))
    effects = [(X,) + _law(rng, p)]
    znodes, *_, hw, s, V = smoothed_nodes(effects, M=16, kappa=4.0, T=10.0)
    assert hw.shape == (n,) and np.allclose(np.asarray(hw), np.asarray(hw)[0])
    assert np.isclose(float(hw[0]), 10.0 + 4.0 * np.sqrt(float(V.max())))
    assert np.allclose(np.asarray(znodes[0]), np.asarray(znodes[-1]))
