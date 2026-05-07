from __future__ import annotations

import numpy as np
from jax.experimental import sparse

from gibss_reference.logistic_sparse import elbo_sparse, logistic_susie_fit_sparse_global_jj, xi_update_sparse


def _make_data(seed: int = 0, n: int = 120, p: int = 20):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, p), dtype=np.float32)
    for i in range(n):
        cols = rng.choice(p, size=max(1, p // 5), replace=False)
        X[i, cols] = rng.normal(size=cols.size)
    beta = np.zeros(p, dtype=np.float32)
    beta[0] = 2.5
    beta[3] = -1.75
    logits = X @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return X, y


def test_sparse_global_reference_smoke():
    X, y = _make_data(seed=1, n=100, p=16)
    Xs = sparse.BCOO.fromdense(X)
    q, xi, intercept, elbos, converged = logistic_susie_fit_sparse_global_jj(Xs, y, L=3, max_iter=4)

    assert len(q) == 3
    assert np.isfinite(intercept)
    assert np.all(np.isfinite(xi))
    assert np.all(np.diff(elbos) >= -1e-6)
    assert isinstance(converged, bool)


def test_sparse_global_reference_xi_update_matches_elbo_inputs():
    X, y = _make_data(seed=2, n=80, p=10)
    Xs = sparse.BCOO.fromdense(X)
    q, _xi, intercept, _elbos, _converged = logistic_susie_fit_sparse_global_jj(Xs, y, L=2, max_iter=2)

    xi = xi_update_sparse(Xs, q, intercept)
    assert np.isfinite(elbo_sparse(Xs, y, q, xi, intercept))
