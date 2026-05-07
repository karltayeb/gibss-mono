from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

from .logistic import LogisticSER, _ensure_array, _to_public_q, lambda_xi


def _ensure_bcoo(X) -> sparse.BCOO:
    if isinstance(X, sparse.BCOO):
        return X
    return sparse.BCOO.fromdense(_ensure_array(X))


def _squared_sparse(X: sparse.BCOO) -> sparse.BCOO:
    return sparse.BCOO(
        (jnp.square(X.data), X.indices),
        shape=X.shape,
        indices_sorted=True,
        unique_indices=True,
    )


def ser_update_sparse(X, y, offset, xi, prior_variance):
    X = _ensure_bcoo(X)
    X_sq = _squared_sparse(X)
    y = _ensure_array(y)
    offset = _ensure_array(offset)
    xi = _ensure_array(xi)

    lambd = lambda_xi(xi)
    v = X.T @ (y - 0.5 - 2.0 * offset * lambd)
    r = 2.0 * (X_sq.T @ lambd)
    var0 = 1.0 / r
    beta_hat = v * var0
    sigma2 = 1.0 / (1.0 / prior_variance + r)
    mu = sigma2 * v
    log_bf = 0.5 * (jnp.log(var0) - jnp.log(prior_variance + var0)) + 0.5 * (
        (beta_hat**2) / var0
    ) * prior_variance / (prior_variance + var0)
    lnalpha = log_bf - jax.nn.logsumexp(log_bf)
    alpha = jnp.exp(lnalpha)
    b_bar = alpha * mu
    b2_bar = alpha * (mu**2 + sigma2)
    p = X.shape[1]
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * (
        (sigma2 + mu**2) / prior_variance - jnp.log(sigma2 / prior_variance) - 1.0
    )
    kl = kl_cat + alpha @ kl_gauss
    return LogisticSER(
        np.asarray(mu),
        np.asarray(sigma2),
        np.asarray(alpha),
        np.asarray(b_bar),
        np.asarray(b2_bar),
        float(kl),
        float(prior_variance),
    )


def xi_update_sparse(X, q, intercept):
    X = _ensure_bcoo(X)
    X_sq = _squared_sparse(X)
    b_bar = jnp.stack([_ensure_array(ql.b_bar) for ql in q], axis=0)
    b2_bar = jnp.stack([_ensure_array(ql.b2_bar) for ql in q], axis=0)
    xb = jax.vmap(lambda w: X @ w)(b_bar)
    total_mu_pred = jnp.sum(xb, axis=0) + intercept
    var_contrib = jax.vmap(lambda w2, xbm: X_sq @ w2 - xbm**2)(b2_bar, xb)
    total_var_pred = jnp.sum(var_contrib, axis=0)
    return np.asarray(jnp.sqrt(jnp.maximum(total_mu_pred**2 + total_var_pred, 1e-12)))


def estimate_intercept_sparse(y, offset, xi):
    y = _ensure_array(y)
    offset = _ensure_array(offset)
    xi = _ensure_array(xi)
    lambd = lambda_xi(xi)
    num = jnp.sum(y - 0.5 - 2.0 * lambd * offset)
    den = 2.0 * jnp.sum(lambd)
    return float(num / den)


def elbo_sparse(X, y, q, xi, intercept):
    X = _ensure_bcoo(X)
    X_sq = _squared_sparse(X)
    y = _ensure_array(y)
    b_bar = jnp.stack([_ensure_array(ql.b_bar) for ql in q], axis=0)
    b2_bar = jnp.stack([_ensure_array(ql.b2_bar) for ql in q], axis=0)
    kl = jnp.asarray([ql.kl for ql in q], dtype=float)
    xb = jax.vmap(lambda w: X @ w)(b_bar)
    total_mu_pred = jnp.sum(xb, axis=0) + intercept
    var_contrib = jax.vmap(lambda w2, xbm: X_sq @ w2 - xbm**2)(b2_bar, xb)
    total_var_pred = jnp.sum(var_contrib, axis=0)
    expected_eta_sq = total_mu_pred**2 + total_var_pred
    lambd = lambda_xi(xi)
    ell = jnp.sum(
        (y - 0.5) * total_mu_pred
        - jnp.logaddexp(0.0, -xi)
        - xi / 2.0
        - lambd * (expected_eta_sq - xi**2)
    )
    return float(ell - jnp.sum(kl))


def logistic_susie_fit_sparse_global_jj(
    X,
    y,
    L,
    prior_variance=1.0,
    max_iter=10,
    estimate_prior=False,
    estimate_intercept_flag=True,
    tol=1e-4,
):
    del estimate_prior
    X = _ensure_bcoo(X)
    X_sq = _squared_sparse(X)
    y = _ensure_array(y)
    n, p = X.shape

    alpha = jnp.full((L, p), 1.0 / p)
    mu = jnp.zeros((L, p))
    var = jnp.full((L, p), prior_variance)
    b_bar = alpha * mu
    b2_bar = alpha * (mu**2 + var)
    xb = jax.vmap(lambda w: X @ w)(b_bar)
    var_contrib = jax.vmap(lambda w2, xbm: X_sq @ w2 - xbm**2)(b2_bar, xb)
    kl = jnp.zeros(L)
    prior_var = jnp.full(L, prior_variance)
    intercept = jnp.array(0.0)
    xi = jnp.ones(n)
    total_xb = jnp.sum(xb, axis=0)
    total_var_pred = jnp.sum(var_contrib, axis=0)
    q_public = _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_var)
    elbos = [elbo_sparse(X, y, q_public, xi, float(intercept))]

    converged = False
    for _ in range(max_iter):
        prev = elbos[-1]
        for l in range(L):
            loo_xb = total_xb - xb[l]
            if estimate_intercept_flag:
                intercept = jnp.asarray(estimate_intercept_sparse(y, loo_xb + xb[l], xi))
            ql = ser_update_sparse(X, y, loo_xb + intercept, xi, float(prior_var[l]))
            mu = mu.at[l].set(_ensure_array(ql.mu))
            var = var.at[l].set(_ensure_array(ql.var))
            alpha = alpha.at[l].set(_ensure_array(ql.alpha))
            b_bar = b_bar.at[l].set(_ensure_array(ql.b_bar))
            b2_bar = b2_bar.at[l].set(_ensure_array(ql.b2_bar))
            kl = kl.at[l].set(float(ql.kl))
            xb = xb.at[l].set(X @ b_bar[l])
            var_contrib = var_contrib.at[l].set(X_sq @ b2_bar[l] - xb[l] ** 2)
            total_xb = jnp.sum(xb, axis=0)
            total_var_pred = jnp.sum(var_contrib, axis=0)
            xi = jnp.sqrt(jnp.maximum((total_xb + intercept) ** 2 + total_var_pred, 1e-12))
            q_public = _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_var)
            elbos.append(elbo_sparse(X, y, q_public, xi, float(intercept)))
        if abs(elbos[-1] - prev) <= tol:
            converged = True
            break
    return (
        _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_var),
        np.asarray(xi),
        float(intercept),
        np.asarray(elbos),
        bool(converged),
    )
