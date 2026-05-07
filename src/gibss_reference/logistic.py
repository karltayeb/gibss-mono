from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp as jsp_logsumexp

LogisticSER = namedtuple(
    "LogisticSER",
    ["mu", "var", "alpha", "b_bar", "b2_bar", "kl", "prior_variance"],
)


def _ensure_array(x):
    return jnp.asarray(x, dtype=float)


def _stack_q(q, field: str) -> jnp.ndarray:
    return jnp.stack([_ensure_array(getattr(ql, field)) for ql in q], axis=0)


def _stack_q_scalar(q, field: str) -> jnp.ndarray:
    return jnp.asarray([getattr(ql, field) for ql in q], dtype=float)


def logsumexp(x):
    return jsp_logsumexp(_ensure_array(x))


def softplus(x):
    return jnp.exp(x - jsp_logsumexp(x))


def lambda_xi(xi):
    xi = _ensure_array(xi)
    small = jnp.abs(xi) < 1e-6
    taylor = 0.25 - (xi**2) / 48.0
    stable = jnp.tanh(xi / 2.0) / (4.0 * xi)
    return jnp.where(small, taylor, stable)


def ser_ss(v, r, sigma02):
    s2 = 1.0 / r
    beta_hat = v * s2
    z2 = (beta_hat**2) / s2
    sigma2 = 1.0 / (1.0 / sigma02 + 1.0 / s2)
    mu = sigma2 * v
    log_bf = 0.5 * (jnp.log(s2) - jnp.log(sigma02 + s2)) + 0.5 * z2 * sigma02 / (
        sigma02 + s2
    )
    alpha = softplus(log_bf)
    return alpha, mu, sigma2


def _ser_update_jax(X, y, offset, xi, prior_variance):
    lambd = lambda_xi(xi)
    v = (y - 0.5 - 2.0 * offset * lambd) @ X
    r = 2.0 * (lambd @ (X**2))

    alpha, mu, var = ser_ss(v, r, prior_variance)
    b_bar = alpha * mu
    b2_bar = alpha * (mu**2 + var)

    p = X.shape[1]
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2) / prior_variance - jnp.log(var / prior_variance) - 1.0)
    kl = alpha @ kl_gauss + kl_cat
    return mu, var, alpha, b_bar, b2_bar, kl


def ser_update(X, y, offset, xi, prior_variance):
    mu, var, alpha, b_bar, b2_bar, kl = _ser_update_jax(
        _ensure_array(X),
        _ensure_array(y),
        _ensure_array(offset),
        _ensure_array(xi),
        float(prior_variance),
    )
    return LogisticSER(
        np.asarray(mu),
        np.asarray(var),
        np.asarray(alpha),
        np.asarray(b_bar),
        np.asarray(b2_bar),
        float(kl),
        float(prior_variance),
    )


def xi_update(X, q, intercept):
    X = _ensure_array(X)
    b_bar = _stack_q(q, "b_bar")
    b2_bar = _stack_q(q, "b2_bar")
    xb = b_bar @ X.T
    total_mu_pred = jnp.sum(xb, axis=0) + intercept
    total_var_pred = jnp.sum((X**2) @ b2_bar.T, axis=1) - jnp.sum(xb**2, axis=0)
    return np.asarray(jnp.sqrt(jnp.maximum(total_mu_pred**2 + total_var_pred, 1e-12)))


def elbo(X, y, q, xi, intercept):
    X = _ensure_array(X)
    y = _ensure_array(y)
    b_bar = _stack_q(q, "b_bar")
    b2_bar = _stack_q(q, "b2_bar")
    kl = _stack_q_scalar(q, "kl")

    xb = b_bar @ X.T
    total_mu_pred = jnp.sum(xb, axis=0) + intercept
    total_var_pred = jnp.sum((X**2) @ b2_bar.T, axis=1) - jnp.sum(xb**2, axis=0)
    expected_eta_sq = total_mu_pred**2 + total_var_pred
    lambd = lambda_xi(xi)
    ell = jnp.sum(
        (y - 0.5) * total_mu_pred
        - jnp.logaddexp(0.0, -xi)
        - xi / 2.0
        - lambd * (expected_eta_sq - xi**2)
    )
    return float(ell - jnp.sum(kl))


def _estimate_prior_variance_jax(alpha, mu, var):
    prior_var_est = jnp.maximum(alpha @ (var + mu**2), 1e-8)
    p = alpha.size
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2) / prior_var_est - jnp.log(var / prior_var_est) - 1.0)
    kl = alpha @ kl_gauss + kl_cat
    return prior_var_est, kl


def estimate_prior_variance(q: LogisticSER):
    prior_var_est, kl = _estimate_prior_variance_jax(
        _ensure_array(q.alpha),
        _ensure_array(q.mu),
        _ensure_array(q.var),
    )
    return LogisticSER(
        np.asarray(q.mu),
        np.asarray(q.var),
        np.asarray(q.alpha),
        np.asarray(q.b_bar),
        np.asarray(q.b2_bar),
        float(kl),
        float(prior_var_est),
    )


def _estimate_intercept_jax(y, offset, xi):
    lambd = lambda_xi(xi)
    num = jnp.sum(y - 0.5 - 2.0 * lambd * offset)
    den = 2.0 * jnp.sum(lambd)
    return num / den


def estimate_intercept(y, offset, xi):
    y = _ensure_array(y)
    offset = _ensure_array(offset)
    xi = _ensure_array(xi)
    return float(_estimate_intercept_jax(y, offset, xi))


def _solve_global_jj_intercept_xi(y, total_mean, total_var, intercept):
    y = _ensure_array(y)
    total_mean = _ensure_array(total_mean)
    total_var = _ensure_array(total_var)
    intercept = jnp.asarray(intercept, dtype=float)

    def body_fun(state):
        intercept, _xi, _it = state
        xi = jnp.sqrt(
            jnp.maximum(jnp.square(total_mean + intercept) + total_var, 1e-12)
        )
        intercept_new = _estimate_intercept_jax(y, total_mean, xi)
        return intercept_new, xi, _it + 1

    xi0 = jnp.sqrt(jnp.maximum(jnp.square(total_mean + intercept) + total_var, 1e-12))
    intercept, xi, _ = lax.fori_loop(
        0,
        10,
        lambda _i, state: body_fun(state),
        (intercept, xi0, 0),
    )
    xi = jnp.sqrt(jnp.maximum(jnp.square(total_mean + intercept) + total_var, 1e-12))
    return intercept, xi


def _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_variance):
    q = []
    for l in range(mu.shape[0]):
        q.append(
            LogisticSER(
                np.asarray(mu[l]),
                np.asarray(var[l]),
                np.asarray(alpha[l]),
                np.asarray(b_bar[l]),
                np.asarray(b2_bar[l]),
                float(kl[l]),
                float(prior_variance[l]),
            )
        )
    return q


def logistic_susie_fit(
    X,
    y,
    L,
    prior_variance=1.0,
    max_iter=10,
    estimate_prior=False,
    estimate_intercept_flag=True,
):
    X = _ensure_array(X)
    y = _ensure_array(y)
    n, p = X.shape
    x_sq = X**2

    def _run_fit(X, y):
        alpha = jnp.full((L, p), 1.0 / p)
        mu = jnp.zeros((L, p))
        var = jnp.full((L, p), prior_variance)
        b_bar = alpha * mu
        b2_bar = alpha * (mu**2 + var)
        xb = b_bar @ X.T
        var_contrib = (x_sq @ b2_bar.T).T - xb**2
        kl = jnp.zeros(L)
        prior_var = jnp.full(L, prior_variance)
        intercept = jnp.array(0.0)
        xi = jnp.ones(n)
        total_xb = jnp.sum(xb, axis=0)
        total_var_pred = jnp.sum(var_contrib, axis=0)
        elbos = jnp.zeros(1 + max_iter * L)

        def current_elbo(total_xb, total_var_pred, xi, intercept, kl):
            total_mu_pred = total_xb + intercept
            expected_eta_sq = total_mu_pred**2 + total_var_pred
            lambd = lambda_xi(xi)
            ell = jnp.sum(
                (y - 0.5) * total_mu_pred
                - jnp.logaddexp(0.0, -xi)
                - xi / 2.0
                - lambd * (expected_eta_sq - xi**2)
            )
            return ell - jnp.sum(kl)

        elbos = elbos.at[0].set(
            current_elbo(total_xb, total_var_pred, xi, intercept, kl)
        )

        state = (
            mu,
            var,
            alpha,
            b_bar,
            b2_bar,
            xb,
            var_contrib,
            kl,
            prior_var,
            intercept,
            xi,
            total_xb,
            total_var_pred,
            elbos,
        )

        def outer_body(i, state):
            def inner_body(l, state):
                (
                    mu,
                    var,
                    alpha,
                    b_bar,
                    b2_bar,
                    xb,
                    var_contrib,
                    kl,
                    prior_var,
                    intercept,
                    xi,
                    total_xb,
                    total_var_pred,
                    elbos,
                ) = state

                loo_xb = total_xb - xb[l]
                intercept, xi = lax.cond(
                    estimate_intercept_flag,
                    lambda _: _solve_global_jj_intercept_xi(
                        y, total_xb, total_var_pred, intercept
                    ),
                    lambda _: (
                        intercept,
                        jnp.sqrt(
                            jnp.maximum(
                                (total_xb + intercept) ** 2 + total_var_pred, 1e-12
                            )
                        ),
                    ),
                    operand=None,
                )
                offset = loo_xb + intercept

                (
                    mu_new,
                    var_new,
                    alpha_new,
                    b_bar_new,
                    b2_bar_new,
                    kl_new,
                ) = _ser_update_jax(X, y, offset, xi, prior_var[l])

                def _update_prior(_):
                    return _estimate_prior_variance_jax(alpha_new, mu_new, var_new)

                prior_var_new, kl_new = lax.cond(
                    estimate_prior,
                    _update_prior,
                    lambda _: (prior_var[l], kl_new),
                    operand=None,
                )

                xb_new = X @ b_bar_new
                var_contrib_new = x_sq @ b2_bar_new - xb_new**2

                total_xb = total_xb - xb[l] + xb_new
                total_var_pred = total_var_pred - var_contrib[l] + var_contrib_new

                mu = mu.at[l].set(mu_new)
                var = var.at[l].set(var_new)
                alpha = alpha.at[l].set(alpha_new)
                b_bar = b_bar.at[l].set(b_bar_new)
                b2_bar = b2_bar.at[l].set(b2_bar_new)
                xb = xb.at[l].set(xb_new)
                var_contrib = var_contrib.at[l].set(var_contrib_new)
                kl = kl.at[l].set(kl_new)
                prior_var = prior_var.at[l].set(prior_var_new)

                xi = jnp.sqrt(jnp.maximum((total_xb + intercept) ** 2 + total_var_pred, 1e-12))
                elbo_val = current_elbo(total_xb, total_var_pred, xi, intercept, kl)
                elbos = elbos.at[1 + l + i * L].set(elbo_val)

                return (
                    mu,
                    var,
                    alpha,
                    b_bar,
                    b2_bar,
                    xb,
                    var_contrib,
                    kl,
                    prior_var,
                    intercept,
                    xi,
                    total_xb,
                    total_var_pred,
                    elbos,
                )

            return lax.fori_loop(0, L, inner_body, state)

        return lax.fori_loop(0, max_iter, outer_body, state)

    (
        mu,
        var,
        alpha,
        b_bar,
        b2_bar,
        _xb,
        _var_contrib,
        kl,
        prior_var,
        intercept,
        xi,
        _total_xb,
        _total_var_pred,
        elbos,
    ) = jax.jit(_run_fit)(X, y)

    q = _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_var)
    return q, np.asarray(xi), float(intercept), np.maximum.accumulate(np.asarray(elbos))
