from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp as jsp_logsumexp
from jax.scipy.stats import norm

LinearSER = namedtuple(
    "LinearSER",
    ["mu", "var", "alpha", "mu_predict", "var_predict", "kl", "prior_variance"],
)


def logsumexp(x):
    return jsp_logsumexp(_ensure_array(x))


def softplus(x):
    return jnp.exp(x - jsp_logsumexp(x))


def _ensure_array(x):
    return jnp.asarray(x, dtype=float)


def _stack_q(q, field: str) -> jnp.ndarray:
    return jnp.stack([_ensure_array(getattr(ql, field)) for ql in q], axis=0)


def _stack_q_scalar(q, field: str) -> jnp.ndarray:
    return jnp.asarray([getattr(ql, field) for ql in q], dtype=float)


def _linear_ser_update_jax(X, y, residual_variance, prior_variance):
    tau0 = 1.0 / prior_variance
    d_sq = jnp.sum(X**2, axis=0)
    tau = d_sq / residual_variance
    bhat = (y @ X) / d_sq
    var = 1.0 / (tau + tau0)
    mu = tau * var * bhat

    var_bhat = 1.0 / tau
    lbf = norm.logpdf(bhat, 0.0, jnp.sqrt(prior_variance + var_bhat)) - norm.logpdf(
        bhat, 0.0, jnp.sqrt(var_bhat)
    )
    alpha = softplus(lbf)

    mu_predict = X @ (mu * alpha)
    var_predict = (X**2) @ (alpha * (mu**2 + var)) - mu_predict**2

    p = X.shape[1]
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2) / prior_variance - jnp.log(var / prior_variance) - 1.0)
    kl = alpha @ kl_gauss + kl_cat
    return mu, var, alpha, mu_predict, var_predict, kl


def ser_update(X, y, residual_variance, prior_variance):
    mu, var, alpha, mu_predict, var_predict, kl = _linear_ser_update_jax(
        _ensure_array(X),
        _ensure_array(y),
        float(residual_variance),
        float(prior_variance),
    )
    return LinearSER(
        np.asarray(mu),
        np.asarray(var),
        np.asarray(alpha),
        np.asarray(mu_predict),
        np.asarray(var_predict),
        float(kl),
        float(prior_variance),
    )


def elbo(y, q, intercept, residual_variance):
    y = _ensure_array(y)
    mu_predict = jnp.sum(_stack_q(q, "mu_predict"), axis=0) + intercept
    var_predict = jnp.sum(_stack_q(q, "var_predict"), axis=0)
    erss = jnp.sum((y - mu_predict) ** 2) + jnp.sum(var_predict)
    kl = jnp.sum(_stack_q_scalar(q, "kl"))
    n = y.size
    value = (
        -0.5 * n * jnp.log(2.0 * jnp.pi * residual_variance)
        - 0.5 / residual_variance * erss
        - kl
    )
    return float(value)


def estimate_residual_variance(y, q, intercept):
    y = _ensure_array(y)
    n = y.size
    mu_predict = jnp.sum(_stack_q(q, "mu_predict"), axis=0) + intercept
    var_predict = jnp.sum(_stack_q(q, "var_predict"), axis=0)
    erss = jnp.sum((y - mu_predict) ** 2) + jnp.sum(var_predict)
    return float(jnp.maximum(erss / n, 1e-8))


def estimate_intercept(y, q):
    y = _ensure_array(y)
    total_mu_pred = jnp.sum(_stack_q(q, "mu_predict"), axis=0)
    return float(jnp.mean(y - total_mu_pred))


def _estimate_prior_variance_jax(alpha, mu, var):
    prior_var_est = jnp.maximum(alpha @ (var + mu**2), 1e-8)
    p = alpha.size
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2) / prior_var_est - jnp.log(var / prior_var_est) - 1.0)
    kl = alpha @ kl_gauss + kl_cat
    return prior_var_est, kl


def estimate_prior_variance(q: LinearSER):
    prior_var_est, kl = _estimate_prior_variance_jax(
        _ensure_array(q.alpha),
        _ensure_array(q.mu),
        _ensure_array(q.var),
    )
    return LinearSER(
        np.asarray(q.mu),
        np.asarray(q.var),
        np.asarray(q.alpha),
        np.asarray(q.mu_predict),
        np.asarray(q.var_predict),
        float(kl),
        float(prior_var_est),
    )


def _to_public_q(mu, var, alpha, mu_predict, var_predict, kl, prior_variance):
    q = []
    for l in range(mu.shape[0]):
        q.append(
            LinearSER(
                np.asarray(mu[l]),
                np.asarray(var[l]),
                np.asarray(alpha[l]),
                np.asarray(mu_predict[l]),
                np.asarray(var_predict[l]),
                float(kl[l]),
                float(prior_variance[l]),
            )
        )
    return q


def susie_fit(
    X,
    y,
    L,
    residual_variance=1.0,
    prior_variance=1.0,
    max_iter=10,
    estimate_residual=False,
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
        mu_predict = jnp.zeros((L, n))
        var_init = x_sq @ (alpha[0] * (mu[0] ** 2 + var[0]))
        var_predict = jnp.broadcast_to(var_init, (L, n))
        kl = jnp.zeros(L)
        prior_var = jnp.full(L, prior_variance)
        intercept = jnp.array(0.0)
        res_var = jnp.array(residual_variance)
        total_mu_predict = jnp.sum(mu_predict, axis=0)
        total_var_predict = jnp.sum(var_predict, axis=0)
        elbos = jnp.zeros(1 + max_iter * L)

        def current_elbo(total_mu_predict, total_var_predict, intercept, res_var, kl):
            erss = jnp.sum((y - (total_mu_predict + intercept)) ** 2) + jnp.sum(total_var_predict)
            return (
                -0.5 * n * jnp.log(2.0 * jnp.pi * res_var)
                - 0.5 / res_var * erss
                - jnp.sum(kl)
            )

        elbos = elbos.at[0].set(
            current_elbo(total_mu_predict, total_var_predict, intercept, res_var, kl)
        )

        state = (
            mu,
            var,
            alpha,
            mu_predict,
            var_predict,
            kl,
            prior_var,
            total_mu_predict,
            total_var_predict,
            intercept,
            res_var,
            elbos,
        )

        def outer_body(i, state):
            def inner_body(l, state):
                (
                    mu,
                    var,
                    alpha,
                    mu_predict,
                    var_predict,
                    kl,
                    prior_var,
                    total_mu_predict,
                    total_var_predict,
                    intercept,
                    res_var,
                    elbos,
                ) = state

                intercept = lax.cond(
                    estimate_intercept_flag,
                    lambda _: jnp.mean(y - total_mu_predict),
                    lambda _: intercept,
                    operand=None,
                )

                mu_excl = total_mu_predict - mu_predict[l]
                offset = mu_excl + intercept
                (
                    mu_new,
                    var_new,
                    alpha_new,
                    mu_predict_new,
                    var_predict_new,
                    kl_new,
                ) = _linear_ser_update_jax(X, y - offset, res_var, prior_var[l])

                def _update_prior(_):
                    return _estimate_prior_variance_jax(alpha_new, mu_new, var_new)

                prior_var_new, kl_new = lax.cond(
                    estimate_prior,
                    _update_prior,
                    lambda _: (prior_var[l], kl_new),
                    operand=None,
                )

                total_mu_predict = total_mu_predict - mu_predict[l] + mu_predict_new
                total_var_predict = total_var_predict - var_predict[l] + var_predict_new

                mu = mu.at[l].set(mu_new)
                var = var.at[l].set(var_new)
                alpha = alpha.at[l].set(alpha_new)
                mu_predict = mu_predict.at[l].set(mu_predict_new)
                var_predict = var_predict.at[l].set(var_predict_new)
                kl = kl.at[l].set(kl_new)
                prior_var = prior_var.at[l].set(prior_var_new)

                res_var = lax.cond(
                    estimate_residual,
                    lambda _: jnp.maximum(
                        (
                            jnp.sum((y - (total_mu_predict + intercept)) ** 2)
                            + jnp.sum(total_var_predict)
                        )
                        / n,
                        1e-8,
                    ),
                    lambda _: res_var,
                    operand=None,
                )

                elbo_val = current_elbo(
                    total_mu_predict, total_var_predict, intercept, res_var, kl
                )
                elbos = elbos.at[1 + l + i * L].set(elbo_val)

                return (
                    mu,
                    var,
                    alpha,
                    mu_predict,
                    var_predict,
                    kl,
                    prior_var,
                    total_mu_predict,
                    total_var_predict,
                    intercept,
                    res_var,
                    elbos,
                )

            return lax.fori_loop(0, L, inner_body, state)

        return lax.fori_loop(0, max_iter, outer_body, state)

    (
        mu,
        var,
        alpha,
        mu_predict,
        var_predict,
        kl,
        prior_var,
        _total_mu_predict,
        _total_var_predict,
        intercept,
        res_var,
        elbos,
    ) = jax.jit(_run_fit)(X, y)

    q = _to_public_q(mu, var, alpha, mu_predict, var_predict, kl, prior_var)
    return q, float(intercept), np.maximum.accumulate(np.asarray(elbos)), float(res_var)
