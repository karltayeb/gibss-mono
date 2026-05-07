from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp as jsp_logsumexp

from .logistic import (
    LogisticSER,
    _ensure_array,
    _estimate_intercept_jax,
    _estimate_prior_variance_jax,
    _to_public_q,
    lambda_xi,
)


def _total_prediction_moments_jax(
    X: jnp.ndarray,
    b_bar: jnp.ndarray,
    b2_bar: jnp.ndarray,
    intercept: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    xb = b_bar @ X.T
    total_mu_pred = jnp.sum(xb, axis=0) + intercept
    total_var_pred = jnp.sum((X**2) @ b2_bar.T, axis=1) - jnp.sum(xb**2, axis=0)
    return total_mu_pred, jnp.maximum(total_var_pred, 0.0)


def xi_from_posterior(X, q, intercept):
    X = _ensure_array(X)
    b_bar = jnp.stack([_ensure_array(ql.b_bar) for ql in q], axis=0)
    b2_bar = jnp.stack([_ensure_array(ql.b2_bar) for ql in q], axis=0)
    total_mu_pred, total_var_pred = _total_prediction_moments_jax(
        X,
        b_bar,
        b2_bar,
        _ensure_array(intercept),
    )
    return np.asarray(jnp.sqrt(jnp.maximum(total_mu_pred**2 + total_var_pred, 1e-12)))


def elbo(X, y, q, intercept):
    X = _ensure_array(X)
    y = _ensure_array(y)
    b_bar = jnp.stack([_ensure_array(ql.b_bar) for ql in q], axis=0)
    b2_bar = jnp.stack([_ensure_array(ql.b2_bar) for ql in q], axis=0)
    kl = jnp.asarray([ql.kl for ql in q], dtype=float)

    total_mu_pred, total_var_pred = _total_prediction_moments_jax(
        X,
        b_bar,
        b2_bar,
        _ensure_array(intercept),
    )
    xi = jnp.sqrt(jnp.maximum(total_mu_pred**2 + total_var_pred, 1e-12))
    expected_eta_sq = total_mu_pred**2 + total_var_pred
    lambd = lambda_xi(xi)
    ell = jnp.sum(
        (y - 0.5) * total_mu_pred
        - jnp.logaddexp(0.0, -xi)
        - xi / 2.0
        - lambd * (expected_eta_sq - xi**2)
    )
    return float(ell - jnp.sum(kl))


def _univariate_elbo_tight(
    x: jnp.ndarray,
    y: jnp.ndarray,
    mu: jnp.ndarray,
    var: jnp.ndarray,
    offset_mean: jnp.ndarray,
    offset_var: jnp.ndarray,
    prior_variance: float,
) -> jnp.ndarray:
    expected_logits = x * mu + offset_mean
    expected_logits_sq = expected_logits**2 + x**2 * var + offset_var
    xi = jnp.sqrt(jnp.maximum(expected_logits_sq, 1e-12))
    lambd = lambda_xi(xi)
    expected_log_lik = jnp.sum(
        (y - 0.5) * expected_logits
        - jnp.logaddexp(0.0, -xi)
        - xi / 2.0
        - lambd * (expected_logits_sq - xi**2)
    )
    kl_gauss = 0.5 * (
        (var + mu**2) / prior_variance - jnp.log(var / prior_variance) - 1.0
    )
    return expected_log_lik - kl_gauss


_vectorized_univariate_elbo_tight = jax.jit(
    jax.vmap(_univariate_elbo_tight, in_axes=(1, None, 0, 0, None, None, None))
)


@jax.jit
def _local_jj_batched_step(
    X: jnp.ndarray,
    X_sq: jnp.ndarray,
    y_centered: jnp.ndarray,
    mu: jnp.ndarray,
    var: jnp.ndarray,
    offset_mean: jnp.ndarray,
    offset_mean_sq_plus_var: jnp.ndarray,
    prior_variance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    expected_logits_sq = (
        X_sq * (mu**2 + var)
        + 2.0 * X * mu * offset_mean[:, None]
        + offset_mean_sq_plus_var[:, None]
    )
    xi = jnp.sqrt(jnp.maximum(expected_logits_sq, 1e-12))
    tau = 2.0 * lambda_xi(xi)

    new_var = 1.0 / (1.0 / prior_variance + jnp.sum(tau * X_sq, axis=0))
    new_mu = new_var * jnp.sum(
        y_centered[:, None] * X - tau * X * offset_mean[:, None],
        axis=0,
    )
    return new_mu, new_var


@partial(jax.jit, static_argnames=("max_iter",))
def _vectorized_local_jj_update(
    X: jnp.ndarray,
    X_sq: jnp.ndarray,
    y: jnp.ndarray,
    mu_init: jnp.ndarray,
    var_init: jnp.ndarray,
    offset_mean: jnp.ndarray,
    offset_var: jnp.ndarray,
    prior_variance: float,
    max_iter: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y_centered = y - 0.5
    offset_mean_sq_plus_var = offset_mean**2 + offset_var

    def cond_fun(state):
        _, _, active, it = state
        return (it < max_iter) & jnp.any(active)

    def body_fun(state):
        mu, var, active, it = state
        candidate_mu, candidate_var = _local_jj_batched_step(
            X,
            X_sq,
            y_centered,
            mu,
            var,
            offset_mean,
            offset_mean_sq_plus_var,
            prior_variance,
        )
        new_mu = jnp.where(active, candidate_mu, mu)
        new_var = jnp.where(active, candidate_var, var)
        delta = jnp.maximum(jnp.abs(new_mu - mu), jnp.abs(new_var - var))
        new_active = active & (delta > 1e-6)
        return new_mu, new_var, new_active, it + 1

    init_state = (mu_init, var_init, jnp.ones_like(mu_init, dtype=bool), 0)
    final_mu, final_var, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
    final_elbos = _vectorized_univariate_elbo_tight(
        X,
        y,
        final_mu,
        final_var,
        offset_mean,
        offset_var,
        prior_variance,
    )
    return final_mu, final_var, final_elbos


def local_ser_update(
    X,
    y,
    offset_mean,
    offset_var,
    prior_variance,
    mu_init=None,
    var_init=None,
    max_inner_iter: int = 50,
):
    X = _ensure_array(X)
    y = _ensure_array(y)
    offset_mean = _ensure_array(offset_mean)
    offset_var = _ensure_array(offset_var)
    _, p = X.shape
    if mu_init is None:
        mu_init = jnp.zeros(p)
    else:
        mu_init = _ensure_array(mu_init)
    if var_init is None:
        var_init = jnp.full(p, prior_variance)
    else:
        var_init = _ensure_array(var_init)

    mu, var, log_bf = _vectorized_local_jj_update(
        X,
        X**2,
        y,
        mu_init,
        var_init,
        offset_mean,
        offset_var,
        float(prior_variance),
        max_inner_iter,
    )
    alpha = jnp.exp(log_bf - jsp_logsumexp(log_bf))
    b_bar = alpha * mu
    b2_bar = alpha * (mu**2 + var)
    p = X.shape[1]
    kl_cat = alpha @ jnp.log(jnp.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2) / prior_variance - jnp.log(var / prior_variance) - 1.0)
    kl = alpha @ kl_gauss + kl_cat
    return LogisticSER(
        np.asarray(mu),
        np.asarray(var),
        np.asarray(alpha),
        np.asarray(b_bar),
        np.asarray(b2_bar),
        float(kl),
        float(prior_variance),
    )


def logistic_susie_fit_local_jj(
    X,
    y,
    L,
    prior_variance=1.0,
    max_iter=10,
    tol=1e-4,
    estimate_prior=False,
    estimate_intercept_flag=True,
    inner_max_iter=50,
):
    X = _ensure_array(X)
    y = _ensure_array(y)
    x_sq = X**2

    @partial(
        jax.jit,
        static_argnames=("L", "max_iter", "estimate_prior", "estimate_intercept_flag", "inner_max_iter"),
    )
    def _run_fit(
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        L: int,
        max_iter: int,
        estimate_prior: bool,
        estimate_intercept_flag: bool,
        inner_max_iter: int,
    ):
        n, p = X.shape
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
        total_xb = jnp.sum(xb, axis=0)
        total_var_pred = jnp.sum(var_contrib, axis=0)

        def current_elbo(total_xb, total_var_pred, intercept, kl):
            total_mu_pred = total_xb + intercept
            xi = jnp.sqrt(jnp.maximum(total_mu_pred**2 + total_var_pred, 1e-12))
            expected_eta_sq = total_mu_pred**2 + total_var_pred
            lambd = lambda_xi(xi)
            ell = jnp.sum(
                (y - 0.5) * total_mu_pred
                - jnp.logaddexp(0.0, -xi)
                - xi / 2.0
                - lambd * (expected_eta_sq - xi**2)
            )
            return ell - jnp.sum(kl)

        init_elbo = current_elbo(total_xb, total_var_pred, intercept, kl)
        elbos = jnp.zeros(1 + max_iter * L)
        elbos = elbos.at[0].set(init_elbo)

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
            total_xb,
            total_var_pred,
            elbos,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            init_elbo,
        )

        def outer_body(i, state):
            def run_sweep(state):
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
                        total_xb,
                        total_var_pred,
                        elbos,
                        n_iter,
                        converged,
                        prev_sweep_elbo,
                    ) = state

                    loo_xb = total_xb - xb[l]
                    loo_var = jnp.maximum(total_var_pred - var_contrib[l], 0.0)
                    xi_total = jnp.sqrt(
                        jnp.maximum((total_xb + intercept) ** 2 + total_var_pred, 1e-12)
                    )
                    intercept = lax.cond(
                        estimate_intercept_flag,
                        lambda _: _estimate_intercept_jax(y, total_xb, xi_total),
                        lambda _: intercept,
                        operand=None,
                    )
                    offset_mean = loo_xb + intercept

                    mu_new, var_new, log_bf_new = _vectorized_local_jj_update(
                        X,
                        x_sq,
                        y,
                        mu[l],
                        var[l],
                        offset_mean,
                        loo_var,
                        prior_var[l],
                        inner_max_iter,
                    )
                    alpha_new = jnp.exp(log_bf_new - jsp_logsumexp(log_bf_new))
                    b_bar_new = alpha_new * mu_new
                    b2_bar_new = alpha_new * (mu_new**2 + var_new)
                    p = X.shape[1]
                    kl_cat = alpha_new @ jnp.log(jnp.maximum(alpha_new * p, 1e-15))
                    kl_gauss = 0.5 * (
                        (var_new + mu_new**2) / prior_var[l]
                        - jnp.log(var_new / prior_var[l])
                        - 1.0
                    )
                    kl_new = alpha_new @ kl_gauss + kl_cat

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
                    total_var_pred = jnp.maximum(
                        total_var_pred - var_contrib[l] + var_contrib_new,
                        0.0,
                    )

                    mu = mu.at[l].set(mu_new)
                    var = var.at[l].set(var_new)
                    alpha = alpha.at[l].set(alpha_new)
                    b_bar = b_bar.at[l].set(b_bar_new)
                    b2_bar = b2_bar.at[l].set(b2_bar_new)
                    xb = xb.at[l].set(xb_new)
                    var_contrib = var_contrib.at[l].set(var_contrib_new)
                    kl = kl.at[l].set(kl_new)
                    prior_var = prior_var.at[l].set(prior_var_new)

                    elbo_val = current_elbo(total_xb, total_var_pred, intercept, kl)
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
                        total_xb,
                        total_var_pred,
                        elbos,
                        n_iter,
                        converged,
                        prev_sweep_elbo,
                    )

                state = lax.fori_loop(0, L, inner_body, state)
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
                    total_xb,
                    total_var_pred,
                    elbos,
                    _n_iter,
                    _converged,
                    prev_sweep_elbo,
                ) = state
                sweep_elbo = current_elbo(total_xb, total_var_pred, intercept, kl)
                diff = sweep_elbo - prev_sweep_elbo
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
                    total_xb,
                    total_var_pred,
                    elbos,
                    jnp.asarray(i + 1, dtype=jnp.int32),
                    jnp.abs(diff) <= tol,
                    sweep_elbo,
                )

            return lax.cond(state[14], lambda s: s, run_sweep, state)

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
        _total_xb,
        _total_var_pred,
        elbos,
        n_iter,
        converged,
        _prev_sweep_elbo,
    ) = _run_fit(
        X,
        y,
        L=L,
        max_iter=max_iter,
        estimate_prior=estimate_prior,
        estimate_intercept_flag=estimate_intercept_flag,
        inner_max_iter=inner_max_iter,
    )

    q = _to_public_q(mu, var, alpha, b_bar, b2_bar, kl, prior_var)
    used = 1 + int(n_iter) * max(L, 1)
    return (
        q,
        float(intercept),
        np.maximum.accumulate(np.asarray(elbos[:used])),
        bool(converged),
    )


__all__ = [
    "elbo",
    "local_ser_update",
    "logistic_susie_fit_local_jj",
    "xi_from_posterior",
]
