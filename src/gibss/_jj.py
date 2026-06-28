"""Shared Jaakkola-Jordan primitives for the JJ logistic SER families
(globaljj, localjj).

`lambda_xi` and the null-model bound were duplicated verbatim across both
modules. The null model (beta = 0, eta = offset) is **method-independent** -- it
doesn't involve the candidate feature -- so global and local JJ should (and now
do) score it identically: with a **null-tuned** xi = sqrt(E[eta^2]). At that xi
the JJ bound touches the true log-likelihood (the lambda term vanishes), so it is
the tight null. Using a different xi for the null (e.g. globaljj's old global xi,
tuned to the full strong eta) makes the bound loose at the null and inflates the
Bayes factor.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@jax.jit
def lambda_xi(xi: Any) -> Any:
    """Stable Jaakkola-Jordan lambda(xi) with small-xi Taylor handling."""
    xi = jnp.abs(jnp.asarray(xi))
    small = xi < 1e-6
    taylor = 0.125 - (xi**2) / 192.0
    safe_xi = jnp.where(small, 1.0, xi)
    stable = jnp.tanh(safe_xi / 2.0) / (4.0 * safe_xi)
    return jnp.where(small, taylor, stable)


@jax.jit
def jj_bound_null_log_likelihood(y, offset, xi, offset_var=None) -> Any:
    """JJ lower bound on the intercept-only (beta=0) log-likelihood at given xi."""
    eta_sq = jnp.square(offset)
    if offset_var is not None:
        eta_sq = eta_sq + offset_var
    l_xi = lambda_xi(xi)
    return jnp.sum(
        (y - 0.5) * offset
        - l_xi * (eta_sq - jnp.square(xi))
        - jnp.logaddexp(0.0, xi)
        + 0.5 * xi
    )


def null_tuned_xi(offset, offset_var=None) -> Any:
    """xi tuned to the null predictor: xi^2 = E[eta^2] = offset^2 + var(offset)."""
    eta_sq = jnp.square(jnp.asarray(offset))
    if offset_var is not None:
        eta_sq = eta_sq + jnp.asarray(offset_var)
    return jnp.sqrt(jnp.maximum(eta_sq, 1e-12))


@jax.jit
def jj_null_log_likelihood(y, offset, offset_var=None) -> Any:
    """Tight JJ null log-likelihood: the bound at the null-tuned xi.

    This is the method-independent BF denominator for both global and local JJ.
    At xi^2 = E[eta^2] the lambda term is zero, so this is the tightest JJ null.
    """
    xi = null_tuned_xi(offset, offset_var)
    return jj_bound_null_log_likelihood(y, offset, xi, offset_var)


@partial(jax.jit, static_argnames=("n_iter",))
def jj_profiled_null_log_likelihood(y, offset, offset_var=None, n_iter: int = 25) -> Any:
    """Intercept-PROFILED JJ null: max over (b0, xi) of the bound at eta = offset + b0.

    Required when the effect model profiles a per-feature intercept (e.g. centered
    localjj): the BF must profile the intercept in BOTH numerator and denominator
    (offset-shift invariance). Using the plain null (no intercept) credits the
    feature for the intercept fit and inflates the BF.
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    ov = jnp.zeros_like(offset) if offset_var is None else jnp.asarray(offset_var)

    def body(state):
        b0, it = state
        xi = jnp.sqrt(jnp.maximum((offset + b0) ** 2 + ov, 1e-12))
        tau = 2.0 * lambda_xi(xi)
        return jnp.sum(y - 0.5 - tau * offset) / jnp.sum(tau), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    eta = offset + b0
    return jj_bound_null_log_likelihood(y, eta, null_tuned_xi(eta, ov), ov)


__all__ = [
    "lambda_xi",
    "jj_bound_null_log_likelihood",
    "null_tuned_xi",
    "jj_null_log_likelihood",
    "jj_profiled_null_log_likelihood",
]
