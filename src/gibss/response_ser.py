"""Response-model per-column SER kernel.

One kernel for every family: per-feature MAP (Newton/MM on the response's working
residual + curvature) + Laplace scale + a Gauss-Hermite tail over the effect `b`.
The family enters only through a `ResponseModel.terms(eta, aux) -> (loglik, grad,
weight)`. `order=1` recovers the Laplace evidence.

This is the generic form of the logistic `quadrature_ser`: with `Bernoulli` it
reproduces it (fixed offset). Offset integration is a separate elaboration.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .engine import BaseSERState
from .operators import DesignOperator
from .response import Bernoulli, ResponseModel
from .ser_ops import _gh_rule, _normal_logpdf

__all__ = ["glm_ser", "build_ser_state"]


def build_ser_state(mu, var, feature_log_evidence, coefficient_kl, prior_variance,
                    null_log_likelihood=0.0):
    """Assemble a `BaseSERState` from per-feature glm_ser outputs.

    `feature_log_evidence` is the per-feature log-evidence (log-BF up to a shared,
    feature-independent baseline that cancels in `alpha`). Shared by every family
    built on `glm_ser`; only how the evidence baseline / null are defined differs.
    """
    p = feature_log_evidence.shape[0]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
               + jnp.sum(alpha * coefficient_kl))
    return BaseSERState(
        mu=mu, var=var, alpha=alpha, pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_log_likelihood), kl=kl,
    )


@partial(jax.jit, static_argnames=("order", "n_iter", "response"))
def glm_ser(
    op: DesignOperator,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    order: int = 15,
    n_iter: int = 50,
    tol: float = 1e-8,
):
    """Per-column SER for an arbitrary response. `aux` is per-observation auxiliary
    data (y for Bernoulli/Poisson, llr for the two-group marginal).

    Returns per-feature (mu, var, feature_log_bf, coefficient_kl), the log-BF
    relative to the b=0 fit at `offset` (what alpha uses).
    """
    aux = jnp.asarray(aux)
    offset = jnp.asarray(offset)
    aux_e = op.broadcast_rows(aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def body(state):
        b, _, it = state
        eta = off_e + op.column_linpred(b)
        _, grad_i, w_i = response.terms(eta, aux_e)
        grad = op.local_moment(1, grad_i) - inv_pv * b
        curv = op.local_moment(2, w_i) + inv_pv  # MM/Fisher curvature -> monotone
        step = grad / curv
        return b + step, jnp.max(jnp.abs(step)), it + 1

    b, _, _ = jax.lax.while_loop(
        lambda s: (s[2] < n_iter) & (s[1] > tol), body, (jnp.zeros(op.p), jnp.inf, 0)
    )
    b = jnp.where(jnp.isfinite(b), b, 0.0)

    eta = off_e + op.column_linpred(b)
    _, _, w = response.terms(eta, aux_e)
    precision = op.local_moment(2, w) + inv_pv
    sigma = jnp.sqrt(1.0 / precision)

    nodes_np, log_w_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    log_w = jnp.asarray(log_w_np)
    ll0 = response.terms(off_e, aux_e)[0]  # b=0 baseline (per entry)

    def node_term(node, lw):
        bb = b + jnp.sqrt(2.0) * sigma * node
        eta = off_e + op.column_linpred(bb)
        ll = response.terms(eta, aux_e)[0]
        dll = op.local_moment(0, ll - ll0)  # (p,) loglik diff vs b=0
        logint = lw + node**2 + dll + _normal_logpdf(bb, prior_variance) + jnp.log(
            jnp.sqrt(2.0) * sigma
        )
        return logint, bb, dll

    logint, b_nodes, dll_nodes = jax.vmap(node_term)(nodes, log_w)  # (order, p) x3
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm
    return mu, var, log_norm, coefficient_kl
