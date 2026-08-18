r"""Analytic offset-integrated cumulant for the Poisson (log-link) base -- CAVI in Q2.

For a log-link Poisson the cumulant is `A(eta) = exp(eta)`, so the offset-integrated
cumulant the Q2 effect update needs,

    Atilde(eta) = E_{o}[A(eta + o)] = e^{eta} * E[e^{o}],

is CLOSED FORM: `E[e^{o}]` is just the moment-generating function of the offset law at
`t = 1`. Under the "offset mean lives in eta" convention the mean `s = E[o]` is carried
in the predictor, so we integrate the ZERO-MEAN fluctuation `o0 = o - s` and

    Atilde(eta) = e^{eta} * E[e^{o0}] = e^{eta + logkappa},   logkappa = log E[e^{o0}],

a single per-row scalar `logkappa >= 0` (Jensen). All z-derivatives coincide:
`Atilde = Atilde' = Atilde'' = e^{eta + logkappa}`. There is NO quadrature grid and NO
Chebyshev table -- this is the exact analogue of `cf_offset.CharFnOffset` for the logistic
base, but where that base needs the psi-tamed residual CF quadrature (the logistic density
is not `e^{eta}`), the Poisson MGF is elementary. The MGF is the characteristic function
`phi` evaluated at the single imaginary argument `t = -i`, so `logkappa` is `offset_cf`'s
zero-mean product collapsed to one point -- exact, and cheaper than the logistic path.

Effect space, mean-field product
--------------------------------
In Q2 every other effect `l` is a Gaussian mixture over its selected feature `c`:
`q(b_l | c) = N(mu_lc, var_lc)` with selection weight `alpha_lc`, contributing
`o_li = x_ic b` to row `i`. Under the mean-field factorization the offset MGF is the
PRODUCT over effects (independence), and each effect's MGF is a closed-form mixture of
Gaussian MGFs:

    E[e^{o_li}] = sum_c alpha_lc exp(x_ic mu_lc + 1/2 x_ic^2 var_lc),   (log-normal MGF)
    E[o_li]     = sum_c alpha_lc x_ic mu_lc                            (the mean, in eta)

so the per-effect ZERO-MEAN log contribution is `logsumexp_c(...) - E[o_li]` and

    logkappa_i = sum_{l != j} [ logsumexp_c(log alpha_lc + x_ic mu_lc + 1/2 x_ic^2 var_lc)
                                - E[o_li] ]  +  1/2 * offset_var.

The trailing `offset_var` folds one extra homogeneous zero-mean Gaussian `N(0, offset_var)`
-- the shared intercept's variational factor `q(b0) = N(m0, v0)` on the all-ones column
(its mean `m0` lives in eta) -- whose zero-mean MGF is `exp(v0/2)`, i.e. `+ v0/2` in log.
An unfit effect (`mu = var = 0`, `alpha = 1/p`) contributes `logsumexp(log 1/p) - 0 = 0`,
so empty effects drop out for free (the neutral MGF factor `1`).

`logkappa` is ROW-only (the OTHER effects are the same for every candidate feature of the
effect being updated), so it broadcasts over the design columns exactly like `y` -- the
`vi_gh` kernel consumes `aux = (y, logkappa)` and integrates `b` by Gauss-Hermite over the
per-row cumulant `terms(offset + x b, aux) = (y*eta - Atilde, y - Atilde', Atilde'')`.
Requires x64 for the MGF exponents; gated to the `Poisson` base.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jax.ops import segment_sum
from jax.scipy.special import logsumexp

from .operators import BCOOOperator
from .response import Poisson, ResponseModel, Smoother

__all__ = [
    "log_kappa_dense",
    "log_kappa_sparse",
    "PoissonLogNormalOffset",
]


def _effect_log_mgf_dense(x, alpha, mu, var):
    """Zero-mean log-MGF `logsumexp_c(log alpha_c + x_ic mu_c + 1/2 x_ic^2 var_c) - mean_i`
    of one effect's row contribution `o_li = x_ic b`, `(n,)`. `x` is (n, C); `alpha, mu,
    var` are the effect-space law (C,). The subtraction removes the mean carried in eta."""
    x = jnp.asarray(x)
    alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
    mean = x @ (alpha * mu)  # (n,) = E[o_li]
    # log alpha_c + x_ic mu_c + 1/2 x_ic^2 var_c, reduced over c by logsumexp (stable).
    terms = jnp.log(alpha)[None, :] + x * mu[None, :] + 0.5 * (x**2) * var[None, :]
    return logsumexp(terms, axis=1) - mean


def log_kappa_dense(effects, n, offset_var=0.0):
    """Per-row `logkappa` (n,) for a dense design: sum of the per-effect zero-mean
    log-MGFs plus the intercept factor `1/2 * offset_var`. `effects` is a list of
    `(x, alpha, mu, var)` (design block + effect-space law), the same effect-space
    representation `cf_offset.offset_cf` consumes; empty effects contribute 0."""
    logk = jnp.zeros(n) + 0.5 * jnp.asarray(offset_var)
    for x, alpha, mu, var in effects:
        logk = logk + _effect_log_mgf_dense(x, alpha, mu, var)
    return logk


def log_kappa_sparse(X, effects, offset_var=0.0, entry_chunk=1 << 16):
    """Per-row `logkappa` (n,) on a sparse (BCOO) design via the zero-clumping identity

        E[e^{o_li}] = 1 + sum_{c in supp(i)} alpha_lc (exp(x_ic mu_lc + 1/2 x_ic^2 var_lc) - 1)

    (exact, using `sum_c alpha_lc = 1`: an off-support gene contributes the constant
    `alpha_lc`, so only the nnz entries carry `b`-dependence). The support sum is a per-row
    `segment_sum` over the BCOO entries, chunked to bound peak memory at `O(entry_chunk)`.
    `effects` is a list of `(alpha, mu, var)` (each (p,)) over the shared design `X`."""
    op = BCOOOperator(X)
    idx = X.indices
    rows, cols, vals = idx[:, 0], idx[:, 1], jnp.asarray(X.data)
    n = X.shape[0]
    nnz = vals.shape[0]

    logk = jnp.zeros(n) + 0.5 * jnp.asarray(offset_var)
    for alpha, mu, var in effects:
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        mean = op.matvec(alpha * mu)  # (n,) = E[o_li]
        acc = jnp.zeros(n)
        for k0 in range(0, nnz, entry_chunk):
            k1 = min(k0 + entry_chunk, nnz)
            r, c, xv = rows[k0:k1], cols[k0:k1], vals[k0:k1]
            g = alpha[c] * (jnp.exp(xv * mu[c] + 0.5 * xv**2 * var[c]) - 1.0)
            acc = acc + segment_sum(g, r, num_segments=n)
        # log(1 + acc) = log E[e^{o_li}]; log1p keeps precision when acc is small.
        logk = logk + (jnp.log1p(acc) - mean)
    return logk


@dataclass(frozen=True)
class PoissonLogNormalOffset(Smoother):
    """Analytic Q2 offset-integrated cumulant for the Poisson (log-link) base.

    `Atilde(eta) = E_o[e^{eta+o}] = e^{eta + logkappa}` in closed form (the log-normal /
    MGF identity), with `logkappa` the per-row zero-mean offset log-MGF built from the
    OTHER effects' Gaussian-mixture laws (`build_aux[_sparse]`). A drop-in for
    `CharFnOffset`/`Compress` behind the `vi_gh` (CAVI in Q2) table seam, but with NO
    quadrature grid and NO Chebyshev table -- exact and cheaper. Convex (weight `e^{.} >
    0`); the smoothed loglik is exact, so its evidence is a true ELBO. Requires x64."""

    def validate(self, base):
        if not isinstance(base, Poisson):
            raise TypeError(
                "PoissonLogNormalOffset is the analytic Poisson (log-link) offset "
                f"cumulant; it needs a Poisson base, got {type(base).__name__}. For the "
                "logistic base use CharFnOffset; for a general base use Compress."
            )

    def build_aux(self, base, y, effects, offset_var=0.0):
        """Dense build: `effects = [(x, alpha, mu, var)]` (design block + effect-space
        law). Returns `aux = (y, logkappa)`; the offset mean lives in eta."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_dense(effects, y.shape[0], offset_var=offset_var)

    def build_aux_sparse(self, base, y, X, effects, offset_var=0.0, entry_chunk=1 << 16):
        """Sparse (BCOO) build: `X` shared, `effects = [(alpha, mu, var)]`. Same aux."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_sparse(X, effects, offset_var=offset_var, entry_chunk=entry_chunk)

    def terms(self, base: ResponseModel, eta, aux):
        """Exact Poisson terms with the offset folded into the rate: `Atilde = e^{eta +
        logkappa}`, all derivatives equal. `eta`, `y`, `logkappa` broadcast elementwise
        (the kernels open the GH-node axis on every leaf)."""
        y, logk = aux
        e = jnp.exp(jnp.asarray(eta) + jnp.asarray(logk))
        y = jnp.asarray(y)
        return y * eta - e, y - e, e
