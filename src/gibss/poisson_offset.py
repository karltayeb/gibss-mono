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

import jax
import jax.numpy as jnp

from jax.ops import segment_sum
from jax.scipy.special import logsumexp

from .operators import BCOOOperator
from .response import Poisson, ResponseModel, Smoother

__all__ = [
    "log_kappa_dense",
    "log_kappa_sparse",
    "log_kappa_q1_dense",
    "log_kappa_q1_sparse",
    "PoissonLogNormalOffset",
    "PoissonSelfNormOffset",
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


def log_kappa_sparse(X, effects, offset_var=0.0, entry_chunk=1 << 16,
                     colmean=None, feat_chunk=1 << 13):
    """Per-row `logkappa` (n,) on a sparse (BCOO) design via the zero-clumping identity

        E[e^{o_li}] = 1 + sum_{c in supp(i)} alpha_lc (exp(x_ic mu_lc + 1/2 x_ic^2 var_lc) - 1)

    (exact, using `sum_c alpha_lc = 1`: an off-support gene contributes the constant
    `alpha_lc`, so only the nnz entries carry `b`-dependence). The support sum is a per-row
    `segment_sum` over the BCOO entries, chunked to bound peak memory at `O(entry_chunk)`.
    `effects` is a list of `(alpha, mu, var)` (each (p,)) over the shared design `X`.

    Centering (`colmean` (p,) given): the off-support value is `-c_j` (not 0), so the
    exp(...) is no longer 1 and the "1 +" collapse fails. Recover it EXACTLY -- the MGF
    analogue of the CF baseline+support split -- by splitting the centered value into a
    row-independent baseline (-c_j) plus an on-support jump:

        E[e^{o_li}] = G_l + sum_{c in supp(i)} alpha_lc (h_ic - g_c),
        G_l = sum_c alpha_lc g_c,  g_c = exp(-c_j mu_c + 1/2 c_j^2 var_c),
        h_ic = exp((x_ic - c_j) mu_c + 1/2 (x_ic - c_j)^2 var_c).

    `G_l` is one row-independent reduction over all p (chunked by `feat_chunk`, once per
    effect); the support correction stays O(nnz). The uncentered branch (`c = 0`, `G_l = 1`)
    is left intact as the hot path."""
    op = BCOOOperator(X)
    idx = X.indices
    rows, cols, vals = idx[:, 0], idx[:, 1], jnp.asarray(X.data)
    n = X.shape[0]
    p = X.shape[1]
    nnz = vals.shape[0]
    c_arr = None if colmean is None else jnp.asarray(colmean)

    logk = jnp.zeros(n) + 0.5 * jnp.asarray(offset_var)
    for alpha, mu, var in effects:
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        if c_arr is None:
            mean = op.matvec(alpha * mu)  # (n,) = E[o_li]
            G = 1.0
        else:
            mean = op.matvec(alpha * mu) - jnp.dot(alpha * mu, c_arr)  # centered mean
            G = jnp.zeros(())  # baseline sum_c alpha_c g_c over all p (chunked)
            for j0 in range(0, p, feat_chunk):
                j1 = min(j0 + feat_chunk, p)
                cc = c_arr[j0:j1]
                gc = jnp.exp(-cc * mu[j0:j1] + 0.5 * cc**2 * var[j0:j1])
                G = G + jnp.sum(alpha[j0:j1] * gc)
        acc = jnp.zeros(n)
        for k0 in range(0, nnz, entry_chunk):
            k1 = min(k0 + entry_chunk, nnz)
            r, c, xv = rows[k0:k1], cols[k0:k1], vals[k0:k1]
            if c_arr is None:
                g = alpha[c] * (jnp.exp(xv * mu[c] + 0.5 * xv**2 * var[c]) - 1.0)
            else:
                xc = xv - c_arr[c]  # centered support value
                h = jnp.exp(xc * mu[c] + 0.5 * xc**2 * var[c])
                gon = jnp.exp(-c_arr[c] * mu[c] + 0.5 * c_arr[c] ** 2 * var[c])
                g = alpha[c] * (h - gon)
            acc = acc + segment_sum(g, r, num_segments=n)
        # log E[e^{o_li}] - mean. Uncentered: log1p(acc) keeps precision (G = 1). Centered:
        # E = G + acc > 0 (a sum of positive MGF terms); a tiny floor guards fp cancellation.
        if c_arr is None:
            logE = jnp.log1p(acc)
        else:
            logE = jnp.log(jnp.maximum(G + acc, 1e-300))
        logk = logk + (logE - mean)
    return logk


# --------------------------------------------------------------------------------------
# Free-form (Q1) node folds. A Q1 effect's posterior IS the discrete quadrature law: value
# nodes `b_cm` over features c and nodes m, with self-normalized weights `W_cm = softmax`.
# For the Poisson MGF `E[e^{psi}]` this is an EXACT finite sum `sum_{c,m} W_cm e^{x_ic b_cm}`
# -- no interpolation, the Q1 analogue of the Gaussian MGF above. Used by the analytic ELBO.
# --------------------------------------------------------------------------------------


def _log_node_mgf_intercept(node_law):
    """Zero-mean log-MGF `log(sum_m W_m e^{b_m}) - sum_m W_m b_m` of the free-form intercept
    (all-ones column, so `psi = b0`), a SCALAR. `node_law = (b0_nodes, logW0)` each (Q,)."""
    b0, logW0 = node_law
    b0 = jnp.asarray(b0)
    logw = jnp.asarray(logW0)
    logw = logw - logsumexp(logw)  # normalize
    W0 = jnp.exp(logw)
    return logsumexp(logw + b0) - jnp.sum(W0 * b0)


def log_kappa_q1_dense(X, effects, n, intercept=None):
    """Per-row zero-mean log-MGF (n,) for FREE-FORM (Q1) effects on a dense design.

    `effects` is a list of `(b_nodes, logW)` each `(C, Q)` -- value nodes and log
    unnormalized JOINT weights over (feature, node). `intercept` is an optional
    `(b0_nodes, logW0)` for the shared free-form intercept (all-ones column). Each effect's
    contribution is `log(sum_{c,m} W_cm e^{x_ic b_cm}) - sum_{c,m} W_cm x_ic b_cm`, the log
    of the node-weighted MGF minus the mean carried in eta -- computed stably in log space,
    reduced feature-by-feature with a `lax.scan` (peak memory `O(n Q)`)."""
    X = jnp.asarray(X)
    logk = jnp.zeros(n)
    for b_nodes, logW in effects:
        b_nodes = jnp.asarray(b_nodes)                       # (C, Q)
        logw = jnp.asarray(logW)
        logw = logw - logsumexp(logw)                        # normalize over all (C, Q)
        W = jnp.exp(logw)
        mean = X @ jnp.sum(W * b_nodes, axis=1)              # (n,) = sum_{c,m} W_cm x_ic b_cm

        def body(carry_logS, comp):
            xc, bc, lwc = comp                               # (n,), (Q,), (Q,)
            logS_c = logsumexp(lwc[None, :] + xc[:, None] * bc[None, :], axis=1)  # (n,)
            return jnp.logaddexp(carry_logS, logS_c), None

        logS, _ = jax.lax.scan(
            body, jnp.full(n, -jnp.inf), (X.T, b_nodes, logw)
        )
        logk = logk + (logS - mean)
    if intercept is not None:
        logk = logk + _log_node_mgf_intercept(intercept)
    return logk


def log_kappa_q1_sparse(X, effects, n, intercept=None, entry_chunk=1 << 16,
                        colmean=None, feat_chunk=1 << 13):
    """Sparse (BCOO) analogue of `log_kappa_q1_dense` via the zero-clumping identity: an
    off-support feature (x_ic = 0) contributes its marginal node mass `pi_c = sum_m W_cm`,
    so `E[e^{psi}] = 1 + sum_{c in supp}(sum_m W_cm e^{x_ic b_cm} - pi_c)` (uses sum_c pi_c
    = 1). `effects` is a list of `(b_nodes, logW)` each `(C, Q)`; `intercept` optional.

    Centering (`colmean` (p,) given): the free-form (Q1) analogue of the Q2
    baseline+support split. An off-support feature contributes `g_c = sum_m W_cm e^{-c_j
    b_cm}` (its node MGF at `-c_j`), not `pi_c`, so

        E[e^{psi}] = G_l + sum_{c in supp}(sum_m W_cm e^{(x_ic - c_j) b_cm} - g_c),
        G_l = sum_c g_c   (row-independent, one reduction over all (C, Q), chunked).

    Exact; the uncentered branch (`c = 0`, `g_c = pi_c`, `G_l = 1`) is left intact."""
    op = BCOOOperator(X)
    idx = X.indices
    rows, cols, vals = idx[:, 0], idx[:, 1], jnp.asarray(X.data)
    nnz = vals.shape[0]
    C = jnp.asarray(effects[0][0]).shape[0] if effects else 0
    c_arr = None if colmean is None else jnp.asarray(colmean)

    logk = jnp.zeros(n)
    for b_nodes, logW in effects:
        b_nodes = jnp.asarray(b_nodes)                       # (C, Q)
        logw = jnp.asarray(logW)
        W = jax.nn.softmax(logw.reshape(-1)).reshape(logw.shape)  # (C, Q), joint
        pi = jnp.sum(W, axis=1)                              # (C,) marginal feature mass
        A = jnp.sum(W * b_nodes, axis=1)                     # (C,) per-feature E[b]
        if c_arr is None:
            mean = op.matvec(A)                             # (n,) off-support x=0 -> 0
            G = 1.0
        else:
            mean = op.matvec(A) - jnp.dot(A, c_arr)         # centered mean
            G = jnp.zeros(())  # baseline G_l = sum_c g_c over all C (chunked)
            for j0 in range(0, C, feat_chunk):
                j1 = min(j0 + feat_chunk, C)
                gc = jnp.sum(W[j0:j1] * jnp.exp(-c_arr[j0:j1, None] * b_nodes[j0:j1]),
                             axis=1)  # (chunk,) node MGF at -c_j
                G = G + jnp.sum(gc)
        acc = jnp.zeros(n)
        for k0 in range(0, nnz, entry_chunk):
            k1 = min(k0 + entry_chunk, nnz)
            r, c, xv = rows[k0:k1], cols[k0:k1], vals[k0:k1]
            if c_arr is None:
                # sum_m W_cm e^{x b_cm} - pi_c per support entry
                s = jnp.sum(W[c] * jnp.exp(xv[:, None] * b_nodes[c]), axis=1) - pi[c]
            else:
                xc = xv - c_arr[c]  # centered support value
                h = jnp.sum(W[c] * jnp.exp(xc[:, None] * b_nodes[c]), axis=1)
                gon = jnp.sum(W[c] * jnp.exp(-c_arr[c][:, None] * b_nodes[c]), axis=1)
                s = h - gon
            acc = acc + segment_sum(s, r, num_segments=n)
        if c_arr is None:
            logE = jnp.log1p(acc)
        else:
            logE = jnp.log(jnp.maximum(G + acc, 1e-300))
        logk = logk + (logE - mean)
    if intercept is not None:
        logk = logk + _log_node_mgf_intercept(intercept)
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

    def build_aux_sparse(self, base, y, X, effects, offset_var=0.0, entry_chunk=1 << 16,
                         colmean=None):
        """Sparse (BCOO) build: `X` shared, `effects = [(alpha, mu, var)]`. Same aux.
        `colmean` (p,), if given, treats `X` as centered via the baseline+support split."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_sparse(
            X, effects, offset_var=offset_var, entry_chunk=entry_chunk, colmean=colmean
        )

    def terms(self, base: ResponseModel, eta, aux):
        """Exact Poisson terms with the offset folded into the rate: `Atilde = e^{eta +
        logkappa}`, all derivatives equal. `eta`, `y`, `logkappa` broadcast elementwise
        (the kernels open the GH-node axis on every leaf)."""
        y, logk = aux
        e = jnp.exp(jnp.asarray(eta) + jnp.asarray(logk))
        y = jnp.asarray(y)
        return y * eta - e, y - e, e


@dataclass(frozen=True)
class PoissonSelfNormOffset(Smoother):
    r"""Exact free-form (Q1 / CAVI) offset-integrated cumulant for the Poisson base.

    `Atilde(eta) = E_o[e^{eta + o}] = e^{eta + logkappa}` in closed form, with `logkappa`
    the per-row zero-mean offset log-MGF built from the OTHER effects' TRUE, non-Gaussian
    free-form posteriors -- each effect's raw quadrature node law `(b_nodes, logW)` -- plus
    the shared free-form intercept's node law. The Q1 analogue of `PoissonLogNormalOffset`:
    that class folds Gaussian-MIXTURE effect laws (CAVI in Q2); this one folds the raw
    NODE law by the exact finite-sum MGF `sum_{c,m} W_cm e^{x_ic b_cm}` (CAVI in Q1). Both
    are the same identity `Atilde = e^{eta} E[e^{o}]`, differing only in how `E[e^{o}]` is
    formed -- and `logkappa` is already assembled by `log_kappa_q1_{dense,sparse}` (the
    analytic-ELBO builders).

    A drop-in for `CompressSelfNorm` behind the `quad` kernel's `terms` seam, but with NO
    Chebyshev residual and NO offset quadrature. That is the whole point: `CompressSelfNorm`
    fits the residual `R(z) = A(z + obar) - Atilde(z)` with a fixed-degree Chebyshev series
    on an interval of halfwidth `~ T + kappa sqrt(V)`. For `A = exp` the exact residual
    `R(z) = e^{z+obar}(e^{V/2} - 1)` grows EXPONENTIALLY while the interval only widens as
    `sqrt(V)`, so a fixed-degree polynomial hits a hard width cliff and the Q1 Poisson CAVI
    sweep oscillates (ELBO decreases) once the sequential fold accumulates enough offset
    variance. The closed-form MGF has no such basis mismatch -- exact (`Ahat == Atilde`, so
    the smoothed loglik is a true ELBO) and convex (weight `e^{.} > 0`). Requires x64.

    `aux = (y, logkappa)`, both per-row `(n,)`: `logkappa` is row-only (the OTHER effects
    are shared across the candidate features of the effect being updated), so the `quad`
    kernel broadcasts it over the feature/node axes exactly like `y`. Sparse (BCOO) uses the
    zero-clumping identity; under implicit pre-centering an off-support entry contributes a
    per-feature node MGF `g_c` at `-c_j` (not a constant), recovered by the baseline+support
    split in `log_kappa_q1_sparse` (pass `colmean`) -- so it DOES pre-center sparse.
    """

    def validate(self, base):
        if not isinstance(base, Poisson):
            raise TypeError(
                "PoissonSelfNormOffset is the analytic free-form (Q1) Poisson (log-link) "
                f"offset cumulant; it needs a Poisson base, got {type(base).__name__}. For "
                "a general base use CompressSelfNorm; for Gaussian-q CAVI in Q2 use "
                "PoissonLogNormalOffset."
            )

    def build_aux(self, base, y, X, effects, intercept=None):
        """Dense build: `effects = [(b_nodes, logW)]` each `(C, Q)` -- the OTHER effects'
        free-form node laws over the shared design `X`; `intercept` an optional
        `(b0_nodes, logW0)` node law. Returns `aux = (y, logkappa)`; the offset mean lives
        in eta (the node MGF is zero-mean)."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_q1_dense(X, effects, y.shape[0], intercept=intercept)

    def build_aux_sparse(self, base, y, X, effects, intercept=None, entry_chunk=1 << 16,
                         colmean=None):
        """Sparse (BCOO) build via the zero-clumping identity (`log_kappa_q1_sparse`). Same
        `aux = (y, logkappa)` contract. `X` shared, `effects = [(b_nodes, logW)]`.
        `colmean` (p,), if given, treats `X` as centered via the baseline+support split."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_q1_sparse(
            X, effects, y.shape[0], intercept=intercept, entry_chunk=entry_chunk,
            colmean=colmean,
        )

    def terms(self, base: ResponseModel, eta, aux):
        """Exact Poisson terms with the offset folded into the rate: `Atilde = e^{eta +
        logkappa}`, all derivatives equal. Identical to `PoissonLogNormalOffset.terms`;
        only the `logkappa` builder (node MGF vs Gaussian-mixture MGF) differs."""
        y, logk = aux
        e = jnp.exp(jnp.asarray(eta) + jnp.asarray(logk))
        y = jnp.asarray(y)
        return y * eta - e, y - e, e
