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
`vi_gh` kernel consumes `aux = (y, logkappa)`. The b-expectation is closed form too
(`E_b[e^{eta + x b}] = e^{eta + x m + x^2 v / 2}`), so the engine routes it to
`glm_vi_poisson_[center_]ser` instead of integrating b by Gauss-Hermite.
Requires x64 for the MGF exponents; gated to the `Poisson` base.

Cost per sweep
--------------
`logkappa` is ADDITIVE over effects, so a fold is a sum of per-effect terms `t_l` that
depend only on effect l's law. The smoothers memoize `t_l` per effect slot (`_TermCache`),
so a sweep computes L terms, not L^2 (the analogue of `cf_offset._FactorCache`). On a 0/1
design a term is one (n, 2) SpMM (mean and support sum), with no per-entry exp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from jax.experimental import sparse as jsparse
from jax.ops import segment_sum
from jax.scipy.special import logsumexp

from .cf_offset import _is_binary_design
from .response import Poisson, ResponseModel, Smoother

__all__ = [
    "log_kappa_dense",
    "log_kappa_sparse",
    "log_kappa_q1_dense",
    "log_kappa_q1_sparse",
    "effect_log_mgf_q1_dense",
    "PoissonLogNormalOffset",
    "PoissonSelfNormOffset",
]


# Overflow guard for the Poisson rate. `Atilde = exp(eta + logkappa)` can exceed float64's
# ~exp(709.8) range on a transient bad iterate -- a tail GH node, or a large offset log-MGF
# from heavy-tailed design x strong effects x large L (logkappa >= 0 and grows with L). An
# inf rate poisons grad = y - Atilde and the curvature w = Atilde, turning the whole SER
# update to nan. Cap the log-rate well below the per-element overflow AND below where a
# reduction over rows would overflow (exp(600) ~ 4e260, so a sum over up to ~1e47 rows stays
# finite). The cap is astronomically above any physical Poisson rate, so it never touches a
# converged fit or any realistic iterate; it only turns an overflowing iterate into a large
# finite rate whose (kernel-clipped) Newton step still points the predictor back down. The
# cumulant is exact wherever the cap is inactive (every converged and near-converged state).
_LOGRATE_CAP = 600.0


def _poisson_terms(y, eta, logk):
    """`(loglik, grad, weight)` for the offset-folded Poisson cumulant `Atilde = exp(eta +
    logk)`, shared by `PoissonLogNormalOffset` (Q2) and `PoissonSelfNormOffset` (Q1). The
    linear term `y*eta` uses the UNCLAMPED `eta` (so the evidence is exact); only the rate
    `e` is capped (`_LOGRATE_CAP`), and only on an overflowing iterate."""
    y = jnp.asarray(y)
    eta = jnp.asarray(eta)
    e = jnp.exp(jnp.minimum(eta + jnp.asarray(logk), _LOGRATE_CAP))
    return y * eta - e, y - e, e


class _TermCache:
    """Per-fit memo of the per-effect zero-mean log-MGF terms `t_l` (n,).

    `logkappa` is ADDITIVE over effects (the mean-field offset MGF is a product of
    per-effect MGFs), so a fold is `sum_{l in others} t_l` and each `t_l` depends only on
    effect l's own law. A sweep builds L effect folds (L-1 terms each) plus an intercept
    fold (all L terms): rebuilt from scratch that is O(L^2) term builds per sweep for O(L)
    distinct terms. The Poisson analogue of `cf_offset._FactorCache`: a term is keyed by
    the SLOT the caller names (the effect index, stable across sweeps) and validated by
    the IDENTITY of every input array (design, law, column centers). A SER update replaces
    the effect's law arrays, so a stale term is never served. One (n,) vector per slot, so
    the memo needs no byte cap. The memo also remembers, per design object, whether its
    entries are all 0/1 (`is_binary`), which selects the matvec path, and holds a BCSR
    copy of a BCOO design for that matvec (`matvec_design`).

    Purely a cost concern: a memoized term is the same array the builder would recompute."""

    __slots__ = ("slots", "binary", "csr")

    def __init__(self):
        self.slots = {}  # slot -> (refs, term)
        self.binary = {}  # id(x) -> (x, is_binary)
        self.csr = {}  # id(x) -> (x, BCSR copy)

    def term(self, slot, refs, compute):
        if slot is None:
            return compute()
        e = self.slots.get(slot)
        if e is not None and len(e[0]) == len(refs) and all(
            a is b for a, b in zip(e[0], refs)
        ):
            return e[1]
        t = compute()
        self.slots[slot] = (refs, t)
        return t

    def is_binary(self, x):
        hit = self.binary.get(id(x))
        if hit is not None and hit[0] is x:
            return hit[1]
        flag = _is_binary_design(x)
        self.binary[id(x)] = (x, flag)
        return flag

    def matvec_design(self, x):
        hit = self.csr.get(id(x))
        if hit is not None and hit[0] is x:
            return hit[1]
        xm = _matvec_design(x)
        self.csr[id(x)] = (x, xm)
        return xm


def _matvec_design(x):
    """The layout to multiply `x` by on the 0/1 path: a BCSR copy of a BCOO design (JAX's
    BCSR SpMV is ~6-10x faster than its BCOO matmul), a dense design as is."""
    return jsparse.BCSR.from_bcoo(x) if isinstance(x, jsparse.BCOO) else x


def _binary_flag(x, cache, seen):
    """Is the design `x` all 0/1? Memoized per fit by `cache`, else per call by `seen`."""
    if cache is not None:
        return cache.is_binary(x)
    hit = seen.get(id(x))
    if hit is None or hit[0] is not x:
        hit = (x, _is_binary_design(x))
        seen[id(x)] = hit
    return hit[1]


def _fold_terms(n, offset_var, items, keys, cache):
    """`1/2 offset_var + sum_l t_l` over `items = [(refs, compute)]`, each term served from
    the memo when its slot's refs are unchanged."""
    keys = [None] * len(items) if keys is None else list(keys)
    logk = jnp.zeros(n) + 0.5 * jnp.asarray(offset_var)
    for (refs, compute), slot in zip(items, keys):
        t = compute() if cache is None else cache.term(slot, refs, compute)
        logk = logk + t
    return logk


@jax.jit
def _effect_log_mgf_dense(x, alpha, mu, var):
    """Zero-mean log-MGF `logsumexp_c(log alpha_c + x_ic mu_c + 1/2 x_ic^2 var_c) - mean_i`
    of one effect's row contribution `o_li = x_ic b`, `(n,)`. `x` is (n, C); `alpha, mu,
    var` are the effect-space law (C,). The subtraction removes the mean carried in eta.
    Jitted, so the (n, C) pass compiles once per shape and fuses."""
    mean = x @ (alpha * mu)  # (n,) = E[o_li]
    # log alpha_c + x_ic mu_c + 1/2 x_ic^2 var_c, reduced over c by logsumexp (stable).
    terms = jnp.log(alpha)[None, :] + x * mu[None, :] + 0.5 * (x**2) * var[None, :]
    return logsumexp(terms, axis=1) - mean


@partial(jax.jit, static_argnames=("centered",))
def _effect_log_mgf_binary(x, alpha, mu, var, colmean, centered):
    """Zero-mean log-MGF (n,) of one effect on a 0/1 design `x` (dense or BCOO).

    With `x_ic in {0, 1}` the per-entry MGF `exp(x mu_c + 1/2 x^2 var_c)` takes two values
    per feature, so the effect MGF is AFFINE in the design:

        E[e^{o_li}] = G + sum_c x_ic D_c,   D_c = alpha_c (h_c - g_c),  G = sum_c alpha_c g_c,

    with `g_c, h_c` the MGF at the off-support value `-c_c` and the support value `1 - c_c`
    -- one matvec, no per-entry exp (the MGF analogue of `cf_offset._binary_terms`).
    Uncentered (`centered=False`): `g_c = 1`, `G = 1`, and `log1p` keeps the precision."""
    am = alpha * mu
    if centered:
        x0, x1 = -colmean, 1.0 - colmean
        g = jnp.exp(x0 * mu + 0.5 * x0**2 * var)
        h = jnp.exp(x1 * mu + 0.5 * x1**2 * var)
        M = x @ jnp.stack([am, alpha * (h - g)], axis=1)  # ONE (n, 2) pass: mean, support
        mean = M[:, 0] - jnp.dot(am, colmean)
        logE = jnp.log(jnp.maximum(jnp.sum(alpha * g) + M[:, 1], 1e-300))
    else:
        M = x @ jnp.stack([am, alpha * jnp.expm1(mu + 0.5 * var)], axis=1)
        mean = M[:, 0]
        logE = jnp.log1p(M[:, 1])
    return logE - mean


@partial(jax.jit, static_argnames=("n",))
def _sparse_mgf_chunk(r, c, xv, alpha, mu, var, colmean, n):
    """Per-row support sum `sum_{c in supp(i)} alpha_c (h_ic - g_c)` over one BCOO entry
    chunk: `h_ic` the MGF at the (centered) entry value, `g_c` at the off-support value
    `-c_c` (exactly 1 when `colmean = 0`). Jitted with `n` static so the chunk loop hits
    cache."""
    mc, vc, cc = mu[c], var[c], colmean[c]
    xc = xv - cc  # centered support value
    h = jnp.exp(xc * mc + 0.5 * xc**2 * vc)
    g_on = jnp.exp(-cc * mc + 0.5 * cc**2 * vc)
    return segment_sum(alpha[c] * (h - g_on), r, num_segments=n)


@jax.jit
def _sparse_mean_baseline(X, alpha, mu, var, colmean):
    """Per-row mean (n,) of one effect on the (centered) BCOO design, and its row-
    independent off-support baseline `G = sum_c alpha_c exp(-c_c mu_c + 1/2 c_c^2 var_c)`."""
    am = alpha * mu
    mean = X @ am - jnp.dot(am, colmean)
    G = jnp.sum(alpha * jnp.exp(-colmean * mu + 0.5 * colmean**2 * var))
    return mean, G


def _effect_log_mgf_sparse(X, alpha, mu, var, colmean, entry_chunk):
    """Zero-mean log-MGF (n,) of one effect on a general-valued BCOO design: the baseline
    `G` (1 uncentered) plus the chunked per-entry support sum (`log_kappa_sparse`)."""
    n, p = X.shape
    c_arr = jnp.zeros(p) if colmean is None else colmean
    mean, G = _sparse_mean_baseline(X, alpha, mu, var, c_arr)
    rows, cols, vals = X.indices[:, 0], X.indices[:, 1], X.data
    acc = jnp.zeros(n)
    for k0 in range(0, vals.shape[0], entry_chunk):
        k1 = k0 + entry_chunk
        acc = acc + _sparse_mgf_chunk(
            rows[k0:k1], cols[k0:k1], vals[k0:k1], alpha, mu, var, c_arr, n
        )
    # log E[e^{o_li}] - mean. Uncentered: log1p(acc) keeps precision (G = 1). Centered:
    # E = G + acc > 0 (a sum of positive MGF terms); a tiny floor guards fp cancellation.
    if colmean is None:
        return jnp.log1p(acc) - mean
    return jnp.log(jnp.maximum(G + acc, 1e-300)) - mean


def log_kappa_dense(effects, n, offset_var=0.0, keys=None, cache=None):
    """Per-row `logkappa` (n,) for a dense design: sum of the per-effect zero-mean
    log-MGFs plus the intercept factor `1/2 * offset_var`. `effects` is a list of
    `(x, alpha, mu, var)` (design block + effect-space law), the same effect-space
    representation `cf_offset.offset_cf` consumes; empty effects contribute 0. A 0/1
    design takes the matvec path (`_effect_log_mgf_binary`). `keys` (one slot per effect)
    and a `cache` (`_TermCache`) memoize each effect's term across the folds of a fit."""
    seen = {}
    items = []
    for x, alpha, mu, var in effects:
        x = jnp.asarray(x)
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        if _binary_flag(x, cache, seen):
            compute = partial(_effect_log_mgf_binary, x, alpha, mu, var, None, centered=False)
        else:
            compute = partial(_effect_log_mgf_dense, x, alpha, mu, var)
        items.append(((x, alpha, mu, var), compute))
    return _fold_terms(n, offset_var, items, keys, cache)


def log_kappa_sparse(X, effects, offset_var=0.0, entry_chunk=1 << 16,
                     colmean=None, feat_chunk=None, keys=None, cache=None):
    """Per-row `logkappa` (n,) on a sparse (BCOO) design via the zero-clumping identity

        E[e^{o_li}] = 1 + sum_{c in supp(i)} alpha_lc (exp(x_ic mu_lc + 1/2 x_ic^2 var_lc) - 1)

    (exact, using `sum_c alpha_lc = 1`: an off-support gene contributes the constant
    `alpha_lc`, so only the nnz entries carry `b`-dependence). The support sum is a per-row
    `segment_sum` over the BCOO entries, chunked to bound peak memory at `O(entry_chunk)`;
    on a 0/1 design it is one SpMV (`_effect_log_mgf_binary`). `effects` is a list of
    `(alpha, mu, var)` (each (p,)) over the shared design `X`. `keys`/`cache` memoize the
    per-effect terms across the folds of a fit (`_TermCache`).

    Centering (`colmean` (p,) given): the off-support value is `-c_j` (not 0), so the
    exp(...) is no longer 1 and the "1 +" collapse fails. Recover it EXACTLY -- the MGF
    analogue of the CF baseline+support split -- by splitting the centered value into a
    row-independent baseline (-c_j) plus an on-support jump:

        E[e^{o_li}] = G_l + sum_{c in supp(i)} alpha_lc (h_ic - g_c),
        G_l = sum_c alpha_lc g_c,  g_c = exp(-c_j mu_c + 1/2 c_j^2 var_c),
        h_ic = exp((x_ic - c_j) mu_c + 1/2 (x_ic - c_j)^2 var_c).

    `G_l` is one row-independent reduction over all p (once per effect); the support
    correction stays O(nnz). The uncentered branch (`c = 0`, `G_l = 1`) uses `log1p`."""
    del feat_chunk  # the baseline reduction is one jitted (p,) pass now
    n = X.shape[0]
    c_arr = None if colmean is None else jnp.asarray(colmean)
    binary = _binary_flag(X, cache, {})
    if binary:
        Xm = cache.matvec_design(X) if cache is not None else _matvec_design(X)
    items = []
    for alpha, mu, var in effects:
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        if binary:
            compute = partial(
                _effect_log_mgf_binary, Xm, alpha, mu, var,
                jnp.zeros(X.shape[1]) if c_arr is None else c_arr,
                centered=c_arr is not None,
            )
        else:
            compute = partial(_effect_log_mgf_sparse, X, alpha, mu, var, c_arr, entry_chunk)
        items.append(((X, alpha, mu, var, c_arr), compute))
    return _fold_terms(n, offset_var, items, keys, cache)


# --------------------------------------------------------------------------------------
# Free-form (Q1) node folds. A Q1 effect's posterior IS the discrete quadrature law: value
# nodes `b_cm` over features c and nodes m, with self-normalized weights `W_cm = softmax`.
# For the Poisson MGF `E[e^{psi}]` this is an EXACT finite sum `sum_{c,m} W_cm e^{x_ic b_cm}`
# -- no interpolation, the Q1 analogue of the Gaussian MGF above.
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


@jax.jit
def effect_log_mgf_q1_dense(X, b_nodes, logW):
    """Per-row zero-mean log-MGF (n,) of ONE free-form (Q1) effect's row contribution on a
    dense design: `log(sum_{c,m} W_cm e^{x_ic b_cm}) - sum_{c,m} W_cm x_ic b_cm`. `X` is
    (n, C); `b_nodes, logW` are the effect's node law (C, Q) (unnormalized joint logW).

    The per-row MGF is reduced feature-by-feature with a `lax.scan` (peak memory O(n Q)).
    `@jax.jit` gives that scan a stable, shape-keyed compilation cache: an eager `lax.scan`
    rebuilt from a fresh closure each call re-traces and re-compiles at the same shape
    every time (the node VALUES drift, their shape does not). Under the wrapper it compiles
    once per (n, C, Q). Being a pure function of one effect's node law, its result is the
    per-effect term `t_l` the fold memoizes (`_TermCache`)."""
    X = jnp.asarray(X)
    b_nodes = jnp.asarray(b_nodes)                       # (C, Q)
    logw = jnp.asarray(logW)
    logw = logw - logsumexp(logw)                        # normalize over all (C, Q)
    W = jnp.exp(logw)
    mean = X @ jnp.sum(W * b_nodes, axis=1)              # (n,) = sum_{c,m} W_cm x_ic b_cm

    def body(carry_logS, comp):
        xc, bc, lwc = comp                               # (n,), (Q,), (Q,)
        logS_c = logsumexp(lwc[None, :] + xc[:, None] * bc[None, :], axis=1)  # (n,)
        return jnp.logaddexp(carry_logS, logS_c), None

    logS, _ = jax.lax.scan(body, jnp.full(X.shape[0], -jnp.inf), (X.T, b_nodes, logw))
    return logS - mean


def _node_law(logW):
    """Joint self-normalized node weights `W = softmax(logW)` over all (C, Q)."""
    return jax.nn.softmax(logW.reshape(-1)).reshape(logW.shape)


@partial(jax.jit, static_argnames=("centered",))
def _node_log_mgf_binary(x, b_nodes, logW, colmean, centered):
    """Zero-mean log-MGF (n,) of one free-form effect on a 0/1 design `x` (dense or BCOO):
    `E[e^{psi}] = G + x @ D`, `D_c = h_c - g_c`, `G = sum_c g_c`, with `h_c, g_c` the node
    MGFs `sum_m W_cm e^{v b_cm}` at the support value `v = 1 - c_c` and the off-support
    value `v = -c_c` -- one matvec, no per-entry exp. Uncentered: `g_c = pi_c`, `G = 1`."""
    W = _node_law(logW)
    A = jnp.sum(W * b_nodes, axis=1)  # (C,) per-feature E[b]
    if centered:
        g = jnp.sum(W * jnp.exp(-colmean[:, None] * b_nodes), axis=1)
        h = jnp.sum(W * jnp.exp((1.0 - colmean)[:, None] * b_nodes), axis=1)
        M = x @ jnp.stack([A, h - g], axis=1)  # ONE (n, 2) pass: mean, support
        mean = M[:, 0] - jnp.dot(A, colmean)
        logE = jnp.log(jnp.maximum(jnp.sum(g) + M[:, 1], 1e-300))
    else:
        M = x @ jnp.stack([A, jnp.sum(W * jnp.expm1(b_nodes), axis=1)], axis=1)
        mean = M[:, 0]
        logE = jnp.log1p(M[:, 1])
    return logE - mean


@partial(jax.jit, static_argnames=("n",))
def _sparse_node_mgf_chunk(r, c, xv, b_nodes, W, colmean, n):
    """Per-row support sum `sum_{c in supp(i)} (sum_m W_cm e^{(x_ic - c_c) b_cm} - g_c)`
    over one BCOO entry chunk, `g_c = sum_m W_cm e^{-c_c b_cm}` (= `pi_c` uncentered)."""
    bc, Wc, cc = b_nodes[c], W[c], colmean[c]
    h = jnp.sum(Wc * jnp.exp((xv - cc)[:, None] * bc), axis=1)
    g_on = jnp.sum(Wc * jnp.exp(-cc[:, None] * bc), axis=1)
    return segment_sum(h - g_on, r, num_segments=n)


@jax.jit
def _sparse_node_mean_baseline(X, b_nodes, logW, colmean):
    """Joint node weights `W`, the per-row mean (n,) on the (centered) BCOO design, and the
    off-support baseline `G = sum_c sum_m W_cm e^{-c_c b_cm}`."""
    W = _node_law(logW)
    A = jnp.sum(W * b_nodes, axis=1)
    mean = X @ A - jnp.dot(A, colmean)
    G = jnp.sum(W * jnp.exp(-colmean[:, None] * b_nodes))
    return W, mean, G


def _node_log_mgf_sparse(X, b_nodes, logW, colmean, entry_chunk):
    """Zero-mean log-MGF (n,) of one free-form effect on a general-valued BCOO design."""
    n, p = X.shape
    c_arr = jnp.zeros(p) if colmean is None else colmean
    W, mean, G = _sparse_node_mean_baseline(X, b_nodes, logW, c_arr)
    rows, cols, vals = X.indices[:, 0], X.indices[:, 1], X.data
    acc = jnp.zeros(n)
    for k0 in range(0, vals.shape[0], entry_chunk):
        k1 = k0 + entry_chunk
        acc = acc + _sparse_node_mgf_chunk(
            rows[k0:k1], cols[k0:k1], vals[k0:k1], b_nodes, W, c_arr, n
        )
    if colmean is None:
        return jnp.log1p(acc) - mean
    return jnp.log(jnp.maximum(G + acc, 1e-300)) - mean


def _q1_fold(n, effects, intercept, keys, cache, refs, term):
    """Shared Q1 fold: memoized per-effect node terms (`term(b_nodes, logW)`) plus the
    scalar intercept term. `refs` (one tuple per effect) are the memo's validity keys --
    pass the effect's STORED node arrays when `effects` holds fresh views of them."""
    items = []
    refs = [None] * len(effects) if refs is None else list(refs)
    for (b_nodes, logW), ref in zip(effects, refs):
        b_nodes, logW = jnp.asarray(b_nodes), jnp.asarray(logW)
        ref = (b_nodes, logW) if ref is None else tuple(ref)
        items.append((ref, partial(term, b_nodes, logW)))
    logk = _fold_terms(n, 0.0, items, keys, cache)
    if intercept is not None:
        logk = logk + _log_node_mgf_intercept(intercept)
    return logk


def log_kappa_q1_dense(X, effects, n, intercept=None, keys=None, cache=None, refs=None):
    """Per-row zero-mean log-MGF (n,) for FREE-FORM (Q1) effects on a dense design: the SUM
    over effects of `effect_log_mgf_q1_dense` (additive -- the mean-field offset MGF
    factorizes over effects). `effects` is a list of `(b_nodes, logW)` each `(C, Q)`;
    `intercept` is an optional `(b0_nodes, logW0)` for the shared free-form intercept. A
    0/1 design takes the matvec path. `keys`/`cache`/`refs` memoize the per-effect terms
    across the folds of a fit (`_TermCache`)."""
    X = jnp.asarray(X)
    if _binary_flag(X, cache, {}):
        def term(b, w):
            return _node_log_mgf_binary(X, b, w, None, centered=False)
    else:
        def term(b, w):
            return effect_log_mgf_q1_dense(X, b, w)
    if refs is not None:
        refs = [(X, *r) for r in refs]
    elif cache is not None:
        refs = [(X, jnp.asarray(b), jnp.asarray(w)) for b, w in effects]
    return _q1_fold(n, effects, intercept, keys, cache, refs, term)


def log_kappa_q1_sparse(X, effects, n, intercept=None, entry_chunk=1 << 16,
                        colmean=None, feat_chunk=None, keys=None, cache=None, refs=None):
    """Sparse (BCOO) analogue of `log_kappa_q1_dense` via the zero-clumping identity: an
    off-support feature (x_ic = 0) contributes its marginal node mass `pi_c = sum_m W_cm`,
    so `E[e^{psi}] = 1 + sum_{c in supp}(sum_m W_cm e^{x_ic b_cm} - pi_c)` (uses sum_c pi_c
    = 1). `effects` is a list of `(b_nodes, logW)` each `(C, Q)`; `intercept` optional. On
    a 0/1 design the support sum is one SpMV. `keys`/`cache`/`refs` as in the dense build.

    Centering (`colmean` (p,) given): the free-form (Q1) analogue of the Q2
    baseline+support split. An off-support feature contributes `g_c = sum_m W_cm e^{-c_j
    b_cm}` (its node MGF at `-c_j`), not `pi_c`, so

        E[e^{psi}] = G_l + sum_{c in supp}(sum_m W_cm e^{(x_ic - c_j) b_cm} - g_c),
        G_l = sum_c g_c   (row-independent, one reduction over all (C, Q)).

    Exact; the uncentered branch (`c = 0`, `g_c = pi_c`, `G_l = 1`) uses `log1p`."""
    del feat_chunk  # the baseline reduction is one jitted (C, Q) pass now
    c_arr = None if colmean is None else jnp.asarray(colmean)
    if _binary_flag(X, cache, {}):
        c0 = jnp.zeros(X.shape[1]) if c_arr is None else c_arr
        Xm = cache.matvec_design(X) if cache is not None else _matvec_design(X)

        def term(b, w):
            return _node_log_mgf_binary(Xm, b, w, c0, centered=c_arr is not None)
    else:
        def term(b, w):
            return _node_log_mgf_sparse(X, b, w, c_arr, entry_chunk)
    if refs is not None:
        refs = [(X, c_arr, *r) for r in refs]
    elif cache is not None:
        refs = [(X, c_arr, jnp.asarray(b), jnp.asarray(w)) for b, w in effects]
    return _q1_fold(n, effects, intercept, keys, cache, refs, term)


@dataclass(frozen=True)
class PoissonLogNormalOffset(Smoother):
    """Analytic Q2 offset-integrated cumulant for the Poisson (log-link) base.

    `Atilde(eta) = E_o[e^{eta+o}] = e^{eta + logkappa}` in closed form (the log-normal /
    MGF identity), with `logkappa` the per-row zero-mean offset log-MGF built from the
    OTHER effects' Gaussian-mixture laws (`build_aux[_sparse]`). A drop-in for
    `CharFnOffset`/`Compress` behind the `vi_gh` (CAVI in Q2) table seam, but with NO
    quadrature grid and NO Chebyshev table -- exact and cheaper. Convex (weight `e^{.} >
    0`); the smoothed loglik is exact, so its evidence is a true ELBO. Requires x64.

    `_cache` is the per-fit memo of the per-effect log-MGF terms (`_TermCache`): with
    `keys` a sweep builds each effect's term once, not once per fold. Mutable per-instance
    state, excluded from eq/hash/repr (the smoother is a static jit argument)."""

    _cache: _TermCache | None = field(default=None, compare=False, repr=False, hash=False)

    def __post_init__(self):
        if self._cache is None:
            object.__setattr__(self, "_cache", _TermCache())

    def validate(self, base):
        if not isinstance(base, Poisson):
            raise TypeError(
                "PoissonLogNormalOffset is the analytic Poisson (log-link) offset "
                f"cumulant; it needs a Poisson base, got {type(base).__name__}. For the "
                "logistic base use CharFnOffset; for a general base use Compress."
            )

    def build_aux(self, base, y, effects, offset_var=0.0, keys=None):
        """Dense build: `effects = [(x, alpha, mu, var)]` (design block + effect-space
        law). Returns `aux = (y, logkappa)`; the offset mean lives in eta. `keys` (one
        slot per effect, e.g. its index) enables the per-fit term memo."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_dense(
            effects, y.shape[0], offset_var=offset_var, keys=keys, cache=self._cache
        )

    def build_aux_sparse(self, base, y, X, effects, offset_var=0.0, entry_chunk=1 << 16,
                         colmean=None, keys=None):
        """Sparse (BCOO) build: `X` shared, `effects = [(alpha, mu, var)]`. Same aux.
        `colmean` (p,), if given, treats `X` as centered via the baseline+support split."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_sparse(
            X, effects, offset_var=offset_var, entry_chunk=entry_chunk, colmean=colmean,
            keys=keys, cache=self._cache,
        )

    def terms(self, base: ResponseModel, eta, aux):
        """Exact Poisson terms with the offset folded into the rate: `Atilde = e^{eta +
        logkappa}`, all derivatives equal. `eta`, `y`, `logkappa` broadcast elementwise
        (the kernels open the GH-node axis on every leaf)."""
        y, logk = aux
        return _poisson_terms(y, eta, logk)


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

    `_cache` memoizes the per-effect node terms across the folds of a fit (`_TermCache`),
    as in `PoissonLogNormalOffset`.
    """

    _cache: _TermCache | None = field(default=None, compare=False, repr=False, hash=False)

    def __post_init__(self):
        if self._cache is None:
            object.__setattr__(self, "_cache", _TermCache())

    def validate(self, base):
        if not isinstance(base, Poisson):
            raise TypeError(
                "PoissonSelfNormOffset is the analytic free-form (Q1) Poisson (log-link) "
                f"offset cumulant; it needs a Poisson base, got {type(base).__name__}. For "
                "a general base use CompressSelfNorm; for Gaussian-q CAVI in Q2 use "
                "PoissonLogNormalOffset."
            )

    def build_aux(self, base, y, X, effects, intercept=None, keys=None, refs=None):
        """Dense build: `effects = [(b_nodes, logW)]` each `(C, Q)` -- the OTHER effects'
        free-form node laws over the shared design `X`; `intercept` an optional
        `(b0_nodes, logW0)` node law. Returns `aux = (y, logkappa)`; the offset mean lives
        in eta (the node MGF is zero-mean). `keys` (effect slots) enable the per-fit term
        memo; `refs` are its validity keys (the effects' STORED node arrays -- fresh `.T`
        views would never hit)."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_q1_dense(
            X, effects, y.shape[0], intercept=intercept, keys=keys, cache=self._cache,
            refs=refs,
        )

    def build_aux_sparse(self, base, y, X, effects, intercept=None, entry_chunk=1 << 16,
                         colmean=None, keys=None, refs=None):
        """Sparse (BCOO) build via the zero-clumping identity (`log_kappa_q1_sparse`). Same
        `aux = (y, logkappa)` contract. `X` shared, `effects = [(b_nodes, logW)]`.
        `colmean` (p,), if given, treats `X` as centered via the baseline+support split.
        `keys`/`refs` as in `build_aux`."""
        self.validate(base)
        y = jnp.asarray(y)
        return y, log_kappa_q1_sparse(
            X, effects, y.shape[0], intercept=intercept, entry_chunk=entry_chunk,
            colmean=colmean, keys=keys, cache=self._cache, refs=refs,
        )

    def terms(self, base: ResponseModel, eta, aux):
        """Exact Poisson terms with the offset folded into the rate: `Atilde = e^{eta +
        logkappa}`, all derivatives equal. Identical to `PoissonLogNormalOffset.terms`;
        only the `logkappa` builder (node MGF vs Gaussian-mixture MGF) differs."""
        y, logk = aux
        return _poisson_terms(y, eta, logk)


# --------------------------------------------------------------------------------------
# Gaussian-VI effect kernels with the b-expectation in closed form. For the log link the
# Gaussian expectation over q(b) = N(m, v) is the log-normal MGF again:
#
#     E_b[e^{a_i + x_ij b}] = e^{a_i + x_ij m + 1/2 x_ij^2 v},   a_i = offset_i + logkappa_i,
#
# and every derivative of the rate is the rate itself, so the score, the Price curvature
# and the Stein sums the joint (m, v) Newton needs (`_joint_newton_step`) are moments of
# ONE per-entry rate. `glm_vi_gh_ser` integrates the same quantities by GH over b; here
# they are exact and cost one exp per entry instead of `order` of them.
# --------------------------------------------------------------------------------------


def _capped_exp(z):
    return jnp.exp(jnp.minimum(z, _LOGRATE_CAP))


def _vi_finish(m, v, prior_variance):
    """Reset non-finite features to the prior (as `glm_vi_gh_ser` does) and form the KL."""
    ok = jnp.isfinite(m) & jnp.isfinite(v) & (v > 0)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    return m, v, kl, ok


@partial(jax.jit, static_argnames=("n_iter",))
def glm_vi_poisson_ser(op, y, logk, offset, prior_variance, n_iter=100, tol=1e-8):
    """Gaussian-VI SER for the Poisson base with the offset folded as `logkappa` (n,): the
    closed-form analogue of `glm_vi_gh_ser` on `Smoothed(Poisson(), PoissonLogNormalOffset())`
    (or on the plain `Poisson()` base with `logk = 0`). Same stationarity conditions, same
    joint (m, 1/v) Newton, same return contract `(m, v, elbo_rel, coefficient_kl)`; the only
    difference is that `E_b` is exact rather than `order`-node GH."""
    from .response_ser import _joint_newton_step  # circular-import guard

    x = op.entry_x
    y_e = op.broadcast_rows(y)
    a_e = op.broadcast_rows(offset + logk)
    inv_pv = 1.0 / prior_variance

    def rate(m, v):
        return _capped_exp(a_e + x * op.broadcast_cols(m) + 0.5 * x**2 * op.broadcast_cols(v))

    def body(state):
        m, v, _, it = state
        lam = rate(m, v)
        G, W, S3, S4 = op.local_moments(((1, y_e - lam), (2, lam), (3, lam), (4, lam)))
        m_new, v_new, resid = _joint_newton_step(m, v, G, W, S3, S4, inv_pv)
        return m_new, v_new, resid, it + 1

    # Start v at the Laplace variance at b = 0, not at the prior variance: the exact
    # expectation grows like e^{x^2 v / 2}, so a wide start puts the rate at e^{100+} on a
    # large-x column and Newton crawls back from there. The objective is concave in (m, v)
    # (log-concave likelihood), so the start moves only the path, not the fixed point.
    v0 = 1.0 / (inv_pv + op.local_moment(2, rate(jnp.zeros(op.p), jnp.zeros(op.p))))
    m, v, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), v0, jnp.inf, 0),
    )
    m, v, kl, ok = _vi_finish(m, v, prior_variance)
    # E_q[ll] - ll(b=0) per entry: y x m - (E[rate] - rate0). Off-support entries of a
    # sparse design have x = 0 and contribute 0, so this is O(nnz).
    dll = y_e * x * op.broadcast_cols(m) - (rate(m, v) - _capped_exp(a_e))
    elbo_rel = jnp.where(ok, op.local_moment(0, dll) - kl, 0.0)
    return m, v, elbo_rel, kl


@partial(jax.jit, static_argnames=("n_iter",))
def glm_vi_poisson_center_ser(op, y, logk, offset, center, prior_variance, n_iter=100,
                              tol=1e-8):
    """Sparse-CENTERED `glm_vi_poisson_ser`: `eta = offset + (x_ij - c_j) b` on a BCOO
    design. A zero entry sits at `-c_j b`, and for the log link that off-support fill is
    SEPARABLE -- `sum_{i off} E[e^{a_i - c_j b}] = K_j e^{-c_j m + 1/2 c_j^2 v}` with
    `K_j = sum_{i not in supp(j)} e^{a_i}` -- so the all-rows background is exact in O(n + nnz)
    once per SER, not a per-node Chebyshev surrogate (`glm_vi_gh_center_ser`)."""
    from .response_ser import _joint_newton_step  # circular-import guard

    c = jnp.asarray(center)
    a = offset + logk
    k = _capped_exp(a)  # (n,) b = 0 rate
    y_e = op.broadcast_rows(y)
    a_e = op.broadcast_rows(a)
    k_e = op.broadcast_rows(k)
    xc_e = op.entry_x - op.broadcast_cols(c)  # (x_ij - c_j) at support entries
    # off-support totals per feature: all rows minus the support rows
    K_off = jnp.maximum(jnp.sum(k) - op.local_moment(0, k_e), 0.0)
    Y_off = jnp.sum(y) - op.local_moment(0, y_e)
    logK_off = jnp.log(jnp.maximum(K_off, 1e-300))
    inv_pv = 1.0 / prior_variance

    def rates(m, v):
        lam_e = _capped_exp(
            a_e + xc_e * op.broadcast_cols(m) + 0.5 * xc_e**2 * op.broadcast_cols(v)
        )
        log_shift = -c * m + 0.5 * c**2 * v  # log E[e^{-c b}]
        lam_off = jnp.where(K_off > 0.0, _capped_exp(logK_off + log_shift), 0.0)
        return lam_e, lam_off, log_shift

    def body(state):
        m, v, _, it = state
        lam_e, lam_off, _ = rates(m, v)
        G, W, S3, S4 = op.local_moments((
            (0, xc_e * (y_e - lam_e)),
            (0, xc_e**2 * lam_e),
            (0, xc_e**3 * lam_e),
            (0, xc_e**4 * lam_e),
        ))
        G = G - c * (Y_off - lam_off)
        W = W + c**2 * lam_off
        S3 = S3 - c**3 * lam_off
        S4 = S4 + c**4 * lam_off
        m_new, v_new, resid = _joint_newton_step(m, v, G, W, S3, S4, inv_pv)
        return m_new, v_new, resid, it + 1

    # Laplace-at-b=0 start, as in `glm_vi_poisson_ser`.
    z = jnp.zeros(op.p)
    lam_e0, lam_off0, _ = rates(z, z)
    v0 = 1.0 / (inv_pv + op.local_moment(0, xc_e**2 * lam_e0) + c**2 * lam_off0)
    m, v, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (z, v0, jnp.inf, 0),
    )
    m, v, kl, ok = _vi_finish(m, v, prior_variance)
    lam_e, lam_off, log_shift = rates(m, v)
    d_sup = op.local_moment(0, y_e * xc_e * op.broadcast_cols(m) - (lam_e - k_e))
    # off-support: -c m Y_off - (E[rate] - rate0); expm1 keeps the small-shift precision
    d_off = -c * m * Y_off - jnp.where(
        lam_off < jnp.exp(_LOGRATE_CAP), K_off * jnp.expm1(log_shift), lam_off - K_off
    )
    elbo_rel = jnp.where(ok, d_sup + d_off - kl, 0.0)
    return m, v, elbo_rel, kl
