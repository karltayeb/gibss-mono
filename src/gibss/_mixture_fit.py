"""Minimax fitting of the fixed normal scale mixtures of the logistic law.

The tables `response._PROBIT_MIXTURES` / `_PG_MIXTURES` are the solutions of

    probit:  min_{s, pi}  sup_{t >= 0} | F(t) - sum_k pi_k Phi(s_k t) |
    pg:      min_{w, p}   sup_{t >= 0} 1/2 e^{-t/2} | sech(t/2) - sum_k p_k e^{-w_k t^2/2} |

(F the logistic CDF; both errors are odd/even about 0, so t >= 0 suffices), with the
weights on the simplex (the mixture must reach F's limits) and the scales positive. The
pg weight is e^{-t/2}/2 so the error is that of sigma(-t): the half line where the atom
representation of sigma holds.

Protocol (`fit`): the inner problem in the weights is a linear program, the outer problem
in the scales is a nonconvex nonlinear Chebyshev approximation. Each run is a sequential
linear program (`slp`) on a discrete grid: linearize the error in the log-scales and the
weights, solve the LP `min h s.t. |e + J d| <= h, sum dpi = 0, pi + dpi >= 0` inside a
trust region on the log-scales, accept or shrink. SLP finds the basin and gets the
extrema within ~1e-3 of level, but converges linearly from there, so the finish is the
classical Remez second phase (`level`): locate the continuous extrema of the error by
Brent refinement, take the 2K alternating ones, and solve the leveling system
`e(t_i) = (-1)^i h, sum pi = 1` for (log-scales, weights, h) by Newton; re-locate the
extrema and repeat until the sup error equals |h| to roundoff (quadratic convergence).
From a few hundred log-uniform random starts (`multistart`: a short SLP on a coarse grid
ranks them; the best few are leveled), the best leveled run is the row. `certify` reports
what is then known: the continuous sup error (the number the smoothing schemes rely on),
the equioscillation count against the 2K alternation points a local minimax optimum of
this 2K-1 parameter family needs, and the basin fraction (starts whose coarse value is
within 0.1% of the best) -- the evidence for globality, since there is no certificate
for the nonconvex outer problem.

    python -m gibss._mixture_fit --family probit --K 9            # from the stored K=8 row
    python -m gibss._mixture_fit --family probit --K 8 --chain 2  # every row 2..8 afresh
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import expit, ndtr

COARSE = np.r_[np.arange(0.0, 20.0, 0.02), np.arange(20.0, 60.0001, 0.2)]
FINE = np.r_[np.arange(0.0, 20.0, 0.0025), np.arange(20.0, 60.0001, 0.05)]


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


class Probit:
    """`F(t) - sum pi_k Phi(s_k t)`, scales parametrized as s = e^x."""

    name = "probit"
    log_scale_range = (np.log(0.1), np.log(2.0))

    @staticmethod
    def error(t, x, p):
        return expit(t) - ndtr(np.outer(t, np.exp(x))) @ p

    @staticmethod
    def jac(t, x, p):
        ts = np.outer(t, np.exp(x))
        return -p * _phi(ts) * ts, -ndtr(ts)  # d/dx_k, d/dp_k


class PG:
    """`1/2 e^{-t/2} (sech(t/2) - sum p_k e^{-w_k t^2/2})`, precisions w = e^x."""

    name = "pg"
    log_scale_range = (np.log(0.005), np.log(3.0))

    @staticmethod
    def error(t, x, p):
        e = np.exp(-0.5 * np.outer(t**2, np.exp(x)))
        return 0.5 * np.exp(-t / 2) * (1.0 / np.cosh(t / 2) - e @ p)

    @staticmethod
    def jac(t, x, p):
        w = np.exp(x)
        e = np.exp(-0.5 * np.outer(t**2, w))
        wt = (0.5 * np.exp(-t / 2))[:, None]
        return wt * p * e * (0.5 * np.outer(t**2, w)), -wt * e


FAMILIES = {"probit": Probit, "pg": PG}


@dataclass
class Fit:
    x: np.ndarray  # log-scales, sorted
    p: np.ndarray  # weights on the simplex
    value: float  # sup error on the grid it was fitted on

    @property
    def scales(self):
        return np.exp(self.x)


def lp_weights(fam, t, x):
    """The minimax weights for FIXED scales: a linear program on the simplex (the error is
    linear in the weights), solved in units of the uniform-weight error. The start of every
    SLP run: interior for any reasonable scale set, where a random simplex point lets the
    first steps kill a component."""
    K = len(x)
    e0 = fam.error(t, x, np.ones(K) / K)
    scale = 1.0 / np.max(np.abs(e0))
    _, Jp = fam.jac(t, x, np.ones(K) / K)
    B = -Jp * scale  # e = f - B p  ->  e(p) = f_s - B_s p with f_s the error at p = 0
    f = fam.error(t, x, np.zeros(K)) * scale
    c = np.r_[np.zeros(K), 1.0]
    A_ub = np.vstack([np.hstack([-B, -np.ones((len(t), 1))]), np.hstack([B, -np.ones((len(t), 1))])])
    b_ub = np.r_[-f, f]
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
        bounds=[(0.0, None)] * (K + 1), method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    return res.x[:K] / res.x[:K].sum()


def slp(fam, t, x, p, max_iter=300, tol=1e-12, delta=0.3, p_min=0.0):
    """Sequential LP for the discrete minimax on grid `t` from (x, p). Returns a Fit.
    `p_min` floors the weights during the search: a component whose weight reaches 0 is
    dead (its scale gradient is proportional to its weight), and the run ends on an
    effective K-1 solution; the floor is dropped for the final leveling."""
    x, p = np.asarray(x, float).copy(), np.asarray(p, float).copy()
    K = len(x)
    e = fam.error(t, x, p)
    best = float(np.max(np.abs(e)))
    c = np.r_[np.zeros(2 * K), 1.0]
    A_eq = np.r_[np.zeros(K), np.ones(K), 0.0][None]
    for _ in range(max_iter):
        Jx, Jp = fam.jac(t, x, p)
        # The LP is solved in units of the current sup error: HiGHS's feasibility tolerances
        # are absolute (1e-7 by default), and the errors being leveled are 1e-6 to 1e-8 --
        # unscaled, the returned step "achieves" a level the true error violates by up to
        # the tolerance, every late step is rejected and the trust region collapses with
        # the extrema still ~1% apart.
        scale = 1.0 / best
        J = np.hstack([Jx, Jp]) * scale
        A_ub = np.vstack([np.hstack([J, -np.ones((len(t), 1))]), np.hstack([-J, -np.ones((len(t), 1))])])
        b_ub = np.r_[-e, e] * scale
        bounds = [(-delta, delta)] * K + [(p_min - pk, None) for pk in p] + [(0.0, None)]
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0], bounds=bounds, method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        if res.status != 0:
            delta /= 4.0
            if delta < 1e-10:
                break
            continue
        predicted = best - res.fun / scale
        d = res.x
        x_new = x + d[:K]
        p_new = np.clip(p + d[K : 2 * K], p_min, None)
        p_new /= p_new.sum()
        e_new = fam.error(t, x_new, p_new)
        val = float(np.max(np.abs(e_new)))
        if val < best:
            actual = best - val
            x, p, e, best = x_new, p_new, e_new, val
            delta = min(2.0 * delta, 1.0) if actual > 0.5 * predicted else delta
            if predicted < tol * best:  # relative: the level is converged to roundoff
                break
        else:
            delta /= 4.0
            if delta < 1e-10:
                break
    order = np.argsort(x)
    return Fit(x[order], p[order], best)


def extrema(fam, x, p, t_max=80.0, h=1e-3, floor=1e-4):
    """Continuous local extrema of the error: grid-located, Brent-refined, sorted by t.
    Maxima below `floor` times the largest are dropped: the far tail is roundoff noise
    (1e-17) and every wiggle of it would count."""
    t = np.arange(0.0, t_max, h)
    e = fam.error(t, x, p)
    a = np.abs(e)
    idx = np.where((a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > floor * a.max()))[0] + 1
    out_t, out_e = [], []
    for i in idx:
        f = lambda tt: -abs(fam.error(np.array([tt]), x, p)[0])
        r = minimize_scalar(f, bounds=(t[i - 1], t[i + 1]), method="bounded", options={"xatol": 1e-12})
        out_t.append(r.x)
        out_e.append(fam.error(np.array([r.x]), x, p)[0])
    return np.asarray(out_t), np.asarray(out_e)


def polish_lq(fam, fit, t=COARSE, qs=(8, 32, 128, 512), maxiter=2000):
    """Second-order polish of an SLP result: minimize the (log) L_q norm of the error on
    grid `t` for increasing q by L-BFGS, weights through a softmax. SLP is first-order
    and stalls once the extrema are level to ~1e-3; the smooth L_q objective sees the
    curvature and carries the point to the L_inf basin bottom (q = 512 is within 1e-3 of
    the sup norm), where the Remez leveling finishes."""
    K = len(fit.x)
    scale = 1.0 / fit.value

    def unpack(z):
        a = z[K:]
        w = np.exp(a - a.max())
        return z[:K], w / w.sum()

    def make(q):
        def fobj(z):
            x, p = unpack(z)
            e = fam.error(t, x, p) * scale
            m = np.max(np.abs(e))
            return np.log(m) + np.log(np.sum((np.abs(e) / m) ** q)) / q

        def fgrad(z):
            x, p = unpack(z)
            e = fam.error(t, x, p) * scale
            Jx, Jp = fam.jac(t, x, p)
            m = np.max(np.abs(e))
            r = np.abs(e) / m
            wts = r ** (q - 1) * np.sign(e) / np.sum(r**q) / m
            gx = wts @ (Jx * scale)
            gp = wts @ (Jp * scale)
            return np.r_[gx, p * (gp - gp @ p)]

        return fobj, fgrad

    z = np.r_[fit.x, np.log(np.maximum(fit.p, 1e-12))]
    for q in qs:
        fobj, fgrad = make(q)
        res = minimize(fobj, z, jac=fgrad, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-12})
        if np.all(np.isfinite(res.x)):
            z = res.x
    x, p = unpack(z)
    value = float(np.max(np.abs(fam.error(t, x, p))))
    if value >= fit.value:
        return fit
    order = np.argsort(x)
    return Fit(x[order], p[order], value)


def _reference(t_ext, e_ext, K):
    """The alternating extrema with the largest |e|: compress same-sign runs to their
    largest member, then drop the smaller end until 2K remain. None if fewer than 2K - 1
    alternate (the leveling system is then square or one short; see `level`)."""
    if len(t_ext) == 0:
        return None
    keep_t, keep_e = [t_ext[0]], [e_ext[0]]
    for t, e in zip(t_ext[1:], e_ext[1:]):
        if np.sign(e) == np.sign(keep_e[-1]):
            if abs(e) > abs(keep_e[-1]):
                keep_t[-1], keep_e[-1] = t, e
        else:
            keep_t.append(t)
            keep_e.append(e)
    keep_t, keep_e = np.asarray(keep_t), np.asarray(keep_e)
    while len(keep_t) > 2 * K:
        if abs(keep_e[0]) < abs(keep_e[-1]):
            keep_t, keep_e = keep_t[1:], keep_e[1:]
        else:
            keep_t, keep_e = keep_t[:-1], keep_e[:-1]
    if len(keep_t) < 2 * K - 1:
        return None
    return keep_t, keep_e


def level(fam, fit, rounds=20, newton_iter=30, polish_rounds=3):
    """Remez second phase from an SLP result: Newton on the leveling system over the 2K
    alternating extrema, exchange, repeat. When the error has fewer than 2K alternating
    extrema the point is either short of the basin's bottom (the last extremum has not
    separated yet) or a degenerate local optimum (a weight at 0): a few more SLP steps on
    the fine grid plus the current extrema are tried first (`polish_rounds`), and if the
    reference is still short the Fit is returned as is -- `certify` then reports the short
    alternation."""
    K = len(fit.x)
    x, p = fit.x.copy(), fit.p.copy()
    h = fit.value
    polished = 0
    for _ in range(rounds):
        t_ext, e_ext = extrema(fam, x, p)
        ref = _reference(t_ext, e_ext, K)
        if ref is None:
            if polished >= polish_rounds:
                return Fit(x, p, float(np.max(np.abs(e_ext))))
            f = polish_lq(fam, Fit(x, p, float(np.max(np.abs(e_ext)))), np.unique(np.r_[COARSE, t_ext]))
            x, p = f.x, f.p
            polished += 1
            continue
        t_ref, e_ref = ref
        n_ref = len(t_ref)
        sig = np.sign(e_ref[0]) * (-1.0) ** np.arange(n_ref)
        h = float(np.mean(np.abs(e_ref)))
        for _ in range(newton_iter):
            e_r = fam.error(t_ref, x, p)
            Jx, Jp = fam.jac(t_ref, x, p)
            F = np.r_[e_r - sig * h, p.sum() - 1.0]
            Jac = np.block([[Jx, Jp, -sig[:, None]], [np.zeros((1, K)), np.ones((1, K)), np.zeros((1, 1))]])
            # square with 2K reference points; with 2K - 1 (observed from K = 6 on: every
            # second-order method converges to the same configuration) the minimum-norm
            # step levels the reference on its manifold
            dz = np.linalg.lstsq(Jac, -F, rcond=None)[0]
            if not np.all(np.isfinite(dz)):
                return Fit(x, p, float(np.max(np.abs(e_ext))))
            step = min(1.0, 0.3 / max(np.max(np.abs(dz[:K])), 1e-12))  # damp log-scale moves
            x = x + step * dz[:K]
            p = p + step * dz[K : 2 * K]
            h = h + step * dz[2 * K]
            if np.any(p < 0):
                return Fit(x, p, float(np.max(np.abs(fam.error(FINE, x, p)))))
            if np.max(np.abs(F)) < 1e-15 * max(abs(h), 1e-300) + 1e-300:
                break
        t_ext, e_ext = extrema(fam, x, p)
        sup = float(np.max(np.abs(e_ext)))
        if abs(sup - abs(h)) <= 1e-12 * sup:
            break
    order = np.argsort(x)
    return Fit(x[order], p[order], sup)


@dataclass
class Certificate:
    sup_error: float  # continuous, from the refined extrema
    alternation: int  # longest sign-alternating run of extrema with |e| >= (1 - slack) sup
    needed: int  # 2K: alternation points a local minimax optimum needs
    t_ext: np.ndarray
    e_ext: np.ndarray


def certify(fam, x, p, slack=1e-3):
    t_ext, e_ext = extrema(fam, x, p)
    sup = float(np.max(np.abs(e_ext)))
    big = np.abs(e_ext) >= (1.0 - slack) * sup
    signs = np.sign(e_ext[big])
    run = best = 1 if len(signs) else 0
    for i in range(1, len(signs)):
        run = run + 1 if signs[i] != signs[i - 1] else 1
        best = max(best, run)
    return Certificate(sup, best, 2 * len(x), t_ext, e_ext)


def multistart(fam, K, starts=300, seed=0, keep=5, coarse_iter=150, verbose=False):
    """Random starts (half log-uniform scales, half geometric ladders with random ends),
    each from its LP-optimal weights, a coarse-grid SLP to rank them; the `keep` best
    are leveled. Returns (leveled fits sorted by value, basin fraction: starts whose
    coarse value is within 0.1% of the best coarse value)."""
    rng = np.random.default_rng(seed)
    lo, hi = fam.log_scale_range
    coarse = []
    for i in range(starts):
        if i % 2:
            a = rng.uniform(lo, lo + 0.4 * (hi - lo))
            b = rng.uniform(hi - 0.4 * (hi - lo), hi)
            x0 = np.linspace(a, b, K)
        else:
            x0 = np.sort(rng.uniform(lo, hi, size=K))
        p0 = np.maximum(lp_weights(fam, COARSE, x0), 1e-3)
        coarse.append(slp(fam, COARSE, x0, p0 / p0.sum(), max_iter=coarse_iter))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{starts} starts, best so far {min(f.value for f in coarse):.3e}", flush=True)
    coarse.sort(key=lambda f: f.value)
    basin = float(np.mean([f.value <= 1.001 * coarse[0].value for f in coarse]))
    leveled = [level(fam, polish_lq(fam, f)) for f in coarse[:keep]]
    leveled.sort(key=lambda f: f.value)
    return leveled, basin


def continuation(fam, prev, coarse_iter=300):
    """All K-component fits reachable from the (K-1)-component Fit `prev` by inserting one
    scale: below the smallest log-scale, in each gap, above the largest (K candidates),
    weights by LP, solved by SLP on the coarse grid, polished and leveled. Returns the Fits
    sorted by value."""
    x = prev.x
    gaps = np.diff(x)
    slots = [x[0] - max(gaps[0], 0.3)] + list(x[:-1] + gaps / 2) + [x[-1] + max(gaps[-1], 0.3)]
    out = []
    for xn in slots:
        x0 = np.sort(np.r_[x, xn])
        p0 = np.maximum(lp_weights(fam, COARSE, x0), 1e-3)
        f = slp(fam, COARSE, x0, p0 / p0.sum(), max_iter=coarse_iter)
        out.append(level(fam, polish_lq(fam, f)))
    out.sort(key=lambda f: f.value)
    return out


def fit(family, K, starts=100, seed=0, verbose=False, warm=None):
    """The full protocol for one (family, K): continuation from the (K-1)-component row
    (`warm`, a Fit; by default the stored table row, or a recursive fit for K-1 when the
    table has none) plus `starts` random starts as a cross-check, every candidate leveled,
    the best kept. Returns (Fit, Certificate, agreement): `agreement` is the fraction of
    continuation candidates within 0.1% of the best."""
    fam = FAMILIES[family]
    if K <= 2:
        leveled, basin = multistart(fam, K, starts=max(starts, 50), seed=seed, verbose=verbose)
        return leveled[0], certify(fam, leveled[0].x, leveled[0].p), basin
    if warm is None:
        warm = stored_row(family, K - 1)
        if warm is None:
            warm, _, _ = fit(family, K - 1, starts=starts, seed=seed, verbose=verbose)
    cands = continuation(fam, warm)
    if verbose:
        print(f"  continuation: {[f'{f.value:.3e}' for f in cands]}", flush=True)
    if starts:
        leveled, _ = multistart(fam, K, starts=starts, seed=seed, verbose=verbose)
        if verbose:
            print(f"  random starts, best leveled: {leveled[0].value:.3e}", flush=True)
        cands = sorted(cands + leveled, key=lambda f: f.value)
    best = cands[0]
    agree = float(np.mean([f.value <= 1.001 * best.value for f in cands]))
    return best, certify(fam, best.x, best.p), agree


def stored_row(family, K):
    """The stored table row as a Fit, or None."""
    from .response import _PG_MIXTURES, _PROBIT_MIXTURES

    table = {"probit": _PROBIT_MIXTURES, "pg": _PG_MIXTURES}[family]
    if K not in table:
        return None
    scales, weights, err = table[K]
    return Fit(np.log(np.asarray(scales)), np.asarray(weights), err)


def table_row(fit, cert):
    """The row as it is stored in `response.py`: (scales, weights, sup error rounded up)."""
    sup = float(np.ceil(cert.sup_error * 1e3 / 10 ** np.floor(np.log10(cert.sup_error))) / 1e3 * 10 ** np.floor(np.log10(cert.sup_error)))
    s = ", ".join(f"{v:.13f}" for v in fit.scales)
    p = ", ".join(f"{v:.13f}" for v in fit.p)
    return f"({s}), ({p}), {sup:.2e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--starts", type=int, default=50, help="random starts per K (cross-check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain", type=int, default=None, metavar="K0",
                    help="fit K0..K in sequence, each warm from the previous (K0-1 from the stored table)")
    args = ap.parse_args()
    warm = None
    for K in range(args.chain, args.K + 1) if args.chain else [args.K]:
        best, cert, agree = fit(args.family, K, starts=args.starts, seed=args.seed, verbose=True, warm=warm)
        print(f"{args.family} K={K}: sup error {cert.sup_error:.4e}, alternation {cert.alternation}/{cert.needed}, "
              f"agreement {agree:.2f}, min log-scale gap {np.min(np.diff(best.x)):.3f}, min weight {best.p.min():.2e}",
              flush=True)
        print(f"  {K}: ({table_row(best, cert)}),", flush=True)
        warm = best
