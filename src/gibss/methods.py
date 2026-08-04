"""User-facing front door for GLM SuSiE: named methods and semantic axes.

`fit_glm_susie(X, y)` runs logistic SuSiE; the axes (`family`, `intercept`,
`variational_family`, `offset_integration`) select every other variant, and
`method=` names the points in that grid that have names in the literature
(PRESETS is the canonical statement of what each name means -- presets are
partial configs over the same axes, so `method=` composes with explicit
overrides: user value > preset value > default).

This module only TRANSLATES: strings -> response/smoother objects, axes ->
(kernel, intercept) config. Hard validity rules live downstream
(`GLMFamilyState.__post_init__`, `Smoother.validate`); bypassing this module
loses convenience, never correctness. The only checks here are for combinations
that don't *translate* (unknown names, offset integration on a quadratic
family, Gaussian q without a smoothing scheme).
"""

from __future__ import annotations

from . import glm
from .engine import fit_ibss, fit_ibss_greedy
from .linear import is_bcoo
from .response import (
    GH,
    Bernoulli,
    Compress,
    Gaussian,
    JJEnvelope,
    JJFixed,
    MixtureGH,
    Poisson,
    ResponseModel,
    Smoothed,
    Taylor,
    TaylorFixed,
)

__all__ = ["PRESETS", "fit_glm_susie"]


_FAMILIES = {
    "logistic": Bernoulli(),
    "poisson": Poisson(),
    "gaussian": Gaussian(),
}

# smoother constructors, keyed by public name; each receives the resolved config
_SMOOTHERS = {
    "gh": lambda cfg: GH(order=cfg["offset_quadrature_points"]),
    "compress": lambda cfg: Compress(
        inner=MixtureGH(order=cfg["offset_quadrature_points"]),
        M=cfg["compress_degree"],
    ),
    "taylor2": lambda cfg: Taylor(),
    "taylor_fixed": lambda cfg: TaylorFixed(anchor=cfg["_anchor"]),
    "jj": lambda cfg: JJEnvelope(),
    "jj_fixed": lambda cfg: JJFixed(),
}

_DEFAULTS = {
    "family": "logistic",
    "intercept": "shared",  # "shared" | "profiled" | "null"
    "variational_family": "unconstrained",  # "unconstrained" | "gaussian"
    "offset_integration": "none",  # "none" | key of _SMOOTHERS
    "offset_quadrature_points": 15,
    "compress_degree": 48,  # Chebyshev degree M for offset_integration="compress"
    "effect_quadrature_points": 15,
    "background": "exact",  # "exact" | "chebyshev" (profiled or centered kernels; the
    # front door resolves this adaptively by layout when the user leaves it unspecified)
    # preset-only knobs (no public argument; reachable via method=)
    "_anchor": "update",  # taylor_fixed expansion anchor: "update" | "null"
    "_mean_message": False,  # working-model mode: drop the message variance
}

# Named methods = partial configs over the same axes; user kwargs override.
PRESETS = {
    "logistic": {},
    "poisson": {"family": "poisson"},
    "linear": {"family": "gaussian"},
    "smoothed": {"offset_integration": "gh"},
    "localjj": {"variational_family": "gaussian", "offset_integration": "jj"},
    "globaljj": {"offset_integration": "jj_fixed"},
    "irls": {"offset_integration": "taylor_fixed", "_mean_message": True},
    "score": {
        "offset_integration": "taylor_fixed",
        "_anchor": "null",
        "intercept": "null",
    },
}


def _resolve(cfg):
    """(response, kernel) from the resolved config.

    The kernel is derived, never chosen: quadratic responses take the closed
    form ("linear" -- the exact Gaussian posterior, so both variational
    families coincide); otherwise "unconstrained" -> "quad" (free-form q via
    the GH tail) and "gaussian" -> "vi", except that Gaussian q + the JJ bound
    + a shared intercept dispatches to the conjugate "jj" kernel (classic
    localjj: the fixed-tilt bound with the entry-shaped tilt tuned inside the
    kernel -- same fixed point as vi + JJEnvelope, no Newton, no GH).
    """
    vfam = cfg["variational_family"]
    if vfam not in ("unconstrained", "gaussian"):
        raise ValueError(
            f"unknown variational_family {vfam!r}; use 'unconstrained' or 'gaussian'"
        )

    fam = cfg["family"]
    if isinstance(fam, ResponseModel):
        # object passed through verbatim: offset_integration args not consulted
        response = fam
    else:
        if fam not in _FAMILIES:
            raise ValueError(
                f"unknown family {fam!r}; options: {sorted(_FAMILIES)} "
                f"(or pass a ResponseModel instance)"
            )
        base = _FAMILIES[fam]
        integ = cfg["offset_integration"]
        if integ == "none":
            response = base
        elif integ not in _SMOOTHERS:
            raise ValueError(
                f"unknown offset_integration {integ!r}; "
                f"options: {['none', *sorted(_SMOOTHERS)]}"
            )
        elif fam == "gaussian":
            raise ValueError(
                "family='gaussian' has a quadratic cumulant: offset integration "
                "is exact and free; use offset_integration='none'"
            )
        elif integ == "jj" and vfam == "gaussian" and cfg["intercept"] == "shared":
            # conjugate classic localjj; Smoother.validate rejects non-Bernoulli
            return Smoothed(base, JJFixed()), "jj"
        else:
            response = Smoothed(base, _SMOOTHERS[integ](cfg))

    if getattr(response, "quadratic", False):
        return response, "linear"
    if isinstance(getattr(response, "smoother", None), Compress) and vfam != "unconstrained":
        raise ValueError(
            "offset_integration='compress' needs variational_family='unconstrained' "
            "(the 'quad' kernel): Compress precomputes fixed-per-row offset tables, but "
            "the 'vi' kernel folds the effect's OWN variance into the per-entry offset, "
            "which those tables can't express. Use variational_family='unconstrained'."
        )
    if vfam == "unconstrained":
        return response, "quad"
    if not isinstance(response, Smoothed):
        raise ValueError(
            "variational_family='gaussian' needs an offset-integration scheme "
            "(the scheme is the variational expectation operator E_q): set "
            "offset_integration to 'gh', 'taylor2' or 'jj'"
        )
    return response, "vi"


def fit_glm_susie(
    X,
    y,
    L=5,  # int, or "auto" for greedy forward-selection (grows to max_L)
    method=None,  # name in PRESETS; None = plain axes below
    *,
    # model axes (None = "not specified": preset value, then _DEFAULTS)
    family=None,  # name in _FAMILIES, or a ResponseModel instance
    intercept=None,
    variational_family=None,
    offset_integration=None,
    offset_quadrature_points=None,
    effect_quadrature_points=None,
    background=None,
    # plain values (no preset interaction, ordinary defaults)
    center=None,  # pre-center columns; None = on wherever supported (see below)
    offset=0.0,  # fixed per-row offset (Poisson exposure, etc.)
    prior_variance=1.0,
    estimate_prior_variance=True,
    prior_variance_scale=None,  # half-normal(sigma; s) hyperprior on the prior sd (damps runaway, keeps ARD)
    max_prior_variance=None,  # hard ceiling on the estimated prior variance (None = no cap)
    estimate_intercept=True,
    max_iter=100,
    tol=1e-4,
    max_L=None,  # L="auto": cap on the greedy search (default min(20, n_features))
    tol_L=1.0,  # L="auto": stop when an added effect's ser_log_bf < tol_L (nats)
    stride=1,  # L="auto": effects added per round (>1 brackets coarsely, still exact)
    schedule=None,  # escape hatch; defaults to glm.default_schedule()
):
    """Fit a GLM SuSiE. Returns the fitted `GIBSSState`.

    Every configuration is `method=` (a named point in the grid, see PRESETS)
    and/or the axes, resolved as user value > preset value > default:

        fit_glm_susie(X, y)                          # logistic SuSiE
        fit_glm_susie(X, y, method="localjj")        # classic localjj
        fit_glm_susie(X, y, family="poisson",
                      intercept="profiled")          # profiled Poisson SuSiE
        fit_glm_susie(X, y, family=Smoothed(Bernoulli(), GH(5)))  # object escape

    `effect_quadrature_points` is the GH tail for integrating the effect b
    (unconstrained q only); `offset_quadrature_points` is the nested GH inside
    offset_integration="gh". Axis arguments not consulted by the resolved
    configuration are ignored (tuning knobs), but combinations implying a wrong
    model are rejected -- see `_resolve`.
    """
    explicit = {
        k: v
        for k, v in dict(
            family=family,
            intercept=intercept,
            variational_family=variational_family,
            offset_integration=offset_integration,
            offset_quadrature_points=offset_quadrature_points,
            effect_quadrature_points=effect_quadrature_points,
            background=background,
        ).items()
        if v is not None
    }
    if method is not None and method not in PRESETS:
        raise ValueError(f"unknown method {method!r}; options: {sorted(PRESETS)}")
    # background is layout-adaptive when unspecified: chebyshev for sparse (BCOO), where
    # the profiled/centered row background must avoid the O(n*p) exact sum, and exact for
    # dense, where exact is free and avoids the surrogate's ~1e-6 approximation. A preset
    # or explicit user value still overrides (merged after).
    cfg = {
        **_DEFAULTS,
        "background": "chebyshev" if is_bcoo(X) else "exact",
        **(PRESETS[method] if method else {}),
        **explicit,
    }

    response, kernel = _resolve(cfg)

    # Pre-centering support depends on layout AND kernel: dense X is centered eagerly
    # (every kernel), but on sparse (BCOO) X only quad (via glm_center_ser's row
    # background) and linear (row-wise exact through a CenteredOperator) consume it; the
    # vi/jj kernels' per-entry variance/tilt would drop the off-support fill-in.
    # center=None -> on wherever supported (so the default centers everything it can
    # without crashing a method sweep); an EXPLICIT center=True on an unsupported
    # sparse + vi/jj combo is a clear error, not a fit.
    # compress on a DENSE design can center (the fold reads the eagerly-centered X, so it
    # stays consistent with the kernel). On SPARSE it fundamentally cannot: zero-clumping
    # needs X_ij = 0 to be a true point mass at 0, but centering shifts every entry
    # (zeros included) by -cbar_j, leaving no zeros to clump. Default is uncentered
    # (layout-consistent: dense == sparse); dense users may opt into center=True.
    if isinstance(getattr(response, "smoother", None), Compress):
        if center is True and is_bcoo(X):
            raise ValueError(
                "offset_integration='compress' cannot center a sparse (BCOO) design: the "
                "zero-clumping fold needs X_ij = 0 to be a point mass at 0, but centering "
                "shifts every entry (zeros included) so nothing clumps. Use center=False "
                "(the intercept is handled by intercept='shared'/'profiled'), or densify X."
            )
        if center is None:
            center = False
    else:
        center_supported = (not is_bcoo(X)) or kernel in ("quad", "linear")
        if center is None:
            center = center_supported
        elif center and not center_supported:
            raise ValueError(
                f"center=True is not supported for the resolved kernel={kernel!r} on a "
                f"sparse (BCOO) design (only 'quad' and 'linear' consume sparse "
                f"pre-centering; the vi/jj kernels' per-entry variance/tilt would drop "
                f"the off-support fill-in). For the SAME intercept-decoupling benefit on "
                f"sparse, use intercept='profiled' (a per-feature intercept, invariant to "
                f"column shifts -- so centering is unnecessary). Otherwise pass "
                f"center=False, or densify X."
            )

    data = glm.prep_data(X, y, center=center)
    init = (
        glm.initialize_state_mean_message if cfg["_mean_message"] else glm.initialize_state
    )
    greedy = L == "auto"
    L_alloc = (min(20, X.shape[1]) if max_L is None else int(max_L)) if greedy else int(L)
    state = init(
        data,
        L=L_alloc,
        response=response,
        prior_variance=prior_variance,
        family_state_kwargs=dict(
            kernel=kernel,
            intercept=cfg["intercept"],
            quadrature_order=cfg["effect_quadrature_points"],
            background=cfg["background"],
            glm_offset=offset,
            estimate_intercept=estimate_intercept,
            estimate_prior_variance=estimate_prior_variance,
            prior_variance_scale=prior_variance_scale,
            max_prior_variance=max_prior_variance,
            skl_tolerance=tol,
        ),
    )
    sched = schedule if schedule is not None else glm.default_schedule()
    if greedy:
        return fit_ibss_greedy(data, state, sched, tol_L=tol_L, stride=stride, max_L=L_alloc, max_iter=max_iter)
    return fit_ibss(data, state, sched, max_iter=max_iter)
