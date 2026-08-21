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
from .cf_offset import CharFnOffset
from .poisson_offset import PoissonLogNormalOffset, PoissonSelfNormOffset
from .engine import fit_ibss, fit_ibss_greedy
from .linear import is_bcoo
from .response import (
    GH,
    Bernoulli,
    Compress,
    CompressSelfNorm,
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
    # self-normalized offset: fold each other effect (and the intercept) against its TRUE
    # non-Gaussian quadrature posterior -> the IBSS fixed point is the exact CAVI posterior.
    "compress_selfnorm": lambda cfg: CompressSelfNorm(M=cfg["compress_degree"]),
    "taylor2": lambda cfg: Taylor(),
    "taylor_fixed": lambda cfg: TaylorFixed(anchor=cfg["_anchor"]),
    "jj": lambda cfg: JJEnvelope(),
    "jj_fixed": lambda cfg: JJFixed(),
    # cf: exact characteristic-function offset builder (CAVI in Q2, gaussian only).
    # "compress" (above) doubles as the Q2 peel builder (gaussian) and the Q1 fold
    # (unconstrained); _resolve routes on variational_family.
    "cf": lambda cfg: CharFnOffset(M=cfg["compress_degree"]),
}

_DEFAULTS = {
    "family": "logistic",
    "intercept": "shared",  # "shared" | "profiled" | "null"
    "variational_family": "unconstrained",  # "unconstrained" | "gaussian"
    "offset_integration": "none",  # "none" | key of _SMOOTHERS
    "offset_quadrature_points": 15,
    "effect_quadrature_points": 15,
    "compress_degree": 48,  # Chebyshev degree M for the compress/cf offset tables
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
    "cf_cavi": {"variational_family": "gaussian", "offset_integration": "cf"},
    "compress_cavi": {"variational_family": "gaussian", "offset_integration": "compress"},
    # plug-in Q2 (gIBSS with a Gaussian effect): plug in the other effects' mean, no
    # offset integration; the effect b is Gaussian-VI-integrated (glm_vi_gh_ser, plain base).
    "gibss_gaussian": {"variational_family": "gaussian", "offset_integration": "none"},
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
        elif integ == "cf":
            # CAVI in Q2 ONLY: the EXACT offset-integrated cumulant needs Gaussian effect
            # posteriors (free-form Q1 has no closed-form transform). The offset is
            # integrated over the OTHER effects' Gaussian mixture and b by GH -> vi_gh.
            # The exact builder is family-specific: the logistic base uses the psi-tamed
            # characteristic-function quadrature (CharFnOffset); the Poisson (log-link)
            # base has an ANALYTIC offset cumulant e^{eta+logkappa} (the log-normal MGF =
            # the CF at t=-i), so it uses the closed-form PoissonLogNormalOffset -- exact
            # and cheaper, no quadrature grid.
            if vfam != "gaussian":
                raise ValueError(
                    "offset_integration='cf' (CAVI in Q2) needs variational_family="
                    "'gaussian'; free-form Q1 posteriors have no closed-form "
                    "characteristic function -- use 'compress' or 'compress_selfnorm'."
                )
            offset = (
                PoissonLogNormalOffset()
                if isinstance(base, Poisson)
                else CharFnOffset(M=cfg["compress_degree"])
            )
            return Smoothed(base, offset), "vi_gh"
        elif integ == "compress":
            # CAVI in Q2 (gaussian): the Compress peel as the offset-integrated-cumulant
            # builder, interchangeable with "cf" behind the same table seam; GH over b.
            if vfam == "gaussian":
                return Smoothed(base, _SMOOTHERS["compress"](cfg)), "vi_gh"
            # unconstrained: the old moment-projected Q1 path is REMOVED -- free-form
            # effects with a Gaussian-moment offset is dominated by 'compress_selfnorm'
            # (exact free-form Q1, same fold cost -- it reuses the quad nodes 'compress'
            # would compute and discard) and by 'compress'/'cf' + gaussian (exact Q2).
            raise ValueError(
                "offset_integration='compress' + variational_family='unconstrained' is "
                "not supported (it was a moment-projected Q1 approximation). Use "
                "offset_integration='compress_selfnorm' for EXACT free-form CAVI in Q1, "
                "or variational_family='gaussian' for CAVI in Q2."
            )
        elif integ == "compress_selfnorm":
            # EXACT free-form CAVI in Q1: fold each OTHER effect (and the intercept) against
            # its TRUE non-Gaussian quadrature posterior. Like the "cf" seam above, the
            # builder is family-specific: the Poisson (log-link) base has an ANALYTIC
            # node-weighted offset cumulant e^{eta+logkappa} (PoissonSelfNormOffset) -- exact
            # and cheaper, and it cures the Chebyshev-on-exp width cliff that makes the
            # compressed fold oscillate on the Poisson base; the general base uses the
            # compressed self-normalized fold (CompressSelfNorm).
            offset = (
                PoissonSelfNormOffset()
                if isinstance(base, Poisson)
                else _SMOOTHERS["compress_selfnorm"](cfg)
            )
            response = Smoothed(base, offset)
        else:
            response = Smoothed(base, _SMOOTHERS[integ](cfg))

    if getattr(response, "quadratic", False):
        return response, "linear"
    if (
        isinstance(getattr(response, "smoother", None), (Compress, PoissonSelfNormOffset))
        and vfam != "unconstrained"
    ):
        # reachable only for compress_selfnorm + gaussian (compress + gaussian returned
        # vi_gh above; the Poisson base uses the analytic PoissonSelfNormOffset, still a Q1
        # quad fold). The self-normalized fold IS the free-form Q1 offset; there is no
        # Gaussian-q meaning for it.
        raise ValueError(
            "offset_integration='compress_selfnorm' (exact free-form CAVI in Q1) needs "
            "variational_family='unconstrained' (the 'quad' kernel). For a Gaussian q "
            "(CAVI in Q2) use offset_integration='compress' or 'cf'."
        )
    if vfam == "unconstrained":
        return response, "quad"
    if not isinstance(response, Smoothed):
        # gaussian + offset_integration='none' = PLUG-IN CAVI in Q2 (gIBSS with a Gaussian
        # effect posterior): the offset is the OTHER effects' MEAN (a fixed point, no
        # integration over their uncertainty), and the effect b is integrated by GH. This
        # is glm_vi_gh_ser on a plain base -> the vi_gh kernel, no offset table. The Q1
        # analog (free-form plug-in = classic gIBSS) is unconstrained+none -> the 'quad'
        # kernel above, i.e. the default 'logistic'.
        if cfg["offset_integration"] == "none":
            return response, "vi_gh"
        raise ValueError(
            "variational_family='gaussian' needs an offset-integration scheme "
            "(the scheme is the variational expectation operator E_q): set "
            "offset_integration to 'gh', 'taylor2', 'jj' (single-Gaussian offset), "
            "'cf'/'compress' (exact-mixture CAVI), or 'none' (plug-in mean)."
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
    record=None,  # opt-in gibss.history.History sink; captures full-state fitting history
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

    Pass `record=History()` (see `gibss.history`) to capture the full fitting history --
    a host-numpy snapshot after every single-effect update -- without putting any of it on
    the returned state. Read it off the History object afterwards:

        hist = History()
        state = fit_glm_susie(X, y, method="cf_cavi", record=hist)
        hist.states  # full resumable GIBSSState at each recorded step
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

    # Pre-centering support depends on kernel AND builder. Centering orthogonalizes the
    # intercept from the effects, so the mean-field factorization q(b0) prod q(b_l) -- whose
    # premise IS posterior independence -- becomes accurate; always desirable where the fold
    # can consume it.
    #   vi_gh (Q2): dense -> both builders read the eagerly-centered X. Sparse -> the
    #     Compress peel has a centered fold (the -c_j off-support background split); the CF
    #     fold densifies under centering (off-support entries stop clumping) -> rejected.
    #   compress_selfnorm (Q1, quad): dense + sparse (its self-normalized centered fold).
    #   quad / linear (classic): dense eager; sparse via glm_center_ser / CenteredOperator.
    #   vi / jj: dense only (their per-entry variance/tilt drops the sparse off-support fill).
    # Default (center=None) is uncentered on sparse (layout-consistent), centered where free
    # on dense; an EXPLICIT center=True on an unsupported combo is an error, not a fallback.
    smoother = getattr(response, "smoother", None)
    if kernel == "vi_gh":
        if center and is_bcoo(X) and isinstance(
            smoother, (CharFnOffset, PoissonLogNormalOffset)
        ):
            # Sparse centering: the Compress peel has a centered fold (self-normalized
            # -c_j background) + a centered effect kernel (glm_vi_gh_center_ser). Both the
            # CF product and the Poisson MGF are CLOSED-FORM folds that exploit the exact
            # zeros of the design (off-support entries contribute a neutral factor); under
            # centering an off-support entry becomes -c_j (a per-feature background), so
            # the zero-clumping breaks and the fold densifies. Neither can pre-center.
            raise ValueError(
                f"offset_integration='cf' cannot pre-center a sparse (BCOO) design (the "
                f"{'characteristic-function' if isinstance(smoother, CharFnOffset) else 'Poisson MGF'} "
                "fold densifies under centering); use offset_integration='compress', "
                "center=False, or densify X."
            )
        if center is None:
            center = False  # default uncentered (layout-consistent); center=True honored
    elif isinstance(smoother, PoissonSelfNormOffset):
        # analytic Q1 Poisson fold (quad): DENSE pre-centering only. The sparse (BCOO)
        # node-MGF fold uses the zero-clumping identity (an off-support entry contributes a
        # constant), which breaks under centering (the entry becomes -c_j b) -- the same
        # densification as the CF/MGF Q2 folds. Dense centering is transparent (X is
        # centered eagerly and every fold reads the same X).
        if center and is_bcoo(X):
            raise ValueError(
                "offset_integration='compress_selfnorm' with a Poisson base cannot "
                "pre-center a sparse (BCOO) design (the analytic node-MGF fold's "
                "zero-clumping breaks under centering). Use center=False, "
                "intercept='profiled' (shift-invariant), or densify X."
            )
        if center is None:
            center = False
    elif isinstance(smoother, Compress):  # compress_selfnorm (quad): dense + sparse center
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
        return fit_ibss_greedy(
            data, state, sched, tol_L=tol_L, stride=stride, max_L=L_alloc,
            max_iter=max_iter, recorder=record,
        )
    return fit_ibss(data, state, sched, max_iter=max_iter, recorder=record)
