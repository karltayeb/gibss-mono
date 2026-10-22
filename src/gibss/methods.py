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

import warnings

from . import glm
from .cf_offset import CharFnGaussRef, CharFnOffset, CharFnSelfNorm
from .cf_spectral import CharFnSpectral
from .engine import fit_ibss, fit_ibss_greedy, warm_start_effects
from .linear import is_bcoo
from .negbin import estimate_null_dispersion
from .poisson_offset import PoissonLogNormalOffset, PoissonSelfNormOffset
from .response import (
    GH,
    Adjusted,
    Bernoulli,
    Binomial,
    Compress,
    CompressSelfNorm,
    Gaussian,
    JJEnvelope,
    JJFixed,
    LogNormal,
    MacKay,
    MixtureGH,
    NegativeBinomial,
    PGMixture,
    Poisson,
    PoissonGamma,
    ProbitMixture,
    ResponseModel,
    Smoothed,
    Taylor,
    TaylorFixed,
)

__all__ = ["PRESETS", "fit_glm_susie"]


_FAMILIES = {
    "logistic": Bernoulli(),
    "binomial": Binomial(),  # y_i successes in trials_i; pass trials= (selects it by default)
    # counts with a plug-in dispersion r (size): estimated under the null model unless
    # dispersion= is given; the logistic cumulant at eta - log r, so every logistic scheme
    # applies (see response.NegativeBinomial / negbin)
    "negative_binomial": NegativeBinomial(),
    "poisson": Poisson(),
    "gaussian": Gaussian(),
}

# smoother constructors, keyed by public name; each receives the resolved config
_SMOOTHERS = {
    # gh: Gauss-Hermite on the TRUE cumulant at fixed nodes. In the Gaussian family the
    # same rule also integrates the effect (one lumped variance x^2 v + ov, the "vi"
    # kernel), so its accuracy degrades with that variance: on a near-separated feature
    # with the prior variance in the tens, five nodes put the posterior variance 3-4x off.
    # probit_mixture is exact to 6e-7 at any variance for 1.5x the cost; prefer it there.
    "gh": lambda cfg: GH(order=cfg["offset_quadrature_points"]),
    "compress": lambda cfg: Compress(
        inner=MixtureGH(order=cfg["offset_quadrature_points"]),
        M=cfg["compress_degree"],
    ),
    # self-normalized offset: fold each other effect (and the intercept) against its TRUE
    # non-Gaussian quadrature posterior -> the IBSS fixed point is the exact CAVI posterior.
    "compress_selfnorm": lambda cfg: CompressSelfNorm(M=cfg["compress_degree"]),
    "taylor2": lambda cfg: Taylor(),
    # lognormal: the exact single-Gaussian offset for the Poisson base, E e^{eta+o} =
    # e^{eta+ov/2}; closed form and certified, so it is what "gh" approximates there.
    "lognormal": lambda cfg: LogNormal(),
    # mackay: MacKay's moderated output for the logistic base, kappa softplus(eta/kappa)
    # with kappa = sqrt(1 + pi ov/8) -- gIBSS with each row reweighted by kappa and its
    # predictor divided by kappa. Closed form, no row parameter, convex at every ov; mean
    # error <= 0.016 at any ov (the logistic-probit gap), where taylor2 grows as ov^2.
    "mackay": lambda cfg: MacKay(),
    "taylor_fixed": lambda cfg: TaylorFixed(anchor=cfg["_anchor"]),
    # adjusted gIBSS: the family's own GLM with a per-row response shift, predictor shift
    # and temperature (delta, nu, kappa), tuned each effect update so it matches the
    # offset-smoothed loglik at the current predictor in value, score, curvature and third
    # derivative. The targets are GH averages on `offset_quadrature_points` nodes.
    # `_adjusted_rule` picks where the parameters are matched (see Adjusted.rule).
    # Non-quadratic, so it runs in the free-form quad kernel.
    "adjusted": lambda cfg: Adjusted(
        order=cfg["offset_quadrature_points"],
        target=cfg["_adjusted_target"],
        rule=cfg["_adjusted_rule"],
    ),
    "jj": lambda cfg: JJEnvelope(),
    "jj_fixed": lambda cfg: JJFixed(),
    # fixed normal scale mixtures of the logistic law (Bernoulli/Binomial only): the
    # offset integrates in closed form against each Gaussian component, so the error is
    # the table's sup error (4e-5 at 3 components, 6e-7 at 5, 6e-9 at 8) at ANY offset
    # variance,
    # where "gh" needs more nodes as the variance grows. probit_mixture costs one erfc
    # and one exp per component; pg_mixture (Polya-Gamma atoms) has the same guarantee
    # but needs two bivariate normal orthants per component, ~10x the cost.
    "probit_mixture": lambda cfg: ProbitMixture(order=cfg["mixture_order"]),
    "pg_mixture": lambda cfg: PGMixture(order=cfg["mixture_order"]),
    # poisson_gamma: the negative binomial by its Poisson-Gamma augmentation (NB only), the
    # exposure factor tuned locally -- per row and per candidate feature, like the localjj
    # tilt -- so the base regression problem is Poisson; closed form, no quadrature. The
    # other effects enter through their exact mixture log-MGF ("poisson_gamma") or the
    # moment-matched Gaussian of the leave-one-out message ("poisson_gamma_gaussian", the
    # localjj convention). Its Polya-Gamma counterpart (base problem Gaussian) is "jj".
    "poisson_gamma": lambda cfg: PoissonGamma(offset_law="mixture"),
    "poisson_gamma_gaussian": lambda cfg: PoissonGamma(offset_law="gaussian"),
    # cf: exact characteristic-function offset builder (CAVI in Q2, gaussian only).
    # "compress" (above) doubles as the Q2 peel builder (gaussian) and the Q1 fold
    # (unconstrained); _resolve routes on variational_family.
    "cf": lambda cfg: CharFnOffset(M=cfg["compress_degree"]),
    # cf_gaussref: the same CAVI in Q2 by the Gaussian offset reference (the probit
    # mixture at the offset variance) plus a CF tilt (logistic only, gaussian only).
    "cf_gaussref": lambda cfg: CharFnGaussRef(
        M=cfg["compress_degree"], order=cfg["mixture_order"]
    ),
    # cf_spectral: the same CAVI in Q2 with the effect b folded into the CF, so the effect
    # update is one Fourier sum per column (logistic only; sparse X must take at most
    # cf_spectral.MAX_LEVELS distinct values per column: 0/1 membership, a 0/1/2 dosage).
    # `spectral_reference` picks the Fourier residual's reference law (null / plugin /
    # gaussian; the gaussian one runs under the order-`mixture_order` probit mixture).
    "cf_spectral": lambda cfg: CharFnSpectral(
        M=cfg["compress_degree"], reference=cfg["spectral_reference"],
        mixture_order=cfg["mixture_order"],
    ),
    # spectral_gaussian: the same spectral Q2 update with each row's offset law replaced by
    # its moment-matched Gaussian N(0, V_i) (no per-effect CF factors).
    "spectral_gaussian": lambda cfg: CharFnSpectral(
        M=cfg["compress_degree"], offset_law="gaussian",
        reference=cfg["spectral_reference"], mixture_order=cfg["mixture_order"],
    ),
    # spectral_point: the same kernel with the plug-in offset law (phi = 1): gIBSS-Q2 with
    # the collapsed Newton and no CF build.
    "spectral_point": lambda cfg: CharFnSpectral(
        M=cfg["compress_degree"], offset_law="point",
        reference=cfg["spectral_reference"], mixture_order=cfg["mixture_order"],
    ),
}

_DEFAULTS = {
    "family": "logistic",
    "intercept": "shared",  # "shared" | "profiled" | "null"
    "variational_family": "unconstrained",  # "unconstrained" | "gaussian"
    "offset_integration": "none",  # "none" | key of _SMOOTHERS
    "offset_quadrature_points": 15,
    "effect_quadrature_points": 15,
    "mixture_order": 5,  # components of the probit_mixture / pg_mixture logistic tables
    "compress_degree": 48,  # Chebyshev degree M for the compress/cf offset tables
    "spectral_reference": "null",  # cf_spectral residual reference: "null" | "plugin" | "gaussian"
    "background": "exact",  # "exact" | "chebyshev" (profiled or centered kernels; the
    # front door resolves this adaptively by layout when the user leaves it unspecified)
    # preset-only knobs (no public argument; reachable via method=)
    "_anchor": "update",  # taylor_fixed expansion anchor: "update" | "null"
    "_mean_message": False,  # working-model mode: drop the message variance
    "_adjusted_target": "gh",  # adjusted targets E_o A^(k)(z + o): "gh" | "taylor"
    "_adjusted_rule": "one_point",  # adjusted matching rule: "one_point" | "two_point" | "null"
}

# Named methods = partial configs over the same axes; user kwargs override.
PRESETS = {
    "logistic": {},
    "poisson": {"family": "poisson"},
    "linear": {"family": "gaussian"},
    "smoothed": {"offset_integration": "gh"},
    "localjj": {"variational_family": "gaussian", "offset_integration": "jj"},
    "cf_cavi": {"variational_family": "gaussian", "offset_integration": "cf"},
    # the same CAVI-in-Q2 fixed point by a second route: the closed-form Gaussian-offset
    # probit mixture plus a CF tilt to the true mixture offset (exact for the 8-component
    # probit mixture of the logistic law, 6e-9 from it).
    "cf_cavi_gaussref": {
        "variational_family": "gaussian",
        "offset_integration": "cf_gaussref",
        "mixture_order": 8,
    },
    # the same CAVI-in-Q2 fixed point by the spectral effect update: no Chebyshev table,
    # no GH over b. On a two-level (0/1, possibly centered) or few-valued (0/1/2 dosage,
    # a code; dense or BCOO) design each Newton step costs O(p ntau) independent of n,
    # exactly; on a continuous dense design, a Chebyshev-in-x collapse or O(n p ntau).
    "cf_cavi_spectral": {"variational_family": "gaussian", "offset_integration": "cf_spectral"},
    # the same fixed point with the plug-in point mass as the Fourier residual's reference
    # (the sum carries only the gIBSS plug-in error, so the grid ladder stops lower)...
    "cf_cavi_spectral_plugin": {
        "variational_family": "gaussian",
        "offset_integration": "cf_spectral",
        "spectral_reference": "plugin",
    },
    # ...or the moment-matched Gaussian under the order-8 probit-mixture cumulant (the
    # residual is the offset's non-Gaussian part; the Gaussian kernel tail shortens Tmax).
    "cf_cavi_spectral_gaussref": {
        "variational_family": "gaussian",
        "offset_integration": "cf_spectral",
        "spectral_reference": "gaussian",
        "mixture_order": 8,
    },
    # its Gaussian-offset counterpart: identical kernel and grid, offset law N(0, V_i).
    "spectral_gaussian_offset": {
        "variational_family": "gaussian",
        "offset_integration": "spectral_gaussian",
    },
    # and the plug-in counterpart (gIBSS-Q2 in the spectral form): offset law = its mean.
    "spectral_point_offset": {
        "variational_family": "gaussian",
        "offset_integration": "spectral_point",
    },
    # exact free-form CAVI in Q1 through the CF product over the node laws (the CF
    # replacement for offset_integration="compress_selfnorm").
    "cf_cavi_q1": {"variational_family": "unconstrained", "offset_integration": "cf"},
    "compress_cavi": {"variational_family": "gaussian", "offset_integration": "compress"},
    # plug-in Q2 (gIBSS with a Gaussian effect): plug in the other effects' mean, no
    # offset integration; the effect b is Gaussian-VI-integrated (glm_vi_gh_ser, plain base).
    "gibss_gaussian": {"variational_family": "gaussian", "offset_integration": "none"},
    "globaljj": {"offset_integration": "jj_fixed"},
    "adjusted": {"offset_integration": "adjusted"},
    # the same (delta, nu, kappa) rule fed the Taylor-expanded targets A^(k) + ov/2 A^(k+2):
    # like-for-like with offset_integration="taylor2" (closed form, no quadrature)
    "adjusted_taylor": {"offset_integration": "adjusted", "_adjusted_target": "taylor"},
    # the other matching rules: score at the null end s plus score and curvature at the
    # mode end z ("two_point"), or all four conditions at s ("null")
    "adjusted_two_point": {"offset_integration": "adjusted", "_adjusted_rule": "two_point"},
    "adjusted_null": {"offset_integration": "adjusted", "_adjusted_rule": "null"},
    # Negative binomial SuSiE by data augmentation (NB is the default family of both). Three
    # routes, named by the base regression problem each reduces to; `family=
    # "negative_binomial"` with any logistic method (the default quad kernel, localjj,
    # cf_cavi, the spectral presets, ...) is the first:
    #   logistic      -- NB(y | mu, r) = Binomial(y + r "trials", sigmoid(eta - log r)):
    #                    the logistic cumulant at scale y + r, exact; no augmentation.
    #   poisson_gamma -- y | w ~ Poisson(w e^eta), w ~ Gamma(r, r): the Poisson problem with
    #                    a Gamma exposure factor q(w_i | gamma = j) tuned per row and
    #                    candidate feature (local); closed-form Gaussian-VI effect updates on
    #                    the collapsed bound y eta - (y + r) log(r + E e^eta) (Jensen).
    #   polya_gamma   -- NB(y | psi) = 2^-(y+r) e^{kappa psi} E_w[e^{-w psi^2 / 2}],
    #                    w ~ PG(y + r, 0): the Gaussian (conjugate) problem with the PG factor
    #                    q(w_i | gamma = j) = PG(y + r, xi_ij) tuned per row and feature. The
    #                    PG factor's parameter IS the Jaakkola-Jordan tilt, so this is localjj
    #                    on the NB base (kernel 'jj'); globaljj is the row-global mean field
    #                    q(b) q(w) of the same augmentation.
    "poisson_gamma": {
        "family": "negative_binomial",
        "variational_family": "gaussian",
        "offset_integration": "poisson_gamma",
    },
    "polya_gamma": {
        "family": "negative_binomial",
        "variational_family": "gaussian",
        "offset_integration": "jj",
    },
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
        if vfam == "gaussian" and isinstance(response, Smoothed) and isinstance(
            response.smoother, (CharFnOffset, Compress, PoissonLogNormalOffset)
        ):
            # an explicit Q2 offset-table response (the object form of cf_cavi /
            # compress_cavi): it needs the vi_gh kernel, whose aux is the table -- the
            # generic "vi" fallthrough below would hand it (y, ov) and fail.
            return response, "vi_gh"
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
                # EXACT free-form CAVI in Q1 through the CF product: the quad kernel's
                # self-normalized node measure has a closed-form (finite-sum) CF, so the
                # other effects fold as a product of node-law factors (CharFnSelfNorm) --
                # the CF replacement for the sequential 'compress_selfnorm' peel. The
                # Poisson base keeps its analytic node-MGF fold.
                offset = (
                    PoissonSelfNormOffset()
                    if isinstance(base, Poisson)
                    else CharFnSelfNorm(M=cfg["compress_degree"])
                )
                return Smoothed(base, offset), "quad"
            offset = (
                PoissonLogNormalOffset()
                if isinstance(base, Poisson)
                else CharFnOffset(M=cfg["compress_degree"])
            )
            return Smoothed(base, offset), "vi_gh"
        elif integ == "cf_gaussref":
            # CAVI in Q2 through the Gaussian offset reference + CF tilt (CharFnGaussRef):
            # the logistic table's Gaussian-q counterpart of "cf", same vi_gh table seam.
            if vfam != "gaussian":
                raise ValueError(
                    "offset_integration='cf_gaussref' is CAVI in Q2; it needs "
                    "variational_family='gaussian'."
                )
            if not isinstance(base, Bernoulli):
                raise ValueError(
                    "offset_integration='cf_gaussref' needs the logistic family "
                    f"(the probit mixture tables); got family={fam!r}."
                )
            return Smoothed(base, _SMOOTHERS["cf_gaussref"](cfg)), "vi_gh"
        elif integ in ("cf_spectral", "spectral_gaussian", "spectral_point"):
            # CAVI in Q2 by the collapsed spectral effect update (CharFnSpectral); the
            # intercept keeps the inherited CF table, so it rides the same vi_gh seam.
            if vfam != "gaussian":
                raise ValueError(
                    f"offset_integration={integ!r} is a Q2 update; it needs "
                    "variational_family='gaussian'."
                )
            if not isinstance(base, Bernoulli):
                raise ValueError(
                    f"offset_integration={integ!r} needs the logistic family (the "
                    f"logistic Fourier kernel); got family={fam!r}."
                )
            if cfg["intercept"] == "profiled":
                raise ValueError(
                    f"offset_integration={integ!r} supports intercept='shared' or "
                    "'null', not 'profiled'."
                )
            return Smoothed(base, _SMOOTHERS[integ](cfg)), "vi_gh"
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
        elif integ in ("poisson_gamma", "poisson_gamma_gaussian"):
            if not isinstance(base, NegativeBinomial):
                raise ValueError(
                    f"offset_integration={integ!r} is the Poisson-Gamma augmentation of the "
                    f"negative binomial; it needs family='negative_binomial' (got {fam!r})."
                )
            response = Smoothed(base, _SMOOTHERS[integ](cfg))
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
        isinstance(getattr(response, "smoother", None),
                   (Compress, PoissonSelfNormOffset, CharFnSelfNorm))
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
            "offset_integration to 'gh', 'taylor2', 'mackay', 'jj', 'lognormal' (single-Gaussian "
            "offset), "
            "'cf'/'compress' (exact-mixture CAVI), or 'none' (plug-in mean)."
        )
    return response, "vi"


def _resolve_dispersion(dispersion, y, offset, covariates):
    """The NegativeBinomial plug-in r as a positive float or (n,) array: estimated under the
    null model for None / "null", validated otherwise."""
    import numpy as _np

    if dispersion is None or (isinstance(dispersion, str) and dispersion == "null"):
        est = estimate_null_dispersion(y, offset=offset, covariates=covariates)
        if est.at_cap:
            warnings.warn(
                f"the null-model NB dispersion ran to its cap (r = {est.dispersion:.4g}): the "
                "counts show no overdispersion beyond Poisson. The fit proceeds with the capped "
                "plug-in (NB at this r is Poisson to O(mu^2/r)); family='poisson' is the "
                "better model for these data (exact, and cheaper).",
                stacklevel=3,
            )
        return est.dispersion
    if isinstance(dispersion, str):
        raise ValueError(f"dispersion must be 'null', a number or an (n,) array; got {dispersion!r}")
    r = _np.asarray(dispersion, dtype=float)
    if r.ndim == 1 and r.shape[0] != _np.shape(y)[0]:
        raise ValueError(
            f"dispersion as a per-row vector must have length n={_np.shape(y)[0]} (got {r.shape[0]})"
        )
    if r.ndim > 1:
        raise ValueError(f"dispersion must be a scalar or a length-n vector; got shape {r.shape}")
    if not _np.all(_np.isfinite(r) & (r > 0)):
        raise ValueError("dispersion must be finite and positive")
    return float(r) if r.ndim == 0 else r


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
    mixture_order=None,
    background=None,
    # plain values (no preset interaction, ordinary defaults)
    center=None,  # pre-center columns; None = on wherever supported (see below)
    trials=None,  # Binomial: per-row trial counts n_i (y_i = successes); selects family=
    # 'binomial' unless a family is named. Every method/kernel applies: the logistic
    # cumulant scales per trial, so a Binomial fit equals the Bernoulli fit on the
    # row-expanded data.
    dispersion=None,  # NegativeBinomial: the plug-in size r. None (with the family named) or
    # "null" = estimate it once under the null model (intercept + covariates + offset,
    # negbin.estimate_null_dispersion); a number or an (n,) array = use as given (e.g. edgeR /
    # DESeq2 dispersions). A number/array selects family='negative_binomial' unless a family
    # is named. Never updated by the sweeps.
    offset=0.0,  # fixed per-row offset (Poisson exposure, etc.)
    covariates=None,  # fixed (n, k) covariates Z, fit jointly with the intercept (see below)
    prior_variance=1.0,
    estimate_prior_variance=True,
    prior_variance_scale=None,  # half-normal(sigma; s) hyperprior on the prior sd (damps runaway, keeps ARD)
    max_prior_variance=None,  # hard ceiling on the estimated prior variance (None = no cap)
    freeze_prior_variance=None,  # stop updating an effect once its prior variance < this (None
    # = never); for large L, skips re-fitting effects ARD has driven null. See below.
    estimate_intercept=True,
    random_intercept=False,  # add an iid per-row random intercept b0i ~ N(0, sigma^2_ri)
    random_intercept_variance=1.0,  # sigma^2_ri (fixed value, unless estimated below)
    estimate_random_intercept_variance=False,  # EBNM M-step sigma^2=mean_i(m_i^2+v_i); off by
    # default (sigma^2 is weakly identified for an observation-level random effect -- fix it)
    max_iter=100,
    tol=1e-4,
    max_L=None,  # L="auto": cap on the greedy search (default min(20, n_features))
    tol_L=1.0,  # L="auto": stop when an added effect's ser_log_bf < tol_L (nats)
    stride=1,  # L="auto": effects added per round (>1 brackets coarsely, still exact)
    schedule=None,  # escape hatch; defaults to glm.default_schedule()
    record=None,  # opt-in gibss.history.History sink; captures full-state fitting history
    initial_state=None,  # warm start: seed q from a prior fit; call options still win
):
    """Fit a GLM SuSiE. Returns the fitted `GIBSSState`.

    Every configuration is `method=` (a named point in the grid, see PRESETS)
    and/or the axes, resolved as user value > preset value > default:

        fit_glm_susie(X, y)                          # logistic SuSiE
        fit_glm_susie(X, y, method="localjj")        # classic localjj
        fit_glm_susie(X, y, trials=n)                # Binomial(n_i, p_i) SuSiE
        fit_glm_susie(X, y, family="negative_binomial",
                      offset=log_size_factor)        # NB SuSiE, null-model plug-in r
        fit_glm_susie(X, y, dispersion=r_hat)        # NB SuSiE at a given r (or per-row r_i)
        fit_glm_susie(X, y, dispersion=r_hat,
                      method="poisson_gamma")        # NB by the local Poisson-Gamma augmentation
        fit_glm_susie(X, y, dispersion=r_hat,
                      method="polya_gamma")          # NB by the local Polya-Gamma augmentation
        fit_glm_susie(X, y, family="poisson",
                      intercept="profiled")          # profiled Poisson SuSiE
        fit_glm_susie(X, y, family=Smoothed(Bernoulli(), GH(5)))  # object escape

    `effect_quadrature_points` is the GH tail for integrating the effect b
    (unconstrained q only); `offset_quadrature_points` is the nested GH inside
    offset_integration="gh"; `mixture_order` is the number of components of the
    offset_integration="probit_mixture" / "pg_mixture" logistic tables (2 to 8). In the
    Gaussian family "gh" integrates the effect with those same nodes, so its accuracy
    falls with the per-entry variance x^2 v + ov (a near-separated feature at prior
    variance 100 puts its posterior variance 3-4x off); "probit_mixture" keeps its 6e-7
    at any variance and is the better choice there.
    For the logistic family offset_integration="mackay" is MacKay's closed-form
    moderated output, kappa softplus(eta / kappa) with kappa = sqrt(1 + pi ov / 8): no
    quadrature, convex at every variance, mean error within 0.016 of the exact offset.
    For the Poisson family offset_integration="lognormal" is the exact single-Gaussian
    offset, e^{eta + ov/2}, closed form and certified; "gh" only approximates it.
    Axis arguments not consulted by the resolved configuration are ignored (tuning
    knobs), but combinations implying a wrong model are rejected -- see `_resolve`.

    Pass `record=History()` (see `gibss.history`) to capture the full fitting history --
    a host-numpy snapshot after every single-effect update -- without putting any of it on
    the returned state. Read it off the History object afterwards:

        hist = History()
        state = fit_glm_susie(X, y, method="cf_cavi", record=hist)
        hist.states  # full resumable GIBSSState at each recorded step

    Pass `covariates=Z` (dense (n, k)) to adjust for fixed covariates: eta = b0 + Z gamma
    + sum_l X b_l. (b0, gamma) is ONE Gaussian nuisance factor with a diffuse prior, fit
    like the shared intercept (its mean rides in eta, its per-row spread in the effect
    offset variance -- dropped by the plug-in modes, integrated by exact Q2 CAVI). Shared
    intercept only (not intercept='profiled', not the free-form Q1 intercept). Do not
    include a constant column. `state.family_state.covariate_mean` / `.nuisance_cov` hold
    the posterior; `summarize_fit` reports it.

    Pass `initial_state=<a prior GIBSSState>` to WARM START: the fit resumes from
    that state's per-effect posterior q (its `mu`/`var`/`alpha`), while EVERY other
    option -- family, kernel, intercept, L, prior settings, centering -- comes from
    THIS call and wins over whatever `initial_state` was fit with. This resumes a
    partial fit (more iterations), or refines a cheap fit with a better kernel:

        warm = fit_glm_susie(X, y, method="localjj", max_iter=5)
        state = fit_glm_susie(X, y, method="cf_cavi", initial_state=warm)

    Effects are seeded strongest-first into the L slots this call allocates (extra
    effects grow cold, surplus ones are dropped). The design must have the same
    features (X columns) as `initial_state` was fit on.
    """
    explicit = {
        k: v
        for k, v in {
            "family": family,
            "intercept": intercept,
            "variational_family": variational_family,
            "offset_integration": offset_integration,
            "offset_quadrature_points": offset_quadrature_points,
            "effect_quadrature_points": effect_quadrature_points,
            "mixture_order": mixture_order,
            "background": background,
        }.items()
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
    family_named = "family" in explicit or (method is not None and "family" in PRESETS[method])
    if trials is not None and not family_named:
        cfg["family"] = "binomial"  # trials= means Binomial unless the family is spelled out
    if dispersion is not None and not family_named:
        cfg["family"] = "negative_binomial"  # likewise dispersion= means NegativeBinomial

    response, kernel = _resolve(cfg)
    base = response.base if isinstance(response, Smoothed) else response
    if isinstance(base, NegativeBinomial):
        if trials is not None:
            raise ValueError(
                "trials= is the Binomial trial count; family='negative_binomial' takes dispersion=."
            )
        dispersion = _resolve_dispersion(dispersion, y, offset, covariates)
    elif dispersion is not None:
        raise ValueError(
            f"dispersion= is the NegativeBinomial size, but the resolved family is "
            f"{type(base).__name__}; use family='negative_binomial' (the default when "
            "dispersion= is given) or drop dispersion=."
        )
    if isinstance(base, Binomial) and trials is None:
        raise ValueError(
            "family='binomial' needs trials= (per-row trial counts n_i; y_i are the successes)."
        )
    if trials is not None and not isinstance(base, Binomial):
        raise ValueError(
            f"trials= is the Binomial trial count, but the resolved family is "
            f"{type(base).__name__}; use family='binomial' (the default when trials= is "
            f"given) or drop trials=."
        )

    if random_intercept:
        # random_intercept_variance is a scalar sigma^2 (homogeneous, optionally EM-estimated)
        # OR a length-n vector of KNOWN per-row prior variances var(b0i)=sigma^2_i.
        import numpy as _np
        _riv = _np.asarray(random_intercept_variance, dtype=float)
        if _riv.ndim == 1:
            if _riv.shape[0] != X.shape[0]:
                raise ValueError(
                    f"random_intercept_variance as a per-row vector must have length "
                    f"n={X.shape[0]} (got {_riv.shape[0]}); it is the known prior "
                    f"var(b0i)=sigma^2_i."
                )
            if estimate_random_intercept_variance:
                raise ValueError(
                    "a per-row random_intercept_variance vector is a KNOWN heteroskedastic "
                    "prior; set estimate_random_intercept_variance=False (a per-row variance "
                    "is not estimable from one observation per row)."
                )
            random_intercept_variance = _riv
        elif _riv.ndim != 0:
            raise ValueError(
                "random_intercept_variance must be a scalar or a length-n vector; got shape "
                f"{_riv.shape}."
            )

    # Pre-centering support depends on kernel AND builder. Centering orthogonalizes the
    # intercept from the effects, so the mean-field factorization q(b0) prod q(b_l) -- whose
    # premise IS posterior independence -- becomes accurate; always desirable where the fold
    # can consume it.
    #   vi_gh (Q2): dense -> both builders read the eagerly-centered X. Sparse -> every Q2
    #     fold consumes centering: the Compress peel has the -c_j off-support background
    #     split, and the CF (Bernoulli) and MGF (Poisson) folds recover the same split
    #     closed-form (baseline G_l + support jump), so none densify.
    #   compress_selfnorm (Q1): dense + sparse. Bernoulli via the self-normalized centered
    #     fold; Poisson via the node-MGF baseline+support split (log_kappa_q1_sparse).
    #   quad / linear (classic): dense eager; sparse via glm_center_ser / CenteredOperator.
    #   vi / jj: dense eager; sparse via the 2-D (c m, c^2 v) background of
    #     glm_vi_center_ser / glm_jj_center_ser (their per-entry variance/tilt puts a second
    #     per-feature parameter on the off-support fill).
    # Default (center=None) is CENTERED on BOTH layouts -- the canonical model, so a method
    # name means one model and a dense and a BCOO design fit the same thing.
    if center is None:
        center = True

    data = glm.prep_data(
        X, y, center=center, n_trials=trials, covariates=covariates, dispersion=dispersion
    )
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
        family_state_kwargs={
            "kernel": kernel,
            "intercept": cfg["intercept"],
            "quadrature_order": cfg["effect_quadrature_points"],
            "background": cfg["background"],
            "glm_offset": offset,
            "estimate_intercept": estimate_intercept,
            "random_intercept": random_intercept,
            "random_intercept_prior_variance": random_intercept_variance,
            "estimate_random_intercept_variance": estimate_random_intercept_variance,
            "estimate_prior_variance": estimate_prior_variance,
            "prior_variance_scale": prior_variance_scale,
            "max_prior_variance": max_prior_variance,
            "skl_tolerance": tol,
        },
    )
    if initial_state is not None:
        if greedy:
            raise ValueError(
                "initial_state (warm start) is not supported with L='auto' (greedy "
                "forward-selection decides L itself, so a seeded q would fight the "
                "search). Pass an explicit integer L to warm start."
            )
        state = warm_start_effects(
            state, initial_state, data,
            fixed_prior_variance=None if estimate_prior_variance else prior_variance,
        )

    sched = schedule if schedule is not None else glm.default_schedule()
    if freeze_prior_variance is not None:
        # once an effect's prior variance shrinks below the threshold, drop it from
        # update_order so large-L fits stop re-fitting the null effects (ARD-pruned).
        from functools import partial

        from .engine import add_step, freeze_shrunk_effects_step
        sched = add_step(
            sched,
            after_sweep=partial(
                freeze_shrunk_effects_step, min_prior_variance=float(freeze_prior_variance)
            ),
        )
    if greedy:
        return fit_ibss_greedy(
            data, state, sched, tol_L=tol_L, stride=stride, max_L=L_alloc,
            max_iter=max_iter, recorder=record,
        )
    return fit_ibss(data, state, sched, max_iter=max_iter, recorder=record)
