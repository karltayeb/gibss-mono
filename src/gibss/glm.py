"""Generic GLM-SER engine family: one kernel, any ResponseModel.

`response_ser.glm_ser` fits a per-column SER for an arbitrary per-observation
likelihood; this module wraps it as a `gibss` engine family so a full SuSiE runs on
*any* `ResponseModel`. Logistic SuSiE is `GLM(Bernoulli())`, Poisson SuSiE is
`GLM(Poisson())` -- the same code path, only the response differs.

The two-group enrichment model has its own module (`twogroup`) because it needs an
outer f0/f1 M-step and the special after-effects intercept ordering (its intercept is
degenerate at b=0). For the log-concave families here the intercept is well behaved,
so it is estimated the usual way, before the effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from .engine import (
    GIBSSState,
    MeanMessage,
    Message,
    Schedule,
    add_message_index_step,
    check_alpha_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
)
from .linear import (  # noqa: F401 (re-export)
    is_bcoo,
    prep_data,
    update_prior_variance_index_step,
)
from .operators import CenteredOperator, DenseOperator, as_operator
from .response import Bernoulli, Compress, CompressSelfNorm, ResponseModel, Smoothed
from .response_ser import (
    _profile_null,
    build_ser_state,
    glm_center_ser,
    glm_jj_ser,
    glm_linear_profile_ser,
    glm_linear_ser,
    glm_profile_ser_nodes,
    glm_ser_nodes,
    glm_vi_profile_ser,
    glm_vi_ser,
)

__all__ = [
    "GLMFamilyState",
    "prep_data",
    "initialize_state",
    "initialize_state_mean_message",
    "update_effect_index_step",
    "estimate_intercept_step",
    "update_row_param_step",
    "default_schedule",
]


@dataclass(frozen=True, slots=True)
class GLMFamilyState:
    # response picks the likelihood AND its elaborations. Offset integration is a
    # family, not a flag: pass `Smoothed(Bernoulli(), GH(order=...))` (or Taylor(),
    # JJEnvelope(), JJFixed()) and the engine feeds it `(y, message_var)` as aux,
    # integrating the leave-one-out message o ~ N(mean, var) instead of using the
    # mean only (gIBSS = unwrapped response).
    response: ResponseModel = Bernoulli()  # frozen dataclass -> hashable static arg
    # intercept: HOW the shared intercept is treated -- the logistic instance of the
    # nuisance-treatment axis (cox_poisson's `baseline` is the Cox instance):
    #   "shared"   -- refit each sweep at the total predictor (estimate_intercept_step);
    #                 conditional per-feature curvature. The default.
    #   "profiled" -- a per-feature b0_j is profiled INSIDE the kernel (Schur
    #                 curvature; offset-shift-invariant evidence); no shared step.
    #   "null"     -- fit ONCE at the b = 0 null model (in initialize_state), then
    #                 frozen: the score-analysis intercept.
    intercept: str = "shared"
    intercept_value: float = 0.0
    # intercept_var: posterior variance of the shared intercept -- q(b0) =
    # N(intercept, intercept_var), the flat-prior Gaussian variational factor whose
    # coordinate update is exactly estimate_intercept's Newton + v0 = 1/sum_i w_i.
    # Flows into the smoothers' ov alongside the message variance (a CONSTANT per
    # row, O(1/n)); dropped under MeanMessage (mean-only mode) and irrelevant under
    # kernel="profile" (b0 is profiled per feature inside the kernel instead).
    intercept_var: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    prior_variance_scale: float | None = None  # half-normal(sigma; s) MAP if set; None = MLE
    max_prior_variance: float | None = None  # hard ceiling on sigma^2 (None = no cap)
    quadrature_order: int = 15
    # kernel: HOW b is integrated per column (orthogonal to `intercept`; the old
    # compound names profile/vi_profile/linear_profile were kernel x intercept pairs):
    #   "quad"    glm_ser / glm_profile_ser -- MAP + GH tail on the (smoothed)
    #             cumulant; free-form q. Refuses quadratic responses (-> "linear").
    #   "linear"  glm_linear_ser / glm_linear_profile_ser -- closed-form weighted
    #             linear regression; REQUIRED for quadratic responses (Gaussian base,
    #             TaylorFixed/JJFixed schemes). Gaussian + "linear" IS linear SuSiE;
    #             JJFixed + "linear" is globaljj; TaylorFixed + "linear" is IRLS /
    #             score mode.
    #   "vi"      glm_vi_ser / glm_vi_profile_ser -- Gaussian-restricted q(b|gamma) =
    #             N(m, v); the pointwise scheme doubles as the E_q operator. With
    #             JJEnvelope: classic localjj (shared) / centered localjj (profiled).
    #   "jj"      glm_jj_ser -- conjugate quadratic-bound SER (classic localjj);
    #             the kernel tunes the per-entry tilt itself. Shared intercept only.
    # `background` ("exact" O(n*p) / "chebyshev" O(n*D + D*p)) and `node_intercept`
    # ("linear"|"newton", quad only) apply to the profiled-intercept variants.
    kernel: str = "quad"
    background: str = "exact"
    node_intercept: str = "linear"
    # glm_offset: fixed (or family-refreshed) per-row BASE offset added to the linear
    # predictor: exposure offsets for Poisson, the Breslow log-cumulative-hazard for
    # cox_poisson (refreshed each sweep by its update_breslow_step). Scalar 0.0 = none.
    glm_offset: object = 0.0
    # row_param: the smoother's per-row engine-tuned parameter (JJFixed: the tilt
    # xi = sqrt(E[eta]^2 + V[eta]), globaljj-style; TaylorFixed: the expansion anchor
    # zhat, IRLS "update" or score "null"), refreshed by update_row_param_step from
    # the FULL message. Tuned per-row DATA that rides into the kernels through aux,
    # held as state exactly like the shared `intercept`.
    row_param: object = None
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)
    # shared intercept's raw quadrature posterior, populated only under the
    # self-normalized offset fold (CompressSelfNorm): b0_nodes (order,) and their
    # log-unnormalized weights logW0 (order,). `intercept_value` is then the quadrature
    # posterior MEAN, so the fold folds the zero-mean residual (nodes - mean) as one
    # additive non-Gaussian effect -- the exact analogue of the Gaussian N(0, iv) seed.
    intercept_b_nodes: object = None
    intercept_log_node_weight: object = None

    def __post_init__(self):
        legacy = {"profile": "quad", "vi_profile": "vi", "linear_profile": "linear"}
        if self.kernel in legacy:
            raise ValueError(
                f"kernel={self.kernel!r} was split into orthogonal axes: use "
                f"kernel={legacy[self.kernel]!r}, intercept='profiled'."
            )
        if self.kernel not in ("quad", "linear", "vi", "jj"):
            raise ValueError(
                f"unknown kernel {self.kernel!r}; use 'quad', 'linear', 'vi' or 'jj'"
            )
        if self.intercept not in ("shared", "profiled", "null"):
            raise ValueError(
                f"unknown intercept {self.intercept!r}; use 'shared', 'profiled' "
                f"or 'null'"
            )
        if self.kernel == "jj" and self.intercept == "profiled":
            raise ValueError(
                "kernel='jj' has no profiled-intercept variant (the conjugate "
                "update is shared-intercept only); use kernel='vi', "
                "intercept='profiled' (== centered localjj) instead."
            )
        quadratic = bool(getattr(self.response, "quadratic", False))
        if quadratic and self.kernel == "quad":
            raise ValueError(
                f"response {type(self.response).__name__} is quadratic (conjugate): "
                f"use kernel='linear' -- 'quad' would numerically integrate a "
                f"closed-form Gaussian (Newton + GH tail for one WLS solve)."
            )
        if not quadratic and self.kernel == "linear":
            raise ValueError(
                f"kernel='linear' needs a quadratic response (Gaussian base, or "
                f"Smoothed(base, TaylorFixed()/JJFixed())); "
                f"{type(self.response).__name__} is not."
            )


def _offset_var(state, include_intercept_var=True):
    """Per-row variance of the random offset: the message variance plus the shared
    intercept's variance (a constant; excluded under MeanMessage = mean-only mode,
    and when updating the intercept itself -- mean-field, own factor excluded)."""
    fs = state.family_state
    ov = jnp.asarray(state.total_message.var)
    if include_intercept_var and not isinstance(state.total_message, MeanMessage):
        ov = ov + fs.intercept_var
    return ov


def _aux(data, state, include_intercept_var=True):
    """Kernel aux: y; a Smoothed response adds the offset variance (and, for
    row-parameterized schemes under the quadrature/profile kernels, the per-row
    parameter tuned by update_row_param_step -- the jj kernel builds its own
    per-entry tilt instead)."""
    y = jnp.asarray(data.y)
    fs = state.family_state
    if not isinstance(fs.response, Smoothed):
        return y
    ov = _offset_var(state, include_intercept_var)
    smoother = fs.response.smoother
    if isinstance(smoother, Compress):
        # M1: the offset is the aggregate message, treated as a single Gaussian
        # o ~ N(0, ov) (its mean already lives in the fixed offset). Compress
        # precomputes the K=1 residual tables once here; M2 replaces this with the
        # per-target sequential fold over the OTHER effects' mixtures.
        ov = jnp.broadcast_to(ov, (y.shape[0],))
        z = jnp.zeros((y.shape[0], 1))
        return smoother.build_aux(fs.response.base, y, z, ov[:, None], z)
    if smoother.takes_row_param and fs.kernel != "jj":
        return y, ov, jnp.asarray(fs.row_param)
    return y, ov


def _fit_effect_raw(data, fs, aux, offset, prior_variance, order):
    """(kernel, intercept) dispatch, returning the raw per-feature (mu, var, log_bf,
    coefficient_kl, nodes) so wrappers (e.g. cox_poisson's partial-likelihood read-out)
    can adjust before assembling the SER state. `nodes` is `(b_nodes, log_node_weight)`
    -- the adaptive-GH quadrature representation of the per-feature posterior, each
    (order, p) -- for the plain `quad`/shared kernel (which the self-normalized offset
    fold consumes), else `None`."""
    offset = jnp.asarray(offset)
    op = as_operator(data.X)
    profiled = fs.intercept == "profiled"
    center = getattr(data, "column_center", None)
    if center is not None and not profiled:
        # sparse (BCOO) implicit pre-centering: eta = offset + (x_ij - c_j) b_j with c_j
        # fixed. Profiled is invariant to column shifts, so it ignores `center`.
        if fs.kernel == "quad":
            # nonlinear per-feature Laplace: the off-support fill-in is the 1-D row
            # background (glm_center_ser). column_center is BCOO-only (dense is centered
            # eagerly), so exact O(n*p) defeats the sparsity -> default exact to chebyshev.
            bg = "chebyshev" if fs.background == "exact" else fs.background
            return (*glm_center_ser(
                op, aux, offset, center, prior_variance, fs.response,
                order=order, background=bg,
            ), None)
        if fs.kernel == "linear":
            # quadratic response: w is constant, so centering is EXACT and row-wise --
            # a CenteredOperator's rank-1 rmatvec/moment(2) corrections carry it (O(nnz)
            # on BCOO, no background). Wrap the op and fall through to the linear branch.
            op = CenteredOperator.from_offsets(op, center)
        else:  # vi, jj: the entry-space per-entry variance/tilt (x^2 v) becomes a SECOND
            # per-feature parameter under centering (c_j^2 v_j), which the 1-D background
            # can't express -- a 2-D surrogate is a follow-up.
            raise NotImplementedError(
                "sparse (BCOO) pre-centering is implemented for kernel='quad' and "
                f"'linear'; kernel={fs.kernel!r} would silently fit the UNCENTERED model "
                "(its per-entry variance/tilt adds a second per-feature parameter). Pass "
                "center=False, or use intercept='profiled' (invariant to column shifts)."
            )
    nodes = None  # (b_nodes, log_node_weight) for the plain quad/shared kernel, else None
    if fs.kernel == "quad" and profiled:
        mu, var, log_bf, coefficient_kl, _, _, b_nodes, log_node_weight = (
            glm_profile_ser_nodes(
                op, aux, offset, prior_variance, fs.response, order=order,
                background=fs.background, node_intercept=fs.node_intercept,
            )
        )
        # The profile-marginalized b-posterior nodes: the CompressSelfNorm offset fold
        # folds these against o = X*b (the profiled per-feature intercept is excluded
        # from the message, and there is no shared intercept to fold), so exposing them
        # is all that is needed to make compress_selfnorm work under intercept='profiled'.
        nodes = (b_nodes, log_node_weight)
    elif fs.kernel == "quad":
        mu, var, log_bf, coefficient_kl, b_nodes, log_node_weight = glm_ser_nodes(
            op, aux, offset, prior_variance, fs.response, order=order,
        )
        nodes = (b_nodes, log_node_weight)
    elif fs.kernel == "linear":
        fn = glm_linear_profile_ser if profiled else glm_linear_ser
        out = fn(op, aux, offset, prior_variance, fs.response)
        mu, var, log_bf, coefficient_kl = out[:4]
    elif fs.kernel == "vi" and profiled:
        mu, var, log_bf, coefficient_kl, _, _ = glm_vi_profile_ser(
            op, aux, offset, prior_variance, fs.response,
            background=fs.background,
        )
    elif fs.kernel == "vi":
        mu, var, log_bf, coefficient_kl = glm_vi_ser(
            op, aux, offset, prior_variance, fs.response,
        )
    else:  # jj (shared only, validated at construction)
        mu, var, log_bf, coefficient_kl = glm_jj_ser(
            op, aux, offset, prior_variance, fs.response,
        )
    return mu, var, log_bf, coefficient_kl, nodes


def _null_log_marginal(fs, aux, offset):
    """The SER's b=0 null log marginal on the KERNEL'S OWN scale -- the reference
    feature_log_bf is measured against, stored so the absolute marginal is
    recoverable and cross-kernel comparable. Feature-independent (one scalar per
    SER). Recomputed here (cheap: one terms pass, or a scalar Newton when profiled)
    rather than threaded out of every kernel; matches exactly the baseline each
    kernel subtracts internally:
      profiled -> the profiled null (b0 maximized at b=0), _profile_null;
      jj       -> the b=0 fixed-tilt bound (xi0^2 = offset^2 + ov);
      else     -> the b=0 loglik summed over all rows (smoothed at variance ov)."""
    response = fs.response
    offset = jnp.asarray(offset)
    if fs.intercept == "profiled":
        return float(_profile_null(response, offset, aux)[1])
    if fs.kernel == "jj":
        y, ov = aux if isinstance(aux, tuple) else (aux, 0.0)
        ov = jnp.asarray(ov)
        xi0 = jnp.sqrt(jnp.maximum(offset**2 + ov, 1e-12))
        return float(jnp.sum(response.terms(offset, (y, ov, xi0))[0]))
    return float(jnp.sum(response.terms(offset, aux)[0]))


def _fit_effect(data, fs, aux, offset, prior_variance, order):
    # The kernel returns feature_log_bf relative to the b=0 null at `offset`; pair it
    # with that null's marginal so the state stores the absolute per-feature marginal
    # (comparable across kernels). alpha only needs the relative BF (shift-invariant).
    mu, var, log_bf, coefficient_kl, nodes = _fit_effect_raw(
        data, fs, aux, offset, prior_variance, order
    )
    null = _null_log_marginal(fs, aux, offset)
    b_nodes, log_node_weight = nodes if nodes is not None else (None, None)
    return build_ser_state(
        mu, var, log_bf, coefficient_kl, prior_variance, null_log_marginal=null,
        b_nodes=b_nodes, log_node_weight=log_node_weight,
    )


def _freeze_null_intercept(data, state):
    """intercept="null": fit the shared intercept ONCE at the b = 0 null model (the
    message is zero at init, so estimate_intercept IS the null fit) and freeze it --
    the score-analysis intercept, the logistic analog of cox_poisson's
    baseline="null" (frozen Nelson-Aalen)."""
    if state.family_state.intercept != "null":
        return state
    # Alternate (retune row_param at the current null predictor <-> intercept
    # Newton). Row-tuned schemes (TaylorFixed/JJFixed) hold their parameter fixed
    # inside estimate_intercept, so a single pass gives the one-step estimator;
    # the alternation converges to the exact null MLE with the anchor/tilt at it
    # (for schemes without a row parameter the first pass already is exact).
    b0_prev = None
    for _ in range(300):  # MM (JJ) converges linearly; Taylor anchors are Newton-fast
        # mean-field self-exclusion during the null fit: the intercept's own
        # variance stays out of the tilt/anchor tuning (as it does in
        # estimate_intercept's ov), so the JJ fixed point is the EXACT null MLE
        # (tilt tight at b0); v0 is recorded only once converged.
        state = replace(state, family_state=replace(state.family_state, intercept_var=0.0))
        state = update_row_param_step(data, state)
        b0, v0 = estimate_intercept(data, state)
        state = replace(state, family_state=replace(state.family_state, intercept_value=b0))
        if b0_prev is not None and abs(b0 - b0_prev) < 1e-10:
            break
        b0_prev = b0
    return replace(state, family_state=replace(state.family_state, intercept_var=v0))


def initialize_state(data, L=1, response: ResponseModel = Bernoulli(), family_state_kwargs=None,
                     prior_variance=1.0):
    from .linear import _empty_effect
    p = data.X.shape[1]
    n = data.X.shape[0]
    kw = {"response": response, **({} if family_state_kwargs is None else dict(family_state_kwargs))}
    state = GIBSSState(
        single_effects=[_empty_effect(p, prior_variance) for _ in range(L)],
        total_message=Message(jnp.zeros(n), jnp.zeros(n)),
        family_state=GLMFamilyState(**kw),
    )
    return _freeze_null_intercept(data, state)


def initialize_state_mean_message(data, L=1, response: ResponseModel = Bernoulli(), family_state_kwargs=None,
                                  prior_variance=1.0):
    """Mean-only variant: the state carries a `MeanMessage`, whose add/subtract DROP
    the incoming variance, so `total_message.var` is identically zero and every
    `Smoothed` scheme sees ov = 0 -- the working model of A itself, no offset
    integration. E.g. `Smoothed(Bernoulli(), TaylorFixed())` here is the pure IRLS
    working model (the gIBSS-style fixed-offset global expansion); GH/Taylor collapse
    to the unwrapped base family. Mirrors the message-type convention of the older
    modules (irls/localjj/globaljj/logistic_localtaylor)."""
    from .linear import _empty_effect
    p = data.X.shape[1]
    n = data.X.shape[0]
    kw = {"response": response, **({} if family_state_kwargs is None else dict(family_state_kwargs))}
    state = GIBSSState(
        single_effects=[_empty_effect(p, prior_variance) for _ in range(L)],
        total_message=MeanMessage(jnp.zeros(n)),
        family_state=GLMFamilyState(**kw),
    )
    return _freeze_null_intercept(data, state)


def estimate_intercept(data, state):
    """Concave Newton for a scalar intercept: max_{b0} sum_i loglik_i(mean + b0).

    This is the coordinate update of the flat-prior Gaussian factor q(b0) =
    N(b0, v0): the Newton solves the smoothed score equation and v0 = 1/sum_i w_i is
    the vi stationarity for a scalar all-ones column. Expectations use the EFFECTS'
    variance only (mean-field: own factor excluded from ov). Under the jj kernel the
    tilt is re-tuned at each iterate (xi^2 = eta^2 + ov: the standard JJ-MM
    alternation for a point estimate -- no b to integrate here).
    Returns (b0, v0)."""
    fs = state.family_state
    aux = _aux(data, state, include_intercept_var=False)
    total_mean = jnp.asarray(state.total_message.mean) + fs.glm_offset

    def terms_at(b0):
        eta = total_mean + b0
        if fs.kernel == "jj":
            y, ov = aux
            xi = jnp.sqrt(eta**2 + ov)
            return fs.response.terms(eta, (y, ov, xi))
        return fs.response.terms(eta, aux)

    def body(s):
        b0, it = s
        _, g, w = terms_at(b0)
        step = jnp.sum(g) / jnp.maximum(jnp.sum(w), 1e-8)
        return b0 + jnp.clip(step, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < 50, body, (jnp.asarray(fs.intercept_value), 0))
    v0 = 1.0 / jnp.maximum(jnp.sum(terms_at(b0)[2]), 1e-8)
    return float(b0), float(v0)


def estimate_intercept_quad(data, state, order=15, prior_variance=1e6):
    """Fit the shared intercept's 1-D posterior by adaptive quadrature -- a one-feature SER
    on the all-ones column (flat prior ~ large prior_variance). EXACT CAVI: the intercept
    likelihood is integrated over ALL effects' current posteriors, symmetric to how each
    effect update integrates the OTHER effects. Concretely we fit against the
    self-normalized fold of every fitted effect (no leave-one-out, intercept omitted from
    the offset) on the SMOOTHED response, so the cumulant seen at b0 is E_effects[A(b0 +
    Sum_l X b_l)] -- not the plug-in mean the old code used.

    Returns (m0, v0, b0_nodes, logW0): the posterior MEAN and variance, and the raw GH
    nodes + their log-unnormalized weights (each (order,)). Unlike `estimate_intercept`
    (Newton mode + Laplace var) this keeps the FULL non-Gaussian posterior, so the
    self-normalized offset fold can fold the intercept as one additive effect. Only called
    for the `CompressSelfNorm` path; with zero fitted effects the fold reduces to the base
    cumulant, matching the old plug-in fit exactly."""
    fs = state.family_state
    base = fs.response.base if isinstance(fs.response, Smoothed) else fs.response
    n = data.X.shape[0]
    # effects' MEAN rides in eta (the fold is re-centered to obar=0), matching
    # `_selfnorm_fold_aux`; b0 is integrated against the effects-integrated cumulant.
    offset = jnp.asarray(state.total_message.mean) + fs.glm_offset  # effects only
    ones = DenseOperator(jnp.ones((n, 1)))
    comp = fs.response.smoother if isinstance(fs.response, Smoothed) else None
    if isinstance(comp, CompressSelfNorm):
        fold_aux = _selfnorm_fold_aux(
            data, state, comp, base, jnp.asarray(data.y), n,
            state.single_effects, include_intercept=False,
        )
        mu, var, _, _, b0_nodes, logW0 = glm_ser_nodes(
            ones, fold_aux, offset, prior_variance, fs.response, order=order,
        )
    else:  # non-self-normalized fallback: plug-in fit at the effects' mean
        mu, var, _, _, b0_nodes, logW0 = glm_ser_nodes(
            ones, jnp.asarray(data.y), offset, prior_variance, base, order=order,
        )
    return float(mu[0]), float(var[0]), b0_nodes[:, 0], logW0[:, 0]


def estimate_intercept_step(data, state):
    fs = state.family_state
    # profiled: b0 is per-feature inside the kernel; null: frozen since init
    if fs.intercept != "shared" or not fs.estimate_intercept:
        return state
    if isinstance(getattr(fs.response, "smoother", None), CompressSelfNorm):
        # self-normalized fold: keep the intercept's full (non-Gaussian) posterior; the
        # value in eta is the posterior MEAN, the nodes ride for the zero-mean fold.
        m0, v0, b0n, lw0 = estimate_intercept_quad(data, state, order=fs.quadrature_order)
        return replace(state, family_state=replace(
            fs, intercept_value=m0, intercept_var=v0,
            intercept_b_nodes=b0n, intercept_log_node_weight=lw0,
        ))
    b0, v0 = estimate_intercept(data, state)
    return replace(state, family_state=replace(fs, intercept_value=b0, intercept_var=v0))


def update_row_param_step(data, state):
    """Refresh the smoother's per-row engine-tuned parameter from the FULL message
    (all effects) + intercept, before each effect update. JJFixed: the globaljj tilt
    xi = sqrt(E[eta]^2 + V[eta]); TaylorFixed: the working-model anchor zhat (the
    current predictor for anchor="update" = IRLS, the intercept-only null for
    anchor="null" = score mode). No-op for schemes without a row parameter, and
    under kernel="jj" (which tunes its own per-entry tilt from the b-posterior)."""
    fs = state.family_state
    if (
        fs.kernel == "jj"
        or not isinstance(fs.response, Smoothed)
        or not fs.response.smoother.takes_row_param
    ):
        return state
    param = fs.response.smoother.row_param(
        jnp.asarray(state.total_message.mean),
        _offset_var(state),  # message variance + intercept variance
        fs.intercept_value + fs.glm_offset,  # base = every non-effect part of the predictor
    )
    return replace(state, family_state=replace(fs, row_param=param))


def _effect_offset(fs, state):
    """The fixed offset an effect update sees: LOO message mean + glm_offset, plus
    the shared intercept on the non-profiled kernels."""
    base = fs.glm_offset if fs.intercept == "profiled" else fs.glm_offset + fs.intercept_value
    return base + state.total_message.mean


def _selfnorm_fold_aux(data, state, comp, base, y, n, others, include_intercept=True):
    """Self-normalized (`CompressSelfNorm`) offset aux: fold a list of effects -- and,
    when `include_intercept`, the shared intercept -- against their TRUE, non-Gaussian
    posteriors via raw quadrature `(b_nodes, logW)`, not Gaussian moments. Effects carry
    their quad nodes as (order, p) (transposed to the fold's (p, Q)); unfit/empty effects
    (no nodes) contribute 0 and are skipped. The intercept is one more ADDITIVE effect
    (all-ones column) whose zero-mean residual (nodes - posterior mean, the mean kept in
    eta) seeds the fold. Re-centers to obar=0 like the Gaussian path (the offset mean lives
    in eta).

    Used two ways: for an EFFECT update `others` is the leave-one-out list and the
    intercept is folded in (`include_intercept=True`); for the INTERCEPT update `others`
    is ALL fitted effects and the intercept is omitted (`include_intercept=False`), so the
    fold is the effects-integrated cumulant the intercept's exact-CAVI factor integrates."""
    fs = state.family_state
    # Guard: a FITTED effect must carry quad nodes for the self-normalized fold. An
    # unfit/empty effect legitimately has b_nodes=None (it contributes 0 and is
    # dropped), and is identified by kl=inf (the `_empty_effect` sentinel; a fitted
    # effect has finite kl). A fitted effect with no nodes would be SILENTLY dropped
    # here, folding a zero offset and producing a wrong fit -- e.g. before profiled
    # kernels exposed their nodes. Fail loudly instead.
    for e in others:
        if e.b_nodes is None and math.isfinite(float(e.kl)):
            raise ValueError(
                "CompressSelfNorm offset fold requires per-feature quad nodes on every "
                "fitted effect, but a fitted effect (finite kl) has b_nodes=None. This "
                "kernel/intercept combination does not expose quadrature nodes; the fold "
                "would silently use a zero offset. Use kernel='quad'."
            )
    others = [e for e in others if e.b_nodes is not None]  # skip unfit effects
    # intercept as an additive non-Gaussian effect (zero-mean; mean is intercept_value).
    have_icpt = (
        include_intercept
        and not isinstance(state.total_message, MeanMessage)
        and fs.intercept_b_nodes is not None
        and float(fs.intercept_var) > 0.0
    )
    if have_icpt:
        bc = jnp.asarray(fs.intercept_b_nodes) - float(fs.intercept_value)  # centered (order,)
        lw0 = jnp.asarray(fs.intercept_log_node_weight)
        iv = float(fs.intercept_var)

    if is_bcoo(data.X):
        effects = [(e.b_nodes.T, e.log_node_weight.T) for e in others]
        init = None
        if have_icpt:
            init = comp.build_selfnorm_init(
                base, y, jnp.ones((n, 1)), bc[None, :], lw0[None, :], iv
            )
        aux = comp.build_aux_selfnorm_sequential_sparse(base, y, data.X, effects, init=init)
    else:
        Xd = jnp.asarray(data.X)
        effects = []
        if have_icpt:  # zero-mean intercept effect first (all-ones column)
            effects.append((jnp.ones((n, 1)), bc[None, :], lw0[None, :]))
        effects += [(Xd, e.b_nodes.T, e.log_node_weight.T) for e in others]
        aux = comp.build_aux_selfnorm_sequential(base, y, effects)
    y_, _obar, _center, hw, cll, cg, cw = aux
    z = jnp.zeros_like(hw)  # re-center: obar -> 0, center -> 0 (coeffs unchanged)
    return (y_, z, z, hw, cll, cg, cw)


def _compress_fold_aux(data, state, l):
    """M2: the per-target offset aux for Compress. The EXACT offset of target l is the
    SUM of the other effects, so we fold their per-row mixtures one at a time (sparse
    zero-clumping on BCOO, dense otherwise) -- not a single moment-matched Gaussian.

    The fold's `obar` is the offset mean; the engine keeps that mean in `eta` (the fixed
    offset = LOO message mean + base), so we RE-CENTER the aux to obar=0, center=0 (the
    coefficients are unchanged, since the fit interval was centered at -obar). Then
    `Compress.terms(eta, aux)` evaluates Atilde at eta - obar = the mean-free predictor,
    with no double-counting and no change to the engine's mean-in-eta convention. Null/
    empty effects fold as point masses at 0, so they cost ~nothing and need no special
    casing."""
    fs = state.family_state
    comp, base = fs.response.smoother, fs.response.base
    y = jnp.asarray(data.y)
    n = y.shape[0]
    others = [e for i, e in enumerate(state.single_effects) if i != l]
    if isinstance(comp, CompressSelfNorm):
        return _selfnorm_fold_aux(data, state, comp, base, y, n, others)
    # the shared intercept's posterior uncertainty is part of the offset too (as it is
    # in the GH path's ov); fold it as one extra N(0, intercept_var) Gaussian. Zero
    # under profiled (intercept_var stays 0) and under MeanMessage (mean-only mode).
    iv = 0.0 if isinstance(state.total_message, MeanMessage) else float(fs.intercept_var)
    if is_bcoo(data.X):
        # sparse zero-clumping fold of the X-effects. The shared intercept's N(0, iv) is
        # folded FIRST (as the starting cumulant) rather than appended: convolution
        # commutes, so we seed the fold with the intercept's residual (a cheap K=1 build)
        # and cumulative variance iv, then fold the X-effects on top.
        effects = [(e.alpha, e.mu, e.var) for e in others]
        init = None
        if iv > 0:
            z1 = jnp.zeros((n, 1))
            _, _, _, _, cll0, cg0, cw0 = comp.build_aux(base, y, z1, jnp.full((n, 1), iv), z1)
            init = (-cll0, -cg0, cw0, iv)  # (cr0, cr1, cr2, V0)
        aux = comp.build_aux_sequential_sparse(base, y, data.X, effects, init=init)
    else:
        Xd = jnp.asarray(data.X)
        effects = [
            (Xd * jnp.asarray(e.mu)[None, :],
             Xd**2 * jnp.asarray(e.var)[None, :],
             jnp.broadcast_to(jnp.log(jnp.asarray(e.alpha))[None, :], Xd.shape))
            for e in others
        ]
        if iv > 0:  # intercept as a zero-mean single-Gaussian "effect"
            zc = jnp.zeros((n, 1))
            effects.append((zc, jnp.full((n, 1), iv), zc))
        aux = comp.build_aux_sequential(base, y, effects)
    y_, _obar, _center, hw, cll, cg, cw = aux
    z = jnp.zeros_like(hw)  # re-center: obar -> 0, center -> 0 (coeffs unchanged)
    return (y_, z, z, hw, cll, cg, cw)


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    # profiled: offset is the leave-one-out message (+ base glm_offset; b0 profiled
    # per feature); shared-intercept (quad/jj): also add the estimated intercept. A
    # Smoothed response additionally receives the LOO message variance through aux --
    # or, for Compress, the EXACT per-target offset folded from the other effects.
    offset = _effect_offset(fs, state)
    if isinstance(fs.response, Smoothed) and isinstance(fs.response.smoother, Compress):
        aux = _compress_fold_aux(data, state, l)
    else:
        aux = _aux(data, state)
    new_effect = _fit_effect(data, fs, aux, offset, effect.prior_variance, fs.quadrature_order)
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    return Schedule(
        before_sweep=(snapshot_state_step,),
        before_effect_update=(update_row_param_step, estimate_intercept_step),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
