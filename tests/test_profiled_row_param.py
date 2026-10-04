"""Row-parameterized schemes (globaljj's JJ tilt, IRLS's Taylor anchor) must be tuned at the
model's operating point and must score the b=0 null under a parameter tight AT the null.

Two inconsistencies this pins:
  * intercept="profiled" used to tune the row parameter at the bare message (intercept 0):
    the per-candidate b0_j lives inside the kernel, so with a true intercept of -3 the JJ
    tilt sat at |eta| ~ 0.3 instead of ~3 -- the posterior mean halved and the evidence
    collapsed. The parameter is now tuned at message + the profiled null intercept.
  * the null was scored under the parameter tuned at the full predictor (tight at the
    alternative): a loose bound (JJ) / far-from-anchor expansion (Taylor) that under-scores
    the null and inflates every BF. It is now re-scored under a null-tuned parameter.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from gibss.methods import fit_glm_susie

N, P, CAUSAL, B0 = 2000, 30, 4, -3.0


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P))
    X -= X.mean(0)  # centered: profiled and shared describe the same model
    y = rng.binomial(1, 1 / (1 + np.exp(-(B0 + 1.5 * X[:, CAUSAL]))), N).astype(float)
    return X, y


def _fit(X, y, **kw):
    return fit_glm_susie(
        X, y, L=1, estimate_prior_variance=False, prior_variance=1.0, max_iter=60, **kw
    )


def _top(st):
    e = st.single_effects[0]
    j = int(np.argmax(np.asarray(e.alpha)))
    return j, float(e.mu[j]), float(e.var[j]), float(e.ser_log_bf)


@pytest.mark.parametrize("method", ["globaljj", "irls"])
def test_profiled_row_param_is_tuned_at_the_null_intercept(method):
    X, y = _data()
    st = _fit(X, y, method=method, intercept="profiled")
    rp = np.asarray(st.family_state.row_param)
    message = np.asarray(st.total_message.mean)
    # the operating point is message + c0 with c0 the profiled null intercept GIVEN the
    # fitted message (not the intercept-only logit(ybar)): on this design that is the true
    # intercept B0 = -3, against the bare message (intercept 0) it used to be tuned at.
    if method == "irls":  # anchor zhat = message + c0
        assert abs(np.mean(rp - message) - B0) < 0.15
    else:  # tilt xi = |message + c0|, far from the bare |message| ~ 0.3
        assert np.mean(rp) > 0.8 * abs(B0)


@pytest.mark.parametrize("method", ["globaljj", "irls"])
def test_profiled_posterior_matches_exact_profiled_kernel(method):
    X, y = _data()
    j_ref, mu_ref, var_ref, _ = _top(_fit(X, y, intercept="profiled"))  # exact (quad)
    j, mu, var, _ = _top(_fit(X, y, method=method, intercept="profiled"))
    assert j == j_ref == CAUSAL
    assert abs(mu - mu_ref) < 0.03 * abs(mu_ref)  # was off by a factor ~3.7
    if method == "irls":  # the JJ bound's curvature under-states var by design; IRLS is exact
        assert abs(var - var_ref) < 0.03 * var_ref


@pytest.mark.parametrize("method", ["globaljj", "irls"])
def test_profiled_evidence_is_scored_against_a_tight_null(method):
    # the SER log BF is the alternative (parameter tight there) minus the null scored under a
    # null-tuned parameter at its own profiled intercept: on the exact profiled kernel's
    # scale (same null convention), never inflated above it, and not collapsed (the shipped
    # profiled fits reported ~27 against an exact ~106). Shared fits are not compared: each
    # method's shared intercept differs, so their nulls are not on one scale.
    X, y = _data()
    _, _, _, bf_exact = _top(_fit(X, y, intercept="profiled"))
    _, _, _, bf = _top(_fit(X, y, method=method, intercept="profiled"))
    assert bf <= bf_exact + 1.0
    assert bf >= 0.95 * bf_exact


@pytest.mark.parametrize("method", ["globaljj", "irls"])
@pytest.mark.parametrize("intercept", ["shared", "profiled"])
def test_row_param_schemes_converge_on_the_full_skl(method, intercept):
    # a saturated PIP freezes alpha at sweep 1 while the row parameter and mu are still
    # co-converging (contraction ~0.7-0.85 per sweep): the alpha-only stop left globaljj
    # at mu = 0.59 of a 1.36 fixed point after 2 sweeps. The GLM convergence check uses the
    # full SKL for these schemes, so the default fit reaches the fixed point.
    X, y = _data()
    st = _fit(X, y, method=method, intercept=intercept)
    ref = fit_glm_susie(
        X, y, L=1, method=method, intercept=intercept, estimate_prior_variance=False,
        prior_variance=1.0, max_iter=120, tol=-1.0,
    )
    _, mu, var, _ = _top(st)
    _, mu_ref, var_ref, _ = _top(ref)
    assert st.n_iter > 2
    assert abs(mu - mu_ref) < 0.02 * abs(mu_ref)
    assert abs(var - var_ref) < 0.05 * var_ref


@pytest.mark.parametrize("method", ["globaljj", "irls"])
def test_profiled_and_shared_posteriors_agree_on_a_centered_design(method):
    X, y = _data()
    _, mu_s, _, _ = _top(_fit(X, y, method=method, intercept="shared"))
    _, mu_p, _, _ = _top(_fit(X, y, method=method, intercept="profiled"))
    assert abs(mu_p - mu_s) < 0.05 * abs(mu_s)
