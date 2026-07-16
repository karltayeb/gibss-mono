"""Half-normal(sigma; s) hyperprior on the SER prior variance: damps a separated
feature's runaway sigma^2 to ~s*sqrt(S) (finite, << the MLE S) while preserving
ARD (sigma^2 -> 0 for a null effect)."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss import fit_glm_susie
from gibss.linear import estimate_prior_variance
from gibss.engine import BaseSERState


def _effect(mu, var, alpha):
    # a minimal single-effect state carrying just what estimate_prior_variance reads
    return BaseSERState(
        mu=np.asarray(mu, float), var=np.asarray(var, float), alpha=np.asarray(alpha, float),
        pi=np.asarray(alpha, float), prior_variance=1.0, feature_log_marginal=np.zeros(len(alpha)),
        marginal_log_likelihood=0.0, null_log_marginal=0.0, kl=0.0,
    )


def test_estimate_prior_variance_formula():
    # concentrated posterior on one feature: S = mu^2 + var
    e = _effect([3.0, 0.0], [0.5, 0.1], [1.0, 0.0])
    S = 3.0**2 + 0.5
    assert estimate_prior_variance(e, None).prior_variance == pytest.approx(S)       # MLE
    s = 2.0
    assert estimate_prior_variance(e, s).prior_variance == pytest.approx(s * np.sqrt(s * s + S) - s * s)


def test_ard_preserved_and_damps_runaway():
    s = 4.0
    # null effect: S -> 0 => sigma^2 -> 0 (ARD kept, NOT floored above 0)
    null = _effect([0.0, 0.0], [1e-9, 1e-9], [0.5, 0.5])
    assert estimate_prior_variance(null, s).prior_variance < 1e-3
    # runaway (separated) effect: huge S is damped to ~s*sqrt(S) << S (finite, not the MLE)
    S = 1e6
    sep = _effect([np.sqrt(S)], [0.0], [1.0])
    v = estimate_prior_variance(sep, s).prior_variance
    assert v == pytest.approx(s * np.sqrt(s * s + S) - s * s, rel=1e-6)
    assert v < 0.05 * S  # heavily damped vs the MLE (sigma^2 = S)


def _separated_logistic(seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    x = np.zeros(n); members = rng.choice(n, 120, replace=False); x[members] = 1.0
    y = np.zeros(n); y[members] = 1.0  # every member is a hit -> quasi-separation
    y[rng.choice(np.setdiff1d(np.arange(n), members), 80, replace=False)] = 1.0
    return jnp.asarray(x[:, None]), jnp.asarray(y)


def test_scale_none_is_a_noop():
    X, y = _separated_logistic()
    a = fit_glm_susie(X, y, L=1, method="logistic", prior_variance_scale=None, max_iter=100)
    b = fit_glm_susie(X, y, L=1, method="logistic", max_iter=100)  # default is None
    np.testing.assert_allclose(np.asarray(a.pip), np.asarray(b.pip), atol=1e-12)
    assert float(a.single_effects[0].prior_variance) == float(b.single_effects[0].prior_variance)


def test_front_door_damps_separated_sigma2():
    X, y = _separated_logistic()
    mle = float(fit_glm_susie(X, y, L=1, method="logistic", max_iter=100).single_effects[0].prior_variance)
    damped3 = float(fit_glm_susie(X, y, L=1, method="logistic", prior_variance_scale=3.0,
                                  max_iter=100).single_effects[0].prior_variance)
    damped1 = float(fit_glm_susie(X, y, L=1, method="logistic", prior_variance_scale=1.0,
                                  max_iter=100).single_effects[0].prior_variance)
    assert damped3 < mle      # the hyperprior regularizes the runaway down
    assert damped1 < damped3  # smaller scale -> stronger shrinkage
