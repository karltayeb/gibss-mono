"""Convergence handling: early-exit (tol) + non-finite guard on the mode-finders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gibss.operators import DenseOperator
from gibss.response import Bernoulli, JJFixed, Smoothed
from gibss.response_ser import glm_jj_ser, glm_ser
from gibss.legacy.ser_ops import (
    local_irls,
    local_irls_centered,
    localjj_centered_ser,
    profile_ser,
    quadrature_ser,
)


def _stress():
    # extreme imbalance (offset saturates mu->1) + strong effect: the regime that
    # NaN'd undamped Newton.
    rng = np.random.default_rng(0)
    n, p = 600, 40
    Xd = rng.normal(size=(n, p))
    offset = np.full(n, 4.5) + rng.normal(size=n) * 0.2  # mu ~ 0.99
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 3.0 * Xd[:, 0]))), size=n).astype(float)
    return DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset)


@pytest.mark.parametrize(
    "fn",
    [
        lambda op, y, o: local_irls(op, y, o, 1.0),
        lambda op, y, o: local_irls_centered(op, y, o, 1.0),
        lambda op, y, o: quadrature_ser(op, y, o, 1.0, order=11),
        lambda op, y, o: profile_ser(op, y, o, 1.0, order=11),
        lambda op, y, o: localjj_centered_ser(op, y, o, 1.0),
    ],
)
def test_all_outputs_finite_under_saturation(fn):
    op, y, offset = _stress()
    out = fn(op, y, offset)
    for a in out:
        assert np.all(np.isfinite(np.asarray(a))), "non-finite output under saturation"


def test_early_exit_matches_fixed_iterations():
    # early-exit (tol) must give the same result as running the full cap
    rng = np.random.default_rng(1)
    n, p = 400, 20
    Xd = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 2]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    a = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0, n_iter=200, tol=1e-10)
    b = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0, n_iter=200, tol=0.0)  # no early exit
    for x, z in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(z), atol=1e-8)


def test_nonfinite_feature_nullified():
    # a degenerate column (all zeros) can't blow up but exercises the guard path;
    # force a NaN-prone setup and check the output stays finite + that column -> b=0
    rng = np.random.default_rng(2)
    n, p = 300, 10
    Xd = rng.normal(size=(n, p))
    Xd[:, 5] = 0.0  # degenerate feature
    offset = np.full(n, 5.0)  # saturated
    y = rng.binomial(1, 0.99, size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    b, var, lbf = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    assert np.all(np.isfinite(np.asarray(b)))
    np.testing.assert_allclose(float(np.asarray(b)[5]), 0.0, atol=1e-10)  # degenerate -> 0


def _separated_enrichment():
    # The lifan gene-set regime that broke the `quad` kernel (glm_ser): a strongly
    # enriched indicator column against a strict (top-5%) hit list. Most set members
    # are hits, so the single-effect MLE runs toward +inf -- quasi-separation. Undamped
    # Fisher scoring overshoots to the WRONG sign and exits at n_iter on finite garbage
    # (mu ~ -150, logBF ~ -2e4); the concave JJ bound (glm_jj_ser) gives the sane mode.
    rng = np.random.default_rng(0)
    n, base_rate = 5000, 0.05
    offset0 = np.log(base_rate / (1 - base_rate))  # ~ -2.94 baseline log-odds
    set_size, overlap = 233, 76
    members = rng.choice(n, size=set_size, replace=False)
    x = np.zeros(n)
    x[members] = 1.0
    y = np.zeros(n)
    y[rng.choice(members, size=overlap, replace=False)] = 1.0
    rest = np.setdiff1d(np.arange(n), members)
    y[rng.choice(rest, size=int(base_rate * len(rest)), replace=False)] = 1.0
    op = DenseOperator(jnp.asarray(x[:, None]))
    return op, jnp.asarray(y), jnp.full(n, offset0)


def test_glm_ser_does_not_diverge_on_separated_design():
    # Finite is NOT enough here: the divergence produced a FINITE (mu ~ -150) garbage
    # value that the isfinite guard passed straight through, inverting the ranking. Pin
    # sign + magnitude against the concave-bound reference (localjj), which never blows up.
    op, y, offset = _separated_enrichment()
    mu_q, _, lbf_q, _ = glm_ser(op, y, offset, 1.0, Bernoulli(), order=15)
    mu_j, _, lbf_j, _ = glm_jj_ser(op, y, offset, 1.0, Smoothed(Bernoulli(), JJFixed()))
    mu_q, lbf_q, mu_j, lbf_j = (float(np.asarray(v)[0]) for v in (mu_q, lbf_q, mu_j, lbf_j))
    assert np.isfinite(mu_q) and np.isfinite(lbf_q)
    assert mu_q > 0, f"enriched set must have positive log-odds, got mu={mu_q}"
    assert lbf_q > 0, f"strong enrichment must have positive logBF, got {lbf_q}"
    # quad (free-form q) and localjj (Gaussian q) approximate the same MAP; on a strong,
    # well-identified effect they agree closely -- a loose tol guards the sign/scale.
    np.testing.assert_allclose(mu_q, mu_j, rtol=0.1)
    np.testing.assert_allclose(lbf_q, lbf_j, rtol=0.1)
