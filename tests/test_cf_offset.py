"""Correctness of the Q2 characteristic-function offset engine (`gibss.cf_offset`).

Ground truth = brute-force per-row expectation `E_{o'}[A^(k)(z + o')]` over the FULL
zero-mean offset law, enumerating the product mixture (prod_l C_l Gaussian components)
and integrating each component by high-order Gauss-Hermite. The CF engine must match this
without ever enumerating the product mixture. Bernoulli (logistic) base, dense design.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss.cf_offset import CharFnOffset, offset_cf, smoothed_nodes
from gibss.response import Bernoulli


def _softplus(z):
    return np.logaddexp(0.0, z)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _brute_atilde(effects, znodes, gh_order=96):
    """Exact `(At0, At1, At2)` at `znodes` (n, M+1) for the ZERO-MEAN offset formed by
    `effects`, by enumerating the product mixture and GH-integrating each component."""
    effects = [
        (np.asarray(x), np.asarray(a), np.asarray(m), np.asarray(v))
        for x, a, m, v in effects
    ]
    n = effects[0][0].shape[0]
    Cs = [e[1].shape[0] for e in effects]
    s = np.zeros(n)
    for x, a, m, v in effects:
        s += x @ (a * m)  # total offset mean (removed for the zero-mean law)

    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)

    At0 = np.zeros_like(znodes)
    At1 = np.zeros_like(znodes)
    At2 = np.zeros_like(znodes)
    for combo in itertools.product(*[range(c) for c in Cs]):
        w = 1.0
        m_i = np.zeros(n)
        v_i = np.zeros(n)
        for (x, a, mu, var), c in zip(effects, combo):
            w *= a[c]
            m_i += x[:, c] * mu[c]
            v_i += x[:, c] ** 2 * var[c]
        center = znodes + (m_i - s)[:, None]  # (n, M+1) zero-mean shift
        sd = np.sqrt(2.0 * v_i)[:, None, None]
        pts = center[:, :, None] + sd * nodes[None, None, :]  # (n, M+1, order)
        sig = _sigmoid(pts)
        At0 += w * (wts * _softplus(pts)).sum(-1)
        At1 += w * (wts * sig).sum(-1)
        At2 += w * (wts * (sig * (1.0 - sig))).sum(-1)
    return At0, At1, At2


def _rand_effects(rng, n, L, C, mu_scale=1.0, var_scale=0.3, x_scale=1.0):
    effects = []
    for _ in range(L):
        x = rng.normal(size=(n, C)) * x_scale
        logits = rng.normal(size=C)
        alpha = np.exp(logits - logits.max())
        alpha /= alpha.sum()
        mu = rng.normal(size=C) * mu_scale
        var = np.exp(rng.normal(size=C) * 0.5) * var_scale
        effects.append(
            (jnp.asarray(x), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var))
        )
    return effects


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_effect_offset_matches_brute(seed):
    """L=2 offset (one other effect): CF Atilde and derivs match brute force to ~1e-6."""
    rng = np.random.default_rng(seed)
    n, C = 12, 3
    effects = _rand_effects(rng, n, L=1, C=C)  # offset = a single other effect
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes(effects, M=48)
    b0, b1, b2 = _brute_atilde(effects, np.asarray(znodes))
    assert np.max(np.abs(np.asarray(At0) - b0)) < 1e-6
    assert np.max(np.abs(np.asarray(At1) - b1)) < 1e-6
    assert np.max(np.abs(np.asarray(At2) - b2)) < 1e-6


@pytest.mark.parametrize("L", [2, 3, 4])
def test_multi_effect_offset_matches_brute(L):
    """L=3..5 offsets (2..4 others): CF matches the enumerated product mixture to ~1e-5."""
    rng = np.random.default_rng(100 + L)
    n, C = 10, 3
    effects = _rand_effects(rng, n, L=L, C=C)
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes(effects, M=48)
    b0, b1, b2 = _brute_atilde(effects, np.asarray(znodes))
    assert np.max(np.abs(np.asarray(At0) - b0)) < 1e-5
    assert np.max(np.abs(np.asarray(At1) - b1)) < 1e-5
    assert np.max(np.abs(np.asarray(At2) - b2)) < 1e-5


def test_variance_sweep_accuracy():
    """Auto-sizing stays accurate across a wide offset-variance sweep (tiny -> large)."""
    rng = np.random.default_rng(7)
    n, C = 8, 3
    for var_scale in [1e-3, 1e-2, 0.1, 1.0, 5.0]:
        effects = _rand_effects(rng, n, L=2, C=C, var_scale=var_scale)
        znodes, At0, At1, At2, hw, s, V = smoothed_nodes(effects, M=48)
        # the brute-force GH oracle needs many nodes to converge at large component
        # variance (96 -> 160 self-differs ~4e-5 at var_scale=5); 300 is < 2e-7 converged.
        b0, b1, b2 = _brute_atilde(effects, np.asarray(znodes), gh_order=300)
        err = max(
            np.max(np.abs(np.asarray(At0) - b0)),
            np.max(np.abs(np.asarray(At1) - b1)),
            np.max(np.abs(np.asarray(At2) - b2)),
        )
        # 5e-6 covers the genuine CF weight-channel floor (~1.8e-6) at the largest V;
        # the small-V cases are ~1e-7.
        assert err < 5e-6, f"var_scale={var_scale}: err={err:.2e}"


def test_moments_match_message_convention():
    """`offset_cf`'s reported mean/variance equal the analytic mixture moments (the SuSiE
    message mean/var), summed over effects."""
    rng = np.random.default_rng(3)
    n, C = 20, 4
    effects = _rand_effects(rng, n, L=3, C=C)
    tau = jnp.linspace(0.0, 5.0, 64)
    _, s, V, _ = offset_cf(effects, tau)
    s_ref = np.zeros(n)
    V_ref = np.zeros(n)
    for x, a, mu, var in effects:
        x, a, mu, var = map(np.asarray, (x, a, mu, var))
        mean = x @ (a * mu)
        second = (x**2) @ (a * (mu**2 + var))
        s_ref += mean
        V_ref += second - mean**2
    assert np.allclose(np.asarray(s), s_ref, atol=1e-10)
    assert np.allclose(np.asarray(V), V_ref, atol=1e-10)


def test_cf_at_zero_is_one():
    """phi(0) = 1 (centered), and phi is conjugate-even-consistent (real DC)."""
    rng = np.random.default_rng(5)
    effects = _rand_effects(rng, n=6, L=2, C=3)
    tau = jnp.linspace(0.0, 4.0, 33)
    phi, s, V, _ = offset_cf(effects, tau)
    assert np.allclose(np.asarray(phi[:, 0]), 1.0 + 0.0j, atol=1e-12)


def test_terms_presentation_reconstructs_atilde():
    """`CharFnOffset.terms` (plug-in base + Clenshaw residual) reproduces the Atilde-based
    cumulant terms at the build nodes: ll = y*z - Atilde, g = y - Atilde', w = Atilde''."""
    rng = np.random.default_rng(11)
    n, C = 10, 3
    effects = _rand_effects(rng, n, L=2, C=C)
    y = jnp.asarray((rng.uniform(size=n) < 0.5).astype(float))
    sm = CharFnOffset(M=48)
    base = Bernoulli()
    aux = sm.build_aux(base, y, effects)
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes(effects, M=48)
    # evaluate terms at each column of znodes (broadcast the per-row aux over M+1 nodes)
    y_col = np.asarray(y)
    for k in range(znodes.shape[1]):
        z = znodes[:, k]
        ll, g, w = sm.terms(base, z, aux)
        ll_ref = y_col * np.asarray(z) - np.asarray(At0[:, k])
        assert np.allclose(np.asarray(ll), ll_ref, atol=1e-6)
        assert np.allclose(np.asarray(g), y_col - np.asarray(At1[:, k]), atol=1e-6)
        assert np.allclose(np.asarray(w), np.asarray(At2[:, k]), atol=1e-6)


def test_empty_offset_is_base_cumulant():
    """No other effects -> Atilde = A exactly (the offset is a point mass at 0)."""
    n = 5
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes([], M=32, n=n, ntau=64, Tmax=9.0)
    z = np.asarray(znodes)
    assert np.allclose(np.asarray(At0), _softplus(z), atol=1e-9)
    assert np.allclose(np.asarray(At1), _sigmoid(z), atol=1e-9)
    assert np.allclose(np.asarray(V), 0.0)
