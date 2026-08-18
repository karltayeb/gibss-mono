"""Exact-CAVI shared-intercept update under the self-normalized offset fold.

`glm._intercept_freeform` must integrate the intercept likelihood over ALL effects'
current quadrature posteriors (symmetric to how each effect integrates the OTHERS), not
plug in the effects' MEAN. This checks the fitted intercept posterior (mean AND variance)
against a dense numerical reference

    q*(b0) ∝ exp( Σ_i E_effects[ ℓ_i(b0 + o_i) ] ),

with each effect's offset integrated by its own stored nodes (dense product over L=2),
and confirms it beats the old plug-in fit (which ignores the effect variance)."""

import jax
import jax.numpy as jnp
import numpy as np

from gibss import glm
from gibss.linear import prep_data
from gibss.methods import fit_glm_susie
from gibss.operators import DenseOperator
from gibss.response import Bernoulli, CompressSelfNorm, Smoothed
from gibss.response_ser import glm_ser_nodes
import pytest

jax.config.update("jax_enable_x64", True)


def _sim(seed, n=150, p=8, signals=((2, 2.6), (5, -2.4)), b0=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = 0.6 * Z + 0.4 * rng.normal(size=(n, 1))  # correlated -> spread posterior
    b = np.zeros(p)
    for j, v in signals:
        b[j] = v
    logit = X @ b + b0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return jnp.asarray(X), jnp.asarray(y)


def _dense_reference(state, X, y, b0_grid):
    """q*(b0) ∝ exp(Σ_i E_effects[ℓ_i(b0+o_i)]); effects integrated by their own nodes."""
    X, y = np.asarray(X), np.asarray(y)
    n = X.shape[0]
    atom_vals = np.zeros((n, 1))  # running combined offset atoms over effects
    atom_wts = np.ones(1)
    for e in state.single_effects:
        bn = np.asarray(e.b_nodes)  # (order, p)
        W = np.asarray(jax.nn.softmax(jnp.asarray(e.log_node_weight).reshape(-1)))
        W = W.reshape(bn.shape)  # (order, p), sums to 1
        vals = (X[:, None, :] * bn[None, :, :]).reshape(n, -1)  # (n, order*p)
        w = W.reshape(-1)
        atom_vals = (atom_vals[:, :, None] + vals[:, None, :]).reshape(n, -1)
        atom_wts = (atom_wts[:, None] * w[None, :]).reshape(-1)
    logL = np.array([
        np.sum(atom_wts[None, :] * (y[:, None] * (b0 + atom_vals)
                                    - np.logaddexp(0.0, b0 + atom_vals)))
        for b0 in b0_grid
    ])
    wg = np.exp(logL - logL.max())
    wg /= np.trapezoid(wg, b0_grid)
    m = np.trapezoid(wg * b0_grid, b0_grid)
    v = np.trapezoid(wg * (b0_grid - m) ** 2, b0_grid)
    return float(m), float(v)


def _old_plugin(state, X, y, order):
    """The pre-fix plug-in: ones-column SER on the BASE response at the effects' MEAN."""
    fs = state.family_state
    base = fs.response.base if isinstance(fs.response, Smoothed) else fs.response
    n = X.shape[0]
    offset = jnp.asarray(state.total_message.mean) + fs.glm_offset
    ones = DenseOperator(jnp.ones((n, 1)))
    mu, var, *_ = glm_ser_nodes(ones, jnp.asarray(y), offset, 1e6, base, order=order)
    return float(mu[0]), float(var[0])


@pytest.mark.slow
def test_intercept_quad_matches_dense_reference():
    for seed in (0, 1, 2):
        X, y = _sim(seed)
        s = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm")
        assert s.converged
        order = s.family_state.quadrature_order
        data = prep_data(X, y)
        m_new, v_new, _, _, _ = glm._intercept_freeform(data, s, order=order)
        m_old, v_old = _old_plugin(s, X, y, order)
        grid = np.linspace(m_new - 2.0, m_new + 2.0, 2001)
        m_ref, v_ref = _dense_reference(s, X, y, grid)

        # the exact-CAVI fit tracks the dense reference tightly...
        assert abs(m_new - m_ref) < 2e-3
        assert abs(v_new - v_ref) < 2e-5
        # ...and is no worse than the plug-in on the mean, strictly better on the
        # variance (the plug-in drops the effect uncertainty the CAVI factor integrates).
        assert abs(m_new - m_ref) <= abs(m_old - m_ref) + 1e-9
        assert abs(v_new - v_ref) < abs(v_old - v_ref)


def test_intercept_quad_matches_plugin_with_no_effects():
    """Zero fitted effects -> the fold collapses to the base cumulant, so the exact-CAVI
    intercept must equal the old plug-in fit exactly."""
    X, y = _sim(0)
    data = prep_data(X, y)
    resp = Smoothed(Bernoulli(), CompressSelfNorm(M=48))
    state = glm.initialize_state(data, L=2, response=resp)  # effects unfit (b_nodes None)
    order = state.family_state.quadrature_order
    m_new, v_new, _, _, _ = glm._intercept_freeform(data, state, order=order)
    m_old, v_old = _old_plugin(state, X, y, order)
    assert abs(m_new - m_old) < 1e-9
    assert abs(v_new - v_old) < 1e-9
