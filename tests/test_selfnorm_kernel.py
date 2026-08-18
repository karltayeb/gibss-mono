"""The quad kernel exposes its adaptive-GH quadrature nodes (glm_ser_nodes), and the
engine stores them on the effect state -- the (b_nodes, logW) the self-normalized offset
fold consumes. Round-trip: softmax(log_node_weight) reproduces the kernel's own (mu, var)."""

import jax
import jax.numpy as jnp
import numpy as np

from gibss.glm import _fit_effect, initialize_state
from gibss.linear import LinearData
from gibss.operators import DenseOperator
from gibss.response import Bernoulli
from gibss.response_ser import glm_ser, glm_ser_nodes

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(0)


def test_glm_ser_nodes_roundtrip_to_moments():
    n, p = 40, 6
    X = jnp.asarray(RNG.normal(size=(n, p)))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    off = jnp.asarray(RNG.normal(size=n) * 0.3)
    op = DenseOperator(X)

    mu4, var4, lbf4, ck4 = glm_ser(op, y, off, 1.0, Bernoulli(), order=15)
    mu, var, lbf, ck, b_nodes, logw = glm_ser_nodes(op, y, off, 1.0, Bernoulli(), order=15)

    # glm_ser is the 4-tuple slice of glm_ser_nodes
    assert jnp.allclose(mu, mu4) and jnp.allclose(var, var4)
    assert jnp.allclose(lbf, lbf4) and jnp.allclose(ck, ck4)

    # per-feature posterior moments from the nodes match mu/var (softmax over nodes axis)
    pw = jax.nn.softmax(logw, axis=0)  # (order, p)
    mu_n = jnp.sum(pw * b_nodes, axis=0)
    var_n = jnp.sum(pw * b_nodes**2, axis=0) - mu_n**2
    assert jnp.allclose(mu_n, mu, atol=1e-10)
    assert jnp.allclose(var_n, var, atol=1e-10)

    # GLOBAL softmax(logw) marginal over nodes == alpha (softmax of feature_log_bf)
    W = jax.nn.softmax(logw.reshape(-1)).reshape(logw.shape)
    alpha_n = jnp.sum(W, axis=0)
    alpha = jax.nn.softmax(lbf)
    assert jnp.allclose(alpha_n, alpha, atol=1e-10)


def test_effect_state_carries_nodes_after_fit():
    n, p = 30, 5
    X = jnp.asarray(RNG.normal(size=(n, p)))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    data = LinearData(X=X, y=y)
    state = initialize_state(data, L=1, response=Bernoulli())
    fs = state.family_state
    from gibss.glm import _aux, _effect_offset
    eff = _fit_effect(data, fs, _aux(data, state), _effect_offset(fs, state),
                      1.0, fs.quadrature_order)
    assert eff.b_nodes is not None and eff.log_node_weight is not None
    assert eff.b_nodes.shape == (fs.quadrature_order, p)
    # nodes reproduce the state's own mu
    pw = jax.nn.softmax(eff.log_node_weight, axis=0)
    mu_n = jnp.sum(pw * eff.b_nodes, axis=0)
    assert jnp.allclose(mu_n, eff.mu, atol=1e-9)
