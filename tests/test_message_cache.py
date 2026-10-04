"""The engine caches each effect's message between `add_message_index_step` and the next
`subtract_message_index_step` (one design contraction per update instead of two). The cache
must be exactly the message, never leak into host snapshots, and leave the total message
bookkeeping exact."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from dataclasses import replace

from gibss import glm
from gibss.engine import _execute_sweep, effect_to_numpy
from gibss.response import Bernoulli
from gibss.history import History
from gibss.methods import fit_glm_susie


def _data(seed=0, n=150, p=12):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < 0.3).astype(float)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.5 + 2.0 * X[:, 3]))), n).astype(float)
    return X, y


def test_cache_is_the_message_and_total_message_stays_exact():
    X, y = _data()
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    hist = History(granularity="sweep")
    st = fit_glm_susie(Xs, y, L=3, method="localjj", max_iter=3, tol=-1.0, record=hist)
    data = glm.prep_data(Xs, y, center=True)
    # the live total message after a sweep is the sum of the effects' messages (the cache
    # fed every subtract; a stale or wrong cache would leave a residual)
    snap = hist.by_phase("sweep")[-1].state
    tm_mean = np.zeros(X.shape[0])
    tm_var = np.zeros(X.shape[0])
    for e in snap.single_effects:
        msg = e.message(data)
        tm_mean += np.asarray(msg.mean)
        tm_var += np.asarray(msg.var)
    np.testing.assert_allclose(np.asarray(snap.total_message.mean), tm_mean, atol=1e-10)
    np.testing.assert_allclose(np.asarray(snap.total_message.var), tm_var, atol=1e-10)
    # host snapshots (history AND the returned state) never carry the cache
    assert all(e.cached_message is None for e in snap.single_effects)
    assert all(e.cached_message is None for e in st.single_effects)


def test_live_effect_carries_its_message_after_add():
    X, y = _data(1)
    data = glm.prep_data(X, y, center=True)
    state = glm.initialize_state(
        data, L=2, response=Bernoulli(), family_state_kwargs=dict(kernel="quad", intercept="shared")
    )
    sched = replace(glm.default_schedule(), after_fit=())  # keep the live (jax) state
    state = replace(state, update_order=(0, 1))  # fit_ibss seeds this; a bare sweep must
    live = _execute_sweep(data, state, sched)
    for e in live.single_effects:
        assert e.cached_message is not None
        msg = e.message(data)
        np.testing.assert_allclose(np.asarray(e.cached_message.mean), np.asarray(msg.mean))
        np.testing.assert_allclose(np.asarray(e.cached_message.var), np.asarray(msg.var))
        assert effect_to_numpy(e).cached_message is None
