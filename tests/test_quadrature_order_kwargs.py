import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gibss.logistic_profile as P
import gibss.logistic_quadrature as Q


@pytest.mark.parametrize("mod", [Q, P])
def test_quadrature_order_kwargs_respected(mod):
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = jnp.asarray(rng.normal(size=(n, p)))
    y = jnp.asarray((rng.random(n) < 0.3).astype(float))
    data = mod.prep_data(X, y)
    # family_state_kwargs wins over the default parameter
    st = mod.initialize_state(data, L=1, family_state_kwargs={"quadrature_order": 1})
    assert st.family_state.quadrature_order == 1
    # explicit parameter is the fallback default
    assert mod.initialize_state(data, L=1).family_state.quadrature_order == 15
    # parameter still works when kwargs doesn't set it
    assert mod.initialize_state(data, L=1, quadrature_order=7).family_state.quadrature_order == 7
