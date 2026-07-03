import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gibss.logistic_localtaylor as LT


@pytest.mark.parametrize("center", [False, True])  # quadrature / profile
def test_quadrature_order_kwargs_respected(center):
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = jnp.asarray(rng.normal(size=(n, p)))
    y = jnp.asarray((rng.random(n) < 0.3).astype(float))
    data = LT.prep_data(X, y)
    ck = {"center": center}
    st = LT.initialize_state(data, L=1, family_state_kwargs={"quadrature_order": 1, **ck})
    assert st.family_state.quadrature_order == 1
    assert LT.initialize_state(data, L=1, family_state_kwargs=ck).family_state.quadrature_order == 15
    assert LT.initialize_state(data, L=1, quadrature_order=7, family_state_kwargs=ck).family_state.quadrature_order == 7
