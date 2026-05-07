import jax.numpy as jnp
import numpy as np
from gibss2.linear import fit_univariate_linear_regression

def test_univariate_linear_kernel():
    # Simple one-feature case: y = 2x + e
    x = jnp.array([[1.0, 2.0, 3.0]]).T
    y = jnp.array([2.0, 4.0, 6.0])
    tau = jnp.ones(3) # precision 1.0 (sigma^2 = 1.0)
    offset = jnp.zeros(3)
    prior_var = 10.0
    
    mu, var, logbf = fit_univariate_linear_regression(x, y, tau, offset, prior_var)
    
    # Analytical posterior: prec = 1/10 + (1^2 + 2^2 + 3^2) = 0.1 + 14 = 14.1
    # mu = (1/14.1) * (1*2 + 2*4 + 3*6) = 28 / 14.1 approx 1.9858
    assert mu.shape == (1,)
    np.testing.assert_allclose(mu[0], 28.0 / 14.1, rtol=1e-5)
