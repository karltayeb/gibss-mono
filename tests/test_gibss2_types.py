import jax.numpy as jnp
import numpy as np
from gibss2.types import BaseSER

def test_baseser_predict():
    ser = BaseSER(
        mu=jnp.array([1.0, 2.0]),
        var=jnp.array([0.1, 0.1]),
        alpha=jnp.array([0.7, 0.3]),
        feature_log_evidence=jnp.zeros(2),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        prior_variance=1.0
    )
    X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # E[Xb] = X @ (alpha * mu) = [1*0.7*1 + 0*0.3*2, 0*0.7*1 + 1*0.3*2] = [0.7, 0.6]
    expected = jnp.array([0.7, 0.6])
    np.testing.assert_allclose(ser.predict(X), expected)
