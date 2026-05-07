"""Legacy JJ SER monotonicity coverage for the logisticsusie API."""

import pytest
import numpy as np
import jax.numpy as jnp
from logisticsusie.jj import LogisticSER
from logisticsusie.susie import SER

pytestmark = pytest.mark.legacy

def generate_data(rng, n, p, data_type="dense", offset_scale=0.5):
    if data_type == "dense":
        X = rng.normal(size=(n, p))
    elif data_type == "sparse":
        X = rng.binomial(2, 0.1, size=(n, p)).astype(float)
    elif data_type == "colinear":
        Z = rng.normal(size=(n, 2))
        W = rng.normal(size=(2, p))
        X = Z @ W + rng.normal(scale=0.1, size=(n, p))
    else:
        raise ValueError("Unknown data_type")

    # Generate some beta
    beta = np.zeros(p)
    idx = rng.integers(0, p)
    beta[idx] = rng.normal(loc=1.0, scale=0.5)

    offset_mean = rng.normal(scale=offset_scale, size=n)
    offset_var = rng.uniform(0, 0.1, size=n)
    
    logits = X @ beta + offset_mean
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    return jnp.array(X), jnp.array(y), jnp.array(offset_mean), jnp.array(offset_var)


@pytest.mark.parametrize("data_type", ["dense", "sparse", "colinear"])
@pytest.mark.parametrize("offset_scale", [0.0, 1.0, 5.0])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_jj_ser_update_monotonicity(data_type, offset_scale, seed):
    rng = np.random.default_rng(seed)
    n, p = 150, 20
    X, y, offset_mean, offset_var = generate_data(rng, n, p, data_type, offset_scale)

    ser = LogisticSER.init(X, prior_variance=1.0)
    
    # We will step through the update manually to track the exact ELBO
    # using the optimal xi from compute_elbo
    elbos = [ser.compute_elbo(X, y, offset_mean, offset_var)]
    
    # Do several updates
    for _ in range(15):
        ser = ser.update(X, y, offset_mean, offset_var, estimate_prior_variance=False, max_iter=1)
        elbos.append(ser.compute_elbo(X, y, offset_mean, offset_var))

    elbos = np.array(elbos)
    deltas = np.diff(elbos)

    # Monotonicity check
    # Allow a tiny negative tolerance for numerical precision issues
    tol = 1e-10
    assert np.all(deltas > -tol), f"ELBO decreased: {deltas[deltas < -tol]}"


@pytest.mark.parametrize("seed", [111, 222, 333])
def test_jj_ser_prior_variance_update_monotonicity(seed):
    rng = np.random.default_rng(seed)
    n, p = 100, 15
    X, y, offset_mean, offset_var = generate_data(rng, n, p, data_type="dense", offset_scale=0.5)

    # initialize with a suboptimal prior variance
    ser = LogisticSER.init(X, prior_variance=0.01)
    
    # update q(b) first
    ser = ser.update(X, y, offset_mean, offset_var, estimate_prior_variance=False, max_iter=5)
    
    elbo_before_pv = ser.compute_elbo(X, y, offset_mean, offset_var)
    
    # now estimate prior variance
    new_ser = SER.estimate_prior_variance(ser)
    elbo_after_pv = new_ser.compute_elbo(X, y, offset_mean, offset_var)
    
    assert elbo_after_pv >= elbo_before_pv - 1e-10, f"Prior variance update decreased ELBO: {elbo_before_pv} -> {elbo_after_pv}"

    # Also test the fit method convergence
    ser_fit, elbo_history = LogisticSER.fit(
        X, y, offset_mean, offset_var, prior_variance=0.1, 
        estimate_prior_variance=True, max_iter=20, tol=1e-6, verbose=False
    )
    
    deltas = np.diff(elbo_history)
    assert np.all(deltas > -1e-10), f"fit ELBO decreased: {deltas[deltas < -1e-10]}"
