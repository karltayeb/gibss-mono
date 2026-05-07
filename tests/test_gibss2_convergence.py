import jax.numpy as jnp
import numpy as np
from gibss2.convergence import gaussian_skl, categorical_skl, symmetrized_kl, max_alpha_diff
from gibss2.engine import GIBSSState
from gibss2.types import BaseSER, Message

def test_gaussian_skl():
    # Identical
    assert gaussian_skl(0.0, 1.0, 0.0, 1.0) == 0.0
    # Different
    d1 = gaussian_skl(0.0, 1.0, 1.0, 1.0) # mean shift
    d2 = gaussian_skl(0.0, 1.0, 0.0, 2.0) # var shift
    assert d1 > 0
    assert d2 > 0

def test_categorical_skl():
    p1 = jnp.array([0.1, 0.9])
    p2 = jnp.array([0.1, 0.9])
    assert jnp.allclose(categorical_skl(p1, p2), 0.0)

    p3 = jnp.array([0.2, 0.8])
    assert categorical_skl(p1, p3) > 0.0

def test_max_alpha_diff():
    e1 = BaseSER(alpha=jnp.array([0.1, 0.9]), mu=jnp.zeros(2), var=jnp.ones(2), 
                 feature_log_evidence=jnp.zeros(2), marginal_log_likelihood=0.0,
                 null_log_likelihood=0.0, kl=0.0, prior_variance=1.0)
    e2 = BaseSER(alpha=jnp.array([0.2, 0.8]), mu=jnp.zeros(2), var=jnp.ones(2),
                 feature_log_evidence=jnp.zeros(2), marginal_log_likelihood=0.0,
                 null_log_likelihood=0.0, kl=0.0, prior_variance=1.0)
    
    s1 = GIBSSState(single_effects=[e1], total_message=None, family_state=None)
    s2 = GIBSSState(single_effects=[e2], total_message=None, family_state=None)
    
    assert jnp.allclose(max_alpha_diff(s1, s2, None, None, None), 0.1)

def test_symmetrized_kl():
    e1 = BaseSER(alpha=jnp.array([0.1, 0.9]), mu=jnp.zeros(2), var=jnp.ones(2),
                 feature_log_evidence=jnp.zeros(2), marginal_log_likelihood=0.0,
                 null_log_likelihood=0.0, kl=0.0, prior_variance=1.0)
    e2 = BaseSER(alpha=jnp.array([0.1, 0.9]), mu=jnp.zeros(2), var=jnp.ones(2),
                 feature_log_evidence=jnp.zeros(2), marginal_log_likelihood=0.0,
                 null_log_likelihood=0.0, kl=0.0, prior_variance=1.0)
    
    s1 = GIBSSState(single_effects=[e1], total_message=None, family_state=None)
    s2 = GIBSSState(single_effects=[e2], total_message=None, family_state=None)
    
    assert jnp.allclose(symmetrized_kl(s1, s2, None, None, None), 0.0)
    
    e3 = BaseSER(alpha=jnp.array([0.2, 0.8]), mu=jnp.zeros(2), var=jnp.ones(2),
                 feature_log_evidence=jnp.zeros(2), marginal_log_likelihood=0.0,
                 null_log_likelihood=0.0, kl=0.0, prior_variance=1.0)
    s3 = GIBSSState(single_effects=[e3], total_message=None, family_state=None)
    assert symmetrized_kl(s1, s3, None, None, None) > 0.0
