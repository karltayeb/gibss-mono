import jax.numpy as jnp
import pytest
from gibss2.engine import GIBSSState, Message, MeanMessage
from gibss2.types import BaseSER
from gibss.cox import CoxFamilyState
from gibss.globaljj import GlobalJJFamilyState
from gibss.linear import LinearFamilyState, update_prior_variance_index_step
from gibss.localjj import LocalJJFamilyState
from gibss.logistic_quadrature import QuadratureFamilyState

def test_gibss_state_properties():
    # Mock some effects
    e1 = BaseSER(
        mu=jnp.array([1.0, 2.0]),
        var=jnp.array([0.1, 0.2]),
        alpha=jnp.array([0.4, 0.6]),
        feature_log_evidence=jnp.array([2.5, 2.5]),
        marginal_log_likelihood=2.5,
        null_log_likelihood=1.0,
        kl=0.5,
        prior_variance=2.0
    )
    e2 = BaseSER(
        mu=jnp.array([3.0, 4.0]),
        var=jnp.array([0.3, 0.4]),
        alpha=jnp.array([0.7, 0.3]),
        feature_log_evidence=jnp.array([4.5, 4.5]),
        marginal_log_likelihood=4.5,
        null_log_likelihood=2.0,
        kl=0.5,
        prior_variance=3.0
    )

    
    state = GIBSSState(
        single_effects=[e1, e2],
        total_message=Message(jnp.zeros(2), jnp.zeros(2)),
        family_state=None
    )
    
    # Test stacked properties
    assert jnp.allclose(state.alpha, jnp.stack([e1.alpha, e2.alpha]))
    assert jnp.allclose(state.mu, jnp.stack([e1.mu, e2.mu]))
    assert jnp.allclose(state.var, jnp.stack([e1.var, e2.var]))
    assert jnp.allclose(
        state.feature_log_bayes_factor,
        jnp.array([e1.feature_log_bayes_factor, e2.feature_log_bayes_factor]),
    )
    assert jnp.allclose(
        state.marginal_log_likelihood,
        jnp.array([e1.marginal_log_likelihood, e2.marginal_log_likelihood]),
    )
    assert jnp.allclose(
        state.null_log_likelihood,
        jnp.array([e1.null_log_likelihood, e2.null_log_likelihood]),
    )
    assert jnp.allclose(
        state.ser_log_bayes_factor,
        state.marginal_log_likelihood - state.null_log_likelihood,
    )
    assert jnp.allclose(state.prior_variance, jnp.array([e1.prior_variance, e2.prior_variance]))

def test_gibss_state_get_credible_sets():
    e1 = BaseSER(
        mu=jnp.array([1.0, 2.0]),
        var=jnp.array([0.1, 0.2]),
        alpha=jnp.array([0.1, 0.9]),
        feature_log_evidence=jnp.array([1.5, 1.5]),
        marginal_log_likelihood=2.5,
        null_log_likelihood=1.0,
        kl=0.5,
        prior_variance=2.0
    )
    state = GIBSSState(
        single_effects=[e1],
        total_message=Message(jnp.zeros(2), jnp.zeros(2)),
        family_state=None
    )
    cs = state.get_credible_sets(0.8)
    assert len(cs) == 1
    assert cs[0].feature_indices == (1,)
    assert cs[0].alpha_mass == 0.9

def test_gibss_state_add_effects():
    e1 = BaseSER(
        mu=jnp.array([1.0, 2.0]),
        var=jnp.array([0.1, 0.2]),
        alpha=jnp.array([0.1, 0.9]),
        feature_log_evidence=jnp.array([1.5, 1.5]),
        marginal_log_likelihood=2.5,
        null_log_likelihood=1.0,
        kl=0.5,
        prior_variance=2.0
    )
    m1 = Message(jnp.array([0.5, 0.5]), jnp.array([0.1, 0.1]))
    
    state = GIBSSState(
        single_effects=[e1],
        total_message=m1,
        family_state="some_state"
    )
    
    e2 = BaseSER(
        mu=jnp.array([3.0, 4.0]),
        var=jnp.array([0.3, 0.4]),
        alpha=jnp.array([0.5, 0.5]),
        feature_log_evidence=jnp.array([2.5, 2.5]),
        marginal_log_likelihood=4.5,
        null_log_likelihood=2.0,
        kl=0.5,
        prior_variance=3.0
    )
    
    new_state = state.add_effects([e2])
    
    assert len(new_state.single_effects) == 2
    assert new_state.single_effects[1] == e2
    assert new_state.family_state == "some_state"
    # total_message should be same (since we added 0) but it went through _add_message
    assert jnp.allclose(new_state.total_message.mean, m1.mean)
    assert jnp.allclose(new_state.total_message.var, m1.var)

def test_gibss_state_add_effects_mean_only():
    e1 = BaseSER(
        mu=jnp.array([1.0, 2.0]),
        var=jnp.array([0.1, 0.2]),
        alpha=jnp.array([0.1, 0.9]),
        feature_log_evidence=jnp.array([1.5, 1.5]),
        marginal_log_likelihood=2.5,
        null_log_likelihood=1.0,
        kl=0.5,
        prior_variance=2.0
    )
    m1 = MeanMessage(jnp.array([0.5, 0.5]))
    
    state = GIBSSState(
        single_effects=[e1],
        total_message=m1,
        family_state="some_state"
    )
    
    e2 = BaseSER(
        mu=jnp.array([3.0, 4.0]),
        var=jnp.array([0.3, 0.4]),
        alpha=jnp.array([0.5, 0.5]),
        feature_log_evidence=jnp.array([2.5, 2.5]),
        marginal_log_likelihood=4.5,
        null_log_likelihood=2.0,
        kl=0.5,
        prior_variance=3.0
    )
    
    new_state = state.add_effects([e2])
    
    assert isinstance(new_state.total_message, MeanMessage)
    assert jnp.allclose(new_state.total_message.mean, m1.mean)


@pytest.mark.parametrize(
    ("family_state_cls", "expected_default"),
    [
        (LinearFamilyState, True),
        (GlobalJJFamilyState, True),
        (LocalJJFamilyState, True),
        (QuadratureFamilyState, True),
        (CoxFamilyState, False),
    ],
)
def test_family_state_exposes_estimate_prior_variance_flag(
    family_state_cls, expected_default
):
    family_state = family_state_cls()
    assert hasattr(family_state, "estimate_prior_variance")
    assert family_state.estimate_prior_variance is expected_default


def test_update_prior_variance_index_step_respects_family_flag():
    effect = BaseSER(
        mu=jnp.array([2.0, 1.0]),
        var=jnp.array([0.5, 0.25]),
        alpha=jnp.array([0.25, 0.75]),
        feature_log_evidence=jnp.array([0.0, 0.0]),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        prior_variance=10.0,
    )
    total_message = Message(jnp.zeros(2), jnp.zeros(2))
    expected_prior_variance = 2.0625

    enabled_state = GIBSSState(
        single_effects=[effect],
        total_message=total_message,
        family_state=LinearFamilyState(estimate_prior_variance=True),
    )
    enabled_updated_state = update_prior_variance_index_step(None, 0, enabled_state)
    assert enabled_updated_state.single_effects[0].prior_variance == pytest.approx(
        expected_prior_variance
    )

    disabled_state = GIBSSState(
        single_effects=[effect],
        total_message=total_message,
        family_state=LinearFamilyState(estimate_prior_variance=False),
    )
    disabled_updated_state = update_prior_variance_index_step(None, 0, disabled_state)
    assert disabled_updated_state.single_effects[0].prior_variance == pytest.approx(
        effect.prior_variance
    )
