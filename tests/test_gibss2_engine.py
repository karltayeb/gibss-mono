import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import dataclass, replace
from gibss2.types import BaseSER, Message
from gibss2.engine import fit_ibss, GIBSSState, IBSSFamilyDefaults

@dataclass(frozen=True, slots=True)
class ToyFamily(IBSSFamilyDefaults):
    def initial_state(self, X, y): return None
    def init_effect(self, X, y, family_state) -> BaseSER:
        p = X.shape[1]
        return BaseSER(
            mu=jnp.zeros(p), 
            var=jnp.zeros(p), 
            alpha=jnp.zeros(p), 
            feature_log_evidence=jnp.zeros(p), 
            marginal_log_likelihood=0.0,
            null_log_likelihood=0.0,
            kl=0.0,
            prior_variance=1.0
        )
    def zero_message(self, X, y, family_state): 
        return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    def message(self, effect, X, family_state):
        return Message(effect.predict(X), jnp.zeros(X.shape[0]))
    def update_effect(self, effect, X, y, loo_message, family_state):
        # Extremely simple update for testing: set mu to 1.0 for first feature
        mu = jnp.zeros(X.shape[1]).at[0].set(1.0)
        return replace(effect, mu=mu, alpha=jnp.zeros(X.shape[1]).at[0].set(1.0))


def test_default_family_hooks_return_state_unchanged():
    from gibss2.engine import IBSSFamilyDefaults

    class PassiveFamily(IBSSFamilyDefaults):
        def initial_state(self, X, y):
            return None

        def init_effect(self, X, y, family_state):
            raise NotImplementedError

        def zero_message(self, X, y, family_state):
            return Message(jnp.zeros(0), jnp.zeros(0))

        def update_effect(self, effect, X, y, loo_message, family_state):
            raise NotImplementedError

        def message(self, effect, X, family_state):
            return Message(jnp.zeros(0), jnp.zeros(0))

        def compute_elbo(self, state, X, y):
            return 0.0

    family = PassiveFamily()
    state = GIBSSState(
        single_effects=[],
        total_message=Message(jnp.zeros(3), jnp.zeros(3)),
        family_state={"sentinel": 1},
    )

    assert family.before_fit(state, None, None) is state
    assert family.update_intercept(state, None, None) is state
    assert family.before_sweep(state, None, None) is state
    assert family.after_sweep(state, None, None) is state
    assert family.before_effect_update(state, 0, None, None) is state
    assert family.after_effect_update(state, 0, None, None) is state


def test_default_hook_base_supports_partial_override():
    from gibss2.engine import IBSSFamilyDefaults

    class PassiveFamily(IBSSFamilyDefaults):
        def initial_state(self, X, y):
            return None

        def init_effect(self, X, y, family_state):
            raise NotImplementedError

        def zero_message(self, X, y, family_state):
            return Message(jnp.zeros(0), jnp.zeros(0))

        def update_effect(self, effect, X, y, loo_message, family_state):
            raise NotImplementedError

        def message(self, effect, X, family_state):
            return Message(jnp.zeros(0), jnp.zeros(0))

        def compute_elbo(self, state, X, y):
            return 0.0

    class CountingFamily(PassiveFamily):
        def before_effect_update(self, state, effect_index, X, y):
            return replace(state, family_state=effect_index)

    state = GIBSSState(
        single_effects=[],
        total_message=Message(jnp.zeros(2), jnp.zeros(2)),
        family_state=0,
    )
    family = CountingFamily()

    assert family.before_fit(state, None, None) is state
    assert family.before_sweep(state, None, None) is state
    assert family.after_sweep(state, None, None) is state

    updated = family.before_effect_update(state, 3, None, None)
    assert updated.family_state == 3
    assert family.after_effect_update(updated, 3, None, None) is updated


def test_fit_ibss_uses_inherited_default_hooks():
    from gibss2.engine import IBSSFamilyDefaults

    class HookTraceFamily(IBSSFamilyDefaults):
        def initial_state(self, X, y):
            return {"updated": False}

        def init_effect(self, X, y, family_state):
            p = X.shape[1]
            return BaseSER(
                mu=jnp.zeros(p),
                var=jnp.zeros(p),
                alpha=jnp.zeros(p).at[0].set(1.0),
                feature_log_evidence=jnp.zeros(p),
                marginal_log_likelihood=0.0,
                null_log_likelihood=0.0,
                kl=0.0,
                prior_variance=1.0,
            )

        def zero_message(self, X, y, family_state):
            return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))

        def update_effect(self, effect, X, y, loo_message, family_state):
            return effect

        def message(self, effect, X, family_state):
            return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))

        def after_sweep(self, state, X, y):
            return replace(
                state,
                family_state={**state.family_state, "updated": True},
            )

        def compute_elbo(self, state, X, y):
            return 0.0

    state = fit_ibss(
        X=jnp.zeros((2, 1)),
        y=jnp.zeros(2),
        family=HookTraceFamily(),
        L=1,
        max_iter=1,
    )

    assert state.n_iter == 1
    assert state.family_state["updated"] is True


def test_fit_ibss_toy():
    X = jnp.eye(3)
    y = jnp.array([1., 0., 0.])
    family = ToyFamily()
    result = fit_ibss(X, y, family, L=1, max_iter=2)
    assert result.converged
    np.testing.assert_allclose(result.single_effects[0].mu[0], 1.0)
