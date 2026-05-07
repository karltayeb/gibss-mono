import jax.numpy as jnp
from typing import Any
from dataclasses import dataclass
from gibss2.engine import fit_ibss, IBSSFamily, GIBSSState, BaseSER, Message

@dataclass(frozen=True, slots=True)
class DummySER(BaseSER):
    alpha: jnp.ndarray
    mu: jnp.ndarray
    var: jnp.ndarray
    feature_log_evidence: jnp.ndarray
    marginal_log_likelihood: float
    null_log_likelihood: float
    kl: float
    prior_variance: float

    def get_cs(self, coverage: float = 0.95) -> tuple[int, ...]:
        return ()

class DummyFamily:
    def __init__(self):
        self.before_fit_calls = 0

    def initial_state(self, X: Any, y: Any) -> Any:
        return None

    def init_effect(self, X: Any, y: Any, family_state: Any) -> BaseSER:
        n, p = X.shape
        return DummySER(
            alpha=jnp.ones(p) / p,
            mu=jnp.zeros(p),
            var=jnp.zeros(p),
            feature_log_evidence=jnp.zeros(p),
            marginal_log_likelihood=0.0,
            null_log_likelihood=0.0,
            kl=0.0,
            prior_variance=1.0,
        )

    def zero_message(self, X: Any, y: Any, family_state: Any) -> Message:
        n, p = X.shape
        return Message(jnp.zeros(n), jnp.zeros(n))

    def update_effect(
        self, effect: BaseSER, X: Any, y: Any, loo_message: Message, family_state: Any
    ) -> BaseSER:
        return effect

    def message(self, effect: BaseSER, X: Any, family_state: Any) -> Message:
        n, p = X.shape
        return Message(jnp.zeros(n), jnp.zeros(n))

    def before_fit(self, state: GIBSSState, X: Any, y: Any) -> GIBSSState:
        self.before_fit_calls += 1
        return state

    def update_intercept(self, state: GIBSSState, X: Any, y: Any) -> GIBSSState:
        return state

    def before_sweep(self, state: GIBSSState, X: Any, y: Any) -> GIBSSState:
        return state

    def after_sweep(self, state: GIBSSState, X: Any, y: Any) -> GIBSSState:
        return state

    def before_effect_update(
        self, state: GIBSSState, effect_index: int, X: Any, y: Any
    ) -> GIBSSState:
        return state

    def after_effect_update(
        self, state: GIBSSState, effect_index: int, X: Any, y: Any
    ) -> GIBSSState:
        return state

    def compute_elbo(self, state: GIBSSState, X: Any, y: Any) -> float:
        return 0.0

def test_before_fit_called_once():
    n, p = 10, 5
    X = jnp.zeros((n, p))
    y = jnp.zeros(n)
    family = DummyFamily()
    
    # Run fit_ibss for 2 iterations
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    
    # Assert before_fit was called exactly once
    assert family.before_fit_calls == 1
