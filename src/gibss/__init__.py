import jax
jax.config.update("jax_enable_x64", True)

from ._logistic_intercept import _logistic_intercept

__all__ = ["_logistic_intercept"]
