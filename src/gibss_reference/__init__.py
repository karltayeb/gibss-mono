"""Reference baseline implementations for linear and logistic SuSiE models."""

from .linear import (
    LinearSER,
    elbo as linear_elbo,
    estimate_intercept as estimate_linear_intercept,
    estimate_prior_variance as estimate_linear_prior_variance,
    estimate_residual_variance,
    ser_update as linear_ser_update,
    susie_fit,
)
from .logistic import (
    LogisticSER,
    elbo as logistic_elbo,
    estimate_intercept as estimate_logistic_intercept,
    estimate_prior_variance as estimate_logistic_prior_variance,
    lambda_xi,
    logistic_susie_fit,
    ser_ss,
    ser_update as logistic_ser_update,
    xi_update,
)
from .logistic_local import (
    elbo as logistic_local_elbo,
    local_ser_update,
    logistic_susie_fit_local_jj,
    xi_from_posterior as logistic_local_xi_from_posterior,
)
from .logistic_sparse import (
    elbo_sparse as logistic_sparse_elbo,
    estimate_intercept_sparse as estimate_sparse_logistic_intercept,
    logistic_susie_fit_sparse_global_jj,
    ser_update_sparse as logistic_sparse_ser_update,
    xi_update_sparse as logistic_sparse_xi_update,
)
from .univariate_logistic import (
    UnivariateLogisticRegressionResult,
    fit_univariate_logistic_regression_fixed_intercept,
    fit_univariate_logistic_regression,
)

__all__ = [
    "LinearSER",
    "LogisticSER",
    "UnivariateLogisticRegressionResult",
    "estimate_linear_intercept",
    "estimate_linear_prior_variance",
    "estimate_logistic_intercept",
    "estimate_logistic_prior_variance",
    "estimate_residual_variance",
    "lambda_xi",
    "linear_elbo",
    "linear_ser_update",
    "logistic_elbo",
    "logistic_local_elbo",
    "logistic_local_xi_from_posterior",
    "logistic_sparse_elbo",
    "logistic_sparse_ser_update",
    "logistic_sparse_xi_update",
    "logistic_ser_update",
    "logistic_susie_fit",
    "logistic_susie_fit_local_jj",
    "logistic_susie_fit_sparse_global_jj",
    "local_ser_update",
    "fit_univariate_logistic_regression",
    "fit_univariate_logistic_regression_fixed_intercept",
    "ser_ss",
    "susie_fit",
    "estimate_sparse_logistic_intercept",
    "xi_update",
]
