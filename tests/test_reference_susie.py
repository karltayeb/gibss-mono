import pytest
import numpy as np
from scipy.stats import norm
from collections import namedtuple

SER = namedtuple('SER', ['mu','var', 'alpha', 'mu_predict', 'var_predict', 'kl', 'prior_variance'])

def logsumexp(x):
    return np.log(np.sum(np.exp(x - x.max()))) + x.max()

def softplus(x):
    return np.exp(x - logsumexp(x))

def ser(X, y, residual_variance, prior_variance):
    n, p = X.shape

    tau0 = 1 / prior_variance
    d_sq = (X**2).sum(0)
    
    # Precision of the data term
    tau = d_sq / residual_variance
    
    # OLS estimate: (X.T @ y) / (X**2).sum(0)
    bhat = (y @ X) / d_sq
    
    # Posterior precision and variance
    var = 1 / (tau + tau0)
    mu = tau * var * bhat

    # Log Bayes factor: marginal likelihood vs null
    # Variance of the OLS estimate bhat is 1/tau = residual_variance / d_sq
    var_bhat = 1 / tau
    lbf = norm.logpdf(bhat, 0, np.sqrt(prior_variance + var_bhat)) - norm.logpdf(bhat, 0, np.sqrt(var_bhat))
    alpha = softplus(lbf)

    mu_predict = X @ (mu * alpha)
    var_predict = (X**2) @ (alpha * (mu**2 + var)) - mu_predict**2

    kl_cat = alpha @ np.log(np.maximum(alpha * p, 1e-15))
    kl_gauss = 0.5 * ((var + mu**2)/prior_variance - np.log(var/prior_variance) - 1)
    kl = alpha @ kl_gauss + kl_cat

    return SER(mu, var, alpha, mu_predict, var_predict, kl, prior_variance)

def elbo(X, y, q, residual_variance):    
    n = y.size
    mu_predict = np.sum([ql.mu_predict for ql in q], axis=0)
    var_predict = np.sum([ql.var_predict for ql in q], axis=0)
    erss = np.sum((y - mu_predict)**2) + np.sum(var_predict)

    kl = np.sum([ql.kl for ql in q])
    return -0.5 * n * np.log(2 * np.pi * residual_variance) - 0.5 / residual_variance * erss - kl

def estimate_residual_variance(X, y, q):
    n = y.size
    mu_predict = np.sum([ql.mu_predict for ql in q], axis=0)
    var_predict = np.sum([ql.var_predict for ql in q], axis=0)
    erss = np.sum((y - mu_predict)**2) + np.sum(var_predict)
    return erss / n

def estimate_prior_variance(q):
    # EM update for prior variance: E[b^2 | gamma=1]
    prior_var_est = np.maximum(q.alpha @ (q.var + q.mu**2), 1e-8)
    
    kl_cat = q.alpha @ np.log(np.maximum(q.alpha * q.alpha.size, 1e-15))
    kl_gauss = 0.5 * ((q.var + q.mu**2)/prior_var_est - np.log(q.var/prior_var_est) - 1)
    kl = q.alpha @ kl_gauss + kl_cat
    
    return SER(q.mu, q.var, q.alpha, q.mu_predict, q.var_predict, kl, prior_var_est)

def susie(X, y, L, residual_variance=1.0, prior_variance=1.0, max_iter=10):
    residual = y
    q = []
    elbos = []
    for l in range(L):
        ql = ser(X, residual, residual_variance, prior_variance)
        residual = residual - ql.mu_predict
        q.append(ql)
    elbos.append(elbo(X, y, q, residual_variance))

    for i in range(max_iter):
        for l in range(L):
            residual = residual + q[l].mu_predict
            q[l] = ser(X, residual, residual_variance, q[l].prior_variance)
            residual = residual - q[l].mu_predict
        elbos.append(elbo(X, y, q, residual_variance))
    return q, np.array(elbos)


@pytest.fixture
def simulated_data():
    n, p = 200, 50
    np.random.seed(42)
    X = np.random.normal(size=(n, p))
    # Create some strong signals
    y = 5.0 * X[:, 0] - 3.0 * X[:, 1] + 2.0 * X[:, 2] + np.random.normal(size=n)
    return X, y

def test_ser_monotonicity(simulated_data):
    X, y = simulated_data
    q, elbos = susie(X, y, L=5, max_iter=20, residual_variance=1.5, prior_variance=0.5)
    
    # Check if ELBO is monotonically increasing
    diffs = elbos[1:] - elbos[:-1]
    min_diff = diffs.min()
    assert min_diff >= -1e-6, f"ELBO decreased by {-min_diff}"

def test_residual_variance_update(simulated_data):
    X, y = simulated_data
    q, elbos = susie(X, y, L=3, max_iter=5, residual_variance=2.0)
    
    old_elbo = elbo(X, y, q, 2.0)
    new_res_var = estimate_residual_variance(X, y, q)
    new_elbo = elbo(X, y, q, new_res_var)
    
    assert new_elbo >= old_elbo - 1e-6, f"Residual variance update decreased ELBO by {old_elbo - new_elbo}"

def test_prior_variance_update(simulated_data):
    X, y = simulated_data
    q, elbos = susie(X, y, L=3, max_iter=5, residual_variance=1.5, prior_variance=0.5)
    
    old_elbo = elbo(X, y, q, 1.5)
    
    # Update prior variance for all single effects
    q_new = [estimate_prior_variance(ql) for ql in q]
    new_elbo = elbo(X, y, q_new, 1.5)
    
    assert new_elbo >= old_elbo - 1e-6, f"Prior variance update decreased ELBO by {old_elbo - new_elbo}"

