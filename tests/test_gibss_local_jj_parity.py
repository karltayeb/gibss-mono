import numpy as np

from gibss.engine import fit_ibss
from gibss.localjj import default_schedule, initialize_state, prep_data
from gibss_reference.logistic_local import logistic_susie_fit_local_jj


def test_local_jj_gibss_tracks_reference_on_stable_fixture():
    rng = np.random.default_rng(2)
    n, p, L = 100, 7, 2
    X = rng.normal(size=(n, p))
    beta = np.array([1.8, -1.0, 0.6] + [0.0] * (p - 3))
    logits = -0.2 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob, size=n).astype(float)

    data = prep_data(X, y)
    gibss_state = fit_ibss(
        data,
        init_state=initialize_state(
            data,
            L=L,
            family_state_kwargs={"estimate_prior_variance": False},
        ),
        schedule=default_schedule(),
        max_iter=10,
    )
    q_ref, intercept_ref, _elbos_ref, _converged_ref = logistic_susie_fit_local_jj(
        X,
        y,
        L=L,
        prior_variance=1.0,
        max_iter=10,
        estimate_prior=False,
        estimate_intercept_flag=True,
    )

    alpha_ref = np.stack([effect.alpha for effect in q_ref])
    mu_ref = np.stack([effect.mu for effect in q_ref])
    pip_ref = 1.0 - np.prod(1.0 - alpha_ref, axis=0)
    posterior_mean_ref = np.sum(alpha_ref * mu_ref, axis=0)

    pip = np.asarray(gibss_state.pip)
    posterior_mean = np.asarray(gibss_state.posterior_mean)

    np.testing.assert_allclose(
        gibss_state.family_state.intercept,
        intercept_ref,
        atol=1e-2,
    )
    assert np.argmax(pip) == np.argmax(pip_ref)
    assert np.corrcoef(pip, pip_ref)[0, 1] > 0.99
    assert np.corrcoef(posterior_mean, posterior_mean_ref)[0, 1] > 0.99
