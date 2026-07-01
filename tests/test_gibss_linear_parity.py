import numpy as np

from gibss.engine import fit_ibss
from gibss.linear import default_schedule, initialize_state, prep_data
from gibss_reference.linear import susie_fit


def test_linear_gibss_tracks_reference_on_stable_fixture():
    rng = np.random.default_rng(0)
    n, p, L = 80, 8, 3
    X = rng.normal(size=(n, p))
    beta = np.array([2.5, -1.5, 1.0] + [0.0] * (p - 3))
    y = 0.75 + X @ beta + rng.normal(scale=0.5, size=n)

    data = prep_data(X, y, center=False)
    gibss_state = fit_ibss(
        data,
        init_state=initialize_state(
            data,
            L=L,
            family_state_kwargs={
                "estimate_prior_variance": False,
                "estimate_residual_variance": False,
            },
        ),
        schedule=default_schedule(),
        max_iter=15,
    )
    q_ref, intercept_ref, _elbos_ref, _resid_var_ref = susie_fit(
        X,
        y,
        L=L,
        residual_variance=1.0,
        prior_variance=1.0,
        max_iter=15,
        estimate_residual=False,
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
        atol=2e-4,
    )
    assert np.argmax(pip) == np.argmax(pip_ref)
    assert np.corrcoef(pip, pip_ref)[0, 1] > 0.995
    assert np.corrcoef(posterior_mean, posterior_mean_ref)[0, 1] > 0.995
