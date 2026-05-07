import jax
import jax.numpy as jnp
import numpy as np

from gibss2.distributions import (
    Normal,
    NormalMixture,
    PointMass,
    scale_mixture,
    signed_normal,
)


def test_point_mass_normal_means_likelihood_matches_gaussian_error():
    dist = PointMass(value=0.0)
    bhat = jnp.asarray([0.0, 1.0])
    se = jnp.asarray([1.0, 2.0])

    actual = dist.log_likelihood_nm(bhat, se)
    expected = jax.scipy.stats.norm.logpdf(bhat, 0.0, se)

    np.testing.assert_allclose(actual, expected)


def test_normal_normal_means_likelihood_convolves_observation_error():
    dist = Normal(loc=2.0, scale=3.0)
    bhat = jnp.asarray([1.0, 4.0])
    se = jnp.asarray([0.5, 2.0])

    actual = dist.log_likelihood_nm(bhat, se)
    expected = jax.scipy.stats.norm.logpdf(
        bhat,
        2.0,
        jnp.sqrt(jnp.square(se) + 9.0),
    )

    np.testing.assert_allclose(actual, expected)


def test_normal_update_returns_modified_copy_when_enabled():
    dist = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)
    bhat = jnp.asarray([1.0, 2.0, 3.0])
    se = jnp.asarray([0.1, 0.1, 0.1])
    weights = jnp.asarray([1.0, 1.0, 1.0])

    updated = dist.update_nm(bhat, se, weights)

    assert updated is not dist
    assert dist.loc == 0.0
    assert dist.scale == 1.0
    assert updated.loc == np.float32(2.0)
    assert updated.scale > 0.0


def test_normal_update_respects_fixed_parameters():
    dist = Normal(loc=0.0, scale=1.0, estimate_loc=False, estimate_scale=False)
    updated = dist.update_nm(
        jnp.asarray([10.0, 11.0]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([1.0, 1.0]),
    )

    assert updated == dist


def test_normal_mixture_normal_means_likelihood_uses_logsumexp():
    dist = NormalMixture(
        weights=jnp.asarray([0.25, 0.75]),
        locs=jnp.asarray([-1.0, 1.0]),
        scales=jnp.asarray([0.5, 2.0]),
    )
    bhat = jnp.asarray([0.2, 3.0])
    se = jnp.asarray([1.0, 0.25])

    actual = dist.log_likelihood_nm(bhat, se)
    component_sd = jnp.sqrt(jnp.square(se)[:, None] + jnp.square(dist.scales)[None, :])
    expected = jax.nn.logsumexp(
        jnp.log(dist.weights)[None, :]
        + jax.scipy.stats.norm.logpdf(bhat[:, None], dist.locs[None, :], component_sd),
        axis=1,
    )

    np.testing.assert_allclose(actual, expected)


def test_normal_mixture_update_updates_weights_when_enabled():
    dist = NormalMixture(
        weights=jnp.asarray([0.5, 0.5]),
        locs=jnp.asarray([-4.0, 4.0]),
        scales=jnp.asarray([0.5, 0.5]),
        estimate_weights=True,
    )
    bhat = jnp.asarray([3.8, 4.2, 4.1])
    se = jnp.asarray([0.2, 0.2, 0.2])
    weights = jnp.ones(3)

    updated = dist.update_nm(bhat, se, weights)

    assert updated is not dist
    assert updated.weights[1] > 0.99
    np.testing.assert_allclose(jnp.sum(updated.weights), 1.0)


def test_distribution_convenience_constructors():
    signed = signed_normal(mu=3.0, sigma=0.5)
    np.testing.assert_allclose(signed.weights, [0.5, 0.5])
    np.testing.assert_allclose(signed.locs, [-3.0, 3.0])
    np.testing.assert_allclose(signed.scales, [0.5, 0.5])

    scaled = scale_mixture(scales=jnp.asarray([1.0, 2.0]), weights=jnp.asarray([0.2, 0.8]))
    np.testing.assert_allclose(scaled.weights, [0.2, 0.8])
    np.testing.assert_allclose(scaled.locs, [0.0, 0.0])
    np.testing.assert_allclose(scaled.scales, [1.0, 2.0])


def test_distribution_sampling_shapes():
    rng = np.random.default_rng(1)
    assert PointMass(2.0).sample(rng, 3).shape == (3,)
    assert Normal(0.0, 1.0).sample(rng, (2, 3)).shape == (2, 3)
    assert signed_normal(1.0, 0.1).sample(rng, 4).shape == (4,)
