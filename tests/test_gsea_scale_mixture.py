import numpy as np

from gseasusie import GeneEffects, fit_zero_mean_scale_mixture


def test_fit_zero_mean_scale_mixture_recovers_dominant_scale():
    rng = np.random.default_rng(0)
    n = 400
    se = np.full(n, 0.05, dtype=float)
    true_effects = rng.normal(loc=0.0, scale=1.5, size=n)
    observed = true_effects + rng.normal(loc=0.0, scale=se, size=n)
    gene_effects = GeneEffects(
        gene_ids=[f"g{i}" for i in range(n)],
        effect_estimates=observed,
        standard_errors=se,
    )

    fitted = fit_zero_mean_scale_mixture(
        gene_effects,
        scales=[0.25, 0.5, 1.5, 3.0],
    )

    assert fitted.scales.tolist() == [0.25, 0.5, 1.5, 3.0]
    assert fitted.weights.sum() == 1.0
    assert fitted.scales[int(np.argmax(fitted.weights))] == 1.5


def test_normal_scale_mixture_sample_respects_requested_size():
    gene_effects = GeneEffects(
        gene_ids=["g1", "g2", "g3"],
        effect_estimates=np.asarray([1.0, -1.2, 0.8]),
        standard_errors=np.asarray([0.2, 0.2, 0.2]),
    )
    fitted = fit_zero_mean_scale_mixture(
        gene_effects,
        scales=[0.75],
    )

    samples = fitted.sample(25, rng=np.random.default_rng(1))

    assert samples.shape == (25,)
    assert np.all(np.isfinite(samples))
