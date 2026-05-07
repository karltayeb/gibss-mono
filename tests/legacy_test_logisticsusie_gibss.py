"""Legacy logisticsusie gIBSS coverage kept out of the default suite."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from logisticsusie.gibss import generalized_ibss, symmetrized_kl_ser
from logisticsusie.local_jj import LogisticSER as LocalLogisticSER
from logisticsusie.quadrature_ser import LogisticSER as QuadratureLogisticSER
from logisticsusie.susie import BaseSER

pytestmark = pytest.mark.legacy


def test_symmetrized_kl_ser_zero_for_identical():
    X = jnp.ones((8, 4))
    ser = LocalLogisticSER.init(X)
    skl = symmetrized_kl_ser(ser, ser)
    assert np.isclose(skl, 0.0, atol=1e-12)


def test_symmetrized_kl_ser_positive_and_finite():
    X = jnp.ones((6, 4))
    ser = LocalLogisticSER.init(X)
    lnalpha = ser.lnalpha + jnp.array([0.2, -0.1, 0.0, -0.05])
    lnalpha = lnalpha - jax.nn.logsumexp(lnalpha)
    ser2 = replace(ser, mu=ser.mu + 0.1, var=ser.var * 1.1, lnalpha=lnalpha)
    skl = symmetrized_kl_ser(ser, ser2)
    assert np.isfinite(skl)
    assert skl > 0.0


def test_generalized_ibss_local_jj_runs_and_tracks_history():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 9))
    y = rng.binomial(1, 0.5, size=60)
    result = generalized_ibss(
        X,
        y,
        L=3,
        ser_cls=LocalLogisticSER,
        max_iter=3,
        tol=1e-12,
    )

    assert len(result.single_effects) == 3
    assert len(result.history) == result.n_iter
    assert result.n_iter >= 1

    for ser in result.single_effects:
        assert jnp.isfinite(ser.mu).all()
        assert jnp.isfinite(ser.var).all()
        assert jnp.isfinite(ser.lnalpha).all()
        assert np.isclose(float(jnp.sum(ser.alpha)), 1.0, atol=1e-4)


def test_generalized_ibss_quadrature_backend_runs():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(50, 7))
    y = rng.binomial(1, 0.5, size=50)

    result = generalized_ibss(
        X,
        y,
        L=2,
        ser_cls=QuadratureLogisticSER,
        max_iter=2,
        tol=0.0,
        ser_update_kwargs={"max_iter": 15, "tol": 1e-5},
    )

    assert len(result.single_effects) == 2
    assert result.n_iter == 2
    for ser in result.single_effects:
        assert jnp.isfinite(ser.mu).all()
        assert jnp.isfinite(ser.var).all()
        assert jnp.isfinite(ser.lnalpha).all()
        assert np.isfinite(float(ser.kl))


def test_generalized_ibss_converges_immediately_with_large_tol():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 6))
    y = rng.binomial(1, 0.5, size=40)

    result = generalized_ibss(
        X,
        y,
        L=2,
        ser_cls=LocalLogisticSER,
        max_iter=5,
        tol=1e8,
    )

    assert result.converged
    assert result.n_iter == 1


@dataclass(frozen=True, slots=True)
class DummySER(BaseSER):
    @classmethod
    def init(cls, X, prior_variance: float = 1.0, **kwargs):
        del kwargs
        p = X.shape[1]
        pi = jnp.ones(p) / p
        return cls(
            mu=jnp.zeros(p),
            var=jnp.ones(p),
            lnalpha=jnp.log(pi),
            prior_mean=jnp.zeros(p),
            prior_variance=prior_variance,
            pi=pi,
            kl=0.0,
        )

    @classmethod
    def fit(cls, X, y, offset_mean, offset_var=0.0, **kwargs):
        del y, offset_var, kwargs
        ser = cls.init(X)
        return ser.update(X, y=None, offset_mean=offset_mean, offset_var=0.0)

    def update(self, X, y, offset_mean, offset_var=0.0, **kwargs):
        del X, y, offset_var, kwargs
        new_mu = jnp.array([jnp.asarray(offset_mean)[0] + 1.0])
        return replace(self, mu=new_mu, var=jnp.ones_like(new_mu))


def test_generalized_ibss_uses_leave_one_out_offset():
    X = jnp.array([[1.0]])
    y = jnp.array([1.0])

    result = generalized_ibss(
        X,
        y,
        L=2,
        ser_cls=DummySER,
        max_iter=1,
        tol=0.0,
    )

    m0 = float(result.single_effects[0].posterior_mean_effect[0])
    m1 = float(result.single_effects[1].posterior_mean_effect[0])
    assert np.isclose(m0, 1.0)
    assert np.isclose(m1, 2.0)
