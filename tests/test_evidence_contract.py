"""The evidence contract: feature_log_marginal + null_log_marginal (same scale),
feature_log_bf / ser_log_bf derived, comparable across kernels.

Guards the pinned definition (`notes` / build_ser_state): every kernel stores an
absolute per-feature log marginal and the SER's b=0 null log marginal on ONE
scale, so the derived Bayes factors are comparable by construction rather than by
each kernel happening to agree.
"""

import numpy as np
import pytest

from gibss import glm
from gibss.engine import fit_ibss
from gibss.response import GH, Bernoulli, Smoothed


def _data(seed=0, n=400, p=10, causal=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, causal]))), n).astype(float)
    return glm.prep_data(X, y), causal


def _fit(d, **kw):
    return fit_ibss(d, glm.initialize_state(d, L=2, **kw), glm.default_schedule(), max_iter=50)


def test_feature_log_bf_is_marginal_minus_null():
    d, _ = _data()
    st = _fit(d, response=Bernoulli())
    for e in st.single_effects:
        np.testing.assert_allclose(
            np.asarray(e.feature_log_bf),
            np.asarray(e.feature_log_marginal) - e.null_log_marginal,
            rtol=1e-12,
        )


def test_null_log_marginal_is_scalar_per_ser():
    # feature-independent: one scalar per single-effect (not a per-feature array)
    d, _ = _data()
    st = _fit(d, response=Bernoulli())
    for e in st.single_effects:
        assert np.ndim(np.asarray(e.null_log_marginal)) == 0
        assert np.asarray(e.feature_log_marginal).shape[0] == 10


def test_ser_log_bf_derivation_and_alias():
    d, _ = _data()
    st = _fit(d, response=Bernoulli())
    per_effect = np.array([e.marginal_log_likelihood - e.null_log_marginal
                           for e in st.single_effects])
    np.testing.assert_allclose(np.asarray(st.ser_log_bf), per_effect, rtol=1e-12)
    # historical alias must agree
    np.testing.assert_array_equal(
        np.asarray(st.ser_log_bf), np.asarray(st.ser_log_bayes_factor)
    )


def test_marginal_is_absolute_null_recoverable():
    # feature_log_marginal is the ABSOLUTE marginal: subtracting the null gives the
    # BF, and null_log_marginal equals the true b=0 loglik at the effect's offset.
    d, _ = _data()
    st = _fit(d, response=Bernoulli())
    e = st.single_effects[0]
    # the b=0 marginal (no effect) must be << any feature's marginal here (real signal)
    assert float(e.null_log_marginal) < float(np.max(np.asarray(e.feature_log_marginal)))
    # and the BF of the top feature is positive
    assert float(np.max(np.asarray(e.feature_log_bf))) > 0.0


@pytest.mark.parametrize(
    "kw",
    [
        {"response": Bernoulli()},  # quad
        {"response": Smoothed(Bernoulli(), GH(9)), "family_state_kwargs": {"kernel": "vi"}},
        {"response": Bernoulli(), "family_state_kwargs": {"intercept": "profiled"}},
    ],
)
def test_ser_log_bf_comparable_across_kernels(kw):
    # different kernels on the SAME data must report ser_log_bf on the same (b=0)
    # reference: the top-effect BFs agree to within approximation error, not off by
    # an arbitrary per-kernel constant.
    d, causal = _data()
    fam = kw.pop("family_state_kwargs", {})
    st = _fit(d, family_state_kwargs=fam, **kw)
    ref = _fit(d, response=Bernoulli())  # quad, shared intercept: the reference
    # both recover the causal feature
    assert max(float(np.asarray(e.alpha)[causal]) for e in st.single_effects) > 0.9
    # the leading SER log-BF is on the same scale (same order of magnitude, not an
    # arbitrary offset). Shared-intercept kernels match tightly; profiled uses its
    # own (profiled) null, so only require the same sign and rough magnitude.
    top = float(np.max(np.asarray(st.ser_log_bf)))
    top_ref = float(np.max(np.asarray(ref.ser_log_bf)))
    assert top > 0 and top_ref > 0
    assert abs(top - top_ref) < 0.5 * abs(top_ref) + 5.0


def test_shared_intercept_kernels_agree_on_ser_log_bf():
    # quad vs vi (both shared intercept, exact-ish): ser_log_bf should be close,
    # since both measure against the same b=0 null.
    d, _ = _data()
    quad = _fit(d, response=Bernoulli())
    vi = _fit(d, response=Smoothed(Bernoulli(), GH(15)),
              family_state_kwargs={"kernel": "vi"})
    # the top log-BF agrees to a few nats (vi is an ELBO lower bound on the quad
    # marginal, so slightly smaller, but the same reference)
    assert abs(float(np.max(np.asarray(quad.ser_log_bf)))
               - float(np.max(np.asarray(vi.ser_log_bf)))) < 5.0
