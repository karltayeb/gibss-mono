"""Per-effect fit summary (`gibss.summary.summarize_fit`).

Checks the table against the raw state (columns are the right functions of alpha/mu/var/
ser_log_bf), that it tells the real effect from the null ones, and that the shape toggles
(expand_cs / long), purity, exponentiate, filtering, names, and the Poisson/sparse paths
behave.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss.methods import fit_glm_susie
from gibss.summary import summarize_fit, FitSummary


def _fit_logistic(seed=0, n=400, p=20, causal=3, corr_with=8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, corr_with] = X[:, causal] + 0.05 * rng.standard_normal(n)  # a genuine 2-feature CS
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.3 + 1.8 * X[:, causal])))).astype(float)
    return fit_glm_susie(X, y, L=4, max_iter=60), X, causal, corr_with


def test_columns_match_the_raw_state():
    st, X, causal, _ = _fit_logistic()
    tab = summarize_fit(st, X).table
    assert tab.height == len(st.single_effects)
    for row in tab.to_dicts():
        e = st.single_effects[row["component"] - 1]
        alpha = np.asarray(e.alpha); mu = np.asarray(e.mu); var = np.asarray(e.var)
        top = int(np.argmax(alpha))
        assert row["top"] == top
        assert row["max_pip"] == pytest.approx(float(alpha[top]))
        assert row["beta"] == pytest.approx(float(mu[top]))
        assert row["beta_sd"] == pytest.approx(float(np.sqrt(var[top])))
        assert row["beta_ser"] == pytest.approx(float(alpha @ mu))
        assert row["ser_log_bf"] == pytest.approx(float(e.ser_log_bf))
        assert row["prior_variance"] == pytest.approx(float(e.prior_variance))
        assert row["cs_coverage"] >= 0.95 - 1e-9
        assert row["cs_size"] == len(e.get_cs(coverage=0.95))


def test_header_fields():
    st, X, _, _ = _fit_logistic()
    s = summarize_fit(st, X)
    assert isinstance(s, FitSummary)
    assert (s.n, s.p, s.L) == (400, 20, 4)
    assert s.family == "logistic" and s.kernel == "quad"
    assert s.intercept == pytest.approx(float(st.family_state.intercept_value))
    assert s.random_intercept is None  # not fit here
    assert "GIBSS fit summary" in repr(s) and "n=400" in repr(s)


def test_separates_real_effect_from_nulls():
    st, X, causal, corr_with = _fit_logistic()
    tab = summarize_fit(st, X).table
    real = tab.sort("ser_log_bf", descending=True).row(0, named=True)
    assert real["active"] and real["ser_log_bf"] > 10
    assert real["top"] in (causal, corr_with)
    assert real["cs_size"] <= 3 and real["purity"] > 0.9
    # the null components: weak evidence, diffuse CS, near-zero prior variance, low purity
    nulls = tab.filter(~tab["active"]).to_dicts()
    assert nulls, "expected some null components with L=4 and one causal"
    for row in nulls:
        assert row["cs_size"] > 5 and row["prior_variance"] < 0.1 and row["purity"] < 0.5


def test_expand_cs_and_long_forms():
    st, X, _, _ = _fit_logistic()
    wide = summarize_fit(st).table
    assert "cs" not in wide.columns  # dropped by default
    exp = summarize_fit(st, expand_cs=True).table
    for row in exp.to_dicts():
        assert len(row["cs"]) == row["cs_size"] == len(row["cs_pip"]) == len(row["cs_beta"])
    long = summarize_fit(st, long=True).table
    assert long.height == sum(len(e.get_cs(coverage=0.95)) for e in st.single_effects)
    assert set(long.columns) >= {"component", "feature", "pip", "beta"}


def test_purity_requires_x_and_singletons_are_one():
    st, X, _, _ = _fit_logistic()
    with pytest.raises(ValueError, match="purity"):
        summarize_fit(st, purity=True)  # no X
    assert "purity" not in summarize_fit(st).table.columns  # auto-skip without X
    for row in summarize_fit(st, X).table.to_dicts():
        if row["cs_size"] == 1:
            assert row["purity"] == pytest.approx(1.0)


def test_exponentiate_and_feature_names():
    st, X, _, _ = _fit_logistic()
    names = [f"g{j}" for j in range(20)]
    tab = summarize_fit(st, feature_names=names, expand_cs=True, exponentiate=True).table
    for row in tab.to_dicts():
        assert row["exp_beta"] == pytest.approx(float(np.exp(row["beta"])))
        assert isinstance(row["top"], str) and row["top"].startswith("g")
        assert all(isinstance(c, str) for c in row["cs"])


def test_min_ser_log_bf_filters_but_header_keeps_L():
    st, X, _, _ = _fit_logistic()
    s = summarize_fit(st, min_ser_log_bf=1.0)
    assert s.table.height == s.n_active >= 1
    assert (s.table["ser_log_bf"] >= 1.0).all()
    assert s.L == 4  # header reports the allocated L, not the filtered count


def test_poisson_and_sparse():
    rng = np.random.default_rng(3)
    n, p, causal = 400, 20, 4
    Xd = (rng.uniform(size=(n, p)) < 0.2).astype(float)
    y = rng.poisson(np.exp(-0.3 + 1.2 * Xd[:, causal])).astype(float)
    Xs = jsparse.BCOO.fromdense(Xd)
    st = fit_glm_susie(Xs, y, family="poisson", L=3, method="cf_cavi", max_iter=40)
    s = summarize_fit(st, Xs, exponentiate=True)
    assert s.family == "poisson"
    real = s.table.sort("ser_log_bf", descending=True).row(0, named=True)
    assert real["top"] == causal and real["exp_beta"] == pytest.approx(np.exp(real["beta"]))
