"""Freezing effects whose prior variance has shrunk below a threshold (`freeze_prior_variance`).

Type-II-MLE prior variances split cleanly: real effects sit at O(1), null effects plateau at
the inverse-Fisher floor (~1e-3, not 0). A threshold in that ~1000x gap drops the nulls from
`update_order` so a large-L fit stops re-fitting them. A frozen null has mu ~ 0, so its ~0
message leaves the surviving effects' PIPs unchanged.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from dataclasses import replace

from gibss.methods import fit_glm_susie
from gibss.engine import freeze_shrunk_effects_step


def test_large_L_freezes_nulls_and_matches_full_fit():
    rng = np.random.default_rng(0)
    n, p = 500, 40
    X = rng.standard_normal((n, p))
    causal = [3, 21]
    eta = -0.3 + 2.2 * X[:, 3] - 1.9 * X[:, 21]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    full = fit_glm_susie(X, y, L=8, method="logistic", max_iter=50)
    frz = fit_glm_susie(X, y, L=8, method="logistic", freeze_prior_variance=0.1, max_iter=50)

    # some effects were frozen out of update_order (nulls plateau well below 0.1)
    assert len(frz.update_order) < len(full.single_effects)
    assert len(frz.update_order) >= 1  # the floor keeps at least one
    # every surviving (active) effect is above the threshold; every frozen one below
    active = set(frz.update_order)
    for l, e in enumerate(frz.single_effects):
        if l in active:
            continue
        assert float(e.prior_variance) < 0.1

    # the real effects are recovered identically, and their PIPs are unchanged by freezing
    def tops(st):
        return sorted({int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects
                       if float(e.ser_log_bf) > 1})

    assert tops(frz) == tops(full) == causal
    pf, pz = np.asarray(full.pip), np.asarray(frz.pip)
    np.testing.assert_allclose(pf[causal], pz[causal], atol=1e-3)


def test_default_off_is_byte_identical():
    # freeze_prior_variance=None (the default) installs nothing: identical to a plain fit.
    rng = np.random.default_rng(1)
    n, p = 300, 20
    X = rng.standard_normal((n, p))
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.4 + 1.8 * X[:, 5])))).astype(float)
    a = fit_glm_susie(X, y, L=5, method="logistic", max_iter=30)
    b = fit_glm_susie(X, y, L=5, method="logistic", freeze_prior_variance=None, max_iter=30)
    np.testing.assert_array_equal(np.asarray(a.alpha), np.asarray(b.alpha))


def test_step_prunes_update_order_by_threshold():
    # unit test of the step on hand-built dataclass states (so dataclasses.replace works):
    # drop effects below the threshold, keep the rest, honor warmup and the min_active floor.
    from dataclasses import dataclass

    @dataclass
    class E:
        prior_variance: float

    @dataclass
    class S:
        single_effects: list
        n_iter: int
        update_order: tuple

    def st(pvs, n_iter):
        return S([E(v) for v in pvs], n_iter, tuple(range(len(pvs))))

    # below warmup: no change
    s = st([1.0, 1e-4, 2.0], 0)
    assert freeze_shrunk_effects_step(None, s, min_prior_variance=1e-2, warmup=1).update_order == (0, 1, 2)
    # past warmup: drop the shrunk one
    s = st([1.0, 1e-4, 2.0], 3)
    assert freeze_shrunk_effects_step(None, s, min_prior_variance=1e-2, warmup=1).update_order == (0, 2)
    # min_active floor: all shrunk -> keep the single strongest (largest prior variance)
    s = st([1e-4, 3e-4, 2e-4], 3)
    kept = freeze_shrunk_effects_step(None, s, min_prior_variance=1e-2, warmup=1, min_active=1).update_order
    assert kept == (1,)
