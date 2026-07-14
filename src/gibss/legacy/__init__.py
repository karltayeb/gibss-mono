"""Legacy logistic SuSiE families, retained as reference / parity oracles.

These modules predate the generic response-model engine (`gibss.glm` +
`gibss.response` + `gibss.response_ser`). They are each a hand-written,
logistic-specific specialization of what the generic engine now does for any
`ResponseModel`, and are kept ONLY to validate the generic kernels against an
independent implementation (the `test_gibss_*jj`, `test_gibss_irls`,
`test_glm` parity tests). New code should use `gibss.methods.fit_glm_susie`
(or `gibss.glm` directly); do not build on these.

The generic equivalents (see notes/response_kernel_tables.md):

    legacy.localjj              == glm kernel="jj"  (glm_jj_ser: the conjugate
                                   fixed-tilt JJ bound, classic localjj)
    legacy.globaljj            == Smoothed(Bernoulli(), JJFixed()) + kernel="linear"
    legacy.irls                == Smoothed(Bernoulli(), TaylorFixed()) + kernel="linear"
                                   (mean-message = the fixed-offset IRLS working model)
    legacy.logistic_localtaylor == kernel="quad" (GH tail over b; profile mode ==
                                   intercept="profiled")

`legacy.ser_ops` and `legacy._jj` also live here: the legacy logistic KERNELS
(quadrature_ser, profile_ser, localjj_ser, localjj_centered_ser, local_irls,
local_irls_centered, local_gaussian_ser) and the Jaakkola-Jordan helpers these
modules import. The five genuinely CORE utilities that used to share
`ser_ops` were extracted so the core does not depend on legacy:
`global_gaussian_ser` -> `gibss.linear`, and the numeric primitives `_gh_rule`,
`_normal_logpdf`, `_cheb_fit_matrix`, `_clenshaw` -> `gibss._numerics`.
`legacy.ser_ops` re-imports those from core for the kernels' use.

What remains for full retirement: repoint the parity tests from
generic-vs-legacy to generic-vs-brute-force (the brute integrals already exist
in test_response / test_quadrature_ser), after which this whole subpackage can
be deleted.
"""
