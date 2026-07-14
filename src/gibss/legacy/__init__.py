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

NOT moved here (they cannot be, yet): `ser_ops` and `_jj`. `ser_ops` mixes the
legacy logistic KERNELS (quadrature_ser, profile_ser, localjj_ser,
localjj_centered_ser, local_irls, local_irls_centered, local_gaussian_ser --
which these modules import) with genuinely CORE utilities that `gibss.linear`
and `gibss.response_ser` depend on (global_gaussian_ser, _gh_rule,
_normal_logpdf, _cheb_fit_matrix, _clenshaw). Splitting that core/legacy seam is
the next step of the retirement (Tier 2 proper); until then `ser_ops` and its
helper `_jj` stay in the core package and these modules reach up to them via
`from ..ser_ops import ...`.
"""
