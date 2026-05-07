from __future__ import annotations

import jax
import jax.numpy as jnp


def _logistic_intercept(
    b0_init,
    y,
    offset,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
):
    """
    Solve logistic intercept with fixed offset using monotone root finding.

    Returns `(b0, log_likelihood, n_iter, step)` for compatibility with the
    historical helper expected by the tests.
    """
    y = jnp.asarray(y, dtype=float)
    offset = jnp.asarray(offset, dtype=float)
    b0_init = float(b0_init)
    bracket_limit = 60.0

    def gradient(b0: float) -> float:
        eta = offset + b0
        return float(jnp.sum(y - jax.nn.sigmoid(eta)))

    lo = min(b0_init, 0.0)
    hi = max(b0_init, 0.0)
    g_lo = gradient(lo)
    g_hi = gradient(hi)

    while g_lo < 0.0 and lo > -bracket_limit:
        lo = max(lo * 2.0 if lo < 0.0 else -1.0, -bracket_limit)
        g_lo = gradient(lo)
    while g_hi > 0.0 and hi < bracket_limit:
        hi = min(hi * 2.0 if hi > 0.0 else 1.0, bracket_limit)
        g_hi = gradient(hi)

    n_iter = 0
    if g_lo < 0.0:
        b0 = lo
        step = 0.0
    elif g_hi > 0.0:
        b0 = hi
        step = 0.0
    else:
        b0 = 0.5 * (lo + hi)
        step = hi - lo
        for it in range(1, max_iter + 1):
            b0 = 0.5 * (lo + hi)
            g_mid = gradient(b0)
            step = 0.5 * (hi - lo)
            n_iter = it
            if abs(g_mid) < tol or abs(step) < tol:
                break
            if g_mid > 0.0:
                lo = b0
            else:
                hi = b0

    eta = offset + b0
    log_likelihood = jnp.sum(y * eta - jnp.logaddexp(0.0, eta))
    return float(b0), float(log_likelihood), int(n_iter), float(step)
