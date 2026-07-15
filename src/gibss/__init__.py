import os

import jax

jax.config.update("jax_enable_x64", True)


def enable_compilation_cache(path: str | None = None) -> str:
    """Enable JAX's persistent (on-disk) compilation cache.

    XLA compilations are keyed by computation + shapes, so a fresh process /
    repeated run with the same shapes skips recompilation. Pure performance --
    no effect on results.

    Path resolution: explicit `path` arg, else $GIBSS_JAX_CACHE_DIR, else
    ~/.cache/gibss/jax. `min_compile_time_secs` is set to 0 because the gibss
    kernels compile in well under JAX's 1s default threshold (they would
    otherwise never be cached).
    """
    if path is None:
        path = os.environ.get("GIBSS_JAX_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "gibss", "jax"
        )
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path


# Auto-enable unless opted out (GIBSS_NO_JAX_CACHE=1).
if os.environ.get("GIBSS_NO_JAX_CACHE", "").lower() not in ("1", "true", "yes"):
    try:
        enable_compilation_cache()
    except Exception:
        # Never let cache setup break import (e.g. unwritable home dir).
        pass

from ._logistic_intercept import _logistic_intercept
from .cox_poisson import fit_cox_susie
from .methods import PRESETS, fit_glm_susie
from .twogroup import fit_twogroup_susie

__all__ = [
    "PRESETS",
    "_logistic_intercept",
    "enable_compilation_cache",
    "fit_glm_susie",
    "fit_cox_susie",
    "fit_twogroup_susie",
]
