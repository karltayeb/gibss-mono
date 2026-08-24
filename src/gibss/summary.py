"""Per-effect summary table for a fitted GIBSS model.

`summarize_fit(state)` returns a `FitSummary` with one row per single effect (SER component).
Each row carries the credible set, the evidence for the effect, its size, and the effect
scale. That is what you read to decide which effects are real and what they do. It works on
any `GIBSSState` from `gibss.methods.fit_glm_susie` and knows nothing about gene sets.
Gene-set enrichment summaries (gene lists, set names, overlap) live in `gseasusie.summary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import jax.numpy as jnp
import polars as pl

from .response import Poisson, Smoothed

__all__ = ["FitSummary", "summarize_fit"]

# ser_log_bf (nats) below which an effect is treated as null: no support over the b=0 model.
# Matches the greedy-L stopping threshold (`fit_ibss_greedy` tol_L).
_ACTIVE_LOG_BF = 1.0

_FAMILY = {"Bernoulli": "logistic", "Poisson": "poisson", "Gaussian": "gaussian"}


@dataclass(frozen=True)
class FitSummary:
    """A fitted model's per-effect summary. `table` holds the per-effect rows; the other
    fields are the fit-level header that does not vary by effect. `print(summary)` shows the
    header above the table, and `summary.table` is the raw polars frame to filter or join."""

    table: pl.DataFrame
    n: int
    p: int
    L: int
    n_active: int
    family: str
    kernel: str
    intercept: float
    intercept_sd: float
    coverage: float
    random_intercept: str | None = None

    def __repr__(self) -> str:
        head = [
            f"GIBSS fit summary  ({self.family}, kernel={self.kernel})",
            f"  n={self.n}  p={self.p}  L={self.L} ({self.n_active} active)  "
            f"coverage={self.coverage:g}",
            f"  intercept={self.intercept:.3g} +/- {self.intercept_sd:.3g}",
        ]
        if self.random_intercept is not None:
            head.append(f"  random intercept: {self.random_intercept}")
        return "\n".join(head) + "\n" + str(self.table)


def _base(fs) -> Any:
    return fs.response.base if isinstance(fs.response, Smoothed) else fs.response


def _columns(X, idx: Sequence[int]) -> np.ndarray:
    """The `(n, len(idx))` dense block of columns `idx` from a dense or BCOO design."""
    idx = list(idx)
    if hasattr(X, "todense"):  # BCOO / sparse
        try:
            return np.asarray(X[:, jnp.asarray(idx)].todense())
        except Exception:
            return np.asarray(X.todense())[:, idx]
    return np.asarray(X)[:, idx]


def _purity(X, cs: Sequence[int]) -> float:
    """Min absolute pairwise correlation among the credible-set columns (SuSiE purity): 1.0
    for a singleton, low when the set mixes uncorrelated variables (an uninformative CS).
    Constant columns yield NaN correlations, which are dropped."""
    if len(cs) <= 1:
        return 1.0
    cols = _columns(X, cs)
    c = np.corrcoef(cols, rowvar=False)
    off = np.abs(c[np.triu_indices(len(cs), k=1)])
    off = off[np.isfinite(off)]
    return float(off.min()) if off.size else 1.0


def _random_intercept_header(fs) -> str | None:
    if not getattr(fs, "random_intercept", False):
        return None
    s2 = getattr(fs, "random_intercept_prior_variance", None)
    if s2 is None:
        return "on"
    a = np.asarray(s2)
    return f"sigma^2={float(a):.3g}" if a.ndim == 0 else f"per-row (mean sigma^2={float(a.mean()):.3g})"


def summarize_fit(
    state,
    X=None,
    *,
    coverage: float = 0.95,
    feature_names: Sequence[str] | None = None,
    expand_cs: bool = False,
    long: bool = False,
    purity: bool | None = None,
    exponentiate: bool = False,
    min_ser_log_bf: float | None = None,
) -> FitSummary:
    """Summarize a fitted `GIBSSState`, one row per single effect (SER component).

    Each effect is one credible set with one effect direction. The default columns answer, in
    order, the questions you ask of a fit. Is the effect real (`ser_log_bf`)? Which variable
    or variables (`cs_size`, `top`, and `purity` when `X` is passed)? How sure within the set
    (`max_pip`)? How large (`beta` and `beta_sd`)?

        component     the effect index l (1-based)
        cs_size       number of features in the `coverage` credible set
        top           the top feature (max alpha); a name when `feature_names` is given
        max_pip       alpha of the top feature, i.e. how concentrated the CS is
        cs_coverage   the CS's actual cumulative alpha (at least `coverage`)
        ser_log_bf    SER log Bayes factor against the b=0 model; the evidence for the effect
        beta          the top feature's posterior mean effect, on the link scale
        beta_sd       its posterior standard deviation
        beta_ser      the SER effect sum_j alpha_j mu_j. It equals `beta` for a singleton CS
                      and shrinks for a diffuse one, so a gap from `beta` flags an unsure CS
        prior_variance  the estimated effect scale; near 0 marks a null component
        active        whether ser_log_bf >= 1 nat
        purity        min |correlation| among CS features, computed only when `X` is passed

    `beta` is on the link scale, log-odds for logistic and log-rate for Poisson. Pass
    `exponentiate=True` to add `exp_beta`, the odds ratio or rate ratio. It is skipped for a
    Gaussian base.

    Set `expand_cs=True` to add the per-feature list columns `cs`, `cs_pip`, `cs_beta`. Set
    `long=True` for the tidy form instead, one row per (component, CS feature) with columns
    `component, feature, pip, beta` next to the component-level `ser_log_bf, cs_size,
    prior_variance, active`.

    `min_ser_log_bf` drops components below that evidence, so `min_ser_log_bf=1` keeps only
    active effects. The header still reports the original `L`. `purity` defaults to computing
    when `X` is passed and skipping otherwise. Pass `purity=True` to require it (it raises
    without `X`) or `purity=False` to skip it. `feature_names` maps feature indices to names
    in `top`, `cs`, and the long-form `feature`.

    Returns a `FitSummary`, the polars `table` plus the fit-level header.
    """
    fs = state.family_state
    effects = list(state.single_effects)
    if not effects:
        raise ValueError("summarize_fit: state has no single effects.")
    p = int(np.asarray(effects[0].alpha).shape[0])
    n = int(np.asarray(state.total_message.mean).shape[0])
    L = len(effects)

    if purity is None:
        purity = X is not None
    elif purity and X is None:
        raise ValueError("purity=True needs the design X (to correlate credible-set columns).")

    def name(j: int):
        return feature_names[j] if feature_names is not None else int(j)

    rows: list[dict[str, Any]] = []
    n_active = 0
    for l, e in enumerate(effects, start=1):
        alpha = np.asarray(e.alpha, dtype=float)
        mu = np.asarray(e.mu, dtype=float)
        var = np.asarray(e.var, dtype=float)
        cs = [int(j) for j in e.get_cs(coverage=coverage)]
        top = int(np.argmax(alpha))
        ser_log_bf = float(e.ser_log_bf)
        active = ser_log_bf >= _ACTIVE_LOG_BF
        n_active += active
        row: dict[str, Any] = {
            "component": l,
            "cs_size": len(cs),
            "top": name(top),
            "max_pip": float(alpha[top]),
            "cs_coverage": float(alpha[cs].sum()),
            "ser_log_bf": ser_log_bf,
            "beta": float(mu[top]),
            "beta_sd": float(np.sqrt(max(var[top], 0.0))),
            "beta_ser": float(alpha @ mu),
            "prior_variance": float(e.prior_variance),
            "active": active,
        }
        if exponentiate and not _is_gaussian(fs):  # OR (logistic) / RR (poisson); skip Gaussian
            row["exp_beta"] = float(np.exp(mu[top]))
        if purity:
            row["purity"] = _purity(X, cs)
        # per-feature lists (kept for expand_cs / long; dropped from the default wide table)
        row["cs"] = [name(j) for j in cs]
        row["cs_pip"] = [float(alpha[j]) for j in cs]
        row["cs_beta"] = [float(mu[j]) for j in cs]
        rows.append(row)

    df = pl.from_dicts(rows)
    if min_ser_log_bf is not None:
        df = df.filter(pl.col("ser_log_bf") >= float(min_ser_log_bf))

    list_cols = ["cs", "cs_pip", "cs_beta"]
    if long:
        keep = ["component", "ser_log_bf", "cs_size", "prior_variance", "active"]
        df = (
            df.select(keep + list_cols)
            .explode(list_cols)
            .rename({"cs": "feature", "cs_pip": "pip", "cs_beta": "beta"})
        )
    elif not expand_cs:
        df = df.drop(list_cols)

    return FitSummary(
        table=df,
        n=n,
        p=p,
        L=L,
        n_active=n_active,
        family=_FAMILY.get(type(_base(fs)).__name__, type(_base(fs)).__name__),
        kernel=getattr(fs, "kernel", "?"),
        intercept=float(getattr(fs, "intercept_value", 0.0)),
        intercept_sd=float(np.sqrt(max(float(getattr(fs, "intercept_var", 0.0)), 0.0))),
        coverage=coverage,
        random_intercept=_random_intercept_header(fs),
    )


def _is_gaussian(fs) -> bool:
    return type(_base(fs)).__name__ == "Gaussian"
