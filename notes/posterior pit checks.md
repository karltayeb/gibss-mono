# Posterior predictive PIT calibration checks

`gibss.calibration.posterior_pit(data, state)` returns, per observation, the
probability-integral-transform value

    PIT_i = F_i(y_i) = E_{eta_i ~ q}[ F_base(y_i | eta_i) ],

the fitted posterior predictive CDF evaluated at the observed datum. If the model
is correctly specified and calibrated, `{PIT_i} ~ Uniform(0, 1)`; `result.ks`
reports the Kolmogorov-Smirnov statistic and p-value against the uniform.

## The load-bearing piece: offset uncertainty

`q(eta_i) = N(m_i, v_i)` is the fitted posterior over the linear predictor, and
`v_i` is what the request called the *offset uncertainty*. The engine already
tracks it per observation:

    m_i = intercept + glm_offset + total_message.mean_i
    v_i = total_message.var_i (+ intercept_var)      == glm._offset_var(state)

`total_message.var` is moment-matched over BOTH the coefficient posterior and the
feature-selection (`alpha`) uncertainty, and effects' variances add under the
mean-field aggregation. Integrating `F_base(y_i | eta)` over `q(eta_i)` reuses the
same Gauss-Hermite quadrature the `Smoothed` responses use during fitting.

`offset_uncertainty=False` plugs in `eta_i = m_i`. That narrows the predictive and,
if `v_i` is a real fraction of the predictive spread, pushes the PIT toward 0/1
(a U-shaped histogram, i.e. overconfidence). This is exactly the failure the check
guards against, so the flag is a built-in demonstration of why the term matters.

**Empirical note.** In the SER/SuSiE regime with many observations, `v_i` is
typically O(1/n) small relative to the observation noise, so the on/off toggle
barely moves the PIT (the O(1/n) leverage). The offset term matters when the
message variance is comparable to the observation noise: small `n`, strong priors,
or high collinearity with genuine selection ambiguity. The tests exercise the
mechanism by artificially inflating `total_message.var` and checking the integrated
predictive against a brute-force reference.

## Per-family treatment

| Family | `F_base(y \| eta)` | Continuity | Offset uncertainty |
|--------|-------------------|-----------|--------------------|
| linear-Gaussian | `Phi((y - eta)/sigma)` | continuous | closed form (normal convolution; no quadrature) |
| glm-Gaussian | same, fixed `variance` | continuous | GH over eta |
| two-group | `sigma(eta) F1(bhat) + (1-sigma(eta)) F0(bhat)` | continuous | GH-averaged enrichment prob `E[sigma(eta)]` |
| Bernoulli | `1 - sigma(eta)` step | discrete | GH over eta, then randomized PIT |
| Poisson | `Q(floor(y)+1, exp(eta))` | discrete | GH over eta, then randomized PIT |
| Cox (partial) | Cox-Snell: `S_i = exp(-H0(t_i) exp(eta_i))` | continuous time, censoring | GH over eta (see below) |

### Discrete families: randomized PIT

For integer `y` the CDF has jumps, so no single transform is uniform. We use the
randomized PIT (Dunn-Smyth / Czado):

    PIT_i = F(y_i - 1) + V_i (F(y_i) - F(y_i - 1)),   V_i ~ Uniform(0, 1),

with `F` offset-integrated. `randomized=False` substitutes `V_i = 1/2` (the
deterministic *mid-PIT*): reproducible for QQ inspection but NOT uniform - a binary
response makes this obvious (KS ~ 0.15). `seed` fixes the jitter.

### Cox: Cox-Snell residuals

`r_i = H0(t_i) exp(eta_i)` is unit-exponential under the true model, so the
predictive survival `S_i = exp(-r_i)` gives `PIT = 1 - S_i` for an event and, for a
right-censored observation (`T > t_i`, so the transform lies in `(1 - S_i, 1)`), the
randomized `(1 - S_i) + V_i S_i`.

- `H0` is the Breslow baseline from `(time, event, eta)` with Breslow tie-sharing.
- **Offset uncertainty IS available** for Cox even though the engine aggregates
  effects with a `MeanMessage` (variance dropped): each `CoxEffect` still carries
  `(alpha, mu, var)`, so `_reconstruct_eta_var` rebuilds `v_i` (the message-variance
  formula) and `S_i = E_{eta ~ N(m, v)}[exp(-H0 exp(eta))]` is GH-integrated.
- **Caveat:** the Breslow `H0` is held at its plug-in value; its own dependence on
  the `eta_j` (the risk-set denominators) is not propagated. `H0` is the profiled
  nuisance, so this is a deliberate, documented approximation.
- Use `method="partial"` (the `CoxFamilyState`). The `method="poisson"` reduction
  expands to risk-set pseudo-observations, so its glm state is not a survival PIT.

## Caveats (all families)

1. **Mixture-vs-Gaussian.** `q(eta_i)` is a moment-matched single Gaussian of a
   genuinely mixture posterior (the SER selection). GH over the Gaussian is therefore
   an approximation of the exact predictive; a Tier-2 exact mixture integration is
   possible but not implemented.
2. **In-sample optimism.** The check is on the same data used to fit, so uniformity
   holds only approximately (strictly it wants leave-one-out predictives). For the
   effects the influence of a single `y_i` is O(1/n); for shared parameters (the
   intercept, `f0`/`f1`, residual variance) it is not, so this is a self-consistency
   diagnostic, not a frequentist coverage guarantee.
3. **Hyperparameters point-estimated.** `q(eta_i)` is conditional on the fitted
   prior variance, `f0`/`f1`, residual variance, and `nullweight`; their uncertainty
   is not integrated.
