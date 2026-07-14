# Baseline-hazard uncertainty in cox_poisson: mean-field integration is a no-op

Resolves the "middle rung" question (shared baseline + tracked variance, the Cox
analog of `intercept_var`): DO NOT BUILD. Under the factorization q(o)q(b) with a
correctly calibrated q(o), integrating over baseline uncertainty changes the
b-update not at all -- exactly, not approximately. The rung is logistic-specific.

## The derivation (per-row information accounting)

Collapsed likelihood, conditional on the baseline (o_j = log Lambda0(t_j)):

    l(b; o) = sum_j y_j (b x_j + o_j) - e^{b x_j + o_j} + [b-free]

Integrating over q(o):

    E_q l = sum_j y_j (b x_j + obar_j) - e^{b x_j} E[e^{o_j}] + C

1. The collapsed loglik is SEPARABLE in the increments b_0k, and each h_k =
   e^{b_0k} has an EXACT Gamma(d_k, S0_k) posterior under a flat prior:
   E[h_k] = d_k / S0_k -- the Breslow estimate IS the posterior mean, and
   Var(h_k) = d_k / S0_k^2 exactly (no delta method).
2. e^{o_j} = Lambda0(t_j) = sum_{k<=j} h_k is LINEAR in the increments, so
   E[e^{o_j}] = sum E[h_k] = Lambda0-hat(t_j): the plug-in, exactly.
3. Hence E_q l has IDENTICAL b-gradient and b-Hessian to the point-estimate fit;
   the differences (sum y_j obar_j, KL(q(o))) are b-free and cancel in alpha.

Why Poisson is special: E[e^{z+o}] = e^z E[e^o] -- the exponential cumulant is
within-family under ANY offset convolution (offset integration = exposure rescale
= offset shift). It is the unique cumulant with this property; for softplus the
convolution genuinely reshapes grad/weight, which is why `intercept_var` is
meaningful for logistic and this rung is not meaningful for Cox/Poisson.
(Corollary: Smoothed(Poisson(), Gaussian) == Poisson with offset += ov/2 exactly;
GH(k) on Poisson numerically approximates an offset shift.)

## The information reversal (why mean-field can't be fixed by integration)

Where information sits, per formulation (unique event times, ~n events):

- Cox PL, per EVENT k: Var_{R_k}(x) -- O(1) while risk sets are large, -> 0 for
  the last events (the last event is a choice among 1: zero information).
- Collapsed Poisson (shared baseline), per SUBJECT j: x_j^2 Lambda0(t_j) e^{bx_j}
  -- Lambda0 ~ H_n - H_{n-j} ~ log n for the longest survivors, ~ 1/n for the
  earliest. The totals agree (I_cond = sum_k d_k S2_k/S0_k, same sum reorganized),
  but the ATTRIBUTION is reversed: the conditional form credits long survivors
  with the largest single-row curvature; the PL says late events carry none.
- The conditional-vs-Schur gap sum_k d_k xbar_k^2 CONCENTRATES in late events
  (at the last event S2/S0 = xbar^2 = x^2: the entire conditional contribution is
  spurious baseline information). Predicts when shared-baseline overconfidence
  bites: censoring patterns leaving small late risk sets.

The reversal is caused by the FACTORIZATION q(o)q(b), not by point-vs-integrated
o. The PL encodes the o-b coupling (the baseline soaks up what late risk sets say
about b); profiling is a degenerate q(o|b) tracking b_0*(b), whose linearization
is exactly the Schur complement.

## E[exp(o)] vs exp(E[o]): why the gradient asymmetry is in BOTH

Possible confusion: the derivation above obtains the late-arrival exposure profile
c_j ~ log n - log j by INTEGRATING (c_j = E[e^{o_j}]), which can suggest the
asymmetry is a product of integration. It is not: the plug-in already IS E[e^o],
because the Breslow increment is the hazard-scale posterior mean, so
exp(o_plug) = Lambda0-hat = E[Lambda0] identically. The two candidates differ by
the Jensen factor E[e^o] = e^{E[o]} e^{v/2} with v_j = Var(log Lambda0(t_j)) ~
1/(events before t_j): at most e^{1/2} ~ 1.65 on the earliest rows (where the
exposure is ~1/n, so absolutely negligible) and -> 1 late. Both scale as
log(n/j). The mean-scale choice (Breslow, what the code uses) is canonical: it
preserves the exact PL score identity at the anchor; the log-scale choice
perturbs the score by O(1/n) early-row terms.

Summary: gradient -- right under plug-in and integrated alike (same scaling, same
fixed point). Curvature/evidence attribution -- wrong under both (mean-field),
fixed only by leaving the factorization (baseline="profiled").

## The structural obstruction

I_PL = sum_k d_k Var_{R_k}(x) is NOT diagonal in subjects: no per-row weights,
centering, or ov-channel consumption on the collapsed n-row Poisson can represent
it. The profiled-baseline treatment must live in risk-set space -- which is why
`baseline="profiled"` routes through the sorted risk-set kernel (the PL read-out)
rather than the ov channel.

## Resulting three-rung table (final form)

| nuisance treatment      | logistic intercept          | Cox baseline                     |
|-------------------------|-----------------------------|----------------------------------|
| shared point estimate   | quad kernel (pre-var-track) | baseline="shared"                |
| shared + tracked var    | quad + intercept_var        | VACUOUS (this note): E[Lambda0] = plug-in |
| profiled per feature    | profile kernel (Schur)      | baseline="profiled" (default)    |

See `response_kernel_tables.md` (Cox section) and milestone 21 in
`response_models.md`.
