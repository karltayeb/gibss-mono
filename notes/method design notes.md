# Using the Response design

1. Unsmoothed responses do not access offset variance



## Basic workflow

1. prepare the data
2. prepare the model state
3. mutate the model stat via a schedule


`Bernoulli()`, `Poisson()` are the base models. These responses treat the offset as fixed. The option we have is to profile the intercept or treat it as fixed. 

`Smooth` transforms the response model, usually by approximating the cumulant function. This enables integration over a random offset.
Different options are available for difference responses. Generic options include `Taylor`, `TaylorFixed`, and `GH`. Logistic specific options include `JJ` `JJFixed`, `JJEnvelope`. 
It should be noted that all of these cumulant approximations depend on the first two moments of the linear predictor, and so should only be expected to perform well when the offset is unimodal. 

In principal, given a good quadrature rule we can fit an SER with arbitrary offset distribution to machine precision. However, the exact posterior distribution is no longer gaussian.
It seems reasonable to solve the Gaussian posterior SER, where each $q(b | \gamma = e_j) = N(b| \mu_j, \sigma^2_j)$. In this case we know the distribution of the offset. It is a univariate gaussian mixture with a great number of components. Although we don't want to integrate over all the components explicitly, we can reason about what approximations are reasonable to make (e.g. a gaussian working model, a gaussian mixture with a small nubmer of components).
