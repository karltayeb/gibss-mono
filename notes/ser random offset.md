**SER with a random offset**
$$
	\psi_{i} = \mathbf{b}^T \mathbf{x}_{i} + o_{i}, \quad o_{i} \sim p_{i}.
$$
We will treat the offset distribution as known. We use the variational approximation:
$$
{\mathcal Q} := \{ q(\mathbf{o, \mathbf{b}}): p(\mathbf{o})q(\mathbf{b}) \}.
$$

$$
\begin{align}
\mathbb{E}_{q} \log p(y | \mathbf{b}, \mathbf{o})  
&= \mathbb{E}_{\mathbf{b}, \mathbf{o}}\left[ \sum y_{i}\psi_{i} - A(\psi_{i}) \right] - KL(q\|p) \\
&= \mathbb{E}_{\mathbf{b}}\left[ \sum y_{i} (\mathbf{b}^T \mathbf{x}_{i} + \mathbb{E} o_{i}) - \mathbb{E}A(\mathbf{b}^Tx +  o_{i}) \right] \\
&= \mathbb{E}_{\mathbf{b}}\left[ \sum y_{i}\mathbf{b}^Tx_{i} - \tilde{A}_i(\mathbf{b}^Tx) \right] + \mathbf{y}^T \mathbb{E}_{\mathbf{o}}\mathbf{o} \\
\end{align}
$$

The data dependent term has the form of an SER without an offset, with log cumulant from the original problem is replaced with the convolution
$$
	\tilde{A}(z) = \mathbb{E}_{o} A(z + o_{i}).
$$
Exact computation of $\tilde{A}_{i}$ is often intractable. A special exception is the Gaussian case. Then $\tilde{A}$ is expressed in terms of the first and second moments of $o$. In the non-gaussian case we will resort to approximation. Let $\hat{A}_{i}$ denote an approximation for $\tilde{A}_{i}$, 
$$
\begin{align}
\mathbb{E}_{q} \log p(y | \mathbf{b}, \mathbf{o})  
&\approx \mathbb{E}_{\mathbf{b}}\left[ \sum y_{i}\mathbf{b}^Tx_{i} - \hat{A}_i(\mathbf{b}^Tx) \right] + C\\
\end{align}
$$

Computation of the exact SER posterior with $\tilde{A}_{i}$ would constitutes coordinate ascent of the ELBO in $q(\mathbf{b})$. We instead approximate this by $\hat{A}_{i}$. We might hope that accurate approximations $\hat{A}$ provide similar inferences.

For greater accuracy, we can adapt the approximation to each choice of effect variable. 
$$
\begin{align}
\mathbb{E}_{q} \log p(y | \mathbf{b}, \mathbf{o})  
&\approx \sum q(\gamma = e_{j}) \mathbb{E}\left[ \sum y_{i}\mathbf{b}^Tx_{i} - \hat{A}_{i, j}(b \cdot e_{j}^Tx) | \gamma = e_{j} \right] + C\\
\end{align}
$$

We will consider several choices of approximation for $\hat{A}$. To compare them it is helpful to outline some desiderate.

1. $\hat{A}$ is convex. $\tilde{A}$ is convex, and it makes the optimization problem easier to handle.
2. $\hat{A} \ge \tilde{A}_{i}$, or $\hat{A} \ge \tilde{A} - \epsilon$. When the surrogate is an upper bound, the resulting approximation is an evidence lower bound.
3. Tunable: we can invest more computation for a more accurate approximation.
4. Approximate symmetry: swapping the roles of $o$ and $z$ does not change the marginal likelihood approximation too much.
5. Neutrality: the approximation must not systematically favor some variables over others.



In the context of SuSiE, $o_i$ is a mixture distribution. We resort to approximation. 

**gIBSS** the gIBSS proposal is to ignore uncertainty in $o$. 
$$
	\hat{A}_{\text{gIBSS}}(z) =  A(z + \mathbb{E}o)
$$
This is a simple approximation. It provides a procedure for approximate Bayesian variable selection given a univariate solver. The only requirements are that the solver accepts a fixed offset, and returns a Bayes factor approximation and the posterior mean of the effect. 

**Taylor Approximation/Delta method**, 
$$
\begin{align}
	\hat{A}_{T}(z) &:= \mathbb{E}_{o} \left[ A(z + \mathbb{E}o) + A'(z + \mathbb{E} o) (o - \mathbb{E} o) + \frac{1}{2}A''(z + \mathbb{E}o)(o - \mathbb{E}o)^2 \right] \\
	& =  A(z + \mathbb{E}o) + \frac{1}{2}  A''(z + \mathbb{E}o) V_{o}.
\end{align}
$$

As $V(o) \to 0$ we recover the gIBSS procedure, which uses a posterior mean message. When we fit each univariate regression by iteratively reweighted least squares, we further expand about a well chosen point $\hat{z}$
$$
	\hat{\hat{A}} = \hat{A}(\hat{z}) + \hat{A}'(\hat{z})(z - \hat{z}) + \hat{A}''(\hat{z})(z-\hat{z})^2
$$

This is exactly the the approximation of the non-linearity used in local-IRLS, global-IRLS.

**Polya-Gamma/Jaakola-Jordan approximation**
$$
	\hat{A}_{\xi}(z) := \log(2) + \frac{z + \mathbb{E}o}{2} + \frac{\tanh(\xi / 2)}{4\xi} (\mathbb{E}_{o}(z + o)^2 - \xi^2) + \cosh(\xi/2).
$$

$\xi$ can be tuned globally, with observation specific $\xi$, or locally, with $\xi_{i,j}$. Implmented by JJ-local, JJ-global. 

**Quadrature** given a quadrature rule $R = \{ (w_k, o_k)\}_{k=1}^K$ 
$$
	\hat{A}_{R}(z) = \sum_{k} w_{k} A(z + o_{k})
$$
deferring discussion on how $R$ is selected, we can note that if the weights $w_k$ are non-negative, than $\hat{A}$ preserves the convexity of $A$. 

To compute the posterior mean and Bayes factor, we can find the mode via IRLS and then perform adaptive Gauss hermite quadrature. 

One proposal is to use an adaptive Gauss-Hermite quadrature rule, on the basis of the working model $o \sim N(\mu_o, \sigma^2_o)$. 
$$
	\hat{A}(z) = \frac{1}{\sqrt{ \pi }} \sum w_{k} A(z + \mu_{o} + \sqrt{ 2 }\sigma_{o}o_{k} )
$$

Generalizing to a Gaussian mixture with a small number of components $o \sim \sum_{j} \pi_{j} N(\mu_{oj}, \sigma^2_{oj})$
$$
	\hat{A}(z) =  \sum_{j} \pi_{j}\left[\frac{1}{\sqrt{ \pi }} \cdot \sum_{k} w_{k}  A(z + \mu_{oj} + \sqrt{ 2 }\sigma_{oj}o_{k} ) \right].
$$
Below we consider the context that the random offset SER is being applied to make reasonable decisions about the quadrature rule.


**SuSiE**. In the sum of single effects regression, the linear predictor is the sum of $L$ single effect contributions (an intercept/fixed effects). Under the variational approximation the components are independent. The ELBO is not easily computed, owing to the non-linearity of the log cumulant. 
$$
	\mathbb{E} A \left( \sum_{l} \psi_{l}  \right)
$$
Naively, taking the product of $K$-point quadrature rules over each dimension $l = 1, \dots, L$ results in an explosion of evaluations. However, a coordinate update to $q(\mathbf{b}_{l})$ is given by fitting a SER with a random offset, where the offset is given by the linear predictions of the non-focal effects: $\mathbf{\psi}_{-l} = X\left( \sum_{l' \ne l} \mathbf{b}_{l'} \right)$. It's not necessary to consider the contributions of each single effect separately. Can we find a good quadrature rule that does not require $O(K^L)$ points?

In the SuSiE model, $\psi_{li} = \mathbf{b}_{l}^T\mathbf{x}_{i}$ is a $p$-component mixture. In practice, each component is approximately Gaussian (or Gaussianity is imposed by the variational approximation). Furthermore, many of these components have overlapping density (as in correlated covariates), and the posterior mass will be concentrated on exactly these overlapping components. In an idealized setting, $\psi_{li}$ is well approximated by a single Gaussian distribution. Then, so too is their sum. We might reasonably approximate the distribution of the offset by as Gaussian distribution. This makes an adaptive Gauss-Hermite quadrature rule, determined by the first two moments of the offset, a reasonable choice. 

If we have a working model for the offsets, a Gaussian mixture with a small number of components $o \sim \sum_{j} \pi_{j} N(\mu_{oj}, \sigma^2_{oj})$
$$
	\hat{A}(z) =  \sum_{j} \pi_{j}\left[\frac{1}{\sqrt{ \pi }} \cdot \sum_{k} w_{k}  A(z + \mu_{oj} + \sqrt{ 2 }\sigma_{oj}o_{k} ) \right].
$$

This provides an attractive surrogate ELBO. $\hat{A}$ is log-concave, computation of the gradient and Hessian are just as simple as in the GLM case. The quality of the approximation is be excellent if we can get a good working model for the offset. 

**Review**:

Using either global or local variants of JJ and Taylor approximation, we get simple methods for fitting the logistic SER. The Taylor approximation method, though not providing a lower bound is generalizable to other GLMs. In a separate line, we can consider approximating $\tilde{A}$ by a quadrature rule. An adaptive Gauss-Hermite quadrature rule using $N(\mu_0, \sigma^2_{0})$ as a working model is an attractive alternative to the Taylor approximation. It preserves convexity, and provides a tunable knob trading off accuracy and cost. Insofar as the the Gaussian working model is good, a modest number of quadrature points can lead to good behavior. The cost, is multiplicative in the number of quadrature points. 

A conceptually simple generalization is to adopt a Gaussian mixture working model. However, this requires distilling a high dimension mixture down to a few components, and then repeatedly updating the working model as each SER component updates. 

**Shape blind**: the JJ approximation, Taylor approximation, and GH quadrature with a Gaussian working model are all blind to the shape of the offset distribution. They only depend on the first two moments. Thus, it will give the same SER posterior regardless of the distribution of $o$.  This criticism extends to gIBSS as well. The GH approach can use more sophisticated quadrature rules to be unblinded by moving to a mixture working model.


**SER without an intercept**

A simple way to implement variable selection is to consider an additive model where some of the components are SERs. 
$$
	\psi_{i} = \mathbf{a}^T \mathbf{x}_{i} + \sum \mathbf{b}_{l}^T \mathbf{ x}_{i}
$$



**JJ bound as Polya-Gamma**. Let $w \sim \text{PG}(1, 0)$ denote the random variable with Laplace transform $\mathbb{E}[\exp(-wt)]= \frac{1}{\cosh\left( \sqrt{ \frac{t}{2} } \right)}$. Let $\text{PG}(1, c)$ denote the exponential tilting $p(w | 1, c) \propto \exp(-c^2/2)p(w | 1, 0)$.  It can be shown that $\mathbb{E}_{c}[w] = \frac{\tanh(c/2)}{2c}$. Then,
$$
\begin{align}
	-\log \cosh\left( \frac{x}{2} \right) &= \log \frac{1}{\cosh\left( \frac{x}{2} \right)}  \\
	&= \log \mathbb{E}_{\text{PG}(1, 0)}\left[ \exp\left( -\frac{x^2}{2} w \right) \right] \\
	&= \log \mathbb E_{\text{PG}(1, c)}\left[ \exp\left( -\frac{x^2}{2}w \right) \frac{p(w | 1, 0)}{p(w |1, c)} \right] \\
	&= \log \mathbb E_{\text{PG}(1, c)}\left[ \exp\left( -\frac{x^2}{2}w \right) \frac{1}{\cosh\left( \frac{c}{2}\right) \exp\left( -\frac{c^2}{2}w \right)} \right] \\
	&\ge -\frac{\mathbb{E}_{c}[w]x^2}{2} - \log \cosh\left( \frac{c}{2} \right) +\mathbb{E}_{c}[w] \frac{c^2}{2} \\
	&= -\frac{\mathbb{E}_{c}[w]}{2}(x^2 - c^2) - \log \cosh\left( \frac{c}{2} \right).
\end{align}
$$
which is the JJ bound.

**Scale mixture approximation.** Consider approximating the integration over $PG(1, 0)$ with a quadrature rule. The quadrature rule can be identified with a (scaled) mixture density:
$$
	\mathbb{E}_{\text{PG}(1, 0)} [\exp(-c w x^2)] \approx \sum w_{k}\exp\left( -\frac{w_{k}}{2} x^2 \right) = C\sum \pi_{k} \phi(\sqrt{ w_{k} } x)
$$

