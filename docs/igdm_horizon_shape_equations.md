# IGDM horizon-shape experiment

This note records the equations used in the IGDM horizon-shape study
([exp10](../experiments/exp10_horizon_shape.jl)).

## Fixed IGDM mapping

All designs use the same stationary-by-lag mapping with $K=2$ and three
weights $\omega=(\omega_0,\omega_1,\omega_2)$:

$$
\beta_{n,k} =
\begin{cases}
\omega_{n-k}, & 0 \le n-k \le 2,\\
0, & n-k > 2.
\end{cases}
$$

The update is therefore

$$
x_{n+1}
= x_n - \sum_{\ell=0}^{\min(2,n-1)} \omega_\ell d_{n-\ell},
\qquad
\|d_n-\nabla f(x_n)\| \le 0.3\|\nabla f(x_n)\|.
$$

The mapping is fixed as the horizon changes; only the three weights are
optimized.

## Window objectives

For each horizon $N$, let $W_N(\omega)$ be the certified PEP value for
the mapping above, and let $\widehat W_N$ be the best per-horizon value
found within this mapping. Define the normalized ratio

$$
r_N(\omega) = \frac{W_N(\omega)}{\widehat W_N}.
$$

The window is $\mathcal H=\{2,3,\ldots,32\}$. Window-max chooses one
weight vector by minimizing the largest ratio on this set:

$$
\min_{\omega\in\mathbb R^3}\ \max_{N\in\mathcal H} r_N(\omega).
$$

Window-mean instead minimizes the arithmetic mean of the ratios, assigning
equal weight to each of the 31 horizons:

$$
\min_{\omega\in\mathbb R^3}\ \frac{1}{31}\sum_{N\in\mathcal H} r_N(\omega).
$$

The plotted curve for either design is its pointwise ratio $r_N(\omega)$
for each $N=2,\ldots,48$; “max” and “mean” refer to the design objective,
not to a running statistic along the plotted curve.

The per-horizon reference $\widehat W_N$ is a best-known numerical value,
not a proof of global optimality. In this study, no evaluated design
improved the tailored SLM reference.
