# IGDM memory-size experiment: objective functions

This note records the equations behind the memory-size study
([exp11](../experiments/exp11_memory_anytime.jl)) and its figure
`experiments/results/igdm_memory_anytime/memory_panels.png`.

**Summary.** Every design minimizes the **maximum** normalized worst-case
value over the horizons. No mean objective is optimized; the "mean" columns
in the result tables are descriptive statistics only.

## 1. Mapping with memory $K$

The stationary-by-lag mapping has $K+1$ weights
$\omega=(\omega_0,\dots,\omega_K)$:

$$
x_{n+1} = x_n - \sum_{\ell=0}^{\min(K,\,n-1)} \omega_\ell \, d_{n-\ell},
\qquad
\|d_n-\nabla f(x_n)\| \le 0.3\,\|\nabla f(x_n)\|.
$$

The tested memory sizes are $K\in\mathcal K=\{0,1,2,3,4,6,8,12\}$.

## 2. Worst-case value at a fixed horizon

For a horizon $N$, the certified PEP value of the weights $\omega$ is

$$
W_N^K(\omega) =
\sup_{f,\,x_1,\,d_1,\dots,d_N}
\ \min_{i=2,\dots,N+1} \|\nabla f(x_i)\|^2 .
$$

The supremum runs over convex $1$-smooth functions $f$ with
$f(x_1)-f_\star \le 1$ and over all admissible inexact directions $d_n$.

The inner quantity is the **smallest** squared gradient norm along the
trajectory, not the largest one. The outer supremum makes it a worst case.

## 3. Tailored values (panel a)

For each pair $(N,K)$, the weights are optimized for that horizon only,
with a local SLM solver:

$$
\widehat W_N^K = \min_{\omega\in\mathbb R^{K+1}} W_N^K(\omega).
$$

The common reference is the best value over all memory sizes:

$$
W_N^{\mathrm{ref}} = \min_{K\in\mathcal K} \widehat W_N^K .
$$

The run also took the minimum with every design value. No design improved
the reference, so $W_N^{\mathrm{ref}}=\widehat W_N^{12}$ in practice.

Panel (a) plots $\widehat W_N^K / W_N^{\mathrm{ref}}$. Here the weights
change with $N$.

## 4. Anytime designs (panels b and c)

Let $\mathcal H=\{2,3,\dots,32\}$ be the design grid. For each $K$, one
weight vector is chosen for all $N\in\mathcal H$. Two different problems
are solved.

**Common design (panel b).** Normalize by the common reference:

$$
\omega_K^{\mathrm{common}} \in
\arg\min_{\omega\in\mathbb R^{K+1}}
\ \max_{N\in\mathcal H}
\ \frac{W_N^K(\omega)}{W_N^{\mathrm{ref}}} .
$$

Panel (b) plots $W_N^K(\omega_K^{\mathrm{common}}) / W_N^{\mathrm{ref}}$.
This compares absolute quality across $K$.

**Within design (panel c).** Normalize by the tailored value of the same
memory size:

$$
\omega_K^{\mathrm{within}} \in
\arg\min_{\omega\in\mathbb R^{K+1}}
\ \max_{N\in\mathcal H}
\ \frac{W_N^K(\omega)}{\widehat W_N^K} .
$$

Panel (c) plots $W_N^K(\omega_K^{\mathrm{within}}) / \widehat W_N^K$.
This measures the price of not knowing $N$ inside the prior $K$.

Panels (b) and (c) use **different weight vectors**, not only different
denominators.

Horizons $N=40$ and $N=48$ are not in $\mathcal H$. The designs are only
evaluated there, to measure extrapolation.

## 5. Worst-case summary (panel d)

Each curve in panel (d) takes the maximum over the design grid:

$$
\begin{aligned}
\text{gray dotted:}\quad
& \max_{N\in\mathcal H} \frac{\widehat W_N^K}{W_N^{\mathrm{ref}}},\\[4pt]
\text{black solid:}\quad
& \max_{N\in\mathcal H} \frac{W_N^K(\omega_K^{\mathrm{common}})}{W_N^{\mathrm{ref}}},\\[4pt]
\text{black dashed:}\quad
& \max_{N\in\mathcal H} \frac{W_N^K(\omega_K^{\mathrm{within}})}{\widehat W_N^K}.
\end{aligned}
$$

The black curves are the optimal values of the two minimax problems of
Section 4.

## 6. How the minimax problem is solved

Write $r_N(\omega)=W_N^K(\omega)/\widehat W_N$, with $\widehat W_N$ the
chosen normalizer. Each iteration of the trust-region sequential LP solves

$$
\min_{t,\ \delta}\ t
\quad\text{s.t.}\quad
r_N(\omega) + \nabla r_N(\omega)^{\top}\delta \le t
\ \ \text{for all } N\in\mathcal H,
\qquad
\|\delta\|_\infty \le \Delta .
$$

The step $\omega+\delta$ is accepted if the true value
$\max_N r_N(\omega+\delta)$ decreases enough, and the radius $\Delta$ is
then enlarged or shrunk.

The gradients $\nabla r_N$ come from the PEP envelope (dual) gradients.

## Caveats

- The SLM and SLP solvers are local. The values $\widehat W_N^K$ and
  $W_N^{\mathrm{ref}}$ are best-known numerical values, not proven global
  optima.
- For $K\le 1$, the common problem is degenerate. Its maximum is attained
  at $N=32$, where no $\omega$ beats the tailored value, so the
  minimizer is not determined by the other horizons.
