# PEPDesign Manual

Complete reference for the PEPDesign toolbox: parametric Performance
Estimation Problems (PEPs) with automatically derived exact sensitivities,
and step-size policy design by first-order, second-order, SSDP, and
multi-horizon (HRDP) methods.

Companion guides: [experiments.md](experiments.md) (paper experiments),
[cluster.md](cluster.md) (SLURM / CECI pipeline).

---

## Contents

1. [Installation and environments](#1-installation-and-environments)
2. [Mathematical background](#2-mathematical-background)
3. [Architecture overview](#3-architecture-overview)
4. [The symbolic layer (`params.jl`)](#4-the-symbolic-layer)
5. [The Gram-space DSL and the high-level frontend (`gram.jl`, `classes.jl`, `frontend/`)](#5-the-gram-space-dsl)
6. [Compilation (`compile.jl`)](#6-compilation)
7. [Solving (`solve.jl`)](#7-solving)
8. [Sensitivities (`sensitivity.jl`)](#8-sensitivities)
9. [Policies (`policies.jl`)](#9-policies)
10. [Built-in methods (`methods/`)](#10-built-in-methods)
11. [Design optimizers (`design/`)](#11-design-optimizers)
12. [Multi-horizon robust design (HRDP)](#12-multi-horizon-robust-design-hrdp)
13. [Adding a new method](#13-adding-a-new-method)
14. [Adding a new policy](#14-adding-a-new-policy)
15. [Validation suite](#15-validation-suite)
16. [Performance notes](#16-performance-notes)
17. [Pitfalls and troubleshooting](#17-pitfalls-and-troubleshooting)

---

## 1. Installation and environments

Requirements: Julia ≥ 1.10 and a Mosek license (free for academia; the
open-source fallback is any JuMP SDP solver through `GenericBackend`).

Two environments exist:

| Environment | Path | Purpose |
|---|---|---|
| **Root** (recommended) | `Opt_Methods/Project.toml` | Package deps **plus** Plots/JLD2/LaTeXStrings for scripts. All dev/experiment/cluster scripts activate this one. |
| Package | `PEPDesign/Project.toml` | Core deps only (JuMP, Mosek, MosekTools, ForwardDiff, SparseArrays…). Use if you embed PEPDesign elsewhere. |

```powershell
cd Opt_Methods
julia --project=. -e "import Pkg; Pkg.instantiate()"
julia --project=. --threads=4 PEPDesign/dev/run_all_tests.jl   # ~2 min, needs Mosek
```

> **Warning.** Do not run `Pkg.resolve()`/`Pkg.up()` on the root
> environment: a pre-existing registry compat conflict (SpecialFunctions)
> makes re-resolution fail. The pinned `Manifest.toml` works as-is.

Loading the package in a script:

```julia
import Pkg; Pkg.activate("path/to/Opt_Methods"; io = devnull)
include("path/to/Opt_Methods/PEPDesign/src/PEPDesign.jl")
using .PEPDesign
```

## 2. Mathematical background

A fixed-step first-order method with coefficient array `η ∈ R^p` has a tight
worst-case value `W(η, N)` given by a semidefinite program (the PEP) over a
Gram matrix `G ⪰ 0` of inner products and a vector `F` of function values:

```
W(η) = max_{G⪰0, F}  tr(C(η)G) + b'F + c0
       s.t.          tr(A_c(η)G) + f_c'F + c0_c ≤ 0   (or = 0),  c = 1..m_c
```

**The key structural fact** exploited by PEPDesign: when iterates are kept as
Gram basis vectors (with updates encoded as `‖residual‖² = 0`) or substituted
as affine combinations of basis vectors, every constraint matrix is **at most
quadratic** in η:

```
A_c(η) = A0_c + Σ_r η_r·A1_c[r] + Σ_{r≤s} η_r η_s·A2_c[(r,s)]
```

The sparse constant tensors `(A0, A1, A2)` are extracted **once** per
(method, horizon). Everything downstream is exact linear algebra:

- **Value**: assemble `A_c(η)` by sparse axpy, solve one SDP.
- **Gradient** (envelope theorem, at a primal–dual solution `(G, F, λ, γ)`):
  `∂W/∂η_r = −Σ_c λ_c tr(G ∂A_c/∂η_r) + Σ_k γ_k tr(G ∂C_k/∂η_r)`.
- **Frozen-certificate curvature** `H_cert`: same contraction with
  `∂²A_c/∂η_r∂η_s` — this is the ω–ω block of the NSDP Lagrangian Hessian,
  **not** the Hessian of the value function (paper, Remark on
  fixed-certificate curvature).
- **Policy pullback**: for `η = α(ω)` with Jacobian `J` and per-coefficient
  Hessians `Hpol`: `∇_ω W = J'∇_η W`, `H^ω = J'H^η J + Σ_r (∇_η W)_r Hpol[r]`.

Design then minimizes `W(α(ω), N)` over the low-dimensional `ω` (PDP), or the
normalized multi-horizon average `σ(𝓗, ω)` (HRDP), or attacks the joint
nonlinear SDP in `(λ, ω)` directly (SSDP).

## 3. Architecture overview

```
                 ┌───────────────────────────────────────────────┐
 methods/*.jl    │  PEPModel  (points, f-values, coefficients,   │  ← you write this
 (DSL)           │             constraints, objective)           │     (~40–80 lines)
                 └──────────────────────┬────────────────────────┘
                                        │ compile(m)          [once per (method, N)]
                 ┌──────────────────────▼────────────────────────┐
                 │  CompiledPEP  — exact sparse (A0, A1, A2)     │
                 └───────┬──────────────────────────┬────────────┘
        solve_pep(cp, η) │                          │ grad_hess_eta(cp, η, sol)
                 ┌───────▼────────┐        ┌────────▼───────────┐
                 │  PEPSolution   │───────▶│ ∇W, H_cert (η)     │
                 │  (G, F, λ, γ)  │        └────────┬───────────┘
                 └────────────────┘                 │ pullback via policy (J, Hpol)
                                          ┌─────────▼──────────┐
                                          │ ∇_ω W, H_cert^ω    │
                                          └─────────┬──────────┘
              design_fom / design_som / design_ssdp │ HRDPObjective / SampledHRDP
                                          ┌─────────▼──────────┐
                                          │  DesignTrace /     │
                                          │  SSDPTrace         │
                                          └────────────────────┘
```

Include order (see `src/PEPDesign.jl`): `params → gram → compile → solve →
sensitivity → classes → policies → design/{oracle,trace,fom,som,ssdp,hrdp} →
methods/{ogd,item,igdm}`.

## 4. The symbolic layer

`src/params.jl` — an exact degree-≤2 polynomial algebra in the coefficient
vector η. No computer-algebra system, no runtime AD.

| Symbol | Description |
|---|---|
| `PAff` | affine scalar `c0 + Σ_r lin[r]·η_r` (fields `c0::Float64`, `lin::Dict{Int,Float64}`) |
| `coeff(r)` | the affine symbol η_r |
| `PQuad` | quadratic scalar; adds `quad::Dict{(r≤s),Float64}` for monomials η_r·η_s |
| `affmul(a, b)` | product of two `PAff`s **when at least one is constant** (errors otherwise — see §17) |
| `quadmul(a, b)` | exact product `PAff × PAff → PQuad` |
| `evaluate(x, η)` | numeric value of a `PAff`/`PQuad` at η |
| `isconst(x)` | true if independent of η |

Standard `+ − *` arithmetic with `Real`s is defined on both types.

## 5. The Gram-space DSL

`src/gram.jl` — PEPit-style modeling.

### Building blocks

| Symbol | Description |
|---|---|
| `PEPModel()` | empty model; tracks Gram dimension `dim`, #f-values `nf`, #coefficients `np` |
| `point!(m; name)` | declare a new Gram basis vector, returns a `PointExpr` |
| `PointExpr()` | the **zero point** — use it to pin `x_* = 0`, `g_* = 0` (removes the basis vector entirely; improves conditioning) |
| `fval!(m)` | declare a function value, returns an `FVal` |
| `coeffs!(m, k)` | register k coefficients, returns `Vector{PAff}` |

### Expressions

`PointExpr` supports `+ − ` and scaling by `Real`/`PAff` (coefficients must
stay affine in η). Scalar expressions (`QExpr`):

| Symbol | Description |
|---|---|
| `inner(p, q)` (= `LinearAlgebra.dot`) | inner product ⟨p, q⟩; entries become `PQuad` |
| `sqnorm(p)` | ‖p‖² |
| `FVal` arithmetic | `f[i] - f[j]`, `2.0*f[i]`, `β*f[i]` all yield `QExpr` |
| `qexpr(x)` | promote `Real`/`PAff`/`FVal` to `QExpr` |

### Constraints and objective

| Symbol | Description |
|---|---|
| `add_le!(m, e; name)` | add `e ≤ 0` |
| `add_eq!(m, e; name)` | add `e = 0` |
| `objective_max!(m, e)` | maximize `e` |
| `objective_maxmin!(m, [e_1, …, e_K])` | maximize `min_k e_k` (epigraph; used by IGDM `:min_grad`) |

### Function classes (`src/classes.jl`)

```julia
add_interpolation_fmuL!(m, ys, gs, fs; L, μ = 0.0, name = "ic")
```

adds the 𝓕_{μ,L} interpolation inequalities for the triples
`(ys[k], gs[k], fs[k])` over all ordered pairs `k ≠ l`. `μ = 0` gives plain
L-smooth convex interpolation. Points may be zero `PointExpr()`s (e.g. the
optimum with `g_* = 0`).

### PSD (LMI) blocks

```julia
add_psd!(m, mat::AbstractMatrix; name = "psd…")   # mat ⪰ 0, QExpr entries
```

registers a matrix constraint `T(G, F, η) ⪰ 0` (entries symmetrized at
compile time). Used by the quadratic and linear-operator interpolation
classes below; free scalar slacks inside LMI entries are plain `fval!`
variables. Supported by `solve_pep`/`grad_hess_eta`; **not** by `design_ssdp`
(clear error).

### High-level frontend (`src/frontend/`, ported from PEPit.jl)

The frontend puts a PEPit.jl-style object layer on top of the DSL — same
compiled tensors, zero changes to the design machinery. Ported from
[PEPit.jl](https://github.com/PerformanceEstimation/PEPit.jl) (MIT License,
© 2025 Shuvomoy Das Gupta and contributors).

**Oracles** (`frontend/core.jl`). Every class object records triples
`(x, g, f)` on its model and emits its interpolation constraints once, at
`compile` time:

| Symbol | Description |
|---|---|
| `oracle!(f, x) → (g, fx)` | (sub)gradient + value at `x`; gradients reused iff the class is differentiable, values always |
| `gradient!(f, x)`, `value!(f, x)` | the two components of `oracle!` |
| `stationary_point!(f) → (x, g≡0, fx)` | fresh optimum with zero (sub)gradient |
| `fixed_point!(op) → (x, x, fx)` | fixed point `T x = x` of an operator |
| `add_oracle_point!(f, x, g[, fv])` | register a custom triple (pinned gradients, e.g. Ψ* = −Σ g*) |
| `f1 + f2`, `a * f`, `f / a` | linear combinations; oracles decompose onto leaves, registered triples back-substitute into the last leaf |
| `model_of`, `npoints`, `triples` | bookkeeping accessors |

**Function classes** (`frontend/functions.jl`), constructors take the model:
`ConvexFunction(m)`, `StronglyConvexFunction(m; μ)`, `SmoothFunction(m; L)`,
`SmoothConvexFunction(m; L)`, `SmoothStronglyConvexFunction(m; μ, L)`,
`ConvexLipschitzFunction(m; M)`, `SmoothConvexLipschitzFunction(m; L, M)`,
`ConvexIndicatorFunction(m; D, R, center)`, `ConvexSupportFunction(m; M)`,
`ConvexQGFunction(m; L)`, `RsiEbFunction(m; μ, L)`,
`SmoothStronglyConvexQuadraticFunction(m; μ, L)` [LMI],
`SmoothQuadraticLojasiewiczFunctionCheap(m; μ, L, α)`,
`SmoothQuadraticLojasiewiczFunctionExpensive(m; μ, L)` [LMI].

**Operator classes** (`frontend/operators.jl`):
`MonotoneOperator(m)`, `StronglyMonotoneOperator(m; μ)`,
`LipschitzOperator(m; L)`, `CocoerciveOperator(m; β)`,
`NonexpansiveOperator(m; v)`, `NegativelyComonotoneOperator(m; ρ)`,
`LipschitzStronglyMonotoneOperatorCheap/Expensive(m; μ, L)` [Expensive: LMI],
`CocoerciveStronglyMonotoneOperatorCheap/Expensive(m; μ, β)` [Expensive: LMI],
`LinearOperator(m; L)` (+ `adjoint_oracle!`) [LMI],
`SymmetricLinearOperator(m; μ, L)` [LMI], `SkewSymmetricLinearOperator(m; L)`
[LMI].

**Primitive steps** (`frontend/steps.jl`), all accepting symbolic step sizes
`γ :: Union{Real, PAff}` unless noted:

| Step | Semantics |
|---|---|
| `proximal_step!(x0, f, γ)` | `x = x0 − γ·gx`, `gx ∈ ∂f(x)` (also the resolvent of an operator) |
| `inexact_gradient!(f, x, ε; notion)` | `d ≈ ∇f(x)`, `:absolute` or `:relative` error |
| `inexact_gradient_step!(x0, f, γ, ε; notion)` | step along the inexact gradient |
| `inexact_proximal_step!(x0, f, γ; opt)` | `:PD_gapI/II/III` criteria (symbolic γ for `:PD_gapII` only) |
| `exact_linesearch_step!(x0, f, dirs)` | `∇f(x) ⟂ span(dirs, x − x0)` |
| `linear_optimization_step!(dir, ind)` | LMO: `−dir ∈ ∂ind(x)` |
| `shifted_optimization_step!(dir, f)` | `dir ∈ ∂f(x)` |
| `bregman_gradient_step!(gx0, sx0, h, γ)` | mirror step `∇h(x) = sx0 − γ·gx0` |
| `bregman_proximal_step!(sx0, h, f, γ)` | proximal mirror step |
| `epsilon_subgradient_step!(x0, f, γ)` | step along `g ∈ ∂_ε f(x0)` (Fenchel encoding) |

A symbolic `γ` multiplying an η-dependent point raises the `affmul` error —
the residual-style encoding of §13 applies unchanged.

See [examples/gradient_descent.jl](../examples/gradient_descent.jl) for the
canonical end-to-end use, and `dev/test_frontend_classes.jl` for one known
tight rate per class family.

## 6. Compilation

`src/compile.jl`.

| Symbol | Description |
|---|---|
| `compile(m::PEPModel) → CompiledPEP` | emits registered class constraints (`finalize_classes!`), then one-time exact extraction of every constraint into a `ParamMatrix` |
| `ParamMatrix` | `A0::SpMat`, `A1::Vector{Pair{Int,SpMat}}` (η_r ⇒ matrix), `A2::Vector{Tuple{Int,Int,SpMat}}` ((r ≤ s) ⇒ coefficient matrix of η_r·η_s) |
| `CompiledPEP` | fields `dim`, `nf`, `np`, `cons::Vector{CompiledConstraint}`, `psd::Vector{CompiledPSD}`, `obj::Vector{CompiledExpr}` |
| `CompiledPSD` | one LMI block: symmetric matrix of `CompiledExpr` entries |
| `assemble(M, η) → SpMat` | numeric `A(η)` (sparse axpy) |
| `dmat(M, r, η) → SpMat` | exact `∂A/∂η_r` |
| `d2mat(M, r, s) → SpMat` | exact `∂²A/∂η_r∂η_s` (constant) |
| `trprod(G, S)` | fast `tr(G·S)` for dense `G`, sparse `S` |
| `is_param_dependent(M)` | skip-guard for constant constraints |

**Cache `CompiledPEP` objects** — compile once per (method, N) and reuse
across all design iterations, policies, and starts. Compilation cost at
OGD N = 40 is ~4 s; every later assembly is milliseconds.

Because all entries are polynomials of degree ≤ 2, central finite differences
of `assemble` reproduce `dmat`/`d2mat` **exactly** (up to roundoff) — this
identity is the backbone of the validation suite.

## 7. Solving

`src/solve.jl`.

### Backends

```julia
MosekBackend(; tol = 1e-8, feas_tol = 1e-9, threads = 1, verbose = false)
GenericBackend(factory; attributes = Pair{String,Any}[], verbose = false)
```

- `MosekBackend` is the default. `threads = 1` pins Mosek so that **outer**
  Julia threading (over horizons / policies / starts) scales without
  over-subscription. Raise `threads` only for single large solves.
- `GenericBackend(Clarabel.Optimizer)` (license-free) or any JuMP SDP solver;
  this is also the hook for future GPU backends (e.g. Loraine.jl).

### Solving

```julia
sol = solve_pep(cp::CompiledPEP, η; backend = MosekBackend(), warn = true)
```

`PEPSolution` fields:

| Field | Meaning |
|---|---|
| `obj` | worst-case value `W(η)` |
| `G`, `F` | primal Gram matrix and function values |
| `duals` | constraint multipliers, **sign-normalized**: λ ≥ 0 for `:le`, free for `:eq` |
| `psd_duals` | one KKT matrix Λ_b ⪰ 0 per LMI block (aligned with `CompiledPEP.psd`) |
| `obj_duals` | epigraph weights γ_k (`[1.0]` for single objectives; Σγ = 1 for max-min) |
| `status` | `MOI.OPTIMAL`, `MOI.SLOW_PROGRESS`, … |

> **Dual sign convention.** JuMP/MOI duals of *maximization* problems are the
> negatives of classic KKT multipliers for scalar rows; `solve_pep` negates
> them so the envelope formula in §2 holds verbatim. PSD-cone duals come out
> as the KKT Λ ⪰ 0 directly (no negation). Both conventions are
> finite-difference-validated (`dev/parity_ogd.jl`,
> `dev/test_frontend_classes.jl`). If you bypass `solve_pep`, handle this
> yourself.

## 8. Sensitivities

`src/sensitivity.jl`.

```julia
g, H = grad_hess_eta(cp, η, sol; hess = true)
gω, Hω = pullback(g, H, J, Hpol)
```

- `g[r] = ∂W/∂η_r` — exact wherever `W` is differentiable; at kinks it is a
  certificate-dependent subgradient (see §17).
- `H` — frozen-certificate curvature `H_cert` (ω–ω NSDP Lagrangian block),
  used as the second-order model by `design_som` and `design_ssdp`.
- `pullback` applies the chain rule through a policy: `J` is the p×κ
  Jacobian, `Hpol` a vector of κ×κ per-coefficient Hessians or `nothing`
  (linear policy).

## 9. Policies

`src/policies.jl`. A policy maps `(ω, N) → η` and reports its derivatives:

```julia
η, J, Hpol = evaluate_policy(pol, ω, N)     # Hpol === nothing ⇔ linear in ω
κ = nparams(pol, N)
policy_label(pol)                            # display name
```

### Catalog (scalar schedules, `η_n = f(ω, n, N)`)

| Constructor | κ | Formula | Horizon-consistent |
|---|---|---|---|
| `IdentityPolicy()` | N | η = ω (unrestricted / Kamri baseline) | — |
| `ConstantPolicy()` | 1 | η_n = ω₁ | yes |
| `PowerLawPolicy()` | 3 | a/(n^b + c) — closed-form derivatives | yes |
| `sum_of_exp_policy(k=2)` | 2k+1 | Σᵢ aᵢe^{−bᵢn} + c | yes |
| `log_poly_policy(d=2; δ=1.0)` | d+1 | exp(Σⱼ cⱼ log(n+δ)ʲ) | yes |
| `dct_policy(K=3)` | K+1 | Σₖ cₖ cos(πk(n−1)/(N−1)) | **no** |
| `cosine_decay_policy()` | 3 | a(1+cos(πn/N))ᵖ + c | **no** |
| `rational_policy()` | 4 | (a₀+a₁n)/(1+b₁n+b₂n²) | yes |
| `warped_chebyshev_policy(K=3)` | K+1 | Σₖ aₖTₖ(2log(n+1)/log(N+1)−1) | **no** |
| `piecewise_exp_policy(; n_split)` | 4 | two exp regimes split at n_split (default N÷2) | yes |
| `FunctionPolicy(f, κ; name)` | κ | any `f(ω, n, N)`; ForwardDiff derivatives | — |

### Combinators

| Constructor | Use |
|---|---|
| `ProductPolicy(p1, p2)` | stack two coefficient sequences, `ω = [ω₁; ω₂]` — e.g. ITEM β ⊗ δ |
| `MappedPolicy(base, dof, index)` | scatter a base schedule into a larger flat vector (zeros elsewhere) |

### IGDM structured policies (defined in `methods/igdm.jl`)

| Constructor | κ | Structure |
|---|---|---|
| `igdm_diagonal_policy(base)` | κ(base) | memoryless: β_{n,n} = base(ω)[n] |
| `IGDMKMemoryPolicy(K)` | Σₙ min(n, K+1) | free coefficients on the last K+1 lags |
| `IGDMStationaryPolicy(K)` | K+1 | shift-invariant β_{n,n−ℓ} = ω̄_ℓ |
| `IGDMLagPolicy(K, base)` | (K+1)·κ(base) | one independent schedule per lag |

## 10. Built-in methods

### OGD — `methods/ogd.jl`

```julia
m, α = ogd_pep(N, L, D);  cp = compile_ogd(N, L, D)
```

Regret PEP for Online Gradient Descent (N convex L-Lipschitz losses on a
diameter-D set, x* = 0, projections via indicator subgradients Ψ).
η = α ∈ R^N (α_N is unused by the recursion; its gradient is 0). Gram
dimension 3N+1; objective `max Σₙ f_n − f*_n`. Substitution-style encoding —
all constraint matrices affine or quadratic in α.

### ITEM — `methods/item.jl`

```julia
m, θ = item_pep(N, L, μ, D);  cp = compile_item(N, L, μ, D)
β⋆, δ⋆, Wfac = item_optimal_params(N, L, μ)   # reference; W ⪅ Wfac·D²
```

Information-Theoretic Exact Method over 𝓕_{μ,L}; residual-style encoding
(β, δ enter quadratically). θ = [β₁..β_N, δ₁..δ_N]. Pinning: z₁ = y₀ = 0 and
g_* = 0 are **eliminated from the basis** (dim 3N+2), which markedly improves
Mosek conditioning.

> The compact two-sequence rewrite omits the paper's δ₀ half-step: code slot n
> carries paper β_n together with paper δ_{n−1}, and `Wfac = 1/(1+qA_N)` is a
> near-tight *reference*, not the exact optimum of this parameterization
> (deviation < 0.2 % at small N, growing slowly as values decay).

### IGDM — `methods/igdm.jl`

```julia
m, β = igdm_pep(N, L, ε; D = 1.0, objective = :last_grad)   # or :min_grad
cp = compile_igdm(N, L, ε; D, objective)
```

Inexact Gradient Descent with Memory: `x_{n+1} = x_n − Σ_{k≤n} β_{n,k} d_k`,
`‖d_n − g_n‖ ≤ ε‖g_n‖`, f L-smooth convex, `f(x₁) − f(x_*) ≤ D`.
Coefficients are the flattened lower triangle, `η[igdm_flat_index(n,k)] =
β_{n,k}`, `p = igdm_dof(N) = N(N+1)/2`. Objectives: `:last_grad` maximizes
‖g_{N+1}‖² (legacy convention), `:min_grad` the epigraph min over
‖g_k‖², k = 2..N+1. Helpers: `igdm_diag_indices(N)`, `igdm_hmem(ε)` (the
memoryless reference step `h(ε)/L`).

## 11. Design optimizers

All optimizers work on any `AbstractDesignObjective` — single-horizon
`DesignProblem` or multi-horizon `HRDPObjective`/`SampledHRDP` — through
`eval_all(obj, ω; hess) → (f, g, H, nsolves)`.

### DesignProblem

```julia
dp = DesignProblem(cp, policy, N; backend = MosekBackend())
W, sol, η = pep_value(dp, ω)                        # one SDP solve
gω, Hω = pep_grad_hess(dp, ω, η, sol; hess = true)  # free (contractions)
```

### First-order — `design_fom`

```julia
tr = design_fom(dp, ω0; iters, steps, method = :SD,
                adam_beta1 = 0.9, adam_beta2 = 0.999, adam_eps = 1e-8)
```

- `steps`: vector or function `t → Float64`. `:SD` = normalized steepest
  descent (`0.1/√t` is a good default); `:Adam` uses `steps` as the learning
  rate (constant `1e-3`–`1e-2` typical).
- Cost: exactly 1 SDP solve per iteration (+1 for the initial point).

### Second-order — `design_som`

```julia
tr = design_som(dp, ω0; iters, linesearch = :ArmijoZG,
                sigma_met = -1.0, theta_met = 2.0, M_met = 10)
```

Damped Newton on `H_cert` with LM-regularized Cholesky directions and either
the Aminifard–Grapiglia non-monotone Armijo (`:ArmijoZG`; `sigma_met = 0`
recovers plain Armijo, `< 0` uses |f(ω₀)|) or Strong Wolfe (`:Wolfe`,
bracketing + zoom). Line searches cost extra SDP solves (typically 2–5 per
iteration); a per-ω evaluation cache avoids duplicates. Degenerate points
(solver failures) return +∞ and are backtracked over, never crash.

### Traces

`DesignTrace`: `ω_hist`, `values` (aligned), `solves_hist` (cumulative SDP
solves at each point — **the fair budget axis**), `nsolves`.
`best_point(tr) → (ω_best, W_best)`.

### SSDP — `design_ssdp` (paper Alg. 2)

```julia
ts = design_ssdp(dp, ω0; iters = 30, δ = 1e-6, ρ = 10.0, σ = 1e-4,
                 tol = 1e-6, hessian_mode = :block, λreg = 1e-3)
```

Joint sequential-SDP descent on ϑ = (λ, ω) of the NSDP

```
min  c0_obj − Σ_c λ_c c0_c   s.t.  Σ_c λ_c A_c(η(ω)) − C(η(ω)) ⪰ 0,
                                    b_obj − Σ_c λ_c f_c = 0,  λ_le ≥ 0.
```

Single-piece objectives only, and **no PSD (LMI) blocks** (the LMI-dual NSDP
is not implemented; a clear error is raised — use FOM/SOM instead).

- Initialization: one PEP solve at ω0 gives a feasible ϑ0 and multipliers
  `(G0, y0) = (Gram, F)`.
- Tangent subproblem: convexified quadratic model + linearized LMI, solved
  as one SDP per iteration; multipliers updated by convex combination;
  ℓ1 exact-penalty merit with Han–Powell backtracking.
- `hessian_mode = :block` (default): mild λ-regularization `λreg`, exact
  eigenvalue-shifted ω–ω block, λ–ω coupling kept only in the LMI. Converges
  fast (OGD N=3 constant policy: exact optimum in ~6 tangent solves).
  `:full` is the paper-verbatim full-space shift — provably ⪰ δI but
  conservative and slow; use it for the diagnostics experiment.
- Restriction: single-piece objectives only (no `:min_grad` epigraph).

`SSDPTrace`: `ω_hist`, `obj_hist` (NSDP dual objective), `kkt_hist`,
`merit_hist`, `ζ_hist` (convexification shifts), `γ_hist` (accepted steps),
`pred_hist`/`act_hist` (predicted vs actual merit reduction), `λ`,
`ntangent`, `W_final` (verification PEP solve at the final ω).

## 12. Multi-horizon robust design (HRDP)

`src/design/hrdp.jl`.

```julia
Wstar = compute_wstar(compile_fn, H; policy = IdentityPolicy(),
                      ω0_fn, iters = 50, method = :both)   # threaded over N
ho = HRDPObjective(compile_fn, policy, H, Wstar; backend = MosekBackend())
tr = design_som(ho, ω0; iters = 30)        # or design_fom
```

- `σ(𝓗, ω) = mean(W(η^{(N)}(ω), N)/W*_N)`; per-horizon solves run under
  `Threads.@threads` (start Julia with `--threads=…`).
- `compute_wstar`: per-horizon optimal values; `ω0_fn(N)` returns one start
  or a vector of starts; `method ∈ (:som, :fom, :both)` keeps the best value
  over starts × methods (multistart strongly recommended — single-start SOM
  can stall on kinks).

Metrics:

| Function | Meaning |
|---|---|
| `snw(ho, ω)` | Dict N ⇒ W̄_N(ω) = W_N(ω)/W*_N (scheme-wise normalized worst case) |
| `wgc(ho, ω)` | Worst Generalization Compromise `max_N W̄_N(ω)` |
| `generalization_ratio(ho_full, ω_sub, ω_full)` | GR = σ(𝓗_full, ω_sub)/σ(𝓗_full, ω_full) ≥ 1 |

Prefix-exact, tail-sampled estimator (paper Alg. 3):

```julia
sh = SampledHRDP(ho, r, m; rng, weights = nothing)   # exact for N ≤ r, m tail samples
tr = design_fom(sh, ω0; iters, steps)                # unbiased ⇒ use SGD-style methods
```

Evaluations are stochastic; pair with `design_fom` (line-search-free), not
`design_som`.

## 13. Adding a new method

Create `PEPDesign/src/methods/<name>.jl` (~40–80 lines):

```julia
function myalg_pep(N::Int, L::Real)
    m = PEPModel()
    h = coeffs!(m, N)                              # symbolic coefficients

    # ── High-level path (recommended): class objects + oracles/steps ────────
    f = SmoothConvexFunction(m; L = L)
    xstar, _, fstar = stationary_point!(f)
    x = point!(m; name = "x1")
    add_le!(m, sqnorm(x - xstar) - 1.0; name = "init")
    for n in 1:N
        gx, _ = oracle!(f, x)                      # or proximal_step!, …
        x = x - h[n] * gx
    end
    _, fN = oracle!(f, x)
    objective_max!(m, fN - fstar)                  # compile() emits the class
    return m, h                                    # interpolation constraints
end
compile_myalg(N, L) = compile(myalg_pep(N, L)[1])
```

The low-level DSL remains available (and is what the frontend generates):

```julia
    # (a) substitution style: iterates as affine combinations
    x = Vector{PointExpr}(undef, N); x[1] = x1
    for n in 2:N
        x[n] = x[n-1] - h[n-1] * g[n-1]
    end
    # (b) residual style (needed when coefficients multiply each other):
    #   xn1 = point!(m); add_eq!(m, sqnorm(xn1 - (x[n] - h[n]*g[n])))

    add_interpolation_fmuL!(m, [x; xstar], [g; PointExpr()], [f; fstar]; L, μ = 0)
    add_le!(m, sqnorm(x[1]) - 1.0; name = "init")
    add_eq!(m, qexpr(fstar); name = "fstar0")
    objective_max!(m, f[N] - fstar)
```

Then: include + export in `src/PEPDesign.jl`; write a `dev/` check script
(value sanity + the exact `assemble`/`dmat`/`d2mat` FD identity + a gradient
FD check away from kinks); optionally register in `policy_suite`
(`experiments/common.jl`) **by appending** to make it available to the
experiments and the cluster pipeline.

Design guidance:

- Prefer the frontend classes/steps; drop to the DSL for pinned oracles
  (`add_oracle_point!`) or bespoke constraint sets.
- Prefer substitution style when coefficients enter linearly (smaller Gram);
  use residual style for nested products (ITEM-like). The algebra throws
  `"point coefficients must remain affine in η…"` when you must switch.
- Pin translation/optimality degrees of freedom with zero `PointExpr()`s
  (the built-in methods pin `x_* = 0` this way and use `stationary_point!`
  or `add_oracle_point!` for the optimum).
- Never hardcode problem constants; take `N, L, D, …` as arguments.
- PEPs with LMI blocks work with FOM/SOM/HRDP but not `design_ssdp` yet.

## 14. Adding a new policy

One-liner for scalar schedules:

```julia
my_policy(κ) = FunctionPolicy((ω, n, N) -> ..., κ; name = "my_policy")
```

Closed-form derivatives: subtype `AbstractPolicy`, implement
`nparams(pol, N)` and `evaluate_policy(pol, ω, N) → (η, J, Hpol)` with
`Hpol = nothing` when linear. Validate with the FD harness in
`dev/test_policies.jl`. Compose with `ProductPolicy` / `MappedPolicy` /
`IGDMLagPolicy` as needed. Register in `policy_suite` by **appending only**
(cluster task ids index that list).

## 15. Validation suite

```
julia --project=. --threads=4 PEPDesign/dev/run_all_tests.jl
```

| Script | Checks |
|---|---|
| `test_policies.jl` | FD on Jacobians/Hessians of every policy; IGDM structured-policy sparsity |
| `test_frontend_classes.jl` | one known tight rate per class family (GD on 𝓕_{0,L}/𝓕_{μ,L}/quadratics, PPA, resolvent, exact line search); FD of the LMI-dual envelope gradient; combination & step smoke tests |
| `parity_ogd.jl` | value parity vs frozen legacy (1e-9); exact matrix FD identities; ∇W vs FD |
| `parity_item.jl` | value parity; analytic ITEM reference band; synthetic-certificate contraction identity |
| `parity_igdm.jl` | value parity; contraction identity; max-min objective (Σγ = 1) |
| `smoke_design.jl` | FOM+SOM recover the OGD conjecture (ω* = 1/√N, W* = √N) |
| `smoke_ssdp.jl` | SSDP reaches the exact optimum; dual = primal |
| `smoke_hrdp.jl` | W*_N ≈ √N; σ, WGC, GR sanity; sampled-estimator unbiasedness |

The legacy reference implementation lives frozen in
`dev/legacy_reference/` (the two legacy derivative bugs — OGD ∇A5 scaling
and a spurious HessA7 — are documented in `parity_ogd.jl`; the new gradients
are the FD-validated ones).

## 16. Performance notes

Measured on a laptop (Julia 1.12, Mosek 11, 1 thread per solve):

| Quantity | Value |
|---|---|
| OGD N=40: compile / solve / grad+H_cert | ~4 s / ~19 s / 0.05 s |
| OGD N=30 solve | ~4.5 s |
| IGDM N=30 solve (465 coefficients) | ~3 s |
| Per design iteration vs legacy toolbox (N=10) | **~4× faster** (0.07 s vs 0.29 s; the frontend OGD carries a few redundant indicator constraints vs the retired bespoke encoding, which measured 7.6×) |

Rules of thumb:

- Compile once, cache the `CompiledPEP` per (method, N).
- The SDP solve dominates everything; sensitivity contractions are free.
- Parallelize **outside** Mosek: `--threads=K` + `MosekBackend(threads=1)`;
  HRDP and `compute_wstar` already thread over horizons.
- Budget accounting: use `DesignTrace.solves_hist` — never wall time — as the
  comparison axis across policies/optimizers.

## 17. Pitfalls and troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: point coefficients must remain affine in η` | Coefficient product in an iterate (ITEM-like). Declare the intermediate iterate with `point!` and add `add_eq!(m, sqnorm(residual))`. |
| Gradients disagree with finite differences | (a) At a kink (nonunique certificate — common for ITEM/IGDM): expected; the gradient is a valid subgradient. Test with the synthetic-certificate contraction identity instead. (b) Custom solver path: check the dual sign normalization (§7). |
| `SLOW_PROGRESS` status | Degenerate SDP (residual equalities). Usually still ~1e-6-accurate. Improve by pinning points via zero `PointExpr()`s; tighten `feas_tol`. |
| SOM stalls above the optimum | Kinked landscape + frozen-certificate model. Use multistart (`compute_wstar(...; method = :both)`), or a FOM polish. |
| SSDP steps rejected immediately | Merit penalty ρ too small relative to multipliers, or `:full` mode over-regularizing. Increase `ρ`, or use `hessian_mode = :block`. |
| `Pkg.resolve` fails on the root env | Known registry conflict; the pinned Manifest works. Don't resolve/up. |
| Everything slow under `@threads` | Mosek over-subscription — ensure `MosekBackend(threads = 1)` (default). |
| `W_final` from SSDP ≠ dual objective | Gap = remaining KKT residual; increase `iters` or loosen `tol` expectations. Check `kkt_hist`. |
