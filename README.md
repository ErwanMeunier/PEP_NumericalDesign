TOOLBOX UNDER DEVELOPMENT with a mix of AI and Human generated content. 

The automated formulation of PEP problems is directly taken from https://github.com/PerformanceEstimation/PEPit.jl 

References: 
https://arxiv.org/abs/2507.20773
https://www.youtube.com/watch?v=2vp65pXp5Xk

# PEPDesign

Parametric Performance Estimation and step-size policy design — the toolbox
accompanying *"Principled Robust Design of First-Order Methods via
Parameterized Step-Size Policies"* which should be released in September 2026. 

Successor of the legacy `src/` (FOM_SOM) library. Given a first-order method
written as a short algebraic recurrence, PEPDesign **automatically derives**
the parametric PEP-SDP, its exact gradients, frozen-certificate curvature, and
the NSDP Lagrangian Hessian blocks with respect to the step-size coefficients
and policy parameters — with zero runtime overhead.

## Documentation

| Document | Content |
|---|---|
| [docs/manual.md](docs/manual.md) | **Full package manual**: installation, math background, complete API reference (DSL, compilation, solving, sensitivities, policies, methods, FOM/SOM/SSDP/HRDP), extension guides, troubleshooting |
| [docs/experiments.md](docs/experiments.md) | Paper experiments: what each script computes, outputs, reproducibility |
| [docs/cluster.md](docs/cluster.md) | **SLURM / CECI guide**: setup (modules, depot, Mosek license), submission, resources, monitoring, resuming, troubleshooting |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Repository-level architecture and coding rules |

## How it works

Every PEP constraint built from Gram inner products of iterates is **at most
quadratic** in the method coefficients η (updates are either substituted
affinely or encoded as `‖residual‖² = 0`). PEPDesign therefore represents all
scalar quantities in a lightweight degree-2 polynomial algebra
([src/params.jl](src/params.jl)) and extracts, **once per (method, horizon)**,

```
A_c(η) = A0_c + Σ_r η_r·A1_c[r] + Σ_{r≤s} η_r η_s·A2_c[(r,s)]
```

with exact sparse constant tensors ([src/compile.jl](src/compile.jl)). After
that:

- assembly at any η is a sparse axpy;
- ∂A/∂η and ∂²A/∂η² are exact and free;
- the envelope gradient ∂W/∂η = −Σ_c λ_c tr(G ∂A_c/∂η) and the
  frozen-certificate curvature are trace contractions
  ([src/sensitivity.jl](src/sensitivity.jl));
- policy pullback ω ↦ η(ω) uses closed-form or ForwardDiff Jacobians/Hessians
  ([src/policies.jl](src/policies.jl));
- the same tensors give the exact NSDP Lagrangian blocks used by SSDP
  ([src/design/ssdp.jl](src/design/ssdp.jl)).

No computer-algebra system, no runtime AD, no hand-coded derivative matrices.

## Quickstart

```julia
include("PEPDesign/src/PEPDesign.jl"); using .PEPDesign

# 1. Compile a built-in method once per horizon (cache and reuse!)
cp = compile_ogd(30, 1.0, 1.0)              # OGD regret PEP, L = D = 1

# 2. Evaluate the worst case at any step-size vector
sol = solve_pep(cp, fill(1/sqrt(30), 30))   # ≈ √30
g, H = grad_hess_eta(cp, fill(1/sqrt(30), 30), sol)

# 3. Design over a policy family
dp = DesignProblem(cp, ConstantPolicy(), 30)
tr = design_som(dp, [0.2]; iters = 30)      # damped Newton on H_cert
ω̂, Ŵ = best_point(tr)                       # → ω̂ ≈ 1/√30 (the conjecture)

# SSDP on the joint NSDP (paper Alg. 2)
ts = design_ssdp(dp, [0.2]; iters = 30)

# Multi-horizon robust design (HRDP)
Wstar = compute_wstar(N -> compile_ogd(N, 1, 1), 2:8;
                      policy = ConstantPolicy(), ω0_fn = N -> [[1/sqrt(N)]])
ho = HRDPObjective(N -> compile_ogd(N, 1, 1), ConstantPolicy(), 2:8, Wstar)
trH = design_som(ho, [0.3]; iters = 20)
snw(ho, best_point(trH)[1])                 # per-horizon normalized profile
```

## Defining a new method (~15 lines)

PEPDesign ships a PEPit.jl-style high-level frontend: **14 function classes**
(convex, strongly convex, smooth, 𝓕_{μ,L}, Lipschitz, indicator, support,
QG⁺, RSI⁻/EB⁺, quadratics, Łojasiewicz …), **13 operator classes**
(monotone, cocoercive, Lipschitz, nonexpansive, linear/symmetric/skew …) and
**9 primitive steps** (proximal, inexact gradient/proximal, exact line
search, LMO, Bregman, ε-subgradient) — all emitting exact degree-2
constraints into the same compiled-tensor machinery, with **symbolic step
sizes** wherever the algebra allows. Ported from
[PEPit.jl](https://github.com/PerformanceEstimation/PEPit.jl) (MIT).

```julia
function gd_pep(N, L, D)                     # see examples/gradient_descent.jl
    m = PEPModel()
    γ = coeffs!(m, N)                        # symbolic step sizes
    f = SmoothConvexFunction(m; L)
    xstar, _, fstar = stationary_point!(f)
    x = point!(m)
    add_le!(m, sqnorm(x - xstar) - D^2)
    for k in 1:N
        gx, _ = oracle!(f, x)
        x = x - γ[k] * gx                    # or proximal_step!, LMO, …
    end
    objective_max!(m, oracle!(f, x)[2] - fstar)
    return m, γ
end
cp = compile(gd_pep(30, 1.0, 1.0)[1])        # everything else is generic
```

Function combinations (`f1 + 0.5*f2`) decompose oracles onto leaves, so
splitting methods (proximal gradient, Douglas–Rachford …) read naturally.
Quadratic/linear-operator classes add PSD (LMI) blocks — supported by
`solve_pep` and the exact sensitivities (not yet by SSDP).

Point coefficients must stay **affine** in η; if a recurrence multiplies
parameters together (like ITEM), declare the intermediate iterate with
`point!` and add `‖residual‖² = 0` — the toolbox raises a clear error
otherwise.

## Layout

| Path | Role |
|---|---|
| [src/params.jl](src/params.jl) | degree-2 polynomial algebra in η (PAff/PQuad) |
| [src/gram.jl](src/gram.jl) | Gram-space DSL: points, f-values, constraints, PSD blocks, PEPModel |
| [src/classes.jl](src/classes.jl) | 𝓕_{μ,L} interpolation generator |
| [src/frontend/](src/frontend) | high-level layer (from PEPit.jl, MIT): oracles & combinations, 14 function classes, 13 operator classes, 9 primitive steps |
| [src/compile.jl](src/compile.jl) | exact (A0, A1, A2) extraction → CompiledPEP |
| [src/solve.jl](src/solve.jl) | SDP backends (Mosek default, GenericBackend for others) |
| [src/sensitivity.jl](src/sensitivity.jl) | envelope gradient (incl. LMI duals), H_cert, policy pullback |
| [src/policies.jl](src/policies.jl) | policy families (identity, constant, power law, catalog…) |
| [src/design/](src/design) | FOM (SD/Adam), SOM (damped Newton), SSDP, HRDP |
| [src/methods/](src/methods) | OGD, ITEM, IGDM built on the frontend |
| [examples/](examples) | high-level usage examples (gradient descent) |
| [experiments/](experiments) | paper experiments (exp1–exp4) — see [docs/experiments.md](docs/experiments.md) |
| [cluster/](cluster) | 4-phase SLURM pipeline for CECI — see [docs/cluster.md](docs/cluster.md) |
| [docs/](docs) | manual, experiments guide, cluster guide |
| [dev/](dev) | parity vs frozen legacy code, FD checks, smoke tests, benchmarks |

## Validation

```
julia --project=. --threads=4 PEPDesign/dev/run_all_tests.jl
```

- **OGD**: value parity with legacy at 1e-9; gradients FD-validated. Two
  legacy derivative bugs (∇A5 scaling, HessA7 ≠ 0) are documented in
  [dev/parity_ogd.jl](dev/parity_ogd.jl) — the new gradients are correct.
- **ITEM**: parity with legacy; matches the analytic ITEM guarantee. Note:
  the PEP value has genuine kinks (nonunique certificates) — gradient tests
  use exact contraction identities, not finite differences.
- **IGDM**: parity with legacy; epigraph (max-min) objective supported.
- **Frontend classes**: one known tight rate per class family
  (GD → LD²/(4N+2), PPA → D²/(4Nγ), resolvent → 1/(1+γμ)², exact line
  search → ((κ−1)/(κ+1))^{2N}, quadratics via LMI) plus FD validation of the
  LMI-dual envelope gradient — [dev/test_frontend_classes.jl](dev/test_frontend_classes.jl).
- **Performance**: ~4× faster per design iteration than the legacy library
  at N = 10 (compile-once extraction); OGD N = 40 solves in ~19 s.

## Solver notes

- Mosek is pinned to 1 thread (`MosekBackend(threads=1)`) so that outer
  parallelism (`Threads.@threads` over horizons/policies/starts) scales.
- Any JuMP SDP solver can be swapped in via `GenericBackend(Clarabel.Optimizer)`
  (license-free) — also the hook for future GPU backends.
