# Frontend validation: known tight rates per class, LMI-block gradient FD,
# combination/step smoke tests.
# Run: julia --project=. PEPDesign/dev/test_frontend_classes.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using LinearAlgebra, Printf

const TOL = 2e-6
pass(msg) = println("  ✓ ", msg)

# ── 1. GD on 𝓕_{0,L}: W = LD²/(4N+2) at γ = 1/L ──────────────────────────────
let N = 3, L = 1.0
    m = PEPModel()
    γ = coeffs!(m, N)
    f = SmoothConvexFunction(m; L = L)
    xstar, _, fstar = stationary_point!(f)
    x = point!(m)
    add_le!(m, sqnorm(x - xstar) - 1.0)
    for k in 1:N
        gx, _ = oracle!(f, x)
        x = x - γ[k] * gx
    end
    _, fN = oracle!(f, x)
    objective_max!(m, fN - fstar)
    cp = compile(m)
    W = solve_pep(cp, fill(1 / L, N)).obj
    @assert abs(W - L / (4N + 2)) < TOL "GD smooth convex: $W vs $(L / (4N + 2))"
    pass(@sprintf("GD 𝓕_{0,L}: W = %.8f = LD²/(4N+2)", W))
end

# ── 2. GD on 𝓕_{μ,L}: ‖x_N − x*‖² = max((1−γμ)², (1−γL)²)^N D² ───────────────
let N = 2, L = 1.0, μ = 0.1, γv = 1.0
    m = PEPModel()
    γ = coeffs!(m, N)
    f = SmoothStronglyConvexFunction(m; μ = μ, L = L)
    xstar, _, _ = stationary_point!(f)
    x = point!(m)
    add_le!(m, sqnorm(x - xstar) - 1.0)
    for k in 1:N
        gx, _ = oracle!(f, x)
        x = x - γ[k] * gx
    end
    objective_max!(m, sqnorm(x - xstar))
    cp = compile(m)
    W = solve_pep(cp, fill(γv, N)).obj
    ref = max((1 - γv * μ)^2, (1 - γv * L)^2)^N
    @assert abs(W - ref) < TOL "GD 𝓕_{μ,L}: $W vs $ref"
    pass(@sprintf("GD 𝓕_{μ,L}: W = %.8f = max((1−γμ)²,(1−γL)²)^N", W))
end

# ── 3. Proximal point on convex CCP: W = D²/(4Nγ) ────────────────────────────
let N = 2, γv = 1.0
    m = PEPModel()
    γ = coeffs!(m, N)
    f = ConvexFunction(m)
    xstar, _, fstar = stationary_point!(f)
    x = point!(m)
    add_le!(m, sqnorm(x - xstar) - 1.0)
    local fx
    for k in 1:N
        x, _, fx = proximal_step!(x, f, γ[k])
    end
    objective_max!(m, fx - fstar)
    cp = compile(m)
    W = solve_pep(cp, fill(γv, N)).obj
    @assert abs(W - 1 / (4 * N * γv)) < TOL "PPA: $W vs $(1 / (4N * γv))"
    pass(@sprintf("PPA convex: W = %.8f = D²/(4Nγ)", W))
end

# ── 4. Resolvent of μ-strongly monotone: ‖Jx − Jy‖² = ‖x − y‖²/(1+γμ)² ───────
let μ = 0.5, γv = 1.3
    m = PEPModel()
    γ = coeffs!(m, 1)
    T = StronglyMonotoneOperator(m; μ = μ)
    w1 = point!(m); w2 = point!(m)
    add_le!(m, sqnorm(w1 - w2) - 1.0)
    x1, _, _ = proximal_step!(w1, T, γ[1])     # x = w − γ·Tx (resolvent)
    x2, _, _ = proximal_step!(w2, T, γ[1])
    objective_max!(m, sqnorm(x1 - x2))
    cp = compile(m)
    W = solve_pep(cp, [γv]).obj
    ref = 1 / (1 + γv * μ)^2
    @assert abs(W - ref) < TOL "resolvent: $W vs $ref"
    pass(@sprintf("resolvent of μ-strongly monotone: W = %.8f = 1/(1+γμ)²", W))
end

# ── 5. Steepest descent + exact line search on 𝓕_{μ,L}: ((κ−1)/(κ+1))^{2N} ──
let N = 1, L = 1.0, μ = 0.1
    m = PEPModel()
    f = SmoothStronglyConvexFunction(m; μ = μ, L = L)
    _, _, fstar = stationary_point!(f)
    x = point!(m)
    gx, fx = oracle!(f, x)
    add_le!(m, fx - fstar - 1.0)
    local fN = fx
    for _ in 1:N
        x, gx, fN = exact_linesearch_step!(x, f, [gx])
    end
    objective_max!(m, fN - fstar)
    cp = compile(m)
    W = solve_pep(cp, Float64[]).obj
    ref = ((L - μ) / (L + μ))^(2N)
    @assert abs(W - ref) < TOL "exact LS: $W vs $ref"
    pass(@sprintf("exact line search on 𝓕_{μ,L}: W = %.8f = ((κ−1)/(κ+1))^{2N}", W))
end

# ── 6. Quadratic class [LMI]: GD contraction + FD-validated gradient ─────────
let N = 2, L = 1.0, μ = 0.1, γv = [0.8, 0.9]
    build = function ()
        m = PEPModel()
        γ = coeffs!(m, N)
        f = SmoothStronglyConvexQuadraticFunction(m; μ = μ, L = L)
        c = f.core
        xstar = c.xs[c.stationary[1]]
        x = point!(m)
        add_le!(m, sqnorm(x - xstar) - 1.0)
        for k in 1:N
            gx, _ = oracle!(f, x)
            x = x - γ[k] * gx
        end
        objective_max!(m, sqnorm(x - xstar))
        return compile(m)
    end
    cp = build()
    @assert !isempty(cp.psd) "quadratic class must add an LMI block"
    sol = solve_pep(cp, γv)
    ref = prod(max((1 - g * μ)^2, (1 - g * L)^2) for g in γv)
    @assert abs(sol.obj - ref) < 1e-5 "GD quadratic: $(sol.obj) vs $ref"
    pass(@sprintf("GD on quadratics [LMI]: W = %.8f = Π max((1−γμ)²,(1−γL)²)", sol.obj))

    g, _ = grad_hess_eta(cp, γv, sol)
    h = 1e-5
    for r in 1:N
        γp = copy(γv); γp[r] += h
        γm = copy(γv); γm[r] -= h
        fd = (solve_pep(cp, γp).obj - solve_pep(cp, γm).obj) / (2h)
        @assert abs(g[r] - fd) < 5e-4 "LMI ∇W[$r]: $(g[r]) vs FD $fd"
    end
    pass("LMI-block envelope gradient matches finite differences")
end

# ── 7. Linear combination + proximal gradient smoke ──────────────────────────
let N = 2, L = 1.0
    m = PEPModel()
    γ = coeffs!(m, N)
    f1 = SmoothConvexFunction(m; L = L)
    f2 = ConvexFunction(m)
    F = f1 + 0.5 * f2
    xstar, _, Fstar = stationary_point!(F)     # registers residual triple on leaves
    x = point!(m)
    add_le!(m, sqnorm(x - xstar) - 1.0)
    local Fx
    for k in 1:N
        g1, _ = oracle!(f1, x)                 # forward on f1
        x, _, _ = proximal_step!(x - γ[k] * g1, 0.5 * f2, 1.0)   # backward on f2
        _, Fx = oracle!(F, x)
    end
    objective_max!(m, Fx - Fstar)
    cp = compile(m)
    sol = solve_pep(cp, fill(1 / L, N))
    @assert isfinite(sol.obj) && sol.obj > 0
    pass(@sprintf("proximal gradient on f1 + 0.5·f2 (combination): W = %.6f", sol.obj))
end

# ── 8. Cheap operator classes: resolvent bound retained ──────────────────────
let μ = 0.5, L = 2.0, γv = 1.0
    m = PEPModel()
    γ = coeffs!(m, 1)
    T = LipschitzStronglyMonotoneOperatorCheap(m; μ = μ, L = L)
    w1 = point!(m); w2 = point!(m)
    add_le!(m, sqnorm(w1 - w2) - 1.0)
    x1, _, _ = proximal_step!(w1, T, γ[1])
    x2, _, _ = proximal_step!(w2, T, γ[1])
    objective_max!(m, sqnorm(x1 - x2))
    W = solve_pep(compile(m), [γv]).obj
    @assert 0 < W <= 1 / (1 + γv * μ)^2 + TOL "SM+Lip cheap: W = $W"
    pass(@sprintf("Lipschitz+strongly monotone (cheap): W = %.8f ≤ 1/(1+γμ)²", W))
end

# ── 9. Indicator + LMO (Frank–Wolfe style) and ε-subgradient smoke ───────────
let
    m = PEPModel()
    ind = ConvexIndicatorFunction(m; D = 1.0)
    f = SmoothConvexFunction(m; L = 1.0)
    x0 = point!(m)
    oracle!(ind, x0)
    g0, _ = oracle!(f, x0)
    y, _, _ = linear_optimization_step!(g0, ind)
    xstar, _, fstar = stationary_point!(f)
    oracle!(ind, xstar)
    _, f1 = oracle!(f, x0 + 0.5 * (y - x0))
    objective_max!(m, f1 - fstar)
    W = solve_pep(compile(m), Float64[]).obj
    @assert isfinite(W)
    pass(@sprintf("indicator + linear_optimization_step!: W = %.6f", W))
end

let γv = 1.0
    m = PEPModel()
    f = ConvexLipschitzFunction(m; M = 1.0)    # bounded subgradients ⇒ finite W
    xstar, _, fstar = stationary_point!(f)
    x0 = point!(m)
    add_le!(m, sqnorm(x0 - xstar) - 1.0)
    x1, _, f0, ε = epsilon_subgradient_step!(x0, f, γv)
    add_le!(m, ε - 0.1)                        # ε-subgradient accuracy budget
    objective_max!(m, f0 - fstar)
    W = solve_pep(compile(m), Float64[]).obj
    @assert isfinite(W)
    pass(@sprintf("epsilon_subgradient_step!: W = %.6f", W))
end

println("test_frontend_classes: ALL PASSED")
