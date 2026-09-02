# Gradient descent on an L-smooth convex function, written with the
# high-level frontend (PEPit.jl-style classes + primitive steps).
#
#   x_{k+1} = x_k − γ_k ∇f(x_k),   f ∈ 𝓕_{0,L},  ‖x_1 − x*‖² ≤ D²
#
# Known tight rate at γ = 1/L:  f(x_{N+1}) − f* = L D² / (4N + 2).
# Run: julia --project=. PEPDesign/examples/gradient_descent.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Printf

const N = 5
const L = 1.0
const D = 1.0

"""GD PEP over 𝓕_{0,L} with symbolic step sizes γ ∈ R^N."""
function gd_pep(N, L, D)
    m = PEPModel()
    γ = coeffs!(m, N)
    f = SmoothConvexFunction(m; L = L)

    xstar, _, fstar = stationary_point!(f)       # (x*, ∇f(x*) = 0, f*)
    x = point!(m; name = "x1")
    add_le!(m, sqnorm(x - xstar) - D^2; name = "init")
    for k in 1:N
        gx, _ = oracle!(f, x)                    # exact gradient step
        x = x - γ[k] * gx
    end
    _, fN = oracle!(f, x)                        # value at the last iterate

    objective_max!(m, fN - fstar)
    return m, γ
end

cp = compile(gd_pep(N, L, D)[1])

# 1. Worst case of the classical tuning γ = 1/L: matches L·D²/(4N+2).
sol = solve_pep(cp, fill(1 / L, N))
@printf("W(1/L)   = %.6f   (theory LD²/(4N+2) = %.6f)\n",
        sol.obj, L * D^2 / (4N + 2))

# 2. Exact gradient/curvature w.r.t. the step sizes — free trace contractions.
g, H = grad_hess_eta(cp, fill(1 / L, N), sol)
@printf("‖∂W/∂γ‖  = %.3e\n", sqrt(sum(abs2, g)))

# 3. Design: constant policy γ_k ≡ ω over the horizon (damped Newton).
dp = DesignProblem(cp, ConstantPolicy(), N)
tr = design_som(dp, [0.5 / L]; iters = 20)
ω̂, Ŵ = best_point(tr)
@printf("SOM: ω̂ = %.4f, W(ω̂) = %.6f  (%d SDP solves)\n",
        ω̂[1], Ŵ, tr.solves_hist[end])
