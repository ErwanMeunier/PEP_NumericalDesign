# Validate the extracted ITEM dual certificate: PSD dual matrix, F-balance,
# complementarity — and probe one-sided directional derivatives for kinks.
# Run: julia --project=. PEPDesign/dev/diag_item_duals.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const MU = 0.5
const D = 1.0
const N = 3

βs, δs, _ = item_optimal_params(N, L, MU)
Random.seed!(7)
θ = [βs; δs] .* (1.0 .+ 0.15 .* (rand(2N) .- 0.5))

cp = compile_item(N, L, MU, D)
sol = solve_pep(cp, θ)
println("status: ", sol.status, "   W = ", sol.obj)

# Dual certificate: S = Σ_c λ_c A_c(θ) − C ⪰ 0,  b − Σ_c λ_c f_c = 0
S = -assemble(cp.obj[1].M, θ)
fbal = copy(cp.obj[1].f)
for (c, con) in enumerate(cp.cons)
    S .+= sol.duals[c] .* assemble(con.expr.M, θ)
    fbal .-= sol.duals[c] .* con.expr.f
end
eig = eigvals(Symmetric(Matrix(S)))
@printf("eigmin(S) = %.3e   ‖F-balance‖∞ = %.3e\n", minimum(eig), norm(fbal, Inf))
@printf("complementarity tr(SG) = %.3e\n", tr(S * sol.G))

# One-sided directional derivatives vs certificate gradient
g, _ = grad_hess_eta(cp, θ, sol)
Random.seed!(11)
for k in 1:4
    d = randn(2N); d ./= norm(d)
    gd = dot(g, d)
    h = 1e-3
    Wp = solve_pep(cp, θ .+ h .* d; warn = false).obj
    Wm = solve_pep(cp, θ .- h .* d; warn = false).obj
    right = (Wp - sol.obj) / h
    left = (sol.obj - Wm) / h
    @printf("dir %d:  g'd = %+.6f   right FD = %+.6f   left FD = %+.6f   kink gap = %.2e\n",
            k, gd, right, left, abs(right - left))
end
