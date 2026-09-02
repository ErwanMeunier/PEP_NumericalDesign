# Diagnose ITEM discrepancies: coefficient indexing conventions, old-vs-new
# value at identical θ, and gradient/FD consistency.
# Run: julia --project=. PEPDesign/dev/diag_item.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using JuMP, Mosek, MosekTools, LinearAlgebra, SparseArrays, Random, Printf

include(joinpath(@__DIR__, "legacy_reference", "PEPTypes.jl"))
include(joinpath(@__DIR__, "legacy_reference", "MatrixComputations_ITEM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "PEPSolvers_ITEM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "ProblemGenerators_ITEM.jl"))

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const MU = 0.5
const D = 1.0
const N = 3

q = MU / L
A = zeros(N + 3)                    # A[k] = A_{k-1}
for k in 2:(N + 3)
    Ap = A[k - 1]
    A[k] = ((1 + q) * Ap + 2 * (1 + sqrt((1 + Ap) * (1 + q * Ap)))) / (1 - q)^2
end
βk(k) = A[k + 1] / ((1 - q) * A[k + 2])                    # paper β_k, k≥0
δk(k) = ((1 - q)^2 * A[k + 2] - (1 + q) * A[k + 1]) /
        (2 * (1 + q + q * A[k + 1]))                       # paper δ_k, k≥0

variants = Dict(
    "beta1:N delta1:N"     => ([βk(n) for n in 1:N], [δk(n) for n in 1:N]),
    "beta1:N delta0:N-1"   => ([βk(n) for n in 1:N], [δk(n - 1) for n in 1:N]),
    "beta0:N-1 delta0:N-1" => ([βk(n - 1) for n in 1:N], [δk(n - 1) for n in 1:N]),
    "beta0:N-1 delta1:N"   => ([βk(n - 1) for n in 1:N], [δk(n) for n in 1:N]),
)
@printf("Analytic factors: 1/(1+qA_N) = %.8f   1/(1+qA_{N+1}) = %.8f\n\n",
        1 / (1 + q * A[N + 1]), 1 / (1 + q * A[N + 2]))

cp = compile_item(N, L, MU, D)
for (label, (βs, δs)) in sort(collect(variants); by = first)
    θ = [βs; δs]
    W_new = solve_pep(cp, θ).obj
    W_old = sdp_pep_ITEM(generate_ITEM(N, L, MU, D, θ)).obj_value
    @printf("%-22s  W_new = %.8f   W_old = %.8f\n", label, W_new, W_old)
end

# Gradient vs FD at a perturbed point, with status reporting
Random.seed!(7)
θ = [[βk(n) for n in 1:N]; [δk(n) for n in 1:N]] .* (1.0 .+ 0.15 .* (rand(2N) .- 0.5))
sol = solve_pep(cp, θ)
println("\nstatus at θ: ", sol.status)
g, _ = grad_hess_eta(cp, θ, sol)
for h in (1e-3, 1e-4, 1e-5)
    fd = zeros(2N)
    ok = true
    for r in 1:(2N)
        e = zeros(2N); e[r] = h
        sp = solve_pep(cp, θ .+ e; warn = false)
        sm = solve_pep(cp, θ .- e; warn = false)
        (sp.status == MOI.OPTIMAL && sm.status == MOI.OPTIMAL) || (ok = false)
        fd[r] = (sp.obj - sm.obj) / (2h)
    end
    @printf("h=%.0e  max|g-fd| = %.3e  (all OPTIMAL: %s)\n", h, norm(g - fd, Inf), ok)
end
println("g  = ", round.(g; digits = 5))

# Old gradient at same θ for reference
prob_old = generate_ITEM(N, L, MU, D, θ)
sol_old = sdp_pep_ITEM(prob_old)
g_old, _ = diff_w_ITEM(θ, sol_old, prob_old)
println("g_old = ", round.(g_old; digits = 5))
