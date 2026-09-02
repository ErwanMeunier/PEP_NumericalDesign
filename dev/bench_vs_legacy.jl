# Wall-time comparison: legacy FOM_SOM vs PEPDesign on the OGD path.
# Run: julia --project=. PEPDesign/dev/bench_vs_legacy.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using JuMP, Mosek, MosekTools, LinearAlgebra, SparseArrays, Random, Printf

include(joinpath(@__DIR__, "legacy_reference", "PEPTypes.jl"))
include(joinpath(@__DIR__, "legacy_reference", "MatrixComputations_OGD.jl"))
include(joinpath(@__DIR__, "legacy_reference", "PEPSolvers_OGD.jl"))
include(joinpath(@__DIR__, "legacy_reference", "ProblemGenerators_OGD.jl"))

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const N = 10
const REPS = 10
Random.seed!(1)
α = 0.05 .+ 0.3 .* rand(N)

# Legacy: rebuild + solve + diff (one design iteration)
prob = generate_OGD(N, 1.0, 1.0, α)
sol = sdp_pep_generic(prob)
diff_w(α, sol, prob)
t_old = @elapsed for _ in 1:REPS
    compute_A_∇A_∇2A!(α, prob)
    s = sdp_pep_generic(prob)
    diff_w(α, s, prob)
end

# New: compile once (cached across the whole design run), then solve + grad
cp = compile_ogd(N, 1.0, 1.0)
s0 = solve_pep(cp, α)
grad_hess_eta(cp, α, s0)
t_compile = @elapsed compile_ogd(N, 1.0, 1.0)
t_new = @elapsed for _ in 1:REPS
    s = solve_pep(cp, α)
    grad_hess_eta(cp, α, s)
end

@printf("legacy  : %.3f s / design iter  (assemble+solve+diff)\n", t_old / REPS)
@printf("PEPDesign: %.3f s / design iter  (+ one-time compile %.3f s)\n",
        t_new / REPS, t_compile)
@printf("ratio new/old = %.2f\n", t_new / t_old)
