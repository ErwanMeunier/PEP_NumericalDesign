# Smoke test: FOM and SOM design on OGD with the constant policy.
# Run from repo root:  julia --project=. PEPDesign/dev/smoke_design.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Printf, Test

const N = 5
const L = 1.0
const D = 1.0

cp = compile_ogd(N, L, D)
dp = DesignProblem(cp, ConstantPolicy(), N)

ω0 = [0.15]
W0, _, _ = pep_value(dp, ω0)

tr_fom = design_fom(dp, ω0; iters = 25, steps = t -> 0.1 / sqrt(t))
ω_fom, W_fom = best_point(tr_fom)

tr_som = design_som(dp, ω0; iters = 25)
ω_som, W_som = best_point(tr_som)

conj = D / (L * sqrt(N))   # conjectured optimal constant step size
@printf("W(ω0=0.15)      = %.6f\n", W0)
@printf("FOM  best: ω = %.5f   W = %.6f   (solves = %d)\n", ω_fom[1], W_fom, tr_fom.nsolves)
@printf("SOM  best: ω = %.5f   W = %.6f   (solves = %d)\n", ω_som[1], W_som, tr_som.nsolves)
@printf("conjecture D/(L√N) = %.5f,  s_N = L√N/D · ω_fom = %.4f\n", conj, sqrt(N) * ω_fom[1])

@testset "design smoke" begin
    @test W_fom < W0
    @test W_som < W0
    @test isapprox(ω_fom[1], conj; rtol = 0.05)          # supports the 1/√N conjecture
    @test isapprox(W_fom, sqrt(N) * L * D; rtol = 0.01)  # W* ≈ √N·L·D
end
println("smoke_design: OK")
