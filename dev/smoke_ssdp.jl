# SSDP smoke test on OGD with the constant policy.
# Run: julia --project=. PEPDesign/dev/smoke_ssdp.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Printf, Test

const N = 3
const L = 1.0
const D = 1.0

cp = compile_ogd(N, L, D)
dp = DesignProblem(cp, ConstantPolicy(), N)

ω0 = [0.2]
W0, _, _ = pep_value(dp, ω0)
tr = design_ssdp(dp, ω0; iters = 30, verbose = true)

@printf("\nW(ω0) = %.6f\n", W0)
@printf("SSDP: ω = %.5f  NSDP obj = %.6f  W(ω_final) = %.6f  tangent solves = %d\n",
        tr.ω_hist[end][1], tr.obj_hist[end], tr.W_final, tr.ntangent)
@printf("conjecture: ω* = %.5f, W* = %.6f\n", 1 / sqrt(N), sqrt(N))
@printf("KKT history: %s\n", join([@sprintf("%.2e", k) for k in tr.kkt_hist], " "))

@testset "SSDP smoke" begin
    @test tr.W_final < W0
    @test isapprox(tr.ω_hist[end][1], 1 / sqrt(N); rtol = 0.05)
    @test isapprox(tr.W_final, sqrt(N); rtol = 0.02)
    @test isapprox(tr.obj_hist[end], tr.W_final; rtol = 0.05)  # dual ≈ primal
end
println("smoke_ssdp: OK")
