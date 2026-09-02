# HRDP smoke test: multi-horizon OGD design with the constant policy,
# transfer metrics, and the sampled estimator.
# Run: julia --project=. --threads=4 PEPDesign/dev/smoke_hrdp.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Printf, Test, Random, Statistics

const L = 1.0
const D = 1.0
const H = 2:6

compile_fn(N) = compile_ogd(N, L, D)

# W*_N via the theoretical optimum of the constant family ≈ √N (validated in
# bench_scaling); per-horizon multistart best-of-FOM/SOM designs.
Wstar = compute_wstar(compile_fn, H; policy = ConstantPolicy(),
                      ω0_fn = N -> [[0.5 / sqrt(N)], [1.5 / sqrt(N)]], iters = 25)
for N in H
    @printf("W*_%d = %.5f  (√N = %.5f)\n", N, Wstar[N], sqrt(N))
end

ho = HRDPObjective(compile_fn, ConstantPolicy(), H, Wstar)

# Multi-horizon design
tr = design_som(ho, [0.3]; iters = 15)
ωH, σH = best_point(tr)
@printf("\nHRDP: ω_H = %.5f   σ(H, ω_H) = %.6f   (solves = %d)\n", ωH[1], σH, tr.nsolves)

prof = snw(ho, ωH)
for N in sort(collect(keys(prof)))
    @printf("  N=%d normalized W̄ = %.5f\n", N, prof[N])
end
@printf("WGC = %.5f\n", wgc(ho, ωH))

# Tailored single-horizon designs and the GR
dp6 = DesignProblem(compile_fn(6), ConstantPolicy(), 6)
ω6, _ = best_point(design_som(dp6, [0.3]; iters = 15))
gr = generalization_ratio(ho, ω6, ωH)
@printf("GR(H, ω_{N=6}) = %.5f\n", gr)

# Sampled estimator: unbiasedness sanity (mean over draws ≈ exact σ)
Random.seed!(1)
sh = SampledHRDP(ho, 3, 2)
σ_exact, g_exact, _, _ = eval_all(ho, ωH; hess = false)
ests = [eval_all(sh, ωH; hess = false)[1] for _ in 1:40]
@printf("sampled σ: mean = %.5f  (exact %.5f, sd %.4f)\n",
        mean(ests), σ_exact, std(ests))

@testset "HRDP smoke" begin
    @test all(N -> isapprox(Wstar[N], sqrt(N); rtol = 5e-3), H)
    @test 1.0 - 1e-3 <= σH < 1.05        # constant policy transfers well here
    @test wgc(ho, ωH) >= σH - 1e-9
    @test gr >= 1.0 - 1e-6
    @test abs(mean(ests) - σ_exact) < 3 * std(ests) / sqrt(length(ests)) + 1e-3
end
println("smoke_hrdp: OK")
