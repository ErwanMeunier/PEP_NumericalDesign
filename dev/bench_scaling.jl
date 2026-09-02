# Timing probe at large horizons: compile / solve / gradient for OGD and IGDM.
# Run: julia --project=. PEPDesign/dev/bench_scaling.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Printf

function bench(label, f)
    f()   # warm-up (compilation)
    t = @elapsed out = f()
    @printf("%-34s %8.2f s\n", label, t)
    return out
end

for N in (20, 30, 40)
    println("── N = $N ─────────────────────────────")
    cp = bench("OGD compile (N=$N)", () -> compile_ogd(N, 1.0, 1.0))
    @printf("   dim=%d  cons=%d  np=%d\n", cp.dim, length(cp.cons), cp.np)
    α = fill(1.0 / sqrt(N), N)
    sol = bench("OGD solve", () -> solve_pep(cp, α))
    @printf("   W = %.4f (√N = %.4f)\n", sol.obj, sqrt(N))
    bench("OGD grad+hess", () -> grad_hess_eta(cp, α, sol))
end

for N in (20, 30)
    println("── IGDM N = $N ────────────────────────")
    cp = bench("IGDM compile (N=$N)", () -> compile_igdm(N, 1.0, 0.3))
    @printf("   dim=%d  cons=%d  np=%d\n", cp.dim, length(cp.cons), cp.np)
    η = zeros(cp.np)
    for n in 1:N
        η[igdm_flat_index(n, n)] = igdm_hmem(0.3)
    end
    sol = bench("IGDM solve", () -> solve_pep(cp, η))
    @printf("   W = %.6f\n", sol.obj)
    bench("IGDM grad+hess", () -> grad_hess_eta(cp, η, sol))
end
