# N=1 ITEM probe: identify the correct (β, δ) index convention against the
# exact guarantee 1/(1+qA_1). Run: julia --project=. PEPDesign/dev/diag_item_n1.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using Printf
include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const MU = 0.5
const D = 1.0
q = MU / L

# A_0..A_3
A = zeros(4)
for k in 2:4
    Ap = A[k - 1]
    A[k] = ((1 + q) * Ap + 2 * (1 + sqrt((1 + Ap) * (1 + q * Ap)))) / (1 - q)^2
end
βp(n) = A[n + 1] / ((1 - q) * A[n + 2])                       # paper β_n
δp(n) = ((1 - q)^2 * A[n + 2] - (1 + q) * A[n + 1]) /
        (2 * (1 + q + q * A[n + 1]))                          # paper δ_n

for N in (1, 2)
    cp = compile_item(N, L, MU, D)
    target = 1 / (1 + q * A[N + 1])
    @printf("\nN=%d  target 1/(1+qA_N) = %.8f\n", N, target)
    for (lab, βs, δs) in [
        ("β(1..N)   δ(0..N-1)", [βp(n) for n in 1:N], [δp(n - 1) for n in 1:N]),
        ("β(0..N-1) δ(0..N-1)", [βp(n - 1) for n in 1:N], [δp(n - 1) for n in 1:N]),
        ("β(1..N)   δ(1..N)  ", [βp(n) for n in 1:N], [δp(n) for n in 1:N]),
        ("β(0..N-1) δ(1..N)  ", [βp(n - 1) for n in 1:N], [δp(n) for n in 1:N]),
    ]
        W = solve_pep(cp, [βs; δs]; warn = false).obj
        @printf("  %s  W = %.8f   ratio = %.5f\n", lab, W, W / target)
    end
    # Numerical optimum over free (β, δ) via SOM from the best variant
    dp = DesignProblem(cp, IdentityPolicy(), N)
    θ0 = [[βp(n) for n in 1:N]; [δp(n - 1) for n in 1:N]]
    tr = design_som(dp, θ0; iters = 30)
    θb, Wb = best_point(tr)
    @printf("  numerical min over free θ:  W = %.8f   (ratio %.5f)\n", Wb, Wb / target)
    println("  θ* = ", round.(θb; digits = 5), "  θ0 = ", round.(θ0; digits = 5))
end
