# Parity and correctness checks for the OGD path: new PEPDesign vs legacy src/.
# Run from the repo root:  julia --project=. PEPDesign/dev/parity_ogd.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using JuMP, Mosek, MosekTools, LinearAlgebra, SparseArrays, Random, Test

# Legacy implementation (frozen copy; only the OGD path).
include(joinpath(@__DIR__, "legacy_reference", "PEPTypes.jl"))
include(joinpath(@__DIR__, "legacy_reference", "MatrixComputations_OGD.jl"))
include(joinpath(@__DIR__, "legacy_reference", "PEPSolvers_OGD.jl"))
include(joinpath(@__DIR__, "legacy_reference", "ProblemGenerators_OGD.jl"))

# New package.
include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const D = 1.0

Random.seed!(42)

@testset "OGD parity and sensitivity" begin
    for N in (2, 3, 5)
        α = 0.05 .+ 0.3 .* rand(N)

        # Legacy value
        prob_old = generate_OGD(N, L, D, α)
        sol_old = sdp_pep_generic(prob_old)

        # New value
        cp = compile_ogd(N, L, D)
        sol_new = solve_pep(cp, α)

        @info "N=$N" W_old = sol_old.obj_value W_new = sol_new.obj
        @test isapprox(sol_new.obj, sol_old.obj_value; rtol = 1e-5)

        # Matrix-level derivative checks (exact for degree-2 polynomials)
        for con in cp.cons
            M = con.expr.M
            PEPDesign.is_param_dependent(M) || continue
            h = 1e-3
            for r in 1:N
                ep = zeros(N); ep[r] = h
                FD = (assemble(M, α .+ ep) .- assemble(M, α .- ep)) ./ (2h)
                @test norm(FD - dmat(M, r, α), Inf) < 1e-9
                for s in r:N
                    es = zeros(N); es[s] = h
                    FD2 = (dmat(M, r, α .+ es) .- dmat(M, r, α .- es)) ./ (2h)
                    @test norm(FD2 - d2mat(M, r, s), Inf) < 1e-9
                end
            end
        end

        # Gradient of W vs central finite differences
        g, H = grad_hess_eta(cp, α, sol_new)
        h = 1e-4
        for r in 1:N
            e = zeros(N); e[r] = h
            Wp = solve_pep(cp, α .+ e).obj
            Wm = solve_pep(cp, α .- e).obj
            fd = (Wp - Wm) / (2h)
            @test isapprox(g[r], fd; atol = 5e-3, rtol = 5e-3)
        end
        @test norm(H - H') < 1e-12
        @info "N=$N gradient check passed" g

        # Legacy gradient (known to carry a 1.5× bug on the A5 family) — info only
        g_old, _ = diff_w(α, sol_old, prob_old)
        @info "N=$N legacy-vs-new gradient gap (informational)" gap = norm(g_old - g, Inf)
    end
end

println("parity_ogd: all checks completed")
