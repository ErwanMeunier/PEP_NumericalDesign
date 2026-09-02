# IGDM parity and sensitivity checks: new PEPDesign vs legacy src/.
# Run from repo root:  julia --project=. PEPDesign/dev/parity_igdm.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using JuMP, Mosek, MosekTools, LinearAlgebra, SparseArrays, Random, Test

include(joinpath(@__DIR__, "legacy_reference", "PEPTypes.jl"))
include(joinpath(@__DIR__, "legacy_reference", "MatrixComputations_IGDM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "PEPSolvers_IGDM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "ProblemGenerators_IGDM.jl"))

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const EPS = 0.3

Random.seed!(3)

flat_to_mat(η, N) = begin
    β = zeros(N, N)
    for n in 1:N, k in 1:n
        β[n, k] = η[igdm_flat_index(n, k)]
    end
    β
end

@testset "IGDM parity and sensitivity" begin
    for N in (2, 3, 5)
        p = igdm_dof(N)
        η = 0.05 .+ 0.4 .* rand(p)   # random positive lower-triangular β
        β = flat_to_mat(η, N)

        prob_old = generate_IGDM(N, L, EPS, β)
        sol_old = sdp_pep_IGDM(prob_old)

        cp = compile_igdm(N, L, EPS)   # legacy: D = 1, objective ‖g_{N+1}‖²
        sol_new = solve_pep(cp, η)

        @info "N=$N parity" W_old = sol_old.obj_value W_new = sol_new.obj
        # residual equalities make the SDP degenerate (SLOW_PROGRESS); both
        # solvers land within ~1e-4 relative of each other
        @test isapprox(sol_new.obj, sol_old.obj_value; rtol = 5e-4)

        # Matrix-level derivative checks (exact for degree-2 polynomials)
        h = 1e-3
        for con in cp.cons
            M = con.expr.M
            PEPDesign.is_param_dependent(M) || continue
            for r in 1:p
                ep = zeros(p); ep[r] = h
                FD = (assemble(M, η .+ ep) .- assemble(M, η .- ep)) ./ (2h)
                @test norm(FD - dmat(M, r, η), Inf) < 1e-9
            end
        end

        # Exact contraction identity with a synthetic certificate (solver-free)
        g, H = grad_hess_eta(cp, η, sol_new)
        Ĝ = let X = randn(cp.dim, cp.dim); X * X' ./ cp.dim end
        λ̂ = rand(length(cp.cons))
        fake = PEPDesign.PEPSolution(0.0, Ĝ, zeros(cp.nf), λ̂, [1.0], sol_new.status)
        gs, Hs = grad_hess_eta(cp, η, fake)
        gref = zeros(p); Href = zeros(p, p)
        for (c, con) in enumerate(cp.cons)
            M = con.expr.M
            PEPDesign.is_param_dependent(M) || continue
            for r in 1:p
                gref[r] -= λ̂[c] * trprod(Ĝ, dmat(M, r, η))
                for s in 1:p
                    Href[r, s] -= λ̂[c] * trprod(Ĝ, d2mat(M, r, s))
                end
            end
        end
        @test norm(gs - gref, Inf) < 1e-10
        @test norm(Hs - Href, Inf) < 1e-10

        # Directional derivatives (informational: SDP noise + possible kinks)
        for _ in 1:3
            dv = randn(p); dv ./= norm(dv)
            h = 1e-3
            Wp = solve_pep(cp, η .+ h .* dv; warn = false).obj
            Wm = solve_pep(cp, η .- h .* dv; warn = false).obj
            @info "N=$N directional" g_dot_d = dot(g, dv) fd_central = (Wp - Wm) / 2h
        end

        # maxmin objective variant: sup min_k ≤ sup of the last component
        cpm = compile_igdm(N, L, EPS; objective = :min_grad)
        solm = solve_pep(cpm, η)
        @test solm.obj <= sol_new.obj * (1 + 1e-3)
        @test isapprox(sum(solm.obj_duals), 1.0; atol = 1e-5)   # Σγ_k = 1
    end
end

println("parity_igdm: all checks completed")
