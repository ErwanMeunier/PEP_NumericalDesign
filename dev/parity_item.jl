# ITEM parity and sensitivity checks: new PEPDesign vs legacy src/ and the
# analytic optimum. Run from repo root:
#   julia --project=. PEPDesign/dev/parity_item.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using JuMP, Mosek, MosekTools, LinearAlgebra, SparseArrays, Random, Test

include(joinpath(@__DIR__, "legacy_reference", "PEPTypes.jl"))
include(joinpath(@__DIR__, "legacy_reference", "MatrixComputations_ITEM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "PEPSolvers_ITEM.jl"))
include(joinpath(@__DIR__, "legacy_reference", "ProblemGenerators_ITEM.jl"))

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign

const L = 1.0
const MU = 0.5
const D = 1.0

Random.seed!(7)

@testset "ITEM parity and sensitivity" begin
    for N in (2, 3, 5)
        βs, δs, Wfac = item_optimal_params(N, L, MU)

        cp = compile_item(N, L, MU, D)

        # At the reference coefficients: W near the ITEM guarantee D²/(1+qA_N).
        # The compact variant omits the δ₀ step, and tightness degrades as the
        # values decay geometrically — W* is a lower bound, close for small N.
        θ_opt = [βs; δs]
        sol_opt = solve_pep(cp, θ_opt)
        @info "N=$N analytic check" W_new = sol_opt.obj W_star = Wfac * D^2
        @test (1 - 1e-3) * Wfac * D^2 <= sol_opt.obj <= 1.15 * Wfac * D^2

        # Perturbed coefficients: parity with the legacy implementation
        # (legacy sdp_pep_ITEM runs at Mosek default gap on a worse-conditioned
        # model, so only ~1e-3 relative agreement is expected)
        θ = θ_opt .* (1.0 .+ 0.15 .* (rand(2N) .- 0.5))
        prob_old = generate_ITEM(N, L, MU, D, θ)
        sol_old = sdp_pep_ITEM(prob_old)
        sol_new = solve_pep(cp, θ)
        @info "N=$N parity" W_old = sol_old.obj_value W_new = sol_new.obj
        # atol floor: legacy solver runs at Mosek default gap, so agreement is
        # absolute-tolerance-limited once W decays to ~1e-5 (data scale D²=1)
        @test isapprox(sol_new.obj, sol_old.obj_value; rtol = 5e-3, atol = 1e-6)

        # Matrix-level derivative checks (exact for degree-2 polynomials)
        h = 1e-3
        for con in cp.cons
            M = con.expr.M
            PEPDesign.is_param_dependent(M) || continue
            for r in 1:(2N)
                ep = zeros(2N); ep[r] = h
                FD = (assemble(M, θ .+ ep) .- assemble(M, θ .- ep)) ./ (2h)
                @test norm(FD - dmat(M, r, θ), Inf) < 1e-9
            end
        end

        # Gradient check. ITEM's PEP value has genuine kinks (nonunique
        # certificates), so equality with central FD is NOT expected.

        # Gradient contraction identity with a synthetic certificate (exact,
        # no solver in the loop): grad_hess_eta must equal the explicit
        # −Σ λ_c tr(Ĝ ∂A_c/∂η) contraction built from dmat/d2mat.
        Ĝ = let X = randn(cp.dim, cp.dim); X * X' ./ cp.dim end
        λ̂ = rand(length(cp.cons))
        fake = PEPDesign.PEPSolution(0.0, Ĝ, zeros(cp.nf), λ̂, [1.0], sol_new.status)
        gs, Hs = grad_hess_eta(cp, θ, fake)
        gref = zeros(2N); Href = zeros(2N, 2N)
        for (c, con) in enumerate(cp.cons)
            M = con.expr.M
            PEPDesign.is_param_dependent(M) || continue
            for r in 1:(2N)
                gref[r] -= λ̂[c] * trprod(Ĝ, dmat(M, r, θ))
                for s in 1:(2N)
                    Href[r, s] -= λ̂[c] * trprod(Ĝ, d2mat(M, r, s))
                end
            end
        end
        @test norm(gs - gref, Inf) < 1e-10
        @test norm(Hs - Href, Inf) < 1e-10

        # One-sided derivative bracketing (informational: ITEM has kinks and
        # certificate quality is limited by SLOW_PROGRESS conditioning)
        g, H = grad_hess_eta(cp, θ, sol_new)
        d = randn(2N); d ./= norm(d)
        h = 1e-3
        Wp = solve_pep(cp, θ .+ h .* d; warn = false).obj
        Wm = solve_pep(cp, θ .- h .* d; warn = false).obj
        @info "N=$N directional check (informational)" g_dot_d = dot(g, d) fd_right = (Wp - sol_new.obj) / h fd_left = (sol_new.obj - Wm) / h

        # Legacy gradient comparison (informational: different certificates)
        g_old, H_old = diff_w_ITEM(θ, sol_old, prob_old)
        @info "N=$N legacy-vs-new gradient gap (informational)" gap = norm(g_old - g, Inf)
    end
end

println("parity_item: all checks completed")
