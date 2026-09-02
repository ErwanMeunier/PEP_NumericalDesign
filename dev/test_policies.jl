# Policy-layer sanity: Jacobians/Hessians of every catalog family vs
# ForwardDiff-free finite differences, plus structured IGDM policies.
# Run: julia --project=. PEPDesign/dev/test_policies.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

include(joinpath(@__DIR__, "..", "src", "PEPDesign.jl"))
using .PEPDesign
using Test, LinearAlgebra, Random

Random.seed!(5)
const N = 7

function fd_check(pol, ω; hstep = 1e-6)
    η, J, H = evaluate_policy(pol, ω, N)
    κ = length(ω)
    for ν in 1:κ
        e = zeros(κ); e[ν] = hstep
        ηp = evaluate_policy(pol, ω .+ e, N)[1]
        ηm = evaluate_policy(pol, ω .- e, N)[1]
        fd = (ηp .- ηm) ./ (2hstep)
        @test norm(J[:, ν] - fd, Inf) < 1e-5 * max(1, norm(fd, Inf))
    end
    if H !== nothing
        for ν in 1:κ
            e = zeros(κ); e[ν] = hstep
            Jp = evaluate_policy(pol, ω .+ e, N)[2]
            Jm = evaluate_policy(pol, ω .- e, N)[2]
            fdJ = (Jp .- Jm) ./ (2hstep)
            for r in eachindex(η)
                Hr = H[r] === nothing ? zeros(κ, κ) : H[r]
                @test norm(Hr[:, ν] - fdJ[r, :], Inf) < 2e-4 * max(1, norm(fdJ[r, :], Inf))
            end
        end
    end
end

@testset "policy catalog derivatives" begin
    fd_check(ConstantPolicy(), [0.4])
    fd_check(IdentityPolicy(), rand(N))
    fd_check(PowerLawPolicy(), [0.8, 0.6, 0.3])
    fd_check(sum_of_exp_policy(2), [0.5, 0.3, 0.2, 0.1, 0.05])
    fd_check(log_poly_policy(2), [-0.5, -0.3, 0.05])
    fd_check(dct_policy(3), [0.4, 0.1, -0.05, 0.02])
    fd_check(cosine_decay_policy(), [0.4, 1.3, 0.05])
    fd_check(rational_policy(), [0.5, 0.1, 0.3, 0.05])
    fd_check(warped_chebyshev_policy(3), [0.4, 0.1, -0.05, 0.02])
    fd_check(piecewise_exp_policy(), [0.5, 0.2, 0.4, 0.1])
    fd_check(ProductPolicy(PowerLawPolicy(), ConstantPolicy()), [0.8, 0.6, 0.3, 1.2])
end

@testset "IGDM structured policies" begin
    # Memoryless diagonal lift of a power law
    pol = igdm_diagonal_policy(PowerLawPolicy())
    ω = [0.8, 0.5, 0.2]
    η, J, H = evaluate_policy(pol, ω, N)
    @test length(η) == igdm_dof(N)
    s, _, _ = evaluate_policy(PowerLawPolicy(), ω, N)
    @test all(η[igdm_flat_index(n, n)] ≈ s[n] for n in 1:N)
    @test all(η[igdm_flat_index(n, k)] == 0 for n in 1:N for k in 1:(n - 1))

    # K-memory: correct sparsity and identity Jacobian
    K = 2
    pk = IGDMKMemoryPolicy(K)
    κk = nparams(pk, N)
    ωk = rand(κk)
    ηk, Jk, _ = evaluate_policy(pk, ωk, N)
    @test count(!iszero, ηk) == count(!iszero, ωk)
    @test all(ηk[igdm_flat_index(n, k)] == 0 for n in 1:N for k in 1:n if k < n - K)
    @test Jk' * Jk == I

    # Stationary: lag-constant diagonals
    ps = IGDMStationaryPolicy(2)
    ηs, _, _ = evaluate_policy(ps, [0.5, 0.2, 0.1], N)
    @test all(ηs[igdm_flat_index(n, n - 1)] == 0.2 for n in 2:N)

    # Lag policy: power law per lag, FD on Jacobian
    pl = IGDMLagPolicy(1, PowerLawPolicy())
    ωl = [0.8, 0.5, 0.2, 0.1, 0.4, 0.3]
    ηl, Jl, Hl = evaluate_policy(pl, ωl, N)
    hs = 1e-6
    for ν in eachindex(ωl)
        e = zeros(length(ωl)); e[ν] = hs
        fd = (evaluate_policy(pl, ωl .+ e, N)[1] .- evaluate_policy(pl, ωl .- e, N)[1]) ./ (2hs)
        @test norm(Jl[:, ν] - fd, Inf) < 1e-5
    end
end

println("test_policies: OK")
