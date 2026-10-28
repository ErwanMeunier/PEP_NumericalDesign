using LinearAlgebra

@testset "stationary lag mapping" begin
    N, K = 6, 2
    legacy = IGDMStationaryPolicy(K)
    weights = [0.8, 0.2, -0.1]
    eta, J, H = evaluate_policy(legacy, weights, N)
    @test nparams(legacy, N) == K + 1
    @test H === nothing
    for n in 1:N, k in 1:n
        r, lag = igdm_flat_index(n, k), n - k
        @test eta[r] == (lag <= K ? weights[lag + 1] : 0.0)
        @test J[r, :] == [lag == j ? 1.0 : 0.0 for j in 0:K]
    end
    @test evaluate_policy(IGDMStationaryPolicy(K, IdentityPolicy()), weights, N) ==
          (eta, J, H)
    @test evaluate_policy(IGDMLagPolicy(K, ConstantPolicy()), weights, N)[1:2] ==
          (eta, J)
    @test occursin("power", policy_label(IGDMStationaryPolicy(K, PowerLawPolicy()))) ||
          occursin("Power", policy_label(IGDMStationaryPolicy(K, PowerLawPolicy())))

    @test evaluate_policy(IGDMStationaryPolicy(0), [0.5], 1)[1] == [0.5]
    @test evaluate_policy(IGDMStationaryPolicy(4), ones(5), 1)[1] == [1.0]
    @test_throws ArgumentError IGDMStationaryPolicy(-1)
    @test_throws ArgumentError evaluate_policy(legacy, weights, 0)
    @test_throws DimensionMismatch evaluate_policy(legacy, [0.5], N)
    @test_throws DimensionMismatch evaluate_policy(
        IGDMStationaryPolicy(2, ProductPolicy(ConstantPolicy(), ConstantPolicy())),
        [0.5, 0.1], N)
end

@testset "stationary catalog derivatives and fixed-memory transfer" begin
    families = [
        (ConstantPolicy(), [0.4]),
        (IdentityPolicy(), [0.8, 0.2, 0.1, 0.05]),
        (PowerLawPolicy(), [0.8, 0.6, 0.3]),
        (sum_of_exp_policy(1), [0.8, 0.4, 0.1]),
        (log_poly_policy(2), [-0.5, -0.3, 0.05]),
        (dct_policy(3), [0.4, 0.1, -0.05, 0.02]),
        (cosine_decay_policy(), [0.4, 2.0, 0.05]),
        (rational_policy(), [0.5, 0.1, 0.3, 0.05]),
        (warped_chebyshev_policy(3), [0.4, 0.1, -0.05, 0.02]),
        (piecewise_exp_policy(), [0.5, 0.2, 0.4, 0.1]),
        (FunctionPolicy((w, n, N) -> w[1]^2 / n + w[2], 2), [0.8, 0.1]),
    ]
    for (base, weights) in families
        policy = IGDMStationaryPolicy(3, base)
        eta, J, H = evaluate_policy(policy, weights, 6)
        @test nparams(policy, 6) == nparams(policy, 9) == length(weights)
        @test evaluate_policy(policy, weights, 9)[1][1:length(eta)] == eta
        lag_weights = evaluate_policy(base, weights, 4)[1]
        for n in 1:6, k in 1:n
            @test eta[igdm_flat_index(n, k)] ==
                  (n - k <= 3 ? lag_weights[n - k + 1] : 0.0)
        end
        for j in eachindex(weights)
            e = zeros(length(weights)); e[j] = 1e-6
            ep, Jp, _ = evaluate_policy(policy, weights + e, 6)
            em, Jm, _ = evaluate_policy(policy, weights - e, 6)
            @test J[:, j] ≈ (ep - em) / 2e-6 atol = 1e-7 rtol = 1e-5
            if H !== nothing
                for r in eachindex(eta)
                    column = H[r] === nothing ? zeros(length(weights)) : H[r][:, j]
                    @test column ≈ (Jp[r, :] - Jm[r, :]) / 2e-6 atol = 1e-7 rtol = 1e-5
                end
            end
        end
    end
end
