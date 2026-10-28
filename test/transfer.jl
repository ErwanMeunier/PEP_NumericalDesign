@testset "reuse without redesign" begin
    compile_fn = N -> compile_gd(N, 1.0)
    weights = [1.0]
    rows = assess_transfer(compile_fn, ConstantPolicy(), weights, 2, [2, 3, 5])
    @test weights == [1.0]
    @test getproperty.(rows, :N) == [2, 3, 5]
    for row in rows
        @test row.source_N == 2
        @test row.omega == weights
        @test row.coefficients == ones(row.N)
        @test row.reuse_value ≈ 1 / (4row.N + 2) rtol = 2e-5
        @test row.reuse_solution.diagnostics.certified
        @test row.reuse_seconds >= 0
        @test ismissing(row.free_best_value)
        @test ismissing(row.ratio)
        @test isempty(row.free_runs)
        @test row.free_pep_solves == row.free_tangent_solves == 0
    end

    @testset "product mapping transfer" begin
        function compiler(N)
            model = PEPModel()
            coefficients = coeffs!(model, 2N)
            x = point!(model)
            add_le!(model, sqnorm(x) - 1)
            objective_max!(model, sum(sqnorm(c * x) for c in coefficients))
            return compile(model)
        end
        policy = ProductPolicy(ConstantPolicy(), ConstantPolicy())
        rows = assess_transfer(compiler, policy, [0.5, 0.3], 2, [3, 4])
        for row in rows
            @test row.coefficients == [fill(0.5, row.N); fill(0.3, row.N)]
            @test row.reuse_value ≈ row.N * (0.5^2 + 0.3^2) rtol = 1e-6
        end
    end

    dct = dct_policy(1)
    omega = [1.0, 0.1]
    transferred = only(assess_transfer(compile_fn, dct, omega, 2, [4]))
    @test transferred.coefficients == evaluate_policy(dct, omega, 4)[1]
    @test transferred.coefficients[1:2] != evaluate_policy(dct, omega, 2)[1]

    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 0, [2])
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, Int[])
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, [1])
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, [3, 3])
    @test_throws ArgumentError assess_transfer(compile_fn, dct, [NaN, 0.1], 2, [3])
    @test_throws DimensionMismatch assess_transfer(compile_fn, IdentityPolicy(), ones(2), 2, [3])
    @test_throws DimensionMismatch assess_transfer(compile_fn, dct, [1.0], 2, [3])
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, [3]; iters = -1)
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, [3]; max_time = 0.0)
    @test_throws ArgumentError assess_transfer(compile_fn, dct, omega, 2, [3];
                                              free_starts = _ -> Vector{Float64}[])
end

@testset "fresh multistart free comparator" begin
    row = only(assess_transfer(N -> compile_gd(N, 1.0), ConstantPolicy(), [1.0], 1, [2];
                              free_starts = N -> [ones(N), fill(1.2, N)], iters = 5))
    @test length(row.free_runs) == 2
    @test row.free_best_value == minimum(run.solution.obj for run in row.free_runs)
    @test row.free_solution.diagnostics.certified
    @test length(row.free_omega) == 2
    @test row.free_best_value < row.reuse_value
    @test row.ratio ≈ row.reuse_value / row.free_best_value
    @test row.free_pep_solves == sum(run.trace.nsolves + 1 for run in row.free_runs)
    @test row.free_tangent_solves == sum(run.trace.tangent_hist[end] for run in row.free_runs)
    @test row.free_seconds > 0
end

@testset "IGDM stationary reuse and fixed-memory free baseline" begin
    compiler = N -> compile_igdm(N, 1.0, 0.3; objective = :min_grad,
                                encoding = :substitution)
    policy = IGDMStationaryPolicy(1)
    rows = assess_transfer(compiler, policy, [1.0, 0.0], 2, [3, 4];
                           free_policy = IGDMKMemoryPolicy(1),
                           free_starts = N -> [k == n ? 1.0 : 0.0
                                               for n in 1:N for k in max(1, n - 1):n],
                           iters = 0)
    for row in rows
        @test row.reuse_value ≈ 1 / (0.5 + row.N * 0.7) rtol = 3e-5
        @test row.ratio ≈ 1 rtol = 1e-6
        @test row.omega == [1.0, 0.0]
        @test row.reuse_solution.diagnostics.certified
        @test row.free_pep_solves == 2
    end
end
