using LinearAlgebra

function residual_pep(; duplicate::Bool = false)
    model = PEPModel()
    eta = coeffs!(model, 1)
    x = point!(model; name = "x")
    y = point!(model; name = "y")
    residual = y - eta[1] * x
    add_eq!(model, sqnorm(residual); name = "residual")
    duplicate && add_eq!(model, sqnorm(residual); name = "residual_duplicate")
    add_le!(model, sqnorm(x) - 1; name = "radius")
    objective_max!(model, sqnorm(y))
    return compile(model)
end

@testset "facial reduction discovery" begin
    cp = residual_pep(; duplicate = true)
    eta = [0.5]
    info = facial_reduction_info(cp, eta)
    @test facial_reduction_applied(info)
    @test info.original_dim == 2
    @test info.reduced_dim == 1
    @test info.exposing_rank == 1
    @test info.removed_constraints == [1, 2]
    @test norm(info.basis' * [-eta[1], 1.0]) < 1e-10

    none = facial_reduction_info(cp, eta; mode = :none)
    @test !facial_reduction_applied(none)
    @test none.basis == I
    @test_throws ErrorException facial_reduction_info(cp, eta; mode = :invalid)
end

@testset "indefinite and full-face equalities" begin
    model = PEPModel()
    x = point!(model; name = "x")
    y = point!(model; name = "y")
    add_eq!(model, inner(x, y); name = "indefinite")
    add_le!(model, sqnorm(x) + sqnorm(y) - 1)
    objective_max!(model, sqnorm(x))
    cp = compile(model)
    info = facial_reduction_info(cp, Float64[])
    @test !facial_reduction_applied(info)
    @test isempty(info.removed_constraints)

    full = PEPModel()
    u = point!(full; name = "u")
    v = point!(full; name = "v")
    add_eq!(full, sqnorm(u) + sqnorm(v); name = "zero_gram")
    objective_max!(full, sqnorm(u))
    full_cp = compile(full)
    full_info = facial_reduction_info(full_cp, Float64[])
    @test full_info.exposing_rank == 2
    @test full_info.reduced_dim == 0
    full_sol = solve_pep(full_cp, Float64[]; facial_reduction = :explicit)
    @test full_sol.obj ≈ 0 atol = 1e-9
    @test size(full_sol.G) == (2, 2)
    @test norm(full_sol.G) < 1e-9
end

@testset "reduced solve and original-space certification" begin
    cp = residual_pep(; duplicate = true)
    eta = [0.5]
    sol = solve_pep(cp, eta; facial_reduction = :explicit)
    @test sol.obj ≈ eta[1]^2 rtol = 2e-6 atol = 1e-8
    @test sol.diagnostics.certified
    @test sol.diagnostics.max_scalar_violation <= 1e-7
    @test sol.diagnostics.max_cone_violation <= 1e-7
    @test sol.reduction.reduced_dim == 1
    @test size(sol.G) == (cp.dim, cp.dim)
    @test norm(sol.G * [-eta[1], 1.0]) < 1e-7
    @test all(iszero, sol.duals[sol.reduction.removed_constraints])
    @test_throws ErrorException grad_hess_eta(cp, eta, sol)

    plain = solve_pep(cp, eta; facial_reduction = :none, warn = false)
    @test plain.obj ≈ sol.obj rtol = 2e-4 atol = 1e-7
    @test !facial_reduction_applied(plain.reduction)
end

@testset "first-order design uses certified finite differences" begin
    cp = residual_pep()
    dp = DesignProblem(cp, ConstantPolicy(), 1;
                       facial_reduction = :explicit, fd_rel_step = 1e-4)
    value, gradient, H, solves = eval_all(dp, [0.5]; hess = false)
    @test value ≈ 0.25 rtol = 2e-6
    @test gradient[1] ≈ 1.0 rtol = 5e-4
    @test size(H) == (0, 0)
    @test solves == 3
    @test_throws ErrorException eval_all(dp, [0.5]; hess = true)
    @test_throws ErrorException design_som(dp, [0.5]; iters = 1)
    @test_throws ErrorException design_ssdp(dp, [0.5]; iters = 1)

    trace = design_fom(dp, [0.5]; iters = 2, steps = _ -> 0.1)
    @test best_point(trace)[2] < value
    @test trace.nsolves == 9
end
