function _transfer_coefficients(dp::DesignProblem, ω)
    expected = dp.policy isa IdentityPolicy ? dp.cp.np : nparams(dp.policy, dp.N)
    length(ω) == expected ||
        throw(DimensionMismatch("policy at N=$(dp.N) expects $expected parameters, " *
                                "got $(length(ω)); reuse does not pad or fit weights"))
    all(isfinite, ω) || throw(ArgumentError("policy parameters must be finite"))
    η, _, _ = evaluate_policy(dp.policy, ω, dp.N)
    length(η) == dp.cp.np ||
        throw(DimensionMismatch("policy at N=$(dp.N) produced $(length(η)) coefficients, " *
                                "expected $(dp.cp.np)"))
    all(isfinite, η) || throw(ArgumentError("policy generated nonfinite coefficients"))
    return η
end

function _transfer_value(dp::DesignProblem, η)
    sol = solve_pep(dp.cp, η; backend = dp.backend,
                    facial_reduction = dp.facial_reduction,
                    fr_rtol = dp.fr_rtol, fr_atol = dp.fr_atol,
                    cert_tol = dp.cert_tol, gap_tol = dp.gap_tol)
    isfinite(sol.obj) && sol.diagnostics.certified ||
        error("transfer assessment at N=$(dp.N) has no numerical certificate: " *
              "status=$(sol.status), gap=$(sol.diagnostics.relative_gap), " *
              "scalar=$(sol.diagnostics.max_scalar_violation), " *
              "cone=$(sol.diagnostics.max_cone_violation)")
    return sol
end

"""
    assess_transfer(compile_fn, policy, ω, N_train, horizons;
                    free_starts=nothing, free_policy=IdentityPolicy(),
                    iters=100, max_time=Inf, problem_kwargs...)

Assess unchanged parameters learned at `N_train` at each target horizon M:
`W(policy(ω, M), M)`. No redesign, fitting, padding, or rescaling of ω is
performed. Parameters and generated coefficients must have the right dimensions.
Horizon-dependent mappings (e.g. DCT) are reevaluated at M, so this is not
necessarily a prefix-preserving/anytime schedule. A stationary IGDM policy keeps K
fixed. Only target horizons are solved (one PEP per target for reuse).

Optionally compare with fresh SLM designs: `free_starts(M)` returns one parameter
vector or a nonempty vector of vectors for `free_policy`. The default identity
policy frees all compiled coefficients. For a matched memory restriction pass
`IGDMKMemoryPolicy(K)` explicitly. Each start gets `iters` and `max_time`
(a soft limit checked between iterations); reuse is never used to warm-start
the baseline implicitly. Other keywords are forwarded to DesignProblem.

Returns a vector of named tuples with `N`, `source_N`, `omega`, `coefficients`,
`reuse_value`, `reuse_solution`, `reuse_seconds`, `compile_seconds`,
`free_best_value`, `free_omega`, `free_solution`, `free_seconds`,
`free_pep_solves`, `free_tangent_solves`, `free_runs`, and `ratio`.
`free_runs` retains each trace and SLM stats. Free timings include final
reevaluation; reuse timings include mapping evaluation but exclude compilation.
All reported values must pass numerical certification. `ratio` is
reuse / best-found free value, NOT an optimality gap; it may be below 1.
Without a free comparison its value and the free best value are `missing`.
"""
function assess_transfer(compile_fn, policy::AbstractPolicy, ω::AbstractVector{<:Real},
                         N_train::Int, horizons;
                         free_starts = nothing,
                         free_policy::AbstractPolicy = IdentityPolicy(),
                         iters::Int = 100, max_time::Float64 = Inf,
                         problem_kwargs...)
    N_train >= 1 || throw(ArgumentError("training horizon must be positive"))
    targets = collect(Int, horizons)
    isempty(targets) && throw(ArgumentError("provide at least one target horizon"))
    all(>=(N_train), targets) ||
        throw(ArgumentError("target horizons must be at least N_train"))
    length(unique(targets)) == length(targets) ||
        throw(ArgumentError("target horizons must be unique"))
    iters >= 0 || throw(ArgumentError("iters must be nonnegative"))
    max_time > 0 || throw(ArgumentError("max_time must be positive"))
    weights = collect(Float64, ω)
    all(isfinite, weights) || throw(ArgumentError("policy parameters must be finite"))
    if !(policy isa IdentityPolicy)
        length(weights) == nparams(policy, N_train) ||
            throw(DimensionMismatch("parameter count does not match the training policy"))
    end
    results = NamedTuple[]
    for M in targets
        t = time()
        cp = compile_fn(M)
        compile_seconds = time() - t
        dp = DesignProblem(cp, policy, M; problem_kwargs...)
        t = time()
        η = _transfer_coefficients(dp, weights)
        sol = _transfer_value(dp, η)
        reuse_seconds = time() - t

        free_best_value = missing
        free_omega = nothing
        free_solution = nothing
        free_seconds = 0.0
        free_pep_solves = 0
        free_tangent_solves = 0
        free_runs = NamedTuple[]
        if free_starts !== nothing
            t = time()
            fdp = DesignProblem(cp, free_policy, M; problem_kwargs...)
            starts = free_starts(M)
            starts isa AbstractVector{<:Real} && (starts = [starts])
            isempty(starts) && throw(ArgumentError("free_starts($M) is empty"))
            best = Inf
            for start in starts
                _transfer_coefficients(fdp, start)
                stats = Dict{Symbol,Any}()
                trace = design_slm(fdp, start; iters, max_time, stats,
                                   require_certified = true)
                candidate, _ = best_point(trace)
                coefficients = _transfer_coefficients(fdp, candidate)
                candidate_sol = _transfer_value(fdp, coefficients)
                free_pep_solves += trace.nsolves + 1
                free_tangent_solves += trace.tangent_hist[end]
                push!(free_runs, (; trace, stats, solution = candidate_sol))
                if candidate_sol.obj < best
                    best = candidate_sol.obj
                    free_omega = copy(candidate)
                    free_solution = candidate_sol
                end
            end
            best > 0 || error("normalizing transfer requires a positive free best value")
            free_best_value = best
            free_seconds = time() - t
        end
        ratio = ismissing(free_best_value) ? missing : sol.obj / free_best_value
        push!(results, (; N = M, source_N = N_train, omega = copy(weights),
                         coefficients = η, reuse_value = sol.obj,
                         reuse_solution = sol, reuse_seconds, compile_seconds,
                         free_best_value, free_omega, free_solution, free_seconds,
                         free_pep_solves, free_tangent_solves, free_runs, ratio))
    end
    return results
end
