using PEPDesign
using Printf
using Statistics

function report(label, trace, stats, timepoints)
    totals = trace.solves_hist .+ trace.tangent_hist
    statuses = Dict(status => count(==(status), stats[:tangent_status])
                    for status in unique(stats[:tangent_status]))
    @printf("%s W0=%.8f best=%.8f | PEP=%d tangent=%d accepted=%d time=%.1fs\n",
            label, trace.values[1], minimum(trace.values),
            trace.solves_hist[end],
            trace.tangent_hist[end],
            count(identity, stats[:accepted]),
            trace.time_hist[end])
    @printf("  tangent statuses=%s\n", string(statuses))
    @printf("  initial PEP=%.2fs, tangent build=%.2fs/iter, tangent optimize=%.2fs/iter, trial PEP=%.2fs/iter\n",
            stats[:initial_pep_time],
            isempty(stats[:tangent_build_time]) ? 0.0 : mean(stats[:tangent_build_time]),
            isempty(stats[:tangent_solve_time]) ? 0.0 : mean(stats[:tangent_solve_time]),
            isempty(stats[:trial_pep_time]) ? 0.0 : mean(stats[:trial_pep_time]))
    for budget in (10, 20, 50, 100)
        ids = findall(<=(budget), totals)
        value = isempty(ids) ? "not reached" :
                @sprintf("%.8f", minimum(trace.values[ids]))
        @printf("  SDP budget=%d: %s\n", budget, value)
    end
    for elapsed in timepoints
        ids = findall(<=(elapsed), trace.time_hist)
        value = isempty(ids) ? "not reached" :
                @sprintf("%.8f", minimum(trace.values[ids]))
        @printf("  wall time=%ds: %s\n", elapsed, value)
    end
end

function run_case(label, cp, policy, ω, N, max_time)
    dp = DesignProblem(cp, policy, N)
    stats = Dict{Symbol,Any}()
    trace = design_slm(dp, ω; iters = 1000, max_time, stats)
    report(label, trace, stats, (60, 120, max_time))
end

function run_comparison(N, max_time, include_dct5)
    println("=== IGDM substitution N=$N eps=0.30 ===")
    flush(stdout)
    cp = compile_igdm(N, 1.0, 0.3; objective = :min_grad,
                      encoding = :substitution)

    free_policy = IGDMKMemoryPolicy(N - 1)
    ω_free = zeros(nparams(free_policy, N))
    for n in 1:N
        ω_free[igdm_flat_index(n, n)] = 1.0
    end
    @printf("free parameters=%d\n", length(ω_free))
    run_case("free SLM", cp, free_policy, ω_free, N, max_time)
    flush(stdout)

    stationary_policy = IGDMStationaryPolicy(N - 1)
    ω_stationary = zeros(nparams(stationary_policy, N))
    ω_stationary[1] = 1.0
    @printf("stationary-by-lag parameters=%d\n", length(ω_stationary))
    run_case("stationary-by-lag SLM", cp, stationary_policy,
             ω_stationary, N, max_time)
    flush(stdout)

    for K in (include_dct5 ? (3, 5) : (3,))
        prior_policy = IGDMLagPolicy(N - 1, dct_policy(K))
        ω_prior = zeros(nparams(prior_policy, N))
        ω_prior[1] = 1.0
        @printf("lag-DCT%d parameters=%d\n", K, length(ω_prior))
        run_case("lag-DCT$K SLM", cp, prior_policy, ω_prior, N, max_time)
        flush(stdout)
    end
end

run_comparison(30, 180.0, true)
run_comparison(50, 120.0, false)
