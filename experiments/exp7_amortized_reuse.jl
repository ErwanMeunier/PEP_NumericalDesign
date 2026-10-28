# Train one fixed-memory stationary IGDM schedule once, then assess/reuse it
# and optionally free-refine it at multiple target horizons.
# julia --project=. experiments\exp7_amortized_reuse.jl --smoke
# julia --project=. experiments\exp7_amortized_reuse.jl [N_train] [targets_csv] [K] [updates]

using PEPDesign
using TOML
using Printf
using LinearAlgebra
include(joinpath(@__DIR__, "design_data.jl"))

function cold_start(N, K, h)
    return [k == n ? h : 0.0 for n in 1:N for k in max(1, n - K):n]
end

function certified_eval(dp, omega)
    W, sol, eta = pep_value(dp, omega)
    sol.diagnostics.certified ||
        error("uncertified PEP at N=$(dp.N): $(sol.diagnostics)")
    return (; W, sol, eta)
end

function main(args)
    smoke = args == ["--smoke"]
    smoke || length(args) <= 4 ||
        throw(ArgumentError("expected [N_train] [targets_csv] [K] [updates]"))
    Ntrain = smoke ? 3 : isempty(args) ? 10 : parse(Int, args[1])
    targets = smoke ? [4, 5] : length(args) >= 2 ?
        parse.(Int, split(args[2], ",")) : [Ntrain, Ntrain + 5, Ntrain + 10]
    K = smoke ? 1 : length(args) >= 3 ? parse(Int, args[3]) : min(2, Ntrain - 1)
    updates = smoke ? 3 : length(args) >= 4 ? parse(Int, args[4]) : 15
    epsilon = 0.3
    all(>(Ntrain), targets) || throw(ArgumentError("targets must exceed N_train"))
    length(unique(targets)) == length(targets) ||
        throw(ArgumentError("targets must be unique"))
    0 <= K < Ntrain || throw(ArgumentError("require 0 <= K < N_train"))
    updates >= 0 || throw(ArgumentError("updates must be nonnegative"))
    cap = smoke ? 60.0 : 300.0
    compiler(N) = compile_igdm(N, 1.0, epsilon; objective = :min_grad,
                               encoding = :substitution)
    BLAS.set_num_threads(1)

    # Warm up all relevant solver/model paths before timing.
    warm_cp = compiler(3)
    warm_policy = IGDMStationaryPolicy(1)
    design_slm(DesignProblem(warm_cp, warm_policy, 3), [1.0, 0.0];
               iters = 1, require_certified = true)
    design_slm(DesignProblem(warm_cp, IGDMKMemoryPolicy(1), 3),
               cold_start(3, 1, 1.0); iters = 1, require_certified = true)
    design_slm(DesignProblem(warm_cp, IdentityPolicy(), 3), ones(6);
               iters = 1, require_certified = true)

    stationary = IGDMStationaryPolicy(K)
    omega0 = [1.0; zeros(K)]
    t = time()
    cp_train = compiler(Ntrain)
    source_compile_elapsed = time() - t
    dp_train = DesignProblem(cp_train, stationary, Ntrain)
    train_stats = Dict{Symbol,Any}()
    GC.gc()
    t = time()
    train_trace = design_slm(dp_train, omega0; iters = 100, max_time = cap,
                             stats = train_stats, require_certified = true)
    train_design_elapsed = time() - t
    omega_train, Wtrain = best_point(train_trace)
    t = time()
    train_solution = certified_eval(dp_train, omega_train)
    train_verify_elapsed = time() - t
    train_elapsed = train_design_elapsed + train_verify_elapsed
    train_time_including_compile = train_elapsed + source_compile_elapsed
    target_rows = Dict{String,Any}[]
    compile_cache = Dict(Ntrain => cp_train)
    for M in targets
        compile_elapsed = 0.0
        cp = get!(compile_cache, M) do
            started = time()
            compiled = compiler(M)
            compile_elapsed = time() - started
            compiled
        end
        policy_dp = DesignProblem(cp, stationary, M)
        t = time()
        eta_transfer, _, _ = evaluate_policy(stationary, omega_train, M)
        transfer_solution = certified_eval(policy_dp, omega_train)
        transfer_elapsed = time() - t
        free_policy = IGDMKMemoryPolicy(K)
        free_dp = DesignProblem(cp, free_policy, M)
        omega_warm = [transfer_solution.eta[igdm_flat_index(n, k)]
                      for n in 1:M for k in max(1, n - K):n]
        start_diag = cold_start(M, K, 1.0)
        @assert length(omega_warm) == nparams(free_policy, M)
        common = Dict{String,Any}()

        t = time()
        warm_stats = Dict{Symbol,Any}()
        warm_trace = design_slm(free_dp, omega_warm; iters = updates, max_time = cap,
                                stats = warm_stats, require_certified = true)
        warm_design_elapsed = time() - t
        warm_omega, warm_best = best_point(warm_trace)
        t = time()
        warm_solution = certified_eval(free_dp, warm_omega)
        warm_verify_elapsed = time() - t
        warm_elapsed = warm_design_elapsed + warm_verify_elapsed

        t = time()
        cold_stats = Dict{Symbol,Any}()
        cold_trace = design_slm(free_dp, start_diag; iters = updates, max_time = cap,
                                stats = cold_stats, require_certified = true)
        cold_design_elapsed = time() - t
        cold_omega, cold_best = best_point(cold_trace)
        t = time()
        cold_solution = certified_eval(free_dp, cold_omega)
        cold_verify_elapsed = time() - t
        cold_elapsed = cold_design_elapsed + cold_verify_elapsed

        full_cold = Dict{String,Any}()
        full_cold_elapsed = 0.0
        full_cold_peps = 0
        full_cold_tangents = 0
        if smoke
            # Keep smoke inexpensive; small N makes the unrestricted cold baseline feasible.
            idp = DesignProblem(cp, IdentityPolicy(), M)
            t = time()
            full_trace = design_slm(idp, ones(cp.np); iters = updates, max_time = cap,
                                    require_certified = true)
            full_design_elapsed = time() - t
            full_omega, full_value = best_point(full_trace)
            t = time()
            full_sol = certified_eval(idp, full_omega)
            full_cold_elapsed = full_design_elapsed + (time() - t)
            full_cold = Dict("value" => full_value, "omega" => full_omega,
                             "diagnostics" => diagnostic_data(full_sol.sol))
            full_cold_peps = full_trace.nsolves + 1
            full_cold_tangents = full_trace.tangent_hist[end]
        end

        push!(target_rows, Dict(
            "N" => M, "compile_seconds" => compile_elapsed,
            "transfer_coefficients" => eta_transfer,
            "transfer_value" => transfer_solution.W,
            "warm_refine_start_omega" => omega_warm,
            "transfer_seconds" => transfer_elapsed,
            "transfer_diagnostics" => diagnostic_data(transfer_solution.sol),
            "warm_refine_start_value" => transfer_solution.W,
            "warm_refine_best" => warm_best, "warm_refine_seconds" => warm_elapsed,
            "warm_refine_design_seconds" => warm_design_elapsed,
            "warm_refine_verify_seconds" => warm_verify_elapsed,
            "warm_refine_trace" => trace_data(warm_trace),
            "warm_refine_pep_solves" => warm_trace.nsolves + 1,
            "warm_refine_tangent_solves" => warm_trace.tangent_hist[end],
            "warm_refine_diagnostics" => diagnostic_data(warm_solution.sol),
            "cold_free_best" => cold_best, "cold_free_seconds" => cold_elapsed,
            "cold_free_design_seconds" => cold_design_elapsed,
            "cold_free_verify_seconds" => cold_verify_elapsed,
            "cold_free_trace" => trace_data(cold_trace),
            "cold_free_pep_solves" => cold_trace.nsolves + 1,
            "cold_free_tangent_solves" => cold_trace.tangent_hist[end],
            "cold_free_diagnostics" => diagnostic_data(cold_solution.sol),
            "warm_refine_to_cold_ratio" => warm_best / cold_best,
            "transfer_to_cold_ratio" => transfer_solution.W / cold_best,
            "full_cold" => full_cold, "full_cold_seconds" => full_cold_elapsed,
            "full_cold_pep_solves" => full_cold_peps,
            "full_cold_tangent_solves" => full_cold_tangents))
    end

    cumulative = Dict{String,Any}[]
    sum_transfer = sum(r["transfer_seconds"] for r in target_rows)
    sum_warm = sum(r["warm_refine_seconds"] for r in target_rows)
    sum_cold = sum(r["cold_free_seconds"] for r in target_rows)
    sum_full = sum(r["full_cold_seconds"] for r in target_rows)
    for q in eachindex(target_rows)
        prefix = target_rows[1:q]
        push!(cumulative, Dict(
            "targets_served" => q,
            "transfer_only_cost" => train_time_including_compile +
                                    sum(r["transfer_seconds"] for r in prefix),
            "warm_refine_cost" => train_time_including_compile +
                                  sum(r["transfer_seconds"] + r["warm_refine_seconds"] for r in prefix),
            "cold_matched_memory_cost" => sum(r["cold_free_seconds"] for r in prefix),
            "cold_full_free_cost" => sum(r["full_cold_seconds"] for r in prefix)))
    end
    data = Dict(
        "config" => Dict("method" => "IGDM", "N_train" => Ntrain, "targets" => targets,
                         "memory_K" => K, "epsilon" => epsilon, "L" => 1.0, "D" => 1.0,
                         "objective" => "min_grad", "encoding" => "substitution",
                         "outer_updates" => updates, "soft_seconds_per_run" => cap,
                         "reuse_source_cost_includes_source_compilation" => true,
                                      "source_compile_seconds" => source_compile_elapsed,
                         "target_compilation_separate" => true,
                         "runtime" => "sequential; BLAS and Mosek 1 thread",
                         "julia_version" => string(VERSION)),
        "training" => Dict("omega" => omega_train, "W" => Wtrain,
                           "seconds" => train_elapsed,
                           "design_seconds" => train_design_elapsed,
                           "verify_seconds" => train_verify_elapsed,
                           "seconds_including_source_compile" => train_time_including_compile,
                           "trace" => trace_data(train_trace),
                           "diagnostics" => diagnostic_data(train_solution.sol)),
        "targets" => target_rows, "cumulative_cost" => cumulative,
        "interpretation" => Dict("quality" => "best-found local SLM values, not certified global optima",
                                 "cost_excludes" => "machine deployment of fixed coefficient arrays",
                                 "assessment_cost" => "offline PEP solves to certify worst-case values"))
    root = joinpath(@__DIR__, "results")
    mkpath(root)
    output = mktempdir(root; prefix = "amortized_IGDM_N$(Ntrain)_K$(K)_",
                       cleanup = false)
    path = joinpath(output, "amortized.toml")
    open(path * ".tmp", "w") do io
        TOML.print(io, data)
    end
    mv(path * ".tmp", path)
    println("Train once N=$Ntrain, K=$K: $(round(train_elapsed, digits=2))s")
    println("N   transfer W    warm free W   cold free W   t-reuse  t-warm  t-cold")
    for r in target_rows
        @printf("%2d  %.7f  %.7f  %.7f  %.2f  %.2f  %.2f\n",
                r["N"], r["transfer_value"], r["warm_refine_best"],
                r["cold_free_best"], r["transfer_seconds"],
                r["warm_refine_seconds"], r["cold_free_seconds"])
    end
    println("Targets assessed=$(length(target_rows)); compile time excluded at targets, source compile included.")
    println("Costs are offline; deployment of weights itself requires no SDP.")
    println("Saved: ", path)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
