# Repeat source-horizon training from small diagonal-step perturbations and
# assess whether warm-start advantages persist at longer horizons.
# julia --project=. experiments\exp8_transfer_repeatability.jl --smoke
# julia --project=. experiments\exp8_transfer_repeatability.jl 10 15,20 2 8

using PEPDesign
using TOML
using Printf
using LinearAlgebra
include(joinpath(@__DIR__, "design_data.jl"))

function band_start(N, K, h)
    return [k == n ? h : 0.0 for n in 1:N for k in max(1, n - K):n]
end

function checked(dp, omega)
    W, sol, eta = pep_value(dp, omega)
    sol.diagnostics.certified ||
        error("uncertified PEP at N=$(dp.N): $(sol.diagnostics)")
    return (; W, sol, eta)
end

function persist(path, config, cold, runs)
    temp = path * ".tmp"
    open(temp, "w") do io
        TOML.print(io, Dict("config" => config, "cold_baselines" => cold, "runs" => runs))
    end
    mv(temp, path; force = true)
end

function main(args)
    smoke = args == ["--smoke"]
    smoke || length(args) <= 4 ||
        throw(ArgumentError("expected [N_train] [targets_csv] [K] [updates]"))
    Ntrain = smoke ? 3 : isempty(args) ? 10 : parse(Int, args[1])
    targets = smoke ? [4, 5] : length(args) >= 2 ?
        parse.(Int, split(args[2], ",")) : [Ntrain + 5, Ntrain + 10]
    K = smoke ? 1 : length(args) >= 3 ? parse(Int, args[3]) : min(2, Ntrain - 1)
    updates = smoke ? 3 : length(args) >= 4 ? parse(Int, args[4]) : 8
    Ntrain > K >= 0 || throw(ArgumentError("require N_train > K >= 0"))
    all(>(Ntrain), targets) || throw(ArgumentError("target horizons must exceed N_train"))
    length(unique(targets)) == length(targets) || throw(ArgumentError("duplicate target horizon"))
    epsilon, cap = 0.3, smoke ? 60.0 : 300.0
    starts = smoke ? [1.0] : [0.9, 1.0, 1.1]
    BLAS.set_num_threads(1)
    compiler(N) = compile_igdm(N, 1.0, epsilon; objective = :min_grad,
                               encoding = :substitution)
    small = compiler(3)
    warm_policy = IGDMStationaryPolicy(1)
    design_slm(DesignProblem(small, warm_policy, 3), [1.0, 0.0];
               iters = 1, require_certified = true)
    design_slm(DesignProblem(small, IGDMKMemoryPolicy(1), 3), band_start(3, 1, 1.0);
               iters = 1, require_certified = true)
    design_slm(DesignProblem(small, IdentityPolicy(), 3), ones(small.np);
               iters = 1, require_certified = true)

    stationary = IGDMStationaryPolicy(K)
    free_policy = IGDMKMemoryPolicy(K)
    t = time()
    cp_train = compiler(Ntrain)
    source_compile_seconds = time() - t
    train_dp = DesignProblem(cp_train, stationary, Ntrain)
    cp_cache = Dict(Ntrain => cp_train)
    getcp(N) = get!(cp_cache, N) do
        compiler(N)
    end
    cold = Dict{String,Any}[]
    runs = Dict{String,Any}[]
    root = joinpath(@__DIR__, "results")
    mkpath(root)
    out = mktempdir(root; prefix = "transfer_repeat_N$(Ntrain)_K$(K)_",
                    cleanup = false)
    path = joinpath(out, "repeatability.toml")
    config = Dict("method" => "IGDM", "N_train" => Ntrain, "targets" => targets,
                  "K" => K, "epsilon" => epsilon, "objective" => "min_grad",
                  "encoding" => "substitution", "source_initial_diagonal_steps" => starts,
                  "target_cold_start_diagonal_step" => 1.0, "updates" => updates,
                  "soft_seconds_per_run" => cap, "BLAS_threads" => 1,
                  "Mosek_threads" => 1, "source_compile_seconds" => source_compile_seconds,
                  "julia_version" => string(VERSION),
                  "cold_baseline" => "one K-band SLM from h=1, paired with each source fit")
    persist(path, config, cold, runs)

    # Cold baselines do not depend on the source fit. Solve once per target.
    for M in targets
        cp = getcp(M)
        dp = DesignProblem(cp, free_policy, M)
        t = time()
        tr = design_slm(dp, band_start(M, K, 1.0); iters = updates,
                        max_time = cap, require_certified = true)
        design_seconds = time() - t
        best_omega, best = best_point(tr)
        t = time()
        sol = checked(dp, best_omega)
        verify_seconds = time() - t
        push!(cold, Dict("N" => M, "best" => best, "omega" => best_omega,
                         "seconds" => design_seconds + verify_seconds,
                         "design_seconds" => design_seconds,
                         "verify_seconds" => verify_seconds,
                         "trace" => trace_data(tr),
                         "diagnostics" => diagnostic_data(sol.sol)))
        persist(path, config, cold, runs)
    end

    for h in starts
        start = [h; zeros(K)]
        t = time()
        train_stats = Dict{Symbol,Any}()
        tr_train = design_slm(train_dp, start; iters = 100, max_time = cap,
                              stats = train_stats, require_certified = true)
        train_design_seconds = time() - t
        omega, train_best = best_point(tr_train)
        t = time()
        train_sol = checked(train_dp, omega)
        train_verify_seconds = time() - t
        for M in targets
            cp = getcp(M)
            stationary_dp = DesignProblem(cp, stationary, M)
            t = time()
            transferred = checked(stationary_dp, omega)
            transfer_seconds = time() - t
            free_dp = DesignProblem(cp, free_policy, M)
            warm_omega = [transferred.eta[igdm_flat_index(n, k)]
                          for n in 1:M for k in max(1, n - K):n]
            t = time()
            tr_warm = design_slm(free_dp, warm_omega; iters = updates,
                                 max_time = cap, require_certified = true)
            warm_design_seconds = time() - t
            warm_best_omega, warm_best = best_point(tr_warm)
            t = time()
            warm_sol = checked(free_dp, warm_best_omega)
            warm_verify_seconds = time() - t
            baseline = only(filter(r -> r["N"] == M, cold))
            push!(runs, Dict(
                "source_initial_step" => h, "N_train" => Ntrain, "N" => M,
                "source_omega" => omega, "source_best" => train_best,
                "source_seconds" => train_design_seconds + train_verify_seconds,
                "source_design_seconds" => train_design_seconds,
                "source_verify_seconds" => train_verify_seconds,
                "source_diagnostics" => diagnostic_data(train_sol.sol),
                "transfer_value" => transferred.W, "transfer_seconds" => transfer_seconds,
                "transfer_diagnostics" => diagnostic_data(transferred.sol),
                "warm_start_free_value" => transferred.W,
                "warm_best" => warm_best,
                "warm_seconds" => warm_design_seconds + warm_verify_seconds,
                "warm_design_seconds" => warm_design_seconds,
                "warm_verify_seconds" => warm_verify_seconds,
                "warm_trace" => trace_data(tr_warm),
                "warm_diagnostics" => diagnostic_data(warm_sol.sol),
                "cold_best" => baseline["best"],
                "cold_seconds" => baseline["seconds"],
                "cold_trace" => baseline["trace"],
                "reuse_to_cold_ratio" => transferred.W / baseline["best"],
                "warm_to_cold_ratio" => warm_best / baseline["best"],
                "warm_advantage" => (baseline["best"] - warm_best) / baseline["best"]))
            persist(path, config, cold, runs)
            @printf("source h=%.1f N=%d Wreuse=%.7f Wwarm=%.7f Wcold=%.7f advantage=%+.2f%%\n",
                    h, M, transferred.W, warm_best, baseline["best"],
                    100 * (baseline["best"] - warm_best) / baseline["best"])
            flush(stdout)
        end
    end
    println("Saved source-fit repeatability data: ", path)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
