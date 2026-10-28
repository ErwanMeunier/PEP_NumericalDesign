# Compute-only, sequential paired screening. Plot with plot_memory_study.jl.
using PEPDesign
using LinearAlgebra
using Random
using TOML
using Printf
include(joinpath(@__DIR__, "design_data.jl"))

function memory_case(label, N, K, h)
    if label == "free"
        pol = IGDMKMemoryPolicy(K)
        omega = [k == n ? h : 0.0 for n in 1:N for k in max(1, n - K):n]
    elseif label == "stationary"
        pol = IGDMStationaryPolicy(K)
        omega = [h; zeros(K)]
    elseif label in ("lag_dct1", "lag_dct3")
        degree = label == "lag_dct1" ? 1 : 3
        pol = IGDMLagPolicy(K, dct_policy(degree))
        omega = zeros(nparams(pol, N))
        omega[1] = h
    else
        throw(ArgumentError("unknown memory-study policy '$label'"))
    end
    return pol, omega
end

function save_study(path, config, runs)
    open(path * ".tmp", "w") do io
        TOML.print(io, Dict("config" => config, "runs" => runs))
    end
    mv(path * ".tmp", path; force = true)
end

function main(args)
    labels = ["free", "stationary", "lag_dct1", "lag_dct3"]
    if args == ["--smoke"]
        stage, cells, steps, iters, cap = "smoke", [(4, 1)], [1.0], 3, 60.0
    elseif isempty(args) || args == ["--screen"]
        stage = "screen"
        cells = [(N, K) for N in (10, 20) for K in (0, 1, 2, 4, N - 1)]
        steps, iters, cap = [1.0], 16, 120.0
    elseif length(args) == 4 && args[1] == "--confirm"
        N, K = parse.(Int, args[2:3])
        1 <= K < N || throw(ArgumentError("confirmation needs 1 <= K < N"))
        args[4] in labels[2:end] || throw(ArgumentError("choose a prior label"))
        labels = ["free", args[4]]
        stage, cells, steps, iters, cap = "confirm", [(N, K), (N, 0)], [1.0, 0.9, 1.1], 40, 120.0
    else
        throw(ArgumentError("use --smoke, --screen, or --confirm N K prior_label"))
    end
    BLAS.set_num_threads(1)
    compiler = N -> compile_igdm(N, 1.0, 0.3; objective = :min_grad, encoding = :substitution)
    # Exclude Julia specialization from measured comparisons, for every policy type.
    small = compiler(3)
    for label in labels
        pol, omega = memory_case(label, 3, 1, 1.0)
        design_slm(DesignProblem(small, pol, 3), omega; iters = 1,
                   require_certified = true)
    end
    root = joinpath(@__DIR__, "results")
    mkpath(root)
    output = mktempdir(root; prefix = "memory_$(stage)_", cleanup = false)
    path = joinpath(output, "study.toml")
    config = Dict("stage" => stage, "epsilon" => 0.3, "L" => 1.0, "D" => 1.0,
                  "objective" => "min_grad", "encoding" => "substitution",
                  "cells" => [[N, K] for (N, K) in cells], "initial_steps" => steps,
                  "labels" => labels, "iterations" => iters, "soft_seconds" => cap,
                  "julia_version" => string(VERSION), "blas_threads" => 1,
                  "mosek_threads" => 1, "jit_warmup" => true,
                  "seed" => 20261019, "require_certified" => true)
    runs = Dict{String,Any}[]
    cache = Dict{Int,CompiledPEP}()
    rng = Random.Xoshiro(20261019)
    println("Saving incrementally to ", path)
    flush(stdout)
    for (N, K) in cells
        cp = get!(cache, N) do
            compiler(N)
        end
        for (rep, h) in enumerate(steps)
            reference = [k == n ? h : 0.0 for n in 1:N for k in 1:n]
            for label in shuffle(rng, labels)
                pol, omega = memory_case(label, N, K, h)
                eta, J, _ = evaluate_policy(pol, omega, N)
                isapprox(eta, reference; atol = 1e-12, rtol = 0) ||
                    error("initial coefficient mismatch for $label")
                dp = DesignProblem(cp, pol, N)
                stats = Dict{Symbol,Any}()
                GC.gc()
                tr = design_slm(dp, omega; iters, max_time = cap, stats,
                                require_certified = true)
                final_omega, best = best_point(tr)
                verified, sol, _ = pep_value(dp, final_omega)
                sol.diagnostics.certified || error("uncertified final value for $label")
                isapprox(verified, best; atol = 1e-7, rtol = 1e-5) ||
                    error("final verification differs from trace for $label")
                push!(runs, Dict("N" => N, "K" => K, "replicate" => rep, "initial_step" => h,
                                 "policy" => label, "parameters" => length(omega),
                                 "effective_rank" => rank(J),
                                 "trace" => trace_data(tr), "diagnostics" => diagnostic_data(sol),
                                 "verified_value" => verified, "verification_pep_solves" => 1,
                                 "stats" => Dict(string(k) => v for (k, v) in stats)))
                save_study(path, config, runs)
                @printf("N=%d K=%d rep=%d %-11s best=%.8f T=%d t=%.2fs accepted=%d\n",
                        N, K, rep, label, best, length(tr.values) - 1, last(tr.time_hist),
                        count(identity, stats[:accepted]))
                flush(stdout)
            end
        end
    end
    println("Saved ", path)
    return path
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
