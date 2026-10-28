# Standalone compute-only experiment; no plotting dependencies.
# julia --project=. experiments\exp5_horizon_reuse.jl [N_train] [targets_csv] [K] [base]
# Or use --smoke for N_train=3, targets=4,5, K=1.

using PEPDesign
using LinearAlgebra
using Printf
using TOML
include(joinpath(@__DIR__, "design_data.jl"))

function stationary_base(name, K)
    if name == "identity"
        return IdentityPolicy(), [1.0; zeros(K)]
    elseif name == "constant"
        return ConstantPolicy(), [1.0 / (K + 1)]
    elseif name == "power_law"
        return PowerLawPolicy(), [1.0, 2.0, 0.0]
    elseif name == "sum_of_exp"
        return sum_of_exp_policy(1), [1.0, 1.0, 0.0]
    elseif name == "log_poly"
        return log_poly_policy(2), [0.0, -2.0, 0.0]
    elseif name == "dct"
        base = dct_policy(min(3, K))
        _, J, _ = evaluate_policy(base, zeros(nparams(base, K + 1)), K + 1)
        return base, J \ [1.0; zeros(K)]
    elseif name == "cosine_decay"
        return cosine_decay_policy(), [0.5, 1.0, 0.0]
    elseif name == "rational"
        return rational_policy(), [1.0, 0.0, 0.0, 0.1]
    elseif name == "warped_chebyshev"
        return warped_chebyshev_policy(min(3, K)), [1.0; zeros(min(3, K))]
    elseif name == "piecewise_exp"
        return piecewise_exp_policy(), [1.0, 0.5, 0.5, 0.5]
    end
    throw(ArgumentError("unknown stationary base '$name'; see docs/experiments.md"))
end

function main(args)
    smoke = args == ["--smoke"]
    smoke || length(args) <= 4 ||
        throw(ArgumentError("expected [N_train] [targets_csv] [K] [base]"))
    N = smoke ? 3 : isempty(args) ? 10 : parse(Int, args[1])
    targets = smoke ? [4, 5] : length(args) >= 2 ?
        parse.(Int, split(args[2], ",")) : [N + 1, N + 5]
    K = smoke ? 1 : length(args) >= 3 ? parse(Int, args[3]) : min(4, N - 1)
    name = smoke || length(args) < 4 ? "identity" : args[4]
    N >= 1 && 0 <= K < N ||
        throw(ArgumentError("require N_train >= 1 and 0 <= K < N_train"))
    all(>(N), targets) && length(unique(targets)) == length(targets) ||
        throw(ArgumentError("provide distinct targets larger than N_train"))
    iters = smoke ? 8 : 100
    max_time = smoke ? 60.0 : 300.0
    epsilon = 0.3
    base, omega0 = stationary_base(name, K)
    policy = IGDMStationaryPolicy(K, base)
    compiler = M -> compile_igdm(M, 1.0, epsilon; objective = :min_grad,
                                 encoding = :substitution)
    t = time()
    train_dp = DesignProblem(compiler(N), policy, N)
    compile_seconds = time() - t
    training = design_slm(train_dp, omega0; iters, max_time, require_certified = true)
    omega, _ = best_point(training)
    free_starts(M) = [[k == n ? h : 0.0 for n in 1:M for k in max(1, n - K):n]
                      for h in (1.0, igdm_hmem(epsilon))]
    rows = assess_transfer(compiler, policy, omega, N, [N; targets];
                           free_policy = IGDMKMemoryPolicy(K),
                           free_starts, iters, max_time)
    data = Dict(
        "config" => Dict("N_train" => N, "targets" => targets, "memory" => K,
                        "base" => name, "epsilon" => epsilon, "L" => 1.0, "D" => 1.0,
                        "objective" => "min_grad", "encoding" => "substitution",
                        "free_comparator" => "fixed-memory K=$K, fresh two-start SLM",
                        "iters_per_start" => iters, "seconds_per_start_soft" => max_time,
                        "julia_version" => string(VERSION)),
        "training" => Dict("omega" => omega, "initial_omega" => omega0,
                           "compile_seconds" => compile_seconds,
                           "trace" => trace_data(training)),
        "assessment" => [
            Dict("N" => r.N, "reuse_value" => r.reuse_value,
                 "coefficients" => r.coefficients, "reuse_seconds" => r.reuse_seconds,
                 "reuse_diagnostics" => diagnostic_data(r.reuse_solution),
                 "compile_seconds" => r.compile_seconds,
                 "free_best_value" => r.free_best_value, "free_omega" => r.free_omega,
                 "free_diagnostics" => diagnostic_data(r.free_solution),
                 "free_seconds" => r.free_seconds, "free_pep_solves" => r.free_pep_solves,
                 "free_tangent_solves" => r.free_tangent_solves, "ratio" => r.ratio,
                 "free_traces" => [trace_data(run.trace) for run in r.free_runs])
            for r in rows])
    root = joinpath(@__DIR__, "results")
    mkpath(root)
    output = mktempdir(root; prefix = "reuse_IGDM_$(name)_N$(N)_", cleanup = false)
    open(joinpath(output, "assessment.toml"), "w") do io
        TOML.print(io, data)
    end
    println("Stationary base=$name, trained at N=$N, memory K=$K fixed")
    println("N  reused W       free best      ratio    reuse(s) free(s)")
    for r in rows
        @printf("%2d %.8f     %.8f     %.4f   %.3f    %.3f\n",
                r.N, r.reuse_value, r.free_best_value, r.ratio,
                r.reuse_seconds, r.free_seconds)
    end
    println("Source training (one-off): $(training.time_hist[end]) s")
    println("Values are numerically certified; free values are best found, not proven optima.")
    println("Saved: ", joinpath(output, "assessment.toml"))
    return data
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
