# Phase 0 — precompute W*_N per (method, N) with the free baseline.
# SLURM array task = one (method, N) pair; idempotent (skips existing output).
# Usage: julia --project=. --threads=$SLURM_CPUS_PER_TASK PEPDesign/cluster/phase0_wstar.jl [task_id | --local-all]

include(joinpath(@__DIR__, "registry.jl"))

function free_start(method, N)
    method == :ITEM && return begin
        βs, δs, _ = item_optimal_params(N, GRID.L, GRID.μ)
        [[βs; δs], [βs; δs] .* 1.15]
    end
    method == :IGDM && return begin
        h = igdm_hmem(GRID.ε) / GRID.L
        base = [n == k ? h : 0.02 for n in 1:N for k in 1:n]
        [base, base .* 1.2]
    end
    return [fill(1 / sqrt(N), N), fill(1.3 / sqrt(N), N)]
end

free_policy(method) = method == :ITEM ?
    ProductPolicy(IdentityPolicy(), IdentityPolicy()) : IdentityPolicy()

run_tasks(phase0_tasks()) do (method, N)
    dest = out_path("wstar_$(method)_N$(N).jld2")
    isfile(dest) && (@info "phase0: skip (exists)" dest; return)
    t = @elapsed W = compute_wstar(compiler_for(method), [N];
                                   policy = free_policy(method),
                                   ω0_fn = M -> free_start(method, M),
                                   iters = GRID.wstar_iters)
    jldsave(dest; method = string(method), N, Wstar = W[N], walltime = t,
            grid = repr(GRID))
    @info "phase0: done" method N Wstar = W[N] t
end
println("phase0: complete")
