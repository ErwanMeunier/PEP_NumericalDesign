# Phase 1 — single-horizon budgeted designs per (method, policy, N); all
# multi-starts share the compiled PEP inside one task. Idempotent.
# Usage: julia --project=. --threads=$SLURM_CPUS_PER_TASK PEPDesign/cluster/phase1_designs.jl [task_id | --local-all]

include(joinpath(@__DIR__, "registry.jl"))

run_tasks(phase1_tasks()) do (method, pidx, N)
    suite = suite_for(method, N)
    label, pol, starts = suite[pidx]
    dest = out_path("design_$(method)_p$(pidx)_$(label)_N$(N).jld2")
    isfile(dest) && (@info "phase1: skip (exists)" dest; return)
    cp = compiler_for(method)(N)
    dp = DesignProblem(cp, pol, N)
    t = @elapsed solves, bests, ω_best, W_best, per_opt =
        budget_run(dp; starts, iters_fom = GRID.iters_fom,
                   iters_som = GRID.iters_som)
    jldsave(dest; method = string(method), policy = label, pidx, N,
            solves, bests, omega_best = ω_best, W_best,
            W_fom = per_opt[:fom], W_som = per_opt[:som],
            kappa = length(starts[1]), walltime = t, grid = repr(GRID))
    @info "phase1: done" method label N W_best t
end
println("phase1: complete")
