# Phase 2 — HRDP robust design + anytime transfer curves per (method, policy).
# Requires phase 0 outputs (W*_N). Idempotent.
# Usage: julia --project=. --threads=$SLURM_CPUS_PER_TASK PEPDesign/cluster/phase2_hrdp.jl [task_id | --local-all]

include(joinpath(@__DIR__, "registry.jl"))

function load_wstar(method, horizons)
    W = Dict{Int,Float64}()
    for N in horizons
        f = out_path("wstar_$(method)_N$(N).jld2")
        isfile(f) || error("missing phase0 output $f — run phase 0 first")
        W[N] = load(f, "Wstar")
    end
    return W
end

run_tasks(phase2_tasks()) do (method, pidx)
    N_base = GRID.H_train_max
    suite = suite_for(method, N_base)
    label, pol, starts = suite[pidx]
    label == "free" && (@info "phase2: skip free baseline (not fixed-dim)"; return)
    dest = out_path("hrdp_$(method)_p$(pidx)_$(label).jld2")
    isfile(dest) && (@info "phase2: skip (exists)" dest; return)

    H_train = 2:N_base
    H_test = (N_base + 1):(N_base + GRID.iota_max)
    H_all = 2:(N_base + GRID.iota_max)
    Wstar = load_wstar(method, H_all)
    compiler = compiler_for(method)

    t = @elapsed begin
        ho_train = HRDPObjective(compiler, pol, H_train, Wstar)
        ho_all = HRDPObjective(compiler, pol, H_all, Wstar)
        ω_H, σ_train = best_point(design_som(ho_train, starts[1];
                                             iters = GRID.iters_som))
        dpN = DesignProblem(compiler(N_base), pol, N_base)
        ω_N, _ = best_point(design_som(dpN, starts[1]; iters = GRID.iters_som))
        tailored = Float64[]; transfer = Float64[]; reuse = Float64[]
        for M in H_test
            # policy suites at horizon M give the tailored starts
            sM = suite_for(method, M)[pidx][3][1]
            dpM = DesignProblem(compiler(M), pol, M)
            ω_M, W_M = best_point(design_som(dpM, sM; iters = GRID.iters_som))
            push!(tailored, W_M / Wstar[M])
            push!(transfer, pep_value(dpM, ω_H)[1] / Wstar[M])
            push!(reuse, pep_value(dpM, ω_N)[1] / Wstar[M])
        end
        ω_all, _ = best_point(design_som(ho_all, ω_H; iters = GRID.iters_som))
        gr_val = generalization_ratio(ho_all, ω_H, ω_all)
        wgc_val = wgc(ho_all, ω_H)
        jldsave(dest; method = string(method), policy = label, pidx,
                omega_H = ω_H, omega_N = ω_N, sigma_train = σ_train,
                H_test = collect(H_test), tailored, transfer, reuse,
                GR = gr_val, WGC = wgc_val, grid = repr(GRID))
    end
    @info "phase2: done" method label t
end
println("phase2: complete")
