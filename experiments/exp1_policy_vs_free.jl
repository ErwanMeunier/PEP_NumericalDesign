# Experiment 1 — Parameterized policies vs free step-sizes (Kamri baseline)
# under an equal SDP-solve budget.
#
# For each method and horizon: run every policy family and the unrestricted
# free baseline with identical multi-start budgets; record the best-so-far
# worst-case value vs cumulative SDP solves.
#
# Usage:  julia --project=. --threads=4 PEPDesign/experiments/exp1_policy_vs_free.jl [--smoke]

include(joinpath(@__DIR__, "common.jl"))

const SMOKE = is_smoke()
const CFG = (
    methods = SMOKE ? [:OGD] : [:OGD, :ITEM, :IGDM],
    Ns = SMOKE ? [5] : [10, 20, 30, 40],
    L = 1.0, D = 1.0, μ = 0.5, ε = 0.3,
    nstarts = SMOKE ? 2 : 3,
    iters_fom = SMOKE ? 15 : 120,
    iters_som = SMOKE ? 8 : 40,
    seed = 42,
)
@info "exp1 config" CFG SMOKE

setup_plots!()

for method in CFG.methods
    compiler = method_compiler(method; CFG.L, CFG.D, CFG.μ, CFG.ε)
    for N in CFG.Ns
        # ITEM at N=40 is heavy locally; cluster phase1 covers it
        method == :ITEM && N > 30 && !SMOKE && continue
        @info "exp1: $method N=$N (compile)"
        cp = compiler(N)
        suite = policy_suite(method, N; CFG.L, CFG.D, CFG.μ, CFG.ε,
                             nstarts = CFG.nstarts,
                             rng = Random.Xoshiro(CFG.seed))
        curves = Dict{String,Any}()
        rows = Vector{Vector{String}}()
        plt = plot(; xlabel = "SDP solves", ylabel = L"best $W$ so far",
                   title = "$method, N=$N", yscale = :log10, legend = :topright)
        for (i, (label, pol, starts)) in enumerate(suite)
            dp = DesignProblem(cp, pol, N)
            t = @elapsed solves, bests, ω_best, W_best, per_opt =
                budget_run(dp; starts, iters_fom = CFG.iters_fom,
                           iters_som = CFG.iters_som)
            κ = length(starts[1])
            curves[label] = Dict("solves" => solves, "best" => bests,
                                 "omega_best" => ω_best, "W_best" => W_best,
                                 "kappa" => κ, "W_fom" => per_opt[:fom],
                                 "W_som" => per_opt[:som], "walltime" => t)
            push!(rows, [label, string(κ),
                         (@sprintf "%.6f" W_best),
                         (@sprintf "%.6f" per_opt[:fom]),
                         (@sprintf "%.6f" per_opt[:som]),
                         string(solves[end]), (@sprintf "%.1f" t)])
            plot!(plt, solves, max.(bests, 1e-12); label = label,
                  color = PALETTE[i], linestyle = label == "free" ? :dash : :solid,
                  linewidth = label == "free" ? 3 : 2)
            @info "exp1: $method N=$N $label" W_best solves = solves[end] t
        end
        save_result("exp1_$(method)_N$(N)", CFG,
                    Dict{String,Any}("curves" => curves, "N" => N,
                                     "method" => string(method)))
        save_figure(plt, "exp1_budget_$(method)_N$(N)")
        write_latex_table("exp1_$(method)_N$(N)",
            ["Policy", "\$\\kappa\$", "\$W_{\\min}\$", "\$W\$ (FOM)",
             "\$W\$ (SOM)", "solves", "time (s)"], rows;
            caption = "Equal-budget policy comparison, $(method), \$N=$(N)\$.",
            label = "tab:exp1_$(method)_N$(N)")
    end
end

println("exp1: done")
