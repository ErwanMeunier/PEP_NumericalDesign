# Experiment 4 — Second-order model diagnostics (paper §"Second-Order Model
# Diagnostics"): SSDP predicted-vs-actual merit reduction, convexification
# shifts ζ_t, accepted steps γ_t, KKT residuals; solver comparison SD / SOM /
# SSDP from identical starts; matrix-derivative FD checks; timing splits.
#
# Usage:  julia --project=. --threads=4 PEPDesign/experiments/exp4_ssdp_diagnostics.jl [--smoke]

include(joinpath(@__DIR__, "common.jl"))

const SMOKE = is_smoke()
const CFG = (
    N = SMOKE ? 3 : 5,
    L = 1.0, D = 1.0,
    iters = SMOKE ? 10 : 30,
    seed = 42,
)
@info "exp4 config" CFG SMOKE

setup_plots!()
N = CFG.N
cp = compile_ogd(N, CFG.L, CFG.D)

# ── (1) matrix-derivative directional FD errors (exact for degree ≤ 2) ───────
Random.seed!(CFG.seed)
η0 = 0.1 .+ 0.4 .* rand(N)
h = 1e-3
err1 = 0.0
err2 = 0.0
for con in cp.cons
    M = con.expr.M
    PEPDesign.is_param_dependent(M) || continue
    for r in 1:N
        e = zeros(N); e[r] = h
        global err1 = max(err1,
            norm((assemble(M, η0 .+ e) .- assemble(M, η0 .- e)) ./ (2h) .-
                 dmat(M, r, η0), Inf))
        for s in r:N
            es = zeros(N); es[s] = h
            global err2 = max(err2,
                norm((dmat(M, r, η0 .+ es) .- dmat(M, r, η0 .- es)) ./ (2h) .-
                     d2mat(M, r, s), Inf))
        end
    end
end
@info "exp4: matrix FD errors" err1 err2

# ── (2) ∇W vs FD away from kinks ─────────────────────────────────────────────
sol0 = solve_pep(cp, η0)
g0, _ = grad_hess_eta(cp, η0, sol0)
gfd = zeros(N)
for r in 1:N
    e = zeros(N); e[r] = 1e-4
    gfd[r] = (solve_pep(cp, η0 .+ e).obj - solve_pep(cp, η0 .- e).obj) / 2e-4
end
graderr = norm(g0 - gfd, Inf)
@info "exp4: gradient FD error" graderr

# ── (3) solver comparison from identical starts ───────────────────────────────
pol = ConstantPolicy()
dp = DesignProblem(cp, pol, N)
ω0 = [0.5 * CFG.D / (CFG.L * sqrt(N))]

t_fom = @elapsed tr_fom = design_fom(dp, ω0; iters = CFG.iters,
                                     steps = t -> 0.1 / sqrt(t))
t_som = @elapsed tr_som = design_som(dp, ω0; iters = CFG.iters)
t_ssdp = @elapsed tr_ssdp = design_ssdp(dp, ω0; iters = CFG.iters)

rows = [
    ["SD", (@sprintf "%.6f" best_point(tr_fom)[2]), string(tr_fom.nsolves),
     (@sprintf "%.2f" t_fom)],
    ["Damped Newton", (@sprintf "%.6f" best_point(tr_som)[2]),
     string(tr_som.nsolves), (@sprintf "%.2f" t_som)],
    ["SSDP", (@sprintf "%.6f" tr_ssdp.W_final),
     string(tr_ssdp.ntangent) * " (tangent)", (@sprintf "%.2f" t_ssdp)],
]
write_latex_table("exp4_solver_comparison",
    ["Solver", "final \$W\$", "solves", "time (s)"], rows;
    caption = "Design-solver comparison on the OGD constant policy, " *
              "\$N=$(N)\$, identical starts.",
    label = "tab:exp4_solvers")

# ── (4) SSDP per-iteration diagnostics ────────────────────────────────────────
T = length(tr_ssdp.γ_hist)
plt = plot(; xlabel = "SSDP iteration", yscale = :log10,
           title = "SSDP diagnostics (OGD, constant policy, N=$N)",
           legend = :bottomleft)
plot!(plt, 1:T, max.(tr_ssdp.pred_hist, 1e-16); label = "predicted reduction",
      marker = :circle, markersize = 3)
plot!(plt, 1:T, max.(tr_ssdp.act_hist, 1e-16); label = "actual reduction",
      marker = :diamond, markersize = 3)
plot!(plt, 1:T, max.(tr_ssdp.ζ_hist, 1e-16); label = L"\zeta_t",
      linestyle = :dash)
plot!(plt, 1:T, tr_ssdp.γ_hist; label = L"\gamma_t", linestyle = :dot)
plot!(plt, 1:length(tr_ssdp.kkt_hist), max.(tr_ssdp.kkt_hist, 1e-16);
      label = "KKT residual")
save_figure(plt, "exp4_ssdp_diagnostics")

# ── (5) timing split: SDP solve vs derivative assembly vs tangent SDP ─────────
η, _, _ = evaluate_policy(pol, ω0, N)
t_solve = @elapsed for _ in 1:5
    solve_pep(cp, η)
end
s = solve_pep(cp, η)
t_deriv = @elapsed for _ in 1:5
    grad_hess_eta(cp, η, s)
end
t_tangent = t_ssdp / max(tr_ssdp.ntangent, 1)
write_latex_table("exp4_timing",
    ["Component", "time (s)"],
    [["PEP-SDP solve", (@sprintf "%.4f" t_solve / 5)],
     ["derivative assembly", (@sprintf "%.6f" t_deriv / 5)],
     ["SSDP tangent subproblem", (@sprintf "%.4f" t_tangent)]];
    caption = "Per-call timing split, OGD \$N=$(N)\$: the SDP oracle " *
              "dominates; sensitivity contractions are negligible.",
    label = "tab:exp4_timing")

save_result("exp4_diagnostics", CFG, Dict{String,Any}(
    "matrix_fd_err1" => err1, "matrix_fd_err2" => err2,
    "grad_fd_err" => graderr,
    "ssdp" => Dict("pred" => tr_ssdp.pred_hist, "act" => tr_ssdp.act_hist,
                   "zeta" => tr_ssdp.ζ_hist, "gamma" => tr_ssdp.γ_hist,
                   "kkt" => tr_ssdp.kkt_hist, "obj" => tr_ssdp.obj_hist,
                   "W_final" => tr_ssdp.W_final),
    "timing" => Dict("solve" => t_solve / 5, "deriv" => t_deriv / 5,
                     "tangent" => t_tangent)))

println("exp4: done")
