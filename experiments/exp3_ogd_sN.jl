# Experiment 3 — OGD 1/√N conjecture: the scaling statistic
# s_N = (L√N/D)·c_N for the optimized constant policy, and its gap to the
# unrestricted free design under the same budget.
#
# Usage:  julia --project=. --threads=4 PEPDesign/experiments/exp3_ogd_sN.jl [--smoke]

include(joinpath(@__DIR__, "common.jl"))

const SMOKE = is_smoke()
const CFG = (
    Ns = SMOKE ? [3, 5] : [5, 10, 15, 20, 25, 30, 35, 40],
    L = 1.0, D = 1.0,
    iters = SMOKE ? 12 : 40,
    seed = 42,
)
@info "exp3 config" CFG SMOKE

setup_plots!()

sN = Float64[]
Wc = Float64[]
Wfree = Float64[]
rows = Vector{Vector{String}}()
for N in CFG.Ns
    cp = compile_ogd(N, CFG.L, CFG.D)
    c0 = CFG.D / (CFG.L * sqrt(N))

    dp_c = DesignProblem(cp, ConstantPolicy(), N)
    ω_c, W_c = best_point(design_som(dp_c, [0.8 * c0]; iters = CFG.iters))
    W_c2 = best_point(design_fom(dp_c, [1.3 * c0]; iters = CFG.iters,
                                 steps = t -> 0.1 / sqrt(t)))
    W_c2[2] < W_c && ((ω_c, W_c) = W_c2)

    dp_f = DesignProblem(cp, IdentityPolicy(), N)
    _, W_f = best_point(design_som(dp_f, fill(c0, N); iters = CFG.iters))
    W_f2 = best_point(design_fom(dp_f, fill(1.3 * c0, N); iters = CFG.iters,
                                 steps = t -> 0.1 / sqrt(t)))
    W_f = min(W_f, W_f2[2])

    s = CFG.L * sqrt(N) / CFG.D * ω_c[1]
    push!(sN, s)
    push!(Wc, W_c)
    push!(Wfree, W_f)
    gap = (W_c - W_f) / W_f
    push!(rows, [string(N), (@sprintf "%.5f" ω_c[1]), (@sprintf "%.5f" s),
                 (@sprintf "%.6f" W_c), (@sprintf "%.6f" W_f),
                 (@sprintf "%.2e" gap), (@sprintf "%.5f" W_c / sqrt(N))])
    @info "exp3: N=$N" c_N = ω_c[1] s_N = s W_const = W_c W_free = W_f gap
end

save_result("exp3_ogd_sN", CFG,
            Dict{String,Any}("Ns" => collect(CFG.Ns), "sN" => sN,
                             "W_const" => Wc, "W_free" => Wfree))

plt = plot(collect(CFG.Ns), sN; xlabel = L"N",
           ylabel = L"s_N = \frac{L\sqrt{N}}{D}\,c_N", marker = :circle,
           label = L"s_N", title = "OGD constant-policy scaling statistic",
           ylim = (0.8, 1.1))
hline!(plt, [1.0]; linestyle = :dash, color = :black,
       label = L"conjecture $s_N = 1$")
save_figure(plt, "exp3_ogd_sN")

write_latex_table("exp3_ogd_sN",
    ["\$N\$", "\$c_N\$", "\$s_N\$", "\$W(c_N)\$", "\$W_{\\mathrm{free}}\$",
     "rel.\\ gap", "\$W/\\sqrt{N}\$"], rows;
    caption = "Optimized constant OGD step size vs the \$D/(L\\sqrt{N})\$ " *
              "conjecture and the unrestricted design (\$L=D=1\$).",
    label = "tab:exp3_ogd_sN")

println("exp3: done")
