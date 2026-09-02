# Experiment 2 — HRDP anytime improvement: multi-horizon robust design vs
# tailored and reused single-horizon designs.
#
# For each method and horizon-consistent policy family:
#   ω_H  : HRDP minimizer over 𝓗 = {2..N_train}
#   ω_N  : tailored minimizer at the base horizon N_train
#   For test horizons M = N_train+1 .. N_train+ι_max, compare
#     tailored  W̄_M(ω_M) ≤ transfer W̄_M(ω_H) vs reuse W̄_M(ω_{N_train}).
# Reports GR (generalization ratio) and WGC.
#
# Usage:  julia --project=. --threads=4 PEPDesign/experiments/exp2_hrdp_anytime.jl [--smoke]

include(joinpath(@__DIR__, "common.jl"))

const SMOKE = is_smoke()
const CFG = (
    methods = SMOKE ? [:OGD] : [:OGD, :ITEM, :IGDM],
    N_train = SMOKE ? 5 : 20,
    iota_max = SMOKE ? 2 : 10,
    L = 1.0, D = 1.0, μ = 0.5, ε = 0.3,
    iters = SMOKE ? 10 : 40,
    wstar_iters = SMOKE ? 12 : 50,
    seed = 42,
)
@info "exp2 config" CFG SMOKE

setup_plots!()

# Horizon-consistent families only (paper Def. horizon-consistency)
function hrdp_policies(method)
    method == :OGD && return [("constant", ConstantPolicy(), N -> [1 / sqrt(N)]),
        ("power_law", PowerLawPolicy(), N -> [1.0, 0.5, 0.0]),
        ("log_poly2", log_poly_policy(2), N -> [0.0, -0.5, 0.0]),
        ("rational", rational_policy(), N -> [1 / sqrt(N), 0.0, 0.0, 0.0])]
    method == :ITEM && return [("constant", ProductPolicy(ConstantPolicy(), ConstantPolicy()),
         N -> begin
             βs, δs, _ = item_optimal_params(N, CFG.L, CFG.μ)
             [mean(βs), mean(δs)]
         end),
        ("power_law", ProductPolicy(PowerLawPolicy(), PowerLawPolicy()),
         N -> begin
             βs, δs, _ = item_optimal_params(N, CFG.L, CFG.μ)
             [mean(βs), 0.0, 0.0, mean(δs), 0.0, 0.0]
         end)]
    method == :IGDM && return [("diag_constant", igdm_diagonal_policy(ConstantPolicy()),
         N -> [igdm_hmem(CFG.ε) / CFG.L]),
        ("diag_power_law", igdm_diagonal_policy(PowerLawPolicy()),
         N -> [igdm_hmem(CFG.ε) / CFG.L, 0.0, 0.0])]
    error("unknown method")
end

for method in CFG.methods
    compiler = method_compiler(method; CFG.L, CFG.D, CFG.μ, CFG.ε)
    H_train = 2:CFG.N_train
    H_test = (CFG.N_train + 1):(CFG.N_train + CFG.iota_max)
    H_all = 2:(CFG.N_train + CFG.iota_max)

    # W*_N over train ∪ test with the free baseline (multistart FOM+SOM)
    free_pol, free_start = method == :ITEM ?
        (ProductPolicy(IdentityPolicy(), IdentityPolicy()),
         N -> begin
             βs, δs, _ = item_optimal_params(N, CFG.L, CFG.μ)
             [[βs; δs], [βs; δs] .* 1.15]
         end) :
        method == :IGDM ?
        (IdentityPolicy(),
         N -> begin
             h = igdm_hmem(CFG.ε) / CFG.L
             base = [n == k ? h : 0.02 for n in 1:N for k in 1:n]
             [base, base .* 1.2]
         end) :
        (IdentityPolicy(),
         N -> [fill(1 / sqrt(N), N), fill(1.3 / sqrt(N), N)])
    @info "exp2: $method — computing W*_N over $(H_all)"
    Wstar = compute_wstar(compiler, H_all; policy = free_pol,
                          ω0_fn = free_start, iters = CFG.wstar_iters)

    data = Dict{String,Any}("Wstar" => Dict(string(k) => v for (k, v) in Wstar))
    rows = Vector{Vector{String}}()
    plt = plot(; xlabel = L"test horizon $M$", ylabel = L"\overline{W}_M",
               title = "$method anytime, train ≤ $(CFG.N_train)",
               legend = :topleft)
    for (i, (label, pol, ω0_fn)) in enumerate(hrdp_policies(method))
        ho_train = HRDPObjective(compiler, pol, H_train, Wstar)
        ho_all = HRDPObjective(compiler, pol, H_all, Wstar)
        ω_H, σ_train = best_point(design_som(ho_train, ω0_fn(CFG.N_train);
                                             iters = CFG.iters))
        dpN = DesignProblem(compiler(CFG.N_train), pol, CFG.N_train)
        ω_N, _ = best_point(design_som(dpN, ω0_fn(CFG.N_train); iters = CFG.iters))

        tailored = Float64[]
        transfer = Float64[]
        reuse = Float64[]
        for M in H_test
            dpM = DesignProblem(compiler(M), pol, M)
            ω_M, W_M = best_point(design_som(dpM, ω0_fn(M); iters = CFG.iters))
            push!(tailored, W_M / Wstar[M])
            push!(transfer, pep_value(dpM, ω_H)[1] / Wstar[M])
            push!(reuse, pep_value(dpM, ω_N)[1] / Wstar[M])
        end
        gr_val = generalization_ratio(ho_all, ω_H, ω_H)  # ≥1 sanity ≈ 1
        ω_all, _ = best_point(design_som(ho_all, ω_H; iters = CFG.iters))
        gr_sub = generalization_ratio(ho_all, ω_H, ω_all)
        wgc_val = wgc(ho_all, ω_H)

        data[label] = Dict("omega_H" => ω_H, "omega_N" => ω_N,
                           "sigma_train" => σ_train, "tailored" => tailored,
                           "transfer" => transfer, "reuse" => reuse,
                           "GR" => gr_sub, "WGC" => wgc_val,
                           "H_test" => collect(H_test))
        push!(rows, [label, (@sprintf "%.4f" σ_train), (@sprintf "%.4f" gr_sub),
                     (@sprintf "%.4f" wgc_val),
                     (@sprintf "%.4f" mean(transfer)),
                     (@sprintf "%.4f" mean(reuse))])
        plot!(plt, collect(H_test), transfer; label = "$label transfer",
              color = PALETTE[i], linestyle = :solid, marker = :circle,
              markersize = 3)
        plot!(plt, collect(H_test), reuse; label = "$label reuse",
              color = PALETTE[i], linestyle = :dot, marker = :diamond,
              markersize = 3)
        plot!(plt, collect(H_test), tailored; label = "$label tailored",
              color = PALETTE[i], linestyle = :dash, alpha = 0.6)
        @info "exp2: $method $label" σ_train gr_sub wgc_val
    end
    save_result("exp2_$(method)", CFG, data)
    save_figure(plt, "exp2_anytime_$(method)")
    write_latex_table("exp2_$(method)",
        ["Policy", "\$\\sigma_{\\mathrm{train}}\$", "GR", "WGC",
         "\$\\overline{\\mathcal{T}}\$", "\$\\overline{\\mathcal{R}}\$"], rows;
        caption = "HRDP transfer metrics for $(method) " *
                  "(\$\\mathcal{H}_{\\mathrm{train}}=\\{2..$(CFG.N_train)\\}\$, " *
                  "test up to \$N_{\\mathrm{train}}+$(CFG.iota_max)\$).",
        label = "tab:exp2_$(method)")
end

println("exp2: done")
