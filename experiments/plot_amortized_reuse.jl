using TOML
using Plots
using Printf

function main(args)
    length(args) == 1 || throw(ArgumentError("supply amortized.toml"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    targets, cumulative = data["targets"], data["cumulative_cost"]
    N = [r["N"] for r in targets]
    cfg, train = data["config"], data["training"]
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box,
            legendfontsize = 8, guidefontsize = 10, tickfontsize = 8)

    transfer_ratio = [r["transfer_value"] / r["cold_free_best"] for r in targets]
    warm_ratio = [r["warm_refine_best"] / r["cold_free_best"] for r in targets]
    quality = plot(N, transfer_ratio; marker = :circle, label = "reuse, no redesign",
                   xlabel = "Target horizon N", ylabel = "Value / cold-start free best",
                   title = "Target quality; source N=$(cfg["N_train"]), K=$(cfg["memory_K"])")
    plot!(quality, N, warm_ratio; marker = :diamond, label = "reuse + free refinement")
    hline!(quality, [1.0]; color = :gray, linestyle = :dash, label = "cold free baseline")

    q = [r["targets_served"] for r in cumulative]
    cold_cost = [r["cold_matched_memory_cost"] for r in cumulative]
    warm_cost = [r["warm_refine_cost"] for r in cumulative]
    assessment_cost = [r["transfer_only_cost"] for r in cumulative]
    costs = plot(q, cold_cost; marker = :circle, label = "cold free: sum target costs",
                 xlabel = "Number of target horizons served",
                 ylabel = "Cumulative design/assessment time (s)",
                 title = "Amortized offline cost; source training = $(round(train["seconds_including_source_compile"], digits=1)) s")
    plot!(costs, q, warm_cost; marker = :diamond,
          label = "train once + assess + warm-refine")
    plot!(costs, q, assessment_cost; marker = :utriangle,
          label = "train once + assess reuse")
    if all(>(0), cold_cost) && all(>(0), warm_cost)
        cost_ratio = warm_cost ./ cold_cost
        p = plot(q, cost_ratio; marker = :diamond,
                 xlabel = "Number of target horizons served",
                 ylabel = "Cumulative cost ratio",
                 title = "Amortized refinement cost / cold free cost")
        hline!(p, [1.0]; color = :gray, linestyle = :dash, label = "break-even")
        if any(x -> x < 1.0, cost_ratio)
            first_q = q[findfirst(x -> x < 1.0, cost_ratio)]
            vline!(p, [first_q]; color = :gray, linestyle = :dot, label = "first observed break-even")
        end
    else
        p = plot(title = "Cost-ratio comparison unavailable for zero-cost smoke runs")
    end

    out = dirname(path)
    for (name, fig) in (("amortized_quality", quality), ("amortized_cost", costs),
                        ("amortized_break_even", p))
        savefig(fig, joinpath(out, "$name.pdf"))
        savefig(fig, joinpath(out, "$name.png"))
    end
    rows = ["N,transfer_value,warm_refine_start_value,warm_refine_best,cold_free_best,transfer_to_cold,warm_to_cold,transfer_seconds,warm_seconds,cold_seconds,source_seconds,cumulative_transfer_cost,cumulative_warm_cost,cumulative_cold_cost"]
    for (i, r) in enumerate(targets)
        c = cumulative[i]
        push!(rows, join((r["N"], r["transfer_value"], r["warm_refine_start_value"],
            r["warm_refine_best"], r["cold_free_best"], r["transfer_to_cold_ratio"],
            r["warm_refine_to_cold_ratio"], r["transfer_seconds"],
            r["warm_refine_seconds"], r["cold_free_seconds"],
            train["seconds_including_source_compile"], c["transfer_only_cost"],
            c["warm_refine_cost"], c["cold_matched_memory_cost"]), ","))
    end
    write(joinpath(out, "amortized_summary.csv"), join(rows, "\n") * "\n")
    println("Saved amortized quality/cost plots and summary: ", out)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
