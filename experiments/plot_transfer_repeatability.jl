using TOML
using Plots
using Statistics
using Printf

function main(args)
    length(args) == 1 || throw(ArgumentError("supply repeatability.toml"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    runs = data["runs"]
    horizons = sort!(unique([r["N"] for r in runs]))
    starts = sort!(unique([r["source_initial_step"] for r in runs]))
    rows = String["N,source_initial_step,transfer_to_cold,warm_to_cold,warm_advantage,source_best,transfer_value,warm_best,cold_best"]
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box,
            legendfontsize = 8, guidefontsize = 10, tickfontsize = 9)
    colors = distinguishable_colors(length(starts))
    quality = plot(; xlabel = "Target horizon N", ylabel = "Worst-case value / cold free value",
                   title = "Transfer fit sensitivity; source N=$(data["config"]["N_train"])",
                   legend = :topleft)
    warm = plot(; xlabel = "Target horizon N", ylabel = "Signed warm-start advantage",
                title = "Free refinement from transferred weights", legend = :topleft)
    hline!(quality, [1.0]; color = :gray, linestyle = :dash, label = "cold-start free")
    hline!(warm, [0.0]; color = :gray, linestyle = :dash, label = "")
    for (i, h) in enumerate(starts)
        group = sort!(filter(r -> r["source_initial_step"] == h, runs); by = r -> r["N"])
        xs = [r["N"] for r in group]
        tratio = [r["reuse_to_cold_ratio"] for r in group]
        wratio = [r["warm_to_cold_ratio"] for r in group]
        advantage = [r["warm_advantage"] for r in group]
        plot!(quality, xs, tratio; marker = :circle, color = colors[i],
              label = "reuse h=$(h)")
        plot!(quality, xs, wratio; marker = :diamond, linestyle = :dash,
              color = colors[i], label = "warm refine h=$(h)")
        plot!(warm, xs, advantage; marker = :diamond, color = colors[i],
              label = "source h=$(h)")
        for r in group
            push!(rows, join((r["N"], h, r["reuse_to_cold_ratio"],
                r["warm_to_cold_ratio"], r["warm_advantage"], r["source_best"],
                r["transfer_value"], r["warm_best"], r["cold_best"]), ","))
        end
    end
    out = dirname(path)
    savefig(quality, joinpath(out, "transfer_repeatability_quality.pdf"))
    savefig(quality, joinpath(out, "transfer_repeatability_quality.png"))
    savefig(warm, joinpath(out, "transfer_repeatability_advantage.pdf"))
    savefig(warm, joinpath(out, "transfer_repeatability_advantage.png"))
    write(joinpath(out, "transfer_repeatability.csv"), join(rows, "\n") * "\n")
    open(joinpath(out, "source_sensitivity.csv"), "w") do io
        println(io, "N,repetitions,mean_advantage,min_advantage,max_advantage,positive")
        for N in horizons
            group = filter(r -> r["N"] == N, runs)
            a = [r["warm_advantage"] for r in group]
            println(io, join((N, length(a), mean(a), minimum(a), maximum(a),
                              count(x -> x > 0, a)), ","))
        end
    end
    println("Saved repeatability plots and paired source-sensitivity summaries to ", out)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
