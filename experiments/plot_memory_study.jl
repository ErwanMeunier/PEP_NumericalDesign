using PEPDesign
using TOML
using Plots
using Printf
include(joinpath(@__DIR__, "design_data.jl"))

function main(args)
    length(args) == 1 || throw(ArgumentError("supply the path to study.toml"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    runs = data["runs"]
    cells = sort!(unique([(r["N"], r["K"]) for r in runs]))
    output = dirname(path)
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box,
            legendfontsize = 7, guidefontsize = 9, tickfontsize = 8,
            titlefontsize = 11, left_margin = 12Plots.mm,
            right_margin = 3Plots.mm, bottom_margin = 8Plots.mm,
            top_margin = 6Plots.mm)
    rows = String["N,K,replicate,prior,axis,budget,free_best,prior_best,relative_advantage"]
    labels = filter(!=("free"), data["config"]["labels"])
    colors = Dict("stationary" => :darkorange, "lag_dct1" => :royalblue,
                  "lag_dct3" => :seagreen)
    for axis in (:time, :iterations)
        panels = []
        all_advantages = Float64[]
        for (N, K) in cells
            panel = plot(title = "N=$N, K=$K",
                         xlabel = axis == :time ? "Wall time (s)" : "Design updates T",
                         ylabel = "(free - prior) / free", legend = :bottomright)
            hline!(panel, [0.0]; color = :gray, linestyle = :dash, label = "")
            for label in labels
                subset = filter(r -> r["N"] == N && r["K"] == K && r["policy"] == label, runs)
                for r in subset
                    candidates = filter(f -> f["N"] == N && f["K"] == K &&
                        f["replicate"] == r["replicate"] && f["policy"] == "free", runs)
                    length(candidates) == 1 ||
                        error("missing or duplicated free comparator for N=$N K=$K")
                    f = only(candidates)
                    curve = relative_advantage(trace_from_data(f["trace"]),
                                               trace_from_data(r["trace"]); axis)
                    append!(all_advantages, Float64.(curve.advantage))
                    legend = length(subset) == 1 ? label : "$label, h=$(r["initial_step"])"
                    plot!(panel, curve.budgets, Float64.(curve.advantage);
                          label = legend, color = colors[label], seriestype = :steppost,
                          linestyle = (:solid, :dash, :dot)[mod1(r["replicate"], 3)])
                    for i in eachindex(curve.budgets)
                        push!(rows, join((N, K, r["replicate"], label, axis,
                            curve.budgets[i], curve.free_values[i], curve.prior_values[i],
                            curve.advantage[i]), ","))
                    end
                end
            end
            push!(panels, panel)
        end
        lo, hi = min(0.0, minimum(all_advantages)), max(0.0, maximum(all_advantages))
        padding = max(0.02, 0.05 * (hi - lo))
        for panel in panels
            plot!(panel; ylims = (lo - padding, hi + padding))
        end
        ncols = length(cells) > 2 ? 5 : length(cells)
        nrows = cld(length(cells), ncols)
        figure = plot(panels...; layout = (nrows, ncols), size = (400ncols, 400nrows))
        for ext in ("pdf", "png")
            savefig(figure, joinpath(output, "advantage_$(axis).$ext"))
        end
    end
    write(joinpath(output, "matched_curves.csv"), join(rows, "\n") * "\n")
    println("Saved signed advantage figures and matched_curves.csv to ", output)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
