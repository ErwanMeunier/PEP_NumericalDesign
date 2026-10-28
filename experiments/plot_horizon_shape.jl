# Plot the IGDM horizon-shape study produced by exp10_horizon_shape.jl:
# W_N(ω_H)/W*_N against N in three panels (shape at extent [2,32], mean vs
# max, extent). Missing designs are skipped. Output: shape_panels.{pdf,png}
# next to the input.
#
# julia --project=. experiments\plot_horizon_shape.jl <shape.toml>

using TOML
using Plots

const PANELS = [
    ("(a) shape, extent [2,32], minimax",
     ["window_2_32/max", "geomfine_2_32/max", "exchange_2_32/max", "geom_2_32/max",
      "arith_2_32/max", "triple_2_32/max", "ends_2_32/max", "single_32/reuse"]),
    ("(b) objective: mean (dashed) vs max (solid)",
     ["window_2_32/max", "window_2_32/mean", "geom_2_32/max", "geom_2_32/mean",
      "arith_2_32/max", "arith_2_32/mean"]),
    ("(c) extent, minimax",
     ["geom_2_8/max", "geom_2_16/max", "geom_2_32/max", "geom_4_32/max", "geom_8_32/max",
      "window_2_10/max", "window_16_32/max", "single_10/reuse", "single_16/reuse"])]
const TICKS = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]

function main(args)
    length(args) == 1 || throw(ArgumentError("usage: plot_horizon_shape.jl <shape.toml>"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    Ns = Int.(data["config"]["eval_horizons"])
    W = data["Wstar"]
    bykey = Dict("$(d["set"])/$(d["objective"])" => d for d in data["designs"])
    default(fontfamily = "Computer Modern", framestyle = :box, legendfontsize = 7,
            guidefontsize = 10, tickfontsize = 8, titlefontsize = 10)
    ticks = filter(in(Ns), TICKS)
    plots = []
    for (title, keys_) in PANELS
        p = plot(; xscale = :log10, xticks = (ticks, string.(ticks)), xminorticks = false,
                 xlabel = "Horizon N", ylabel = "W(N) / W*(N)", title,
                 legend = :outertopright)
        hline!(p, [1.0]; color = :gray, linestyle = :dash, label = "")
        vline!(p, [32]; color = :gray, linestyle = :dot, label = "")
        sets = unique(first.(split.(keys_, "/")))
        colors = Dict(s => palette(:tab10)[mod1(i, 10)] for (i, s) in enumerate(sets))
        ymax = 1.0
        for k in keys_
            haskey(bykey, k) || continue
            d = bykey[k]
            r = [d["values"][string(N)] / W[string(N)] for N in Ns]
            H = Int.(d["H"])
            ls = d["objective"] == "mean" ? :dash : d["objective"] == "reuse" ? :dot : :solid
            contiguous = length(H) > 3 && H == collect(H[1]:H[end])
            label = "$(first(split(d["set"], '_'))) $(d["objective"]) H=" *
                    (contiguous ? "{$(H[1])..$(H[end])}" : "{" * join(H, ",") * "}")
            ymax = max(ymax, maximum(r))
            plot!(p, Ns, r; color = colors[d["set"]], linestyle = ls, linewidth = 1.8, label)
            inH = [N in H for N in Ns]
            scatter!(p, Ns[inH], r[inH]; color = colors[d["set"]], markersize = 2.5,
                     markerstrokewidth = 0, label = "")
        end
        ylims!(p, (0.99, ymax + 0.1 * (ymax - 0.99)))
        push!(plots, p)
    end
    fig = plot(plots...; layout = (3, 1), size = (950, 1150), left_margin = 6Plots.mm,
               bottom_margin = 4Plots.mm)
    out = dirname(path)
    savefig(fig, joinpath(out, "shape_panels.pdf"))
    savefig(fig, joinpath(out, "shape_panels.png"))
    println("Saved ", joinpath(out, "shape_panels.{pdf,png}"))
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
