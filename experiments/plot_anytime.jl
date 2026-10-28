# Plot normalized worst cases W_N/W*_N against N for an anytime study
# produced by exp9_anytime.jl or exp9_full_range.jl. Thick lines are
# geometric-grid designs; horizons without a value are skipped. Output is
# <stem>_ratios.{pdf,png} next to the input (<stem>_selected_ratios with --only).
#
# julia --project=. experiments\plot_anytime.jl <anytime.toml|full_range.toml> [--only key1,key2]
#
# Design keys are "family/kind/trainset", e.g. "power_law/hrdp_max/geometric".

using TOML
using Plots

const STYLE = Dict("reuse" => (:solid, :circle), "textbook" => (:dot, :utriangle),
                   "classical_anytime" => (:dot, :dtriangle),
                   "hrdp_mean" => (:dash, :diamond), "hrdp_max" => (:dashdot, :square))
const TICKS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)

function main(args)
    (length(args) == 1 || (length(args) == 3 && args[2] == "--only")) ||
        throw(ArgumentError("usage: plot_anytime.jl <toml> [--only key1,key2]"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    cfg = data["config"]
    horizons = Int.(cfg["horizons"])
    Wstar = data["Wstar"]
    designkey(d) = "$(d["family"])/$(d["kind"])/$(d["trainset"])"
    stem = splitext(basename(path))[1]
    if length(args) == 3
        only_ = Set(split(args[3], ","))
        known = Set(designkey.(data["designs"]))
        issubset(only_, known) ||
            throw(ArgumentError("unknown design keys $(setdiff(only_, known)); known: $(known)"))
        data["designs"] = filter(d -> designkey(d) in only_, data["designs"])
        stem *= "_selected"
    end
    ticks = length(horizons) <= 16 ? horizons :
            filter(N -> N in TICKS || N == cfg["N0"], horizons)
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box,
            legendfontsize = 7, guidefontsize = 10, tickfontsize = 9)
    families = sort!(unique([d["family"] for d in data["designs"]]))
    # One color per design when there are few; otherwise color by family, or
    # by design kind when there is a single family.
    colorkey(d) = length(data["designs"]) <= 10 ? designkey(d) :
                  length(families) > 1 ? d["family"] : d["kind"]
    keys_ = unique(colorkey.(data["designs"]))
    colors = Dict(k => palette(:tab10)[mod1(i, 10)] for (i, k) in enumerate(keys_))
    rmax = 1.0
    p = plot(; xlabel = "Horizon N", ylabel = "W(N) / W*(N)", xscale = :log10,
             title = "$(cfg["problem"]): anytime robustness (N0 = $(cfg["N0"]))",
             legend = :outertopright, size = (950, 450), xticks = (ticks, string.(ticks)),
             xminorticks = false, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
    vline!(p, [cfg["N0"]]; color = :gray, linestyle = :dot, label = "N0")
    hline!(p, [1.0]; color = :gray, linestyle = :dash, label = "")
    for d in data["designs"]
        r = [get(d["values"], string(N), NaN) / Wstar[string(N)] for N in horizons]
        ok = isfinite.(r)
        ls, mk = get(STYLE, d["kind"], (:solid, :circle))
        label = "$(d["family"]) $(d["kind"])" *
                (d["trainset"] == "none" ? "" : " [$(d["trainset"])]")
        geometric = d["trainset"] == "geometric"
        rmax = max(rmax, maximum(r[ok]; init = 1.0))
        plot!(p, horizons[ok], r[ok]; color = colors[colorkey(d)], linestyle = ls,
              marker = mk, markersize = length(horizons) > 16 ? 2 : 3,
              linewidth = geometric ? 3 : 1.5, label)
    end
    ylims!(p, (0.99, rmax + 0.02 * (rmax - 0.99)))
    out = dirname(path)
    savefig(p, joinpath(out, "$(stem)_ratios.pdf"))
    savefig(p, joinpath(out, "$(stem)_ratios.png"))
    println("Saved ", joinpath(out, "$(stem)_ratios.{pdf,png}"))
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
