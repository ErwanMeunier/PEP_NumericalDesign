# Plot the IGDM memory study produced by exp11_memory_anytime.jl.
# (a) tailored W^K_N / W^ref_N (value of memory when N is known);
# (b) common designs: W_N(omega_K) / W^ref_N (one design per K, all N);
# (c) within designs: W_N(omega_K) / W^K_N (price of anytime inside prior K);
# (d) worst ratios on the dense grid against K.
# Output: memory_panels.{pdf,png} next to the input.
#
# julia --project=. experiments\plot_memory_anytime.jl <memory.toml>

using TOML
using Plots

const TICKS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48]

function main(args)
    length(args) == 1 || throw(ArgumentError("usage: plot_memory_anytime.jl <memory.toml>"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    Ks = Int.(data["config"]["Ks"])
    dense = Int.(data["config"]["dense"])
    Ns = sort!([dense; Int.(data["config"]["extra"])])
    Wref = Dict(parse(Int, k) => v for (k, v) in data["Wref"])
    WK(N, K) = data["tailored"][string(N)][string(K)]["W"]
    des(obj, K) = get(data["designs"], "$(obj)/K$(K)", nothing)
    default(fontfamily = "Computer Modern", framestyle = :box, legendfontsize = 7,
            guidefontsize = 10, tickfontsize = 8, titlefontsize = 10)
    colors = Dict(K => cgrad(:viridis, length(Ks); categorical = true)[i]
                  for (i, K) in enumerate(Ks))
    ticks = filter(in(Ns), TICKS)
    base(title, ylabel) = begin
        p = plot(; xscale = :log10, xticks = (ticks, string.(ticks)), xminorticks = false,
                 xlabel = "Horizon N", ylabel, title, legend = :outertopright)
        hline!(p, [1.0]; color = :gray, linestyle = :dash, label = "")
        vline!(p, [last(dense)]; color = :gray, linestyle = :dot, label = "")
        p
    end

    pa = base("(a) tailored to each N", "W^K(N) / W^ref(N)")
    for K in Ks
        plot!(pa, Ns, [WK(N, K) / Wref[N] for N in Ns]; color = colors[K], linewidth = 1.8,
              label = "K=$K")
    end
    pb = base("(b) one design per K, vs best over all K", "W(N) / W^ref(N)")
    pc = base("(c) one design per K, vs tailored in the same K", "W(N) / W^K(N)")
    for K in Ks
        for (p, obj, norm) in ((pb, "common", N -> Wref[N]), (pc, "within", N -> WK(N, K)))
            d = des(obj, K)
            d === nothing && continue
            M = [N for N in Ns if haskey(d["values"], string(N))]
            plot!(p, M, [d["values"][string(N)] / norm(N) for N in M]; color = colors[K],
                  linewidth = 1.8, label = "K=$K")
        end
    end
    # Reference in (b): tailored K=2 (knows N).
    2 in Ks && plot!(pb, Ns, [WK(N, 2) / Wref[N] for N in Ns]; color = :black,
                     linestyle = :dot, linewidth = 1.5, label = "tailored K=2")

    pd = plot(; xlabel = "Memory K", ylabel = "worst ratio on 2..$(last(dense))",
              title = "(d) worst case over the design grid", legend = :outertopright,
              xticks = Ks)
    hline!(pd, [1.0]; color = :gray, linestyle = :dash, label = "")
    tw = [maximum(WK(N, K) / Wref[N] for N in dense) for K in Ks]
    plot!(pd, Ks, tw; marker = :circle, color = :gray, linestyle = :dot,
          label = "tailored K, vs W^ref")
    for (obj, lab, ls, norm) in (("common", "anytime K, vs W^ref", :solid, (N, K) -> Wref[N]),
                                 ("within", "anytime K, vs W^K", :dash, (N, K) -> WK(N, K)))
        Kd = [K for K in Ks if des(obj, K) !== nothing]
        isempty(Kd) && continue
        w = [maximum(des(obj, K)["values"][string(N)] / norm(N, K) for N in dense) for K in Kd]
        plot!(pd, Kd, w; marker = :circle, color = :black, linestyle = ls, label = lab)
    end

    for p in (pa, pb, pc)
        top = maximum(maximum(s[:y]) for s in p.series_list if s[:seriestype] == :path)
        ylims!(p, (0.98, 1 + 1.05 * (top - 1)))
    end
    fig = plot(pa, pb, pc, pd; layout = (4, 1), size = (900, 1450), left_margin = 6Plots.mm,
               bottom_margin = 4Plots.mm)
    out = dirname(path)
    savefig(fig, joinpath(out, "memory_panels.pdf"))
    savefig(fig, joinpath(out, "memory_panels.png"))
    println("Saved ", joinpath(out, "memory_panels.{pdf,png}"))
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
