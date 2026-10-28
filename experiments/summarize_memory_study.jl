using PEPDesign
using TOML
using Statistics
using Printf
include(joinpath(@__DIR__, "design_data.jl"))

function main(args)
    length(args) == 1 || throw(ArgumentError("supply study.toml"))
    path = abspath(args[1])
    data = TOML.parsefile(path)
    runs = data["runs"]
    summary = NamedTuple[]
    for p in filter(r -> r["policy"] != "free", runs)
        f = only(filter(r -> r["policy"] == "free" &&
            (r["N"], r["K"], r["replicate"]) == (p["N"], p["K"], p["replicate"]), runs))
        ft, pt = trace_from_data(f["trace"]), trace_from_data(p["trace"])
        c = relative_advantage(ft, pt; axis = :iterations, budgets = 4:16)
        available = collect(skipmissing(c.advantage))
        isempty(available) && error("insufficient updates to score the screen")
        timecurve = relative_advantage(ft, pt; axis = :time)
        x, y = timecurve.budgets, Float64.(timecurve.advantage)
        duration = last(x) - first(x)
        area = duration > 0 ? sum(diff(x) .* y[1:end-1]) / duration : first(y)
        at = relative_advantage(ft, pt; axis = :iterations, budgets = [4, 8, 12, 16])
        push!(summary, (N = p["N"], K = p["K"], rep = p["replicate"],
            prior = p["policy"], initial_step = p["initial_step"],
            mean_T4_16 = mean(available), observed_T_budgets = length(available),
            mean_time = area,
            T4 = at.advantage[1], T8 = at.advantage[2],
            T12 = at.advantage[3], T16 = at.advantage[4],
            free_final = minimum(ft.values), prior_final = minimum(pt.values)))
    end
    sort!(summary; by = r -> (-r.mean_T4_16, r.N, r.K, r.prior))
    open(joinpath(dirname(path), "summary.csv"), "w") do io
        println(io, join(string.(keys(first(summary))), ","))
        for row in summary
            println(io, join(values(row), ","))
        end
    end
    for r in summary
        @printf("N=%2d K=%2d rep=%d %-11s h=%.1f mean(T4:16)=%+.2f%% mean(time)=%+.2f%% | T4=%s T8=%s T16=%s\n",
                r.N, r.K, r.rep, r.prior, r.initial_step,
                100r.mean_T4_16, 100r.mean_time,
                string(r.T4), string(r.T8), string(r.T16))
    end
    groups = Dict{Tuple{Int,Int,String},Vector{NamedTuple}}()
    for row in summary
        push!(get!(groups, (row.N, row.K, row.prior), NamedTuple[]), row)
    end
    open(joinpath(dirname(path), "paired_summary.csv"), "w") do io
        println(io, "N,K,prior,repeats,mean_T4,range_T4,mean_T8,range_T8,mean_T16,range_T16,mean_T4_16,range_T4_16,positive_T4,positive_T8,positive_T16")
        for key in sort!(collect(keys(groups)))
            rows = groups[key]
            finite = field -> filter(x -> !ismissing(x), getproperty.(rows, field))
            stat = function(field)
                vals = finite(field)
                isempty(vals) ? ("missing", "missing") :
                    (@sprintf("%.6f", mean(vals)),
                     @sprintf("[%.6f,%.6f]", minimum(vals), maximum(vals)))
            end
            t4, r4 = stat(:T4); t8, r8 = stat(:T8); t16, r16 = stat(:T16)
            ave, rave = stat(:mean_T4_16)
            positive = field -> count(x -> !ismissing(x) && x > 0,
                                       getproperty.(rows, field))
            println(io, join((key..., length(rows), t4, r4, t8, r8, t16, r16,
                ave, rave, positive(:T4), positive(:T8), positive(:T16)), ","))
        end
    end
    return summary
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
