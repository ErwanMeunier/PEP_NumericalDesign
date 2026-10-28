# Shared design-loop infrastructure: cached PEP evaluator and run traces.

"""
    DesignTrace

History of a design run. `values[k] = W(ω_hist[k])`; `solves_hist[k]` is the
cumulative PEP-solve count when ω_hist[k] was recorded (the fair comparison
budget across methods and policies); `nsolves` is the run total.
`tangent_hist[k]` counts the cumulative tangent (linearized) SDPs of
SLM-type methods (zero for FOM/SOM) and `time_hist[k]` is the elapsed wall
time in seconds.
"""
struct DesignTrace
    ω_hist::Vector{Vector{Float64}}
    values::Vector{Float64}
    solves_hist::Vector{Int}
    nsolves::Int
    tangent_hist::Vector{Int}
    time_hist::Vector{Float64}
end

DesignTrace(ω_hist, values, solves_hist, nsolves) =
    DesignTrace(ω_hist, values, solves_hist, nsolves,
                zeros(Int, length(solves_hist)), fill(NaN, length(solves_hist)))

"""Best visited point: `(ω_best, W_best)`."""
function best_point(tr::DesignTrace)
    W, k = findmin(tr.values)
    return tr.ω_hist[k], W
end

"""
    relative_advantage(free, prior; axis=:time, budgets=nothing)

Compare best-so-far objectives at matched budgets:
`(best_free - best_prior) / best_free`. Positive favors the prior.
`axis=:time` uses elapsed seconds, `:iterations` outer updates (initial point
is iteration 0), and `:solves` total PEP + tangent SDPs. Rejected updates count
as iterations. The default grid is the union of trace knots on their common
observed interval. Explicit budgets outside either trace are marked `missing`,
not extrapolated. Values between knots use the last completed incumbent.

Returns `(budgets, free_values, prior_values, advantage)`. Values are
best found, not global optima. Both traces must use the same problem,
initial coefficients and accounting convention for a controlled comparison.
"""
function relative_advantage(free::DesignTrace, prior::DesignTrace;
                            axis::Symbol = :time, budgets = nothing)
    axis in (:time, :iterations, :solves) ||
        throw(ArgumentError("axis must be :time, :iterations or :solves"))
    coordinates(tr) = axis == :time ? tr.time_hist :
        axis == :iterations ? collect(0:(length(tr.values) - 1)) :
        tr.solves_hist .+ tr.tangent_hist
    xf, xp = coordinates(free), coordinates(prior)
    for (tr, x) in ((free, xf), (prior, xp))
        !isempty(x) && length(x) == length(tr.values) ||
            throw(ArgumentError("trace budgets and values must be nonempty and aligned"))
        all(isfinite, x) && issorted(x) ||
            throw(ArgumentError("trace budgets must be finite and nondecreasing"))
        all(isfinite, tr.values) ||
            throw(ArgumentError("comparison requires finite objective values"))
    end
    bf, bp = accumulate(min, free.values), accumulate(min, prior.values)
    all(>(0), bf) || throw(ArgumentError("relative advantage requires positive free values"))
    if budgets === nothing
        lo, hi = max(first(xf), first(xp)), min(last(xf), last(xp))
        lo <= hi || throw(ArgumentError("traces have no common observed budget interval"))
        grid = sort!(unique(Float64[x for x in [xf; xp] if lo <= x <= hi]))
    else
        grid = collect(Float64, budgets)
        all(isfinite, grid) && issorted(grid) ||
            throw(ArgumentError("budgets must be finite and nondecreasing"))
    end
    f = Vector{Union{Missing,Float64}}(missing, length(grid))
    p = similar(f); fill!(p, missing)
    a = similar(f); fill!(a, missing)
    for (i, b) in enumerate(grid)
        first(xf) <= b <= last(xf) && first(xp) <= b <= last(xp) || continue
        f[i] = bf[searchsortedlast(xf, b)]
        p[i] = bp[searchsortedlast(xp, b)]
        a[i] = (f[i] - p[i]) / f[i]
    end
    return (; budgets = grid, free_values = f, prior_values = p, advantage = a)
end

# Cached evaluator: one evaluation per distinct ω; degenerate points → +Inf.
mutable struct PEPEvaluator{D<:AbstractDesignObjective}
    dp::D
    ω::Vector{Float64}
    f::Float64
    g::Vector{Float64}
    H::Matrix{Float64}
    hess::Bool
    nsolves::Int
end

function PEPEvaluator(dp::AbstractDesignObjective, κ::Int; hess::Bool = true)
    return PEPEvaluator(dp, fill(NaN, κ), NaN, zeros(κ),
                        hess ? zeros(κ, κ) : zeros(0, 0), hess, 0)
end

function refresh!(ev::PEPEvaluator, ω::AbstractVector{Float64})
    ω == ev.ω && return ev
    copyto!(ev.ω, ω)
    try
        f, g, H, ns = eval_all(ev.dp, ω; hess = ev.hess)
        ev.nsolves += ns
        ev.f = f
        copyto!(ev.g, g)
        ev.hess && copyto!(ev.H, H)
    catch err
        err isa InterruptException && rethrow()
        # Degenerate ω (solver failure, Inf coefficients): report +Inf so line
        # searches backtrack instead of crashing.
        ev.f = Inf
        fill!(ev.g, 0.0)
        ev.hess && fill!(ev.H, 0.0)
        fill!(ev.ω, NaN)
    end
    return ev
end
