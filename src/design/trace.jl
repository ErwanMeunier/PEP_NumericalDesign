# Shared design-loop infrastructure: cached PEP evaluator and run traces.

"""
    DesignTrace

History of a design run. `values[k] = W(ω_hist[k])`; `solves_hist[k]` is the
cumulative SDP-solve count when ω_hist[k] was recorded (the fair comparison
budget across methods and policies); `nsolves` is the run total.
"""
struct DesignTrace
    ω_hist::Vector{Vector{Float64}}
    values::Vector{Float64}
    solves_hist::Vector{Int}
    nsolves::Int
end

"""Best visited point: `(ω_best, W_best)`."""
function best_point(tr::DesignTrace)
    W, k = findmin(tr.values)
    return tr.ω_hist[k], W
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
