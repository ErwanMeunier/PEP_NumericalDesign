# High-level frontend: function/operator objects with symbolic oracles.
#
# Ported from PEPit.jl (https://github.com/PerformanceEstimation/PEPit.jl),
# MIT License, Copyright (c) 2025 Shuvomoy Das Gupta and contributors,
# re-expressed on the PEPDesign degree-2 Gram DSL: oracle triples (x, g, f)
# are recorded per class object, and each class emits its interpolation
# constraints once at compile time (`finalize_classes!`). Function values are
# stored as QExpr so linear combinations and residual back-substitution work.

"""
    AbstractPEPFunction

Common supertype of all frontend function and operator classes.
"""
abstract type AbstractPEPFunction end

"""
    PEPFunc

Oracle bookkeeping shared by every leaf class: sampled triples `(x, g, f)`,
stationary markers, and the gradient-reuse convention of the class.
"""
mutable struct PEPFunc
    model::PEPModel
    name::String
    reuse_gradient::Bool
    xs::Vector{PointExpr}
    gs::Vector{PointExpr}
    fs::Vector{QExpr}
    stationary::Vector{Int}      # indices into xs with g ≡ 0
end

PEPFunc(m::PEPModel, name::String, reuse_gradient::Bool) =
    PEPFunc(m, name, reuse_gradient, PointExpr[], PointExpr[], QExpr[], Int[])

_core(f::AbstractPEPFunction) = f.core
"""Model a frontend function/operator belongs to."""
model_of(f::AbstractPEPFunction) = _core(f).model
"""Number of recorded oracle triples."""
npoints(f::AbstractPEPFunction) = length(_core(f).xs)
"""Oracle triples `(x, g, f)` recorded so far."""
triples(f::AbstractPEPFunction) =
    [(c.xs[i], c.gs[i], c.fs[i]) for c in (_core(f),) for i in eachindex(c.xs)]

_register!(m::PEPModel, f::AbstractPEPFunction) = (push!(m.funcs, f); f)

"""
    add_oracle_point!(f, x, g, fv=nothing; stationary=false) -> (g, f::QExpr)

Register the triple `(x, g, f)` on `f` (a fresh function value is created when
`fv === nothing`). Low-level hook used by primitive steps and custom oracles.
"""
function add_oracle_point!(f::AbstractPEPFunction, x::PointExpr, g::PointExpr,
                           fv = nothing; stationary::Bool = false)
    c = _core(f)
    fe = fv === nothing ? qexpr(fval!(c.model)) : qexpr(fv)
    push!(c.xs, x); push!(c.gs, g); push!(c.fs, fe)
    stationary && push!(c.stationary, length(c.xs))
    return g, fe
end

"""
    oracle!(f, x) -> (g, f::QExpr)

(Sub)gradient/operator value and function value of `f` at `x`. Repeated calls
at the same point reuse the gradient iff the class is differentiable
(`reuse_gradient`); the function value is always reused.
"""
function oracle!(f::AbstractPEPFunction, x::PointExpr)
    c = _core(f)
    i = findfirst(==(x), c.xs)
    if i !== nothing
        c.reuse_gradient && return c.gs[i], c.fs[i]
        g = point!(c.model; name = "g_$(c.name)_$(length(c.xs) + 1)")
        push!(c.xs, x); push!(c.gs, g); push!(c.fs, c.fs[i])   # new subgradient, same value
        return g, c.fs[i]
    end
    g = point!(c.model; name = "g_$(c.name)_$(length(c.xs) + 1)")
    return add_oracle_point!(f, x, g)
end

"""(Sub)gradient / operator evaluation of `f` at `x`."""
gradient!(f::AbstractPEPFunction, x::PointExpr) = oracle!(f, x)[1]
"""Function value of `f` at `x` (QExpr)."""
value!(f::AbstractPEPFunction, x::PointExpr) = oracle!(f, x)[2]

"""
    stationary_point!(f) -> (x, g, f::QExpr)

Register a fresh point with zero (sub)gradient (`g ≡ 0`).
"""
function stationary_point!(f::AbstractPEPFunction)
    m = model_of(f)
    x = point!(m; name = "xs_$(_name_of(f))")
    g, fe = add_oracle_point!(f, x, PointExpr(); stationary = true)
    return x, g, fe
end

"""
    fixed_point!(op) -> (x, Tx, f::QExpr)  with  Tx = x

Register a fixed point of an operator (`T x = x`).
"""
function fixed_point!(op::AbstractPEPFunction)
    m = model_of(op)
    x = point!(m; name = "xfix_$(_name_of(op))")
    g, fe = add_oracle_point!(op, x, x)
    return x, g, fe
end

_name_of(f::AbstractPEPFunction) = _core(f).name

# ── linear combinations  a·f₁ + b·f₂ + …  ────────────────────────────────────

"""
    FunLinComb

Weighted sum of leaf function objects. Oracles decompose onto leaf oracles;
registered triples back-substitute the residual into the last leaf
(PEPit.jl convention), so `proximal_step!` & co. work on combinations.
"""
struct FunLinComb <: AbstractPEPFunction
    model::PEPModel
    weights::Vector{Float64}
    leaves::Vector{AbstractPEPFunction}
end

model_of(f::FunLinComb) = f.model
_name_of(f::FunLinComb) = join(["$(w)*$(_name_of(l))" for (w, l) in
                                zip(f.weights, f.leaves)], "+")

_terms(f::FunLinComb) = collect(zip(f.weights, f.leaves))
_terms(f::AbstractPEPFunction) = [(1.0, f)]

function _combine(a::Float64, f, b::Float64, g)
    ta = [(a * w, l) for (w, l) in _terms(f)]
    tb = [(b * w, l) for (w, l) in _terms(g)]
    terms = vcat(ta, tb)
    m = model_of(f)
    m === model_of(g) || error("cannot combine functions from different models")
    return FunLinComb(m, [t[1] for t in terms],
                      AbstractPEPFunction[t[2] for t in terms])
end

Base.:+(f::AbstractPEPFunction, g::AbstractPEPFunction) = _combine(1.0, f, 1.0, g)
Base.:-(f::AbstractPEPFunction, g::AbstractPEPFunction) = _combine(1.0, f, -1.0, g)
function Base.:*(a::Real, f::AbstractPEPFunction)
    terms = [(Float64(a) * w, l) for (w, l) in _terms(f)]
    return FunLinComb(model_of(f), [t[1] for t in terms],
                      AbstractPEPFunction[t[2] for t in terms])
end
Base.:*(f::AbstractPEPFunction, a::Real) = a * f
Base.:/(f::AbstractPEPFunction, a::Real) = (1.0 / a) * f

function oracle!(f::FunLinComb, x::PointExpr)
    g = PointExpr()
    fe = qexpr(0.0)
    for (w, leaf) in zip(f.weights, f.leaves)
        gl, fl = oracle!(leaf, x)
        g = g + w * gl
        fe = fe + w * fl
    end
    return g, fe
end

# Register (x, g, f) on a combination: leaves 1..k−1 get ordinary oracles and
# the last leaf absorbs the weighted residual, keeping the triple exact.
function add_oracle_point!(f::FunLinComb, x::PointExpr, g::PointExpr,
                           fv = nothing; stationary::Bool = false)
    fe = fv === nothing ? qexpr(fval!(f.model)) : qexpr(fv)
    k = length(f.leaves)
    gacc = g
    facc = fe
    for i in 1:(k - 1)
        gl, fl = oracle!(f.leaves[i], x)
        gacc = gacc - f.weights[i] * gl
        facc = facc - f.weights[i] * fl
    end
    wk = f.weights[k]
    add_oracle_point!(f.leaves[k], x, (1.0 / wk) * gacc, (1.0 / wk) * facc)
    return g, fe
end

# Combinations emit nothing themselves; their leaves are registered separately.
add_class_constraints!(::FunLinComb) = nothing
