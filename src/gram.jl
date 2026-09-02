# Gram-space modeling layer: points, function values, quadratic scalar
# expressions, and the PEPModel container (PEPit-style DSL).

"""
    PointExpr

A vector-valued quantity expressed as a linear combination of Gram basis
vectors, with coefficients affine in the method coefficients η.
"""
struct PointExpr
    c::Dict{Int,PAff}
end

PointExpr() = PointExpr(Dict{Int,PAff}())

function Base.:+(p::PointExpr, q::PointExpr)
    out = Dict{Int,PAff}(k => v for (k, v) in p.c)
    for (k, v) in q.c
        nv = get(out, k, PAff()) + v
        if iszero(nv)
            delete!(out, k)
        else
            out[k] = nv
        end
    end
    return PointExpr(out)
end
Base.:-(p::PointExpr) = PointExpr(Dict(k => -v for (k, v) in p.c))
Base.:-(p::PointExpr, q::PointExpr) = p + (-q)
Base.:*(a::PAff, p::PointExpr) = iszero(a) ? PointExpr() :
    PointExpr(Dict(k => affmul(a, v) for (k, v) in p.c))
Base.:*(x::Real, p::PointExpr) = PAff(x) * p
Base.:*(p::PointExpr, a::Union{PAff,Real}) = a * p
Base.zero(::Type{PointExpr}) = PointExpr()
Base.zero(::PointExpr) = PointExpr()
Base.:(==)(p::PointExpr, q::PointExpr) = p.c == q.c
Base.hash(p::PointExpr, h::UInt) = hash(p.c, h)
Base.iszero(p::PointExpr) = isempty(p.c)

"""
    FVal

Reference to one entry of the function-value vector F.
"""
struct FVal
    idx::Int
end

# ──────────────────────────────────────────────────────────────────────────────

"""
    QExpr

Scalar PEP expression: `Σ_{i≤j} gram[(i,j)]*⟨b_i,b_j⟩ + Σ_k f[k]*F_k + cnst`,
with Gram coefficients quadratic in η and F coefficients affine in η.
"""
struct QExpr
    gram::Dict{Tuple{Int,Int},PQuad}
    f::Dict{Int,PAff}
    cnst::PQuad
end

QExpr() = QExpr(Dict{Tuple{Int,Int},PQuad}(), Dict{Int,PAff}(), PQuad())

qexpr(e::QExpr) = e
qexpr(v::FVal) = QExpr(Dict{Tuple{Int,Int},PQuad}(), Dict(v.idx => PAff(1.0)), PQuad())
qexpr(x::Real) = QExpr(Dict{Tuple{Int,Int},PQuad}(), Dict{Int,PAff}(), PQuad(x))
qexpr(a::PAff) = QExpr(Dict{Tuple{Int,Int},PQuad}(), Dict{Int,PAff}(), PQuad(a))

function _qadd!(dst::Dict{K,V}, src::Dict{K,V}, sgn::Float64) where {K,V}
    for (k, v) in src
        nv = get(dst, k, V()) + sgn * v
        if iszero(nv)
            delete!(dst, k)
        else
            dst[k] = nv
        end
    end
    return dst
end

function Base.:+(a::QExpr, b::QExpr)
    gram = Dict{Tuple{Int,Int},PQuad}(k => v for (k, v) in a.gram)
    f = Dict{Int,PAff}(k => v for (k, v) in a.f)
    QExpr(_qadd!(gram, b.gram, 1.0), _qadd!(f, b.f, 1.0), a.cnst + b.cnst)
end
function Base.:-(a::QExpr, b::QExpr)
    gram = Dict{Tuple{Int,Int},PQuad}(k => v for (k, v) in a.gram)
    f = Dict{Int,PAff}(k => v for (k, v) in a.f)
    QExpr(_qadd!(gram, b.gram, -1.0), _qadd!(f, b.f, -1.0), a.cnst - b.cnst)
end
Base.:-(a::QExpr) = -1.0 * a
Base.:*(x::Real, a::QExpr) = QExpr(
    Dict(k => x * v for (k, v) in a.gram),
    Dict(k => x * v for (k, v) in a.f),
    x * a.cnst)
Base.:*(a::QExpr, x::Real) = x * a
Base.:/(a::QExpr, x::Real) = (1.0 / x) * a

# Mixed arithmetic: promote FVal/Real/PAff operands to QExpr.
const QLike = Union{QExpr,FVal,Real,PAff}
for op in (:+, :-)
    @eval Base.$op(a::QExpr, b::Union{FVal,Real,PAff}) = $op(a, qexpr(b))
    @eval Base.$op(a::Union{FVal,Real,PAff}, b::QExpr) = $op(qexpr(a), b)
    @eval Base.$op(a::FVal, b::Union{FVal,Real,PAff}) = $op(qexpr(a), qexpr(b))
    @eval Base.$op(a::Union{Real,PAff}, b::FVal) = $op(qexpr(a), qexpr(b))
end
Base.:-(a::FVal) = -qexpr(a)
Base.:*(x::Real, v::FVal) = x * qexpr(v)
Base.:*(v::FVal, x::Real) = x * qexpr(v)
Base.:*(a::PAff, v::FVal) = QExpr(Dict{Tuple{Int,Int},PQuad}(), Dict(v.idx => a), PQuad())
Base.:*(v::FVal, a::PAff) = a * v

"""
    inner(p, q) :: QExpr

Inner product ⟨p, q⟩ of two point expressions (exact, degree ≤ 2 in η).
"""
function inner(p::PointExpr, q::PointExpr)
    gram = Dict{Tuple{Int,Int},PQuad}()
    for (i, a) in p.c, (j, b) in q.c
        k = quadkey(i, j)
        nv = get(gram, k, PQuad()) + quadmul(a, b)
        if iszero(nv)
            delete!(gram, k)
        else
            gram[k] = nv
        end
    end
    return QExpr(gram, Dict{Int,PAff}(), PQuad())
end

LinearAlgebra.dot(p::PointExpr, q::PointExpr) = inner(p, q)

"""Squared Euclidean norm ‖p‖² as a QExpr."""
sqnorm(p::PointExpr) = inner(p, p)

# ──────────────────────────────────────────────────────────────────────────────

struct PEPConstraint
    expr::QExpr      # normalized: expr ≤ 0  or  expr == 0
    sense::Symbol    # :le or :eq
    name::String
end

"""
    PSDBlock

Matrix LMI constraint `T(G, F, η) ⪰ 0` with QExpr entries (symmetrized at
compile time). Used by quadratic/linear-operator interpolation classes.
"""
struct PSDBlock
    mat::Matrix{QExpr}
    name::String
end

"""
    PEPModel

Container for a parametric PEP: Gram basis, function values, coefficients η,
constraints, PSD blocks, registered function-class objects, and the
(max or max-min) objective.
"""
mutable struct PEPModel
    dim::Int                       # Gram dimension
    nf::Int                        # number of function values
    np::Int                        # number of coefficients η
    point_names::Vector{String}
    cons::Vector{PEPConstraint}
    psd::Vector{PSDBlock}
    funcs::Vector{Any}             # frontend class objects (functions/operators)
    finalized::Bool                # class constraints already emitted
    obj::Vector{QExpr}             # objective pieces: maximize min_k piece_k
end

PEPModel() = PEPModel(0, 0, 0, String[], PEPConstraint[], PSDBlock[], Any[],
                      false, QExpr[])

"""Declare a new Gram basis vector (decision point) and return it."""
function point!(m::PEPModel; name::String = "p$(m.dim + 1)")
    m.dim += 1
    push!(m.point_names, name)
    return PointExpr(Dict(m.dim => PAff(1.0)))
end

"""Declare a new function-value entry and return its reference."""
function fval!(m::PEPModel)
    m.nf += 1
    return FVal(m.nf)
end

"""Register `k` new method coefficients η and return them as affine symbols."""
function coeffs!(m::PEPModel, k::Int)
    base = m.np
    m.np += k
    return [coeff(base + i) for i in 1:k]
end

"""Add the constraint `expr ≤ 0`."""
add_le!(m::PEPModel, e::QLike; name::String = "c$(length(m.cons) + 1)") =
    (push!(m.cons, PEPConstraint(qexpr(e), :le, name)); nothing)

"""Add the constraint `expr == 0`."""
add_eq!(m::PEPModel, e::QLike; name::String = "c$(length(m.cons) + 1)") =
    (push!(m.cons, PEPConstraint(qexpr(e), :eq, name)); nothing)

"""Add the LMI constraint `mat ⪰ 0` (square matrix of QLike entries)."""
function add_psd!(m::PEPModel, mat::AbstractMatrix;
                  name::String = "psd$(length(m.psd) + 1)")
    size(mat, 1) == size(mat, 2) || error("PSD block must be square")
    push!(m.psd, PSDBlock(QExpr[qexpr(e) for e in mat], name))
    return nothing
end

# Frontend hook: class objects registered on the model emit their
# interpolation constraints once, at compile time.
function add_class_constraints! end

"""Emit interpolation constraints of all registered class objects (idempotent)."""
function finalize_classes!(m::PEPModel)
    m.finalized && return nothing
    m.finalized = true          # set first: classes may not re-enter
    for f in m.funcs
        add_class_constraints!(f)
    end
    return nothing
end

"""Set the objective `maximize expr`."""
objective_max!(m::PEPModel, e::QLike) = (m.obj = [qexpr(e)]; nothing)

"""Set the objective `maximize min_k expr_k` (epigraph form)."""
objective_maxmin!(m::PEPModel, es::AbstractVector) =
    (m.obj = QExpr[qexpr(e) for e in es]; nothing)
