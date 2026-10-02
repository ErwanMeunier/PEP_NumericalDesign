# Exact partial facial reduction for Gram SDPs.
#
# The construction specializes Borwein--Wolkowicz facial reduction to exposed
# PSD faces already visible in the compiled PEP equalities. If
#
#     tr(A G) = 0,    A ⪰ 0,    G ⪰ 0,
#
# then range(A) ⊆ ker(G). Stacking those exposed directions in R and taking an
# orthonormal basis Z of ker(R') gives the exact parameterization G = Z H Z',
# H ⪰ 0. See Borwein & Wolkowicz (1981) and Permenter & Parrilo (2018).

const FACIAL_REDUCTION_MODES = (:none, :explicit)

"""
    FacialReductionInfo

Description of an opt-in Gram-cone facial reduction. `basis` is the matrix Z
in `G = Z*H*Z'`; `removed_constraints` contains the compiled equality rows
made redundant by that parameterization.
"""
struct FacialReductionInfo
    mode::Symbol
    original_dim::Int
    reduced_dim::Int
    exposing_rank::Int
    removed_constraints::Vector{Int}
    basis::Matrix{Float64}
    rtol::Float64
    atol::Float64
end

facial_reduction_applied(info::FacialReductionInfo) =
    info.reduced_dim < info.original_dim

function _check_facial_reduction_mode(mode::Symbol)
    mode in FACIAL_REDUCTION_MODES ||
        error("unknown facial_reduction mode :$mode " *
              "(use :none or :explicit)")
    return mode
end

function _no_facial_reduction(dim::Int; mode::Symbol = :none,
                              rtol::Float64 = 0.0,
                              atol::Float64 = 0.0)
    return FacialReductionInfo(mode, dim, dim, 0, Int[],
                               Matrix{Float64}(I, dim, dim), rtol, atol)
end

"""
    facial_reduction_info(cp, η; mode=:explicit, rtol=1e-9, atol=1e-11)

Find the exact PSD face exposed by compiled equality constraints of the form
`tr(A(η)G) = 0` with `A(η) ⪰ 0` and no function-value or constant term.

`mode=:none` returns the identity parameterization. `mode=:explicit` is a
partial facial reduction: it uses only faces explicitly exposed by existing
PEP equalities and does not solve auxiliary facial-reduction SDPs.
"""
function facial_reduction_info(cp::CompiledPEP, η::AbstractVector{<:Real};
                               mode::Symbol = :explicit,
                               rtol::Float64 = 1e-9,
                               atol::Float64 = 1e-11)
    _check_facial_reduction_mode(mode)
    length(η) == cp.np ||
        error("η has length $(length(η)), expected $(cp.np)")
    rtol >= 0 || error("facial-reduction rtol must be nonnegative")
    atol >= 0 || error("facial-reduction atol must be nonnegative")
    mode == :none && return _no_facial_reduction(cp.dim; mode, rtol, atol)

    directions = Vector{Vector{Float64}}()
    removed = Int[]
    for (index, constraint) in enumerate(cp.cons)
        constraint.sense == :eq || continue
        all(iszero, constraint.expr.f) || continue
        iszero(constraint.expr.c0) || continue

        A = Matrix(assemble(constraint.expr.M, η))
        A = 0.5 .* (A .+ A')
        decomposition = eigen(Symmetric(A))
        scale = maximum(abs, decomposition.values; init = 0.0)
        scale > atol || continue
        tolerance = max(atol, rtol * scale)
        minimum(decomposition.values) >= -tolerance || continue
        positive = findall(>(tolerance), decomposition.values)
        isempty(positive) && continue

        append!(directions,
                [Vector(decomposition.vectors[:, i]) for i in positive])
        push!(removed, index)
    end

    isempty(directions) &&
        return _no_facial_reduction(cp.dim; mode, rtol, atol)

    R = hcat(directions...)
    decomposition = svd(R'; full = true)
    scale = maximum(decomposition.S; init = 0.0)
    tolerance = max(atol, rtol * scale)
    exposing_rank = count(>(tolerance), decomposition.S)
    basis = Matrix(decomposition.V[:, (exposing_rank + 1):end])
    return FacialReductionInfo(mode, cp.dim, size(basis, 2), exposing_rank,
                               removed, basis, rtol, atol)
end
