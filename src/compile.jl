# Compilation: PEPModel → CompiledPEP with exact sparse (A0, A1, A2) tensors.
#
# Each scalar expression tr(A(η)G) + f'F + c0 is stored with
#   A(η) = A0 + Σ_r η_r A1[r] + Σ_{r≤s} η_r η_s A2[(r,s)],
# so assembly and all derivatives w.r.t. η are exact sparse operations.

const SpMat = SparseMatrixCSC{Float64,Int}

"""
    ParamMatrix

Symmetric Gram-space matrix with entries polynomial (degree ≤ 2) in η.
"""
struct ParamMatrix
    dim::Int
    A0::SpMat
    A1::Vector{Pair{Int,SpMat}}          # η_r => coefficient matrix of η_r
    A2::Vector{Tuple{Int,Int,SpMat}}     # (r ≤ s, coefficient matrix of η_r*η_s)
end

is_param_dependent(M::ParamMatrix) = !isempty(M.A1) || !isempty(M.A2)

# COO triplet accumulator honoring tr(GA) = Σ_i A_ii G_ii + 2 Σ_{i<j} A_ij G_ij.
function _pushsym!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
                   i::Int, j::Int, c::Float64)
    if i == j
        push!(I, i); push!(J, j); push!(V, c)
    else
        push!(I, i); push!(J, j); push!(V, c / 2)
        push!(I, j); push!(J, i); push!(V, c / 2)
    end
    return nothing
end

function _param_matrix(gram::Dict{Tuple{Int,Int},PQuad}, dim::Int)
    I0, J0, V0 = Int[], Int[], Float64[]
    coo1 = Dict{Int,NTuple{3,Vector}}()
    coo2 = Dict{Tuple{Int,Int},NTuple{3,Vector}}()
    for ((i, j), q) in gram
        q.c0 == 0.0 || _pushsym!(I0, J0, V0, i, j, q.c0)
        for (r, c) in q.lin
            t = get!(coo1, r) do
                (Int[], Int[], Float64[])
            end
            _pushsym!(t[1], t[2], t[3], i, j, c)
        end
        for (rs, c) in q.quad
            t = get!(coo2, rs) do
                (Int[], Int[], Float64[])
            end
            _pushsym!(t[1], t[2], t[3], i, j, c)
        end
    end
    A0 = sparse(I0, J0, V0, dim, dim)
    A1 = [r => sparse(t[1], t[2], t[3], dim, dim) for (r, t) in sort!(collect(coo1); by = first)]
    A2 = [(rs[1], rs[2], sparse(t[1], t[2], t[3], dim, dim))
          for (rs, t) in sort!(collect(coo2); by = first)]
    return ParamMatrix(dim, A0, A1, A2)
end

"""
    CompiledExpr

One compiled scalar expression: `tr(A(η)G) + f'F + c0`.
"""
struct CompiledExpr
    M::ParamMatrix
    f::Vector{Float64}
    c0::Float64
end

struct CompiledConstraint
    expr::CompiledExpr
    sense::Symbol
    name::String
end

"""
    CompiledPSD

One compiled LMI block: symmetric matrix of `CompiledExpr` entries, `T ⪰ 0`.
"""
struct CompiledPSD
    mat::Matrix{CompiledExpr}
    name::String
end

"""
    CompiledPEP

Fully compiled parametric PEP, ready for repeated numeric solves.
"""
struct CompiledPEP
    dim::Int
    nf::Int
    np::Int
    cons::Vector{CompiledConstraint}
    psd::Vector{CompiledPSD}
    obj::Vector{CompiledExpr}          # maximize min_k piece_k (singleton = plain max)
end

function _compile_qexpr(e::QExpr, m::PEPModel)
    M = _param_matrix(e.gram, m.dim)
    fv = zeros(m.nf)
    for (k, a) in e.f
        isconst(a) || error("function-value coefficients cannot depend on η")
        fv[k] = a.c0
    end
    isconst(e.cnst) || error("constant terms cannot depend on η")
    return CompiledExpr(M, fv, e.cnst.c0)
end

"""
    compile(m::PEPModel) :: CompiledPEP

Extract all constant/linear/quadratic constraint structure once. Emits any
registered function-class interpolation constraints first.
"""
function compile(m::PEPModel)
    finalize_classes!(m)
    isempty(m.obj) && error("no objective set; call objective_max! or objective_maxmin!")
    cons = [CompiledConstraint(_compile_qexpr(c.expr, m), c.sense, c.name) for c in m.cons]
    psd = [CompiledPSD(
               [_compile_qexpr(0.5 * (b.mat[i, j] + b.mat[j, i]), m)
                for i in 1:size(b.mat, 1), j in 1:size(b.mat, 2)],
               b.name) for b in m.psd]
    obj = [_compile_qexpr(e, m) for e in m.obj]
    return CompiledPEP(m.dim, m.nf, m.np, cons, psd, obj)
end

# ── numeric assembly and derivatives ─────────────────────────────────────────

"""Assemble the numeric matrix A(η)."""
function assemble(M::ParamMatrix, η::AbstractVector{<:Real})
    A = copy(M.A0)
    for (r, S) in M.A1
        A += η[r] * S
    end
    for (r, s, S) in M.A2
        A += (r == s ? η[r]^2 : η[r] * η[s]) * S
    end
    return A
end

"""Assemble ∂A/∂η_r at η."""
function dmat(M::ParamMatrix, r::Int, η::AbstractVector{<:Real})
    A = spzeros(M.dim, M.dim)
    for (r1, S) in M.A1
        r1 == r && (A += S)
    end
    for (r1, s1, S) in M.A2
        if r1 == s1 == r
            A += 2 * η[r] * S
        elseif r1 == r
            A += η[s1] * S
        elseif s1 == r
            A += η[r1] * S
        end
    end
    return A
end

"""Assemble ∂²A/∂η_r∂η_s (constant in η)."""
function d2mat(M::ParamMatrix, r::Int, s::Int)
    A = spzeros(M.dim, M.dim)
    key = quadkey(r, s)
    for (r1, s1, S) in M.A2
        if (r1, s1) == key
            A += (r == s ? 2.0 : 1.0) * S
        end
    end
    return A
end

"""Fast tr(G*S) for dense symmetric G and sparse S."""
function trprod(G::AbstractMatrix{Float64}, S::SpMat)
    acc = 0.0
    rows = rowvals(S)
    vals = nonzeros(S)
    for j in 1:size(S, 2)
        for k in nzrange(S, j)
            acc += vals[k] * G[j, rows[k]]
        end
    end
    return acc
end
