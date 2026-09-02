# Operator classes: (necessary) interpolation constraints over (x, Tx) pairs.
#
# Ported from PEPit.jl src/operators/ (MIT License, Copyright (c) 2025
# Shuvomoy Das Gupta and contributors). Oracle triples store (x, g, f) with
# g the operator value/selection at x; function values are unused by the
# constraints but kept for API uniformity. Classes marked [LMI] add PSD blocks.

_upairs(n) = ((i, j) for i in 1:n for j in (i + 1):n)

"""
    MonotoneOperator(m; name)

Maximally monotone operators: `⟨g_i − g_j, x_i − x_j⟩ ≥ 0` for all i < j.
"""
struct MonotoneOperator <: AbstractPEPFunction
    core::PEPFunc
end
MonotoneOperator(m::PEPModel; name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, MonotoneOperator(PEPFunc(m, name, false)))

function add_class_constraints!(op::MonotoneOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        add_le!(c.model, -inner(c.gs[i] - c.gs[j], c.xs[i] - c.xs[j]);
                name = "$(c.name):mono[$i,$j]")
    end
end

"""
    StronglyMonotoneOperator(m; μ, name)

μ-strongly (maximally) monotone: `⟨g_i − g_j, x_i − x_j⟩ ≥ μ‖x_i − x_j‖²`.
"""
struct StronglyMonotoneOperator <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
end
StronglyMonotoneOperator(m::PEPModel; μ::Real,
                         name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, StronglyMonotoneOperator(PEPFunc(m, name, false), μ))

function add_class_constraints!(op::StronglyMonotoneOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        add_le!(c.model, op.μ * sqnorm(dx) - inner(c.gs[i] - c.gs[j], dx);
                name = "$(c.name):smono[$i,$j]")
    end
end

"""
    LipschitzOperator(m; L, name)

L-Lipschitz operators: `‖g_i − g_j‖² ≤ L²‖x_i − x_j‖²` for all i < j.
"""
struct LipschitzOperator <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
end
LipschitzOperator(m::PEPModel; L::Real, name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, LipschitzOperator(PEPFunc(m, name, true), L))

function add_class_constraints!(op::LipschitzOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        add_le!(c.model, sqnorm(c.gs[i] - c.gs[j]) -
                         op.L^2 * sqnorm(c.xs[i] - c.xs[j]);
                name = "$(c.name):lip[$i,$j]")
    end
end

"""
    CocoerciveOperator(m; β, name)

β-cocoercive operators: `⟨g_i − g_j, x_i − x_j⟩ ≥ β‖g_i − g_j‖²`.
"""
struct CocoerciveOperator <: AbstractPEPFunction
    core::PEPFunc
    β::Float64
end
CocoerciveOperator(m::PEPModel; β::Real, name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, CocoerciveOperator(PEPFunc(m, name, true), β))

function add_class_constraints!(op::CocoerciveOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        dg = c.gs[i] - c.gs[j]
        add_le!(c.model, op.β * sqnorm(dg) - inner(dg, c.xs[i] - c.xs[j]);
                name = "$(c.name):coco[$i,$j]")
    end
end

"""
    NonexpansiveOperator(m; v=nothing, name)

(Possibly inconsistent) nonexpansive operators: `‖g_i − g_j‖² ≤ ‖x_i − x_j‖²`;
with infimal displacement vector `v`: `‖v‖² ≤ ⟨x_i − g_i, v⟩`.
"""
struct NonexpansiveOperator <: AbstractPEPFunction
    core::PEPFunc
    v::Union{PointExpr,Nothing}
end
NonexpansiveOperator(m::PEPModel; v = nothing,
                     name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, NonexpansiveOperator(PEPFunc(m, name, true), v))

function add_class_constraints!(op::NonexpansiveOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        add_le!(c.model, sqnorm(c.gs[i] - c.gs[j]) - sqnorm(c.xs[i] - c.xs[j]);
                name = "$(c.name):ne[$i,$j]")
    end
    if op.v !== nothing
        for i in eachindex(c.xs)
            add_le!(c.model, sqnorm(op.v) - inner(c.xs[i] - c.gs[i], op.v);
                    name = "$(c.name):idv[$i]")
        end
    end
end

"""
    NegativelyComonotoneOperator(m; ρ, name)

ρ-negatively comonotone operators (necessary conditions):
`⟨g_i − g_j, x_i − x_j⟩ ≥ −ρ‖g_i − g_j‖²`.
"""
struct NegativelyComonotoneOperator <: AbstractPEPFunction
    core::PEPFunc
    ρ::Float64
end
NegativelyComonotoneOperator(m::PEPModel; ρ::Real,
                             name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, NegativelyComonotoneOperator(PEPFunc(m, name, true), ρ))

function add_class_constraints!(op::NegativelyComonotoneOperator)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        dg = c.gs[i] - c.gs[j]
        add_le!(c.model, -op.ρ * sqnorm(dg) - inner(dg, c.xs[i] - c.xs[j]);
                name = "$(c.name):ncomo[$i,$j]")
    end
end

"""
    LipschitzStronglyMonotoneOperatorCheap(m; μ, L, name)

L-Lipschitz μ-strongly monotone operators, cheap necessary conditions
(pairwise strong monotonicity + Lipschitz).
"""
struct LipschitzStronglyMonotoneOperatorCheap <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
LipschitzStronglyMonotoneOperatorCheap(m::PEPModel; μ::Real, L::Real,
                                       name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, LipschitzStronglyMonotoneOperatorCheap(PEPFunc(m, name, true), μ, L))

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorCheap)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        dg = c.gs[i] - c.gs[j]
        add_le!(c.model, op.μ * sqnorm(dx) - inner(dg, dx);
                name = "$(c.name):smono[$i,$j]")
        add_le!(c.model, sqnorm(dg) - op.L^2 * sqnorm(dx);
                name = "$(c.name):lip[$i,$j]")
    end
end

"""
    CocoerciveStronglyMonotoneOperatorCheap(m; μ, β, name)

β-cocoercive μ-strongly monotone operators, cheap necessary conditions.
"""
struct CocoerciveStronglyMonotoneOperatorCheap <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    β::Float64
end
CocoerciveStronglyMonotoneOperatorCheap(m::PEPModel; μ::Real, β::Real,
                                        name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, CocoerciveStronglyMonotoneOperatorCheap(PEPFunc(m, name, true), μ, β))

function add_class_constraints!(op::CocoerciveStronglyMonotoneOperatorCheap)
    c = op.core
    for (i, j) in _upairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        dg = c.gs[i] - c.gs[j]
        add_le!(c.model, op.β * sqnorm(dg) - inner(dg, dx);
                name = "$(c.name):coco[$i,$j]")
        add_le!(c.model, op.μ * sqnorm(dx) - inner(dg, dx);
                name = "$(c.name):smono[$i,$j]")
    end
end

# ── strengthened classes with 7×7 LMI blocks per point triplet [LMI] ─────────

# Shared 7×7 block builder (Rubbens–Hendrickx strengthened conditions): given
# the six pairwise residual expressions and 9 free slacks, assemble T ⪰ 0.
function _sevenbyseven(m::PEPModel, Aij, Aik, Ajk, Bij, Bik, Bjk, M55c, name)
    M14 = qexpr(fval!(m)); M15 = qexpr(fval!(m)); M16 = qexpr(fval!(m))
    M17 = qexpr(fval!(m)); M26 = qexpr(fval!(m)); M27 = qexpr(fval!(m))
    M34 = qexpr(fval!(m)); M37 = qexpr(fval!(m)); M46 = qexpr(fval!(m))
    M25 = -M14; M23 = -M15; M35 = -M16
    M45 = -M27; M56 = -M37; M57 = -M46
    M55 = M55c - 2.0 * M17 - 2.0 * M26 - 2.0 * M34
    z = qexpr(0.0)
    T = Matrix{QExpr}(undef, 7, 7)
    T[1, :] = [-Bij, z, z, M14, M15, M16, M17]
    T[2, :] = [z, -Ajk, M23, z, M25, M26, M27]
    T[3, :] = [z, M23, -Bij, M34, M35, z, M37]
    T[4, :] = [M14, z, M34, -Bjk, M45, M46, z]
    T[5, :] = [M15, M25, M35, M45, M55, M56, M57]
    T[6, :] = [M16, M26, z, M46, M56, -Aik, z]
    T[7, :] = [M17, M27, M37, z, M57, z, -Bik]
    add_psd!(m, T; name = name)
    return nothing
end

"""
    LipschitzStronglyMonotoneOperatorExpensive(m; μ, L, name)

L-Lipschitz μ-strongly monotone operators — strengthened necessary conditions
(two 7×7 LMI blocks with 9 slacks per ordered point triplet). Expensive.
"""
struct LipschitzStronglyMonotoneOperatorExpensive <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
LipschitzStronglyMonotoneOperatorExpensive(m::PEPModel; μ::Real, L::Real,
        name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, LipschitzStronglyMonotoneOperatorExpensive(
        PEPFunc(m, name, true), μ, L))

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorExpensive)
    c = op.core
    m = c.model
    L, μ = op.L, op.μ
    lipres(p, q) = sqnorm(c.gs[p] - c.gs[q]) - L^2 * sqnorm(c.xs[p] - c.xs[q])
    smres(p, q) = 2L * (μ * sqnorm(c.xs[p] - c.xs[q]) -
                        inner(c.gs[p] - c.gs[q], c.xs[p] - c.xs[q]))
    n = length(c.xs)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        for opt in (1, 0)
            if opt == 1
                Aij, Aik, Ajk = lipres(i, j), lipres(i, k), lipres(k, j)
                Bij, Bik, Bjk = smres(i, j), smres(i, k), smres(k, j)
                M55c = Aij + 2μ * Bij - Ajk - Aik
            else
                Bij, Bik, Bjk = lipres(i, j), lipres(i, k), lipres(k, j)
                Aij, Aik, Ajk = smres(i, j), smres(i, k), smres(k, j)
                M55c = Aij + 2μ * Bij - Ajk - Aik
            end
            _sevenbyseven(m, Aij, Aik, Ajk, Bij, Bik, Bjk, M55c,
                          "$(c.name):lsm$(opt)[$i,$j,$k]")
        end
    end
end

"""
    CocoerciveStronglyMonotoneOperatorExpensive(m; μ, β, name)

β-cocoercive μ-strongly monotone operators — strengthened necessary
conditions (two 7×7 LMI blocks with 9 slacks per ordered triplet). Expensive.
"""
struct CocoerciveStronglyMonotoneOperatorExpensive <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    β::Float64
end
CocoerciveStronglyMonotoneOperatorExpensive(m::PEPModel; μ::Real, β::Real,
        name::String = "T$(length(m.funcs) + 1)") =
    _register!(m, CocoerciveStronglyMonotoneOperatorExpensive(
        PEPFunc(m, name, true), μ, β))

function add_class_constraints!(op::CocoerciveStronglyMonotoneOperatorExpensive)
    c = op.core
    m = c.model
    μ, β = op.μ, op.β
    smres(p, q) = μ * sqnorm(c.xs[p] - c.xs[q]) -
                  inner(c.gs[p] - c.gs[q], c.xs[p] - c.xs[q])
    cocres(p, q) = β * sqnorm(c.gs[p] - c.gs[q]) -
                   inner(c.gs[p] - c.gs[q], c.xs[p] - c.xs[q])
    n = length(c.xs)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        for opt in (1, 0)
            if opt == 1
                Aij, Aik, Ajk = smres(i, j), smres(i, k), smres(k, j)
                Bij, Bik, Bjk = cocres(i, j), cocres(i, k), cocres(k, j)
            else
                Bij, Bik, Bjk = smres(i, j), smres(i, k), smres(k, j)
                Aij, Aik, Ajk = cocres(i, j), cocres(i, k), cocres(k, j)
            end
            M55c = Aij - Ajk - Aik - (2.0 * (1 - 2β * μ)) * Bij
            _sevenbyseven(m, Aij, Aik, Ajk, Bij, Bik, Bjk, M55c,
                          "$(c.name):csm$(opt)[$i,$j,$k]")
        end
    end
end

# ── linear operator classes [LMI] ────────────────────────────────────────────

"""
    LinearOperator(m; L, name)

Linear operators `M` with singular values ≤ L. Forward oracles via
`gradient!(op, x)` (pairs (x, Mx)); adjoint oracles via
`adjoint_oracle!(op, u)` (pairs (u, Mᵀu)). Adds cross equalities
`⟨x_i, v_j⟩ = ⟨y_i, u_j⟩` and two Gram LMI blocks `L²⟨x,x⟩ − ⟨y,y⟩ ⪰ 0`.
"""
struct LinearOperator <: AbstractPEPFunction
    core::PEPFunc
    adj::PEPFunc
    L::Float64
end
LinearOperator(m::PEPModel; L::Real, name::String = "M$(length(m.funcs) + 1)") =
    _register!(m, LinearOperator(PEPFunc(m, name, true),
                                 PEPFunc(m, name * "adj", true), L))

"""Adjoint oracle `(v, f) = (Mᵀu, ⋅)` of a `LinearOperator` at `u`."""
function adjoint_oracle!(op::LinearOperator, u::PointExpr)
    c = op.adj
    i = findfirst(==(u), c.xs)
    i !== nothing && return c.gs[i], c.fs[i]
    v = point!(c.model; name = "v_$(c.name)_$(length(c.xs) + 1)")
    fe = qexpr(fval!(c.model))
    push!(c.xs, u); push!(c.gs, v); push!(c.fs, fe)
    return v, fe
end

function add_class_constraints!(op::LinearOperator)
    c = op.core
    m = c.model
    for i in eachindex(c.xs), j in eachindex(op.adj.xs)
        add_eq!(m, inner(c.xs[i], op.adj.gs[j]) - inner(c.gs[i], op.adj.xs[j]);
                name = "$(c.name):adj[$i,$j]")
    end
    for (pts, tag) in ((c, "fwd"), (op.adj, "rev"))
        n = length(pts.xs)
        n == 0 && continue
        T = Matrix{QExpr}(undef, n, n)
        for i in 1:n, j in 1:n
            T[i, j] = op.L^2 * inner(pts.xs[i], pts.xs[j]) -
                      inner(pts.gs[i], pts.gs[j])
        end
        add_psd!(m, T; name = "$(c.name):sv$tag")
    end
end

"""
    SymmetricLinearOperator(m; μ, L, name)

Symmetric linear operators with eigenvalues in [μ, L]: pairwise symmetry
equalities plus the Gram LMI of `(LI − M)(M − μI) ⪰ 0`.
"""
struct SymmetricLinearOperator <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
SymmetricLinearOperator(m::PEPModel; μ::Real, L::Real,
                        name::String = "M$(length(m.funcs) + 1)") =
    _register!(m, SymmetricLinearOperator(PEPFunc(m, name, true), μ, L))

function add_class_constraints!(op::SymmetricLinearOperator)
    c = op.core
    m = c.model
    n = length(c.xs)
    for (i, j) in _upairs(n)
        add_eq!(m, inner(c.xs[i], c.gs[j]) - inner(c.xs[j], c.gs[i]);
                name = "$(c.name):sym[$i,$j]")
    end
    n == 0 && return
    T = Matrix{QExpr}(undef, n, n)
    for i in 1:n, j in 1:n
        T[i, j] = op.L * inner(c.gs[i], c.xs[j]) - inner(c.gs[i], c.gs[j]) -
                  (op.μ * op.L) * inner(c.xs[i], c.xs[j]) +
                  op.μ * inner(c.xs[i], c.gs[j])
    end
    add_psd!(m, T; name = "$(c.name):ev")
end

"""
    SkewSymmetricLinearOperator(m; L, name)

Skew-symmetric linear operators with singular values ≤ L: skew equalities
plus the Gram LMI `L²⟨x,x⟩ − ⟨g,g⟩ ⪰ 0`.
"""
struct SkewSymmetricLinearOperator <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
end
SkewSymmetricLinearOperator(m::PEPModel; L::Real,
                            name::String = "M$(length(m.funcs) + 1)") =
    _register!(m, SkewSymmetricLinearOperator(PEPFunc(m, name, true), L))

function add_class_constraints!(op::SkewSymmetricLinearOperator)
    c = op.core
    m = c.model
    n = length(c.xs)
    for (i, j) in _upairs(n)
        add_eq!(m, inner(c.xs[i], c.gs[j]) + inner(c.xs[j], c.gs[i]);
                name = "$(c.name):skew[$i,$j]")
    end
    for i in 1:n
        add_eq!(m, inner(c.xs[i], c.gs[i]); name = "$(c.name):skew0[$i]")
    end
    n == 0 && return
    T = Matrix{QExpr}(undef, n, n)
    for i in 1:n, j in 1:n
        T[i, j] = op.L^2 * inner(c.xs[i], c.xs[j]) - inner(c.gs[i], c.gs[j])
    end
    add_psd!(m, T; name = "$(c.name):sv")
end
