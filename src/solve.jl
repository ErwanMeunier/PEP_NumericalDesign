# SDP solve layer: swappable backends and dual-normalized solutions.

abstract type SDPBackend end

"""
    MosekBackend(; tol=1e-8, feas_tol=1e-9, threads=1, verbose=false)

Default backend. `threads=1` prevents over-subscription under Julia @threads.
"""
Base.@kwdef struct MosekBackend <: SDPBackend
    tol::Float64 = 1e-8
    feas_tol::Float64 = 1e-9
    threads::Int = 1
    verbose::Bool = false
end

"""
    GenericBackend(factory; attributes=Pair{String,Any}[], verbose=false)

Any JuMP-compatible SDP optimizer (e.g. `Clarabel.Optimizer`), for
license-free or experimental (GPU) setups.
"""
struct GenericBackend <: SDPBackend
    factory::Any
    attributes::Vector{Pair{String,Any}}
    verbose::Bool
end
GenericBackend(factory; attributes = Pair{String,Any}[], verbose = false) =
    GenericBackend(factory, attributes, verbose)

function _make_model(b::MosekBackend)
    model = Model(Mosek.Optimizer)
    b.verbose || set_silent(model)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", b.tol)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS", b.feas_tol)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS", b.feas_tol)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", b.threads)
    return model
end

function _make_model(b::GenericBackend)
    model = Model(b.factory)
    b.verbose || set_silent(model)
    for (k, v) in b.attributes
        set_attribute(model, k, v)
    end
    return model
end

"""
    PEPSolution

Primal-dual PEP solution. `duals` is aligned with `CompiledPEP.cons` and
sign-normalized so that λ ≥ 0 for `:le` constraints and the envelope formula
∂W/∂η_r = −Σ_c λ_c tr(G ∂A_c/∂η_r) + Σ_b tr(Λ_b ∂T_b/∂η_r)
        + Σ_k γ_k tr(G ∂C_k/∂η_r) holds. `psd_duals` (Λ_b ⪰ 0) is aligned
with `CompiledPEP.psd`.
"""
struct PEPSolution
    obj::Float64
    G::Matrix{Float64}
    F::Vector{Float64}
    duals::Vector{Float64}
    psd_duals::Vector{Matrix{Float64}}
    obj_duals::Vector{Float64}
    status::Any
end

# Convenience for PSD-free problems (synthetic certificates, SSDP init).
PEPSolution(obj, G, F, duals, obj_duals, status) =
    PEPSolution(obj, G, F, duals, Matrix{Float64}[], obj_duals, status)

function _jump_expr(ce::CompiledExpr, η, G, F)
    A = assemble(ce.M, η)
    Is, Js, Vs = findnz(A)
    ex = AffExpr(ce.c0)
    for k in eachindex(Vs)
        add_to_expression!(ex, Vs[k], G[Is[k], Js[k]])
    end
    for k in eachindex(ce.f)
        ce.f[k] == 0.0 || add_to_expression!(ex, ce.f[k], F[k])
    end
    return ex
end

"""
    solve_pep(cp, η; backend=MosekBackend(), warn=true) :: PEPSolution

Solve the compiled PEP at coefficient values η.
"""
function solve_pep(cp::CompiledPEP, η::AbstractVector{<:Real};
                   backend::SDPBackend = MosekBackend(), warn::Bool = true)
    length(η) == cp.np ||
        error("η has length $(length(η)), expected $(cp.np)")
    model = _make_model(backend)
    G = @variable(model, [1:cp.dim, 1:cp.dim] in PSDCone())
    F = @variable(model, [1:cp.nf])

    conrefs = Vector{ConstraintRef}(undef, length(cp.cons))
    for (c, con) in enumerate(cp.cons)
        ex = _jump_expr(con.expr, η, G, F)
        conrefs[c] = con.sense == :le ? @constraint(model, ex <= 0) :
                                        @constraint(model, ex == 0)
    end

    psd_refs = Vector{ConstraintRef}(undef, length(cp.psd))
    for (b, blk) in enumerate(cp.psd)
        s = size(blk.mat, 1)
        S = [_jump_expr(blk.mat[i, j], η, G, F) for i in 1:s, j in 1:s]
        psd_refs[b] = @constraint(model, LinearAlgebra.Symmetric(S) in PSDCone())
    end

    piece_refs = ConstraintRef[]
    if length(cp.obj) == 1
        @objective(model, Max, _jump_expr(cp.obj[1], η, G, F))
    else
        t = @variable(model)
        for piece in cp.obj
            push!(piece_refs, @constraint(model, t <= _jump_expr(piece, η, G, F)))
        end
        @objective(model, Max, t)
    end

    optimize!(model)
    status = termination_status(model)
    if warn && status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        @warn "PEP solve did not reach optimality" status
    end

    # MOI duals of a Max problem are the negatives of the classic KKT multipliers
    # for scalar rows; PSD-cone duals already come out as the KKT Λ ⪰ 0 (FD-checked).
    duals = [-dual(conrefs[c]) for c in eachindex(conrefs)]
    psd_duals = [Matrix{Float64}(dual(psd_refs[b])) for b in eachindex(psd_refs)]
    obj_duals = isempty(piece_refs) ? [1.0] : [-dual(r) for r in piece_refs]
    return PEPSolution(objective_value(model), value.(G), value.(F),
                       duals, psd_duals, obj_duals, status)
end
