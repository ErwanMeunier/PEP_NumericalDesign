# SDP solve layer: swappable backends, opt-in exact facial reduction, and
# dual-normalized solutions.

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

"""Numerical certificate and reconstructed-original-problem residuals."""
struct SolveDiagnostics
    primal_status::Any
    dual_status::Any
    raw_status::String
    relative_gap::Float64
    max_scalar_violation::Float64
    max_cone_violation::Float64
    certified::Bool
end

_default_diagnostics(status) = SolveDiagnostics(
    MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS, string(status),
    NaN, Inf, Inf, false)

"""
    PEPSolution

Primal-dual PEP solution. `duals` is aligned with `CompiledPEP.cons` and
sign-normalized so that λ ≥ 0 for `:le` constraints and the envelope formula
∂W/∂η_r = −Σ_c λ_c tr(G ∂A_c/∂η_r) + Σ_b tr(Λ_b ∂T_b/∂η_r)
        + Σ_k γ_k tr(G ∂C_k/∂η_r) holds. `psd_duals` (Λ_b ⪰ 0) is aligned
with `CompiledPEP.psd`.

When facial reduction is applied, `G` is reconstructed in the original Gram
space. Dual entries of eliminated exposing equalities are set to zero; use
`diagnostics` for value certification and do not pass such a solution to
`grad_hess_eta`.
"""
struct PEPSolution
    obj::Float64
    G::Matrix{Float64}
    F::Vector{Float64}
    duals::Vector{Float64}
    psd_duals::Vector{Matrix{Float64}}
    obj_duals::Vector{Float64}
    status::Any
    diagnostics::SolveDiagnostics
    reduction::FacialReductionInfo
end

# Backward-compatible constructors used by synthetic certificates and SSDP.
function PEPSolution(obj, G, F, duals, psd_duals, obj_duals, status)
    reduction = _no_facial_reduction(size(G, 1))
    return PEPSolution(obj, G, F, duals, psd_duals, obj_duals, status,
                       _default_diagnostics(status), reduction)
end
PEPSolution(obj, G, F, duals, obj_duals, status) =
    PEPSolution(obj, G, F, duals, Matrix{Float64}[], obj_duals, status)

function _jump_expr(ce::CompiledExpr, η, G, F, basis)
    A = assemble(ce.M, η)
    ex = AffExpr(ce.c0)
    if basis === nothing
        Is, Js, Vs = findnz(A)
        for k in eachindex(Vs)
            add_to_expression!(ex, Vs[k], G[Is[k], Js[k]])
        end
    else
        reduced = basis' * Matrix(A) * basis
        for j in axes(reduced, 2), i in axes(reduced, 1)
            reduced[i, j] == 0.0 ||
                add_to_expression!(ex, reduced[i, j], G[i, j])
        end
    end
    for k in eachindex(ce.f)
        ce.f[k] == 0.0 || add_to_expression!(ex, ce.f[k], F[k])
    end
    return ex
end

function _compiled_value(ce::CompiledExpr, η, G, F)
    return dot(assemble(ce.M, η), G) + dot(ce.f, F) + ce.c0
end

function _original_residuals(cp::CompiledPEP, η, G, F)
    scalar = 0.0
    for constraint in cp.cons
        value = _compiled_value(constraint.expr, η, G, F)
        violation = constraint.sense == :le ? max(value, 0.0) : abs(value)
        scalar = max(scalar, violation)
    end

    cone = isempty(G) ? 0.0 : max(-eigmin(Symmetric(G)), 0.0)
    for block in cp.psd
        n = size(block.mat, 1)
        matrix = [_compiled_value(block.mat[i, j], η, G, F)
                  for i in 1:n, j in 1:n]
        cone = max(cone, -eigmin(Symmetric(matrix)), 0.0)
    end
    return scalar, cone
end

_feasible_point(status) =
    status == MOI.FEASIBLE_POINT || status == MOI.NEARLY_FEASIBLE_POINT

function _relative_gap(model)
    try
        return relative_gap(model)
    catch
        return NaN
    end
end

function _solve_diagnostics(model, status, cp, η, G, F;
                            cert_tol::Float64, gap_tol::Float64)
    primal = primal_status(model)
    dual = dual_status(model)
    gap = _relative_gap(model)
    scalar, cone = _original_residuals(cp, η, G, F)
    acceptable_status = status in
        (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.SLOW_PROGRESS)
    certified = acceptable_status && _feasible_point(primal) &&
                _feasible_point(dual) && isfinite(gap) &&
                gap <= gap_tol && scalar <= cert_tol && cone <= cert_tol
    return SolveDiagnostics(primal, dual, raw_status(model), gap, scalar,
                            cone, certified)
end

"""
    solve_pep(cp, η; backend=MosekBackend(), warn=true,
              facial_reduction=:none, fr_rtol=1e-9, fr_atol=1e-11,
              cert_tol=1e-7, gap_tol=1e-6) :: PEPSolution

Solve the compiled PEP at coefficient values `η`.

Facial reduction is opt-in. `facial_reduction=:explicit` detects pure-Gram
PSD equalities, parameterizes their exposed face as `G=Z*H*Z'`, solves over
the smaller PSD variable `H`, and reconstructs the original Gram matrix.
Use `sol.diagnostics.certified` to distinguish a certified `SLOW_PROGRESS`
iterate from an uncertified solver value.
"""
function solve_pep(cp::CompiledPEP, η::AbstractVector{<:Real};
                   backend::SDPBackend = MosekBackend(), warn::Bool = true,
                   facial_reduction::Symbol = :none,
                   fr_rtol::Float64 = 1e-9, fr_atol::Float64 = 1e-11,
                   cert_tol::Float64 = 1e-7, gap_tol::Float64 = 1e-6)
    length(η) == cp.np ||
        error("η has length $(length(η)), expected $(cp.np)")
    cert_tol >= 0 || error("cert_tol must be nonnegative")
    gap_tol >= 0 || error("gap_tol must be nonnegative")
    reduction = facial_reduction_info(cp, η; mode = facial_reduction,
                                      rtol = fr_rtol, atol = fr_atol)
    reduced = facial_reduction_applied(reduction)
    basis = reduced ? reduction.basis : nothing
    gram_dim = reduction.reduced_dim

    model = _make_model(backend)
    Gred = if gram_dim == 0
        Matrix{VariableRef}(undef, 0, 0)
    else
        @variable(model, [1:gram_dim, 1:gram_dim] in PSDCone())
    end
    F = @variable(model, [1:cp.nf])

    conrefs = Vector{Union{Nothing,ConstraintRef}}(undef, length(cp.cons))
    fill!(conrefs, nothing)
    removed = Set(reduction.removed_constraints)
    for (c, con) in enumerate(cp.cons)
        c in removed && continue
        ex = _jump_expr(con.expr, η, Gred, F, basis)
        conrefs[c] = con.sense == :le ? @constraint(model, ex <= 0) :
                                        @constraint(model, ex == 0)
    end

    psd_refs = Vector{ConstraintRef}(undef, length(cp.psd))
    for (b, block) in enumerate(cp.psd)
        n = size(block.mat, 1)
        matrix = [_jump_expr(block.mat[i, j], η, Gred, F, basis)
                  for i in 1:n, j in 1:n]
        psd_refs[b] = @constraint(model,
            LinearAlgebra.Symmetric(matrix) in PSDCone())
    end

    piece_refs = ConstraintRef[]
    if length(cp.obj) == 1
        @objective(model, Max, _jump_expr(cp.obj[1], η, Gred, F, basis))
    else
        t = @variable(model)
        for piece in cp.obj
            push!(piece_refs,
                  @constraint(model,
                      t <= _jump_expr(piece, η, Gred, F, basis)))
        end
        @objective(model, Max, t)
    end

    optimize!(model)
    status = termination_status(model)
    has_values(model) || error("PEP solve returned no primal values (status=$status)")

    H = Matrix{Float64}(value.(Gred))
    G = reduced ? reduction.basis * H * reduction.basis' : H
    G = Matrix(Symmetric(G))
    Fvalue = Vector{Float64}(value.(F))
    diagnostics = _solve_diagnostics(model, status, cp, η, G, Fvalue;
                                     cert_tol, gap_tol)

    if warn && facial_reduction != :none && !diagnostics.certified
        @warn "facially reduced PEP solve is not numerically certified" status primal_status = diagnostics.primal_status dual_status = diagnostics.dual_status relative_gap = diagnostics.relative_gap max_scalar_violation = diagnostics.max_scalar_violation max_cone_violation = diagnostics.max_cone_violation facial_reduction reduced_dim = reduction.reduced_dim
    elseif warn && status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        @warn "PEP solve did not reach optimality" status
    end

    # MOI duals of a Max problem are the negatives of the classic KKT
    # multipliers for scalar rows; PSD-cone duals already have KKT sign.
    duals = zeros(length(cp.cons))
    if has_duals(model)
        for c in eachindex(conrefs)
            conrefs[c] === nothing || (duals[c] = -dual(conrefs[c]))
        end
    else
        fill!(duals, NaN)
    end
    psd_duals = has_duals(model) ?
        [Matrix{Float64}(dual(psd_refs[b])) for b in eachindex(psd_refs)] :
        [fill(NaN, size(block.mat)) for block in cp.psd]
    obj_duals = isempty(piece_refs) ? [1.0] :
        has_duals(model) ? [-dual(reference) for reference in piece_refs] :
                           fill(NaN, length(piece_refs))
    return PEPSolution(objective_value(model), G, Fvalue, duals, psd_duals,
                       obj_duals, status, diagnostics, reduction)
end
