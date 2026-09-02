# Sequential semidefinite programming (SSDP) for the PDP-NSDP (paper Alg. 2,
# following Fares–Noll–Apkarian): joint descent on ϑ = (λ, ω) for
#
#   min_{λ,ω}  c0_obj − Σ_c λ_c c0_c
#   s.t.       𝒜(ϑ) = Σ_c λ_c A_c(η(ω)) − C(η(ω)) ⪰ 0
#              h(ϑ) = b_obj − Σ_c λ_c f_c = 0
#              λ_c ≥ 0  (inequality-constraint multipliers)
#
# Convexified quadratic tangent subproblems use the exact NSDP Lagrangian
# Hessian blocks (λω and ωω), which are trace contractions of the compiled
# (A1, A2) tensors — no extra differentiation is needed.

"""
    SSDPTrace

History of an SSDP run: NSDP objective, KKT residuals, merit values, per-step
diagnostics (ζ convexification shift, γ step, predicted/actual merit
reduction), final multipliers, tangent count, and a verification PEP value.
"""
struct SSDPTrace
    ω_hist::Vector{Vector{Float64}}
    obj_hist::Vector{Float64}
    kkt_hist::Vector{Float64}
    merit_hist::Vector{Float64}
    ζ_hist::Vector{Float64}
    γ_hist::Vector{Float64}
    pred_hist::Vector{Float64}
    act_hist::Vector{Float64}
    λ::Vector{Float64}
    ntangent::Int
    W_final::Float64
end

# ‖Π_{S₋}(A)‖_F: Frobenius norm of the negative part.
function _psd_violation(A::AbstractMatrix{Float64})
    e = eigvals(Symmetric(Matrix(A)))
    return sqrt(sum(x -> min(x, 0.0)^2, e))
end

# Assemble all constraint matrices and the aggregated LMI matrix 𝒜(ϑ).
function _assemble_all(cp::CompiledPEP, η::Vector{Float64}, λ::Vector{Float64})
    Acs = [assemble(con.expr.M, η) for con in cp.cons]
    𝒜 = -Matrix(assemble(cp.obj[1].M, η))
    for c in eachindex(Acs)
        λ[c] == 0.0 || (𝒜 .+= λ[c] .* Matrix(Acs[c]))
    end
    return Acs, 𝒜
end

"""
    design_ssdp(dp, ω0; iters=30, δ=1e-6, ρ=10.0, σ=1e-4, tol=1e-6,
                hessian_mode=:block, λreg=1e-3, verbose=false) :: SSDPTrace

Sequential SDP on the joint NSDP. Initialization: one PEP solve at ω0
provides a feasible ϑ0 = (λ0, ω0) and multipliers (G0, y0) = (Gram, F).
Single-piece objectives only (use `:last_grad`-style objectives).

`hessian_mode = :block` (default) uses a block-diagonal convex model
(mild λ-regularization `λreg`, exact convexified ωω block; the λ–ω coupling
remains in the LMI linearization). `:full` shifts the exact joint Lagrangian
Hessian by its minimum eigenvalue (paper Alg. 2 verbatim; conservative).
"""
function design_ssdp(dp::DesignProblem, ω0::AbstractVector{<:Real};
                     iters::Int = 30, δ::Float64 = 1e-6, ρ::Float64 = 10.0,
                     σ::Float64 = 1e-4, tol::Float64 = 1e-6,
                     hessian_mode::Symbol = :block, λreg::Float64 = 1e-3,
                     verbose::Bool = false)
    cp = dp.cp
    length(cp.obj) == 1 ||
        error("design_ssdp supports single-piece objectives only")
    isempty(cp.psd) ||
        error("design_ssdp does not yet support PEPs with PSD (LMI) blocks; " *
              "use design_fom/design_som instead")
    nc = length(cp.cons)
    κ = length(ω0)
    isle = [con.sense == :le for con in cp.cons]
    c0s = [con.expr.c0 for con in cp.cons]
    c0_obj = cp.obj[1].c0
    bobj = cp.obj[1].f
    Fmat = zeros(cp.nf, nc)                      # h(ϑ) = bobj − Fmat λ
    for c in 1:nc
        Fmat[:, c] = cp.cons[c].expr.f
    end

    nsdp_obj(λ) = c0_obj - dot(c0s, λ)
    function merit(λ, ω)
        η, _, _ = evaluate_policy(dp.policy, ω, dp.N)
        _, 𝒜 = _assemble_all(cp, η, λ)
        V = norm(bobj - Fmat * λ) +
            sum(max(-λ[c], 0.0) for c in 1:nc if isle[c]; init = 0.0) +
            _psd_violation(𝒜)
        return nsdp_obj(λ) + ρ * V, V
    end

    # ── initialization from a PEP primal-dual solve at ω0 ────────────────────
    ω = collect(Float64, ω0)
    η, J, Hpol = evaluate_policy(dp.policy, ω, dp.N)
    sol0 = solve_pep(cp, η; backend = dp.backend)
    λ = max.(sol0.duals, [isle[c] ? 0.0 : -Inf for c in 1:nc])   # clip tiny negatives
    G = Matrix(Symmetric(sol0.G))
    y = copy(sol0.F)

    ω_hist = [copy(ω)]
    obj_hist = [nsdp_obj(λ)]
    merit0, _ = merit(λ, ω)
    merit_hist = [merit0]
    kkt_hist = Float64[]
    ζ_hist = Float64[]
    γ_hist = Float64[]
    pred_hist = Float64[]
    act_hist = Float64[]
    ntangent = 0

    for t in 1:iters
        η, J, Hpol = evaluate_policy(dp.policy, ω, dp.N)
        Acs, 𝒜 = _assemble_all(cp, η, λ)

        # Exact Lagrangian Hessian blocks via contraction with (G, λ)
        fake = PEPSolution(0.0, G, zeros(cp.nf), λ, [1.0], nothing)
        gη, Hη = grad_hess_eta(cp, η, fake)
        gω, Hωω = pullback(gη, Hη, J, Hpol)

        # λ–ω block and per-η_r LMI direction kernels K_r
        Hλω = zeros(nc, κ)
        K = Vector{Union{Nothing,Matrix{Float64}}}(nothing, cp.np)
        addK!(r, w, S) = (K[r] === nothing && (K[r] = zeros(cp.dim, cp.dim));
                          K[r] .+= w .* Matrix(S))
        function tracevec!(gc, M::ParamMatrix)
            for (r, S) in M.A1
                gc[r] += trprod(G, S)
            end
            for (r, s, S) in M.A2
                tv = trprod(G, S)
                if r == s
                    gc[r] += 2.0 * η[r] * tv
                else
                    gc[r] += η[s] * tv
                    gc[s] += η[r] * tv
                end
            end
        end
        gc = zeros(cp.np)
        for c in 1:nc
            M = cp.cons[c].expr.M
            is_param_dependent(M) || continue
            fill!(gc, 0.0)
            tracevec!(gc, M)
            Hλω[c, :] = -(J' * gc)
            # K_r accumulation: ∂𝒜/∂η_r = Σ_c λ_c ∂A_c/∂η_r − ∂C/∂η_r
            if λ[c] != 0.0
                for (r, S) in M.A1
                    addK!(r, λ[c], S)
                end
                for (r, s, S) in M.A2
                    if r == s
                        addK!(r, 2.0 * λ[c] * η[r], S)
                    else
                        addK!(r, λ[c] * η[s], S)
                        addK!(s, λ[c] * η[r], S)
                    end
                end
            end
        end
        Mobj = cp.obj[1].M
        for (r, S) in Mobj.A1
            addK!(r, -1.0, S)
        end
        for (r, s, S) in Mobj.A2
            if r == s
                addK!(r, -2.0 * η[r], S)
            else
                addK!(r, -η[s], S)
                addK!(s, -η[r], S)
            end
        end
        Mdir = [begin
                    Mν = zeros(cp.dim, cp.dim)
                    for r in 1:cp.np
                        (K[r] === nothing || J[r, ν] == 0.0) && continue
                        Mν .+= J[r, ν] .* K[r]
                    end
                    Mν
                end for ν in 1:κ]

        # KKT residual
        h = bobj - Fmat * λ
        statλ = [-c0s[c] - trprod(G, Acs[c]) - dot(y, Fmat[:, c]) for c in 1:nc]
        rstat = maximum(c -> isle[c] ? abs(min(λ[c], statλ[c])) : abs(statλ[c]),
                        1:nc)
        Φt, Vt = merit(λ, ω)
        comp = abs(tr(G * 𝒜))
        kkt = max(Vt, comp, rstat, norm(gω, Inf))
        push!(kkt_hist, kkt)
        verbose && @info "SSDP iter $t" obj = nsdp_obj(λ) kkt Vt comp
        kkt <= tol && break

        # Convexification of the quadratic model
        Hs = Symmetric((Hωω + Hωω') / 2)
        if hessian_mode == :block
            ζω = max(0.0, δ - min(minimum(eigvals(Hs)), 0.0))
            Bω = Hs + ζω * LinearAlgebra.I
            ζλ = λreg
            ζrec = ζω
            Xcross = nothing              # coupling handled by the LMI only
        elseif hessian_mode == :full
            lb = min(minimum(eigvals(Hs)), 0.0) - opnorm(Hλω)
            ζ = max(0.0, δ - lb)
            Bω = Hs + ζ * LinearAlgebra.I
            ζλ = ζ
            ζrec = ζ
            Xcross = Hλω
        else
            error("unknown hessian_mode :$hessian_mode (use :block or :full)")
        end

        # ── tangent subproblem ───────────────────────────────────────────────
        model = _make_model(dp.backend)
        @variable(model, dλ[1:nc])
        @variable(model, dω[1:κ])
        for c in 1:nc
            isle[c] && @constraint(model, λ[c] + dλ[c] >= 0)
        end
        mat = Matrix{AffExpr}(undef, cp.dim, cp.dim)
        for j in 1:cp.dim, i in 1:cp.dim
            mat[i, j] = AffExpr(𝒜[i, j])
        end
        for c in 1:nc
            Is, Js, Vs = findnz(Acs[c])
            for k in eachindex(Vs)
                add_to_expression!(mat[Is[k], Js[k]], Vs[k], dλ[c])
            end
        end
        for ν in 1:κ
            Mν = Mdir[ν]
            for j in 1:cp.dim, i in 1:cp.dim
                Mν[i, j] == 0.0 || add_to_expression!(mat[i, j], Mν[i, j], dω[ν])
            end
        end
        lmi = @constraint(model, Symmetric(mat) in PSDCone())
        hcons = @constraint(model, h .- Fmat * dλ .== 0)
        if Xcross === nothing
            @objective(model, Min,
                -dot(c0s, dλ) + 0.5 * ζλ * sum(dλ .^ 2) + 0.5 * dω' * Bω * dω)
        else
            @objective(model, Min,
                -dot(c0s, dλ) + dλ' * Xcross * dω +
                0.5 * dω' * Bω * dω + 0.5 * ζλ * sum(dλ .^ 2))
        end
        optimize!(model)
        ntangent += 1
        st = termination_status(model)
        if st != MOI.OPTIMAL && st != MOI.SLOW_PROGRESS
            verbose && @warn "SSDP tangent subproblem failed" st
            break
        end
        dλv = value.(dλ)
        dωv = value.(dω)
        Ĝ = Matrix(Symmetric(Matrix(dual(lmi))))
        ŷ = collect(Float64, dual.(hcons))

        # ── exact-penalty backtracking (Han–Powell model decrease) ───────────
        Ddir = -dot(c0s, dλv) - ρ * Vt
        Ddir >= -1e-16 && (Ddir = -ζλ * norm(dλv)^2 - dot(dωv, Bω * dωv) - 1e-16)
        γ = 1.0
        accepted = false
        for _ in 1:40
            Φtrial, _ = merit(λ .+ γ .* dλv, ω .+ γ .* dωv)
            if Φtrial <= Φt + σ * γ * Ddir
                accepted = true
                break
            end
            γ *= 0.5
        end
        accepted || (γ = 0.0)
        γ == 0.0 && (verbose && @info "SSDP: no acceptable step at iter $t"; break)

        λ .+= γ .* dλv
        ω .+= γ .* dωv
        G .= (1 - γ) .* G .+ γ .* Ĝ
        y .= (1 - γ) .* y .+ γ .* ŷ
        Φnew, _ = merit(λ, ω)
        push!(ω_hist, copy(ω))
        push!(obj_hist, nsdp_obj(λ))
        push!(merit_hist, Φnew)
        push!(ζ_hist, ζrec)
        push!(γ_hist, γ)
        push!(pred_hist, -γ * Ddir)          # predicted merit reduction
        push!(act_hist, Φt - Φnew)           # actual merit reduction
    end

    W_final, _, _ = pep_value(dp, ω)
    return SSDPTrace(ω_hist, obj_hist, kkt_hist, merit_hist,
                     ζ_hist, γ_hist, pred_hist, act_hist, λ, ntangent, W_final)
end
