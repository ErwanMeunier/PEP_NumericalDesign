# Sensitivity: exact gradient and frozen-certificate curvature of the PEP value
# w.r.t. the coefficients η, and the chain-rule pullback to policy parameters ω.

"""
    grad_hess_eta(cp, η, sol; hess=true) -> (g, H)

Envelope-theorem gradient `g[r] = ∂W/∂η_r` and frozen-certificate curvature
`H[r,s] = −Σ_c λ_c tr(G ∂²A_c/∂η_r∂η_s) + Σ_k γ_k tr(G ∂²C_k/...)` at the
primal-dual solution `sol`. `H` is NOT the Hessian of the value function in
general (certificate held fixed).
"""
function grad_hess_eta(cp::CompiledPEP, η::AbstractVector{<:Real},
                       sol::PEPSolution; hess::Bool = true)
    p = cp.np
    g = zeros(p)
    H = hess ? zeros(p, p) : zeros(0, 0)

    # Constraints contribute with weight −λ_c, objective pieces with +γ_k.
    function accumulate!(M::ParamMatrix, w::Float64)
        w == 0.0 && return
        for (r, S) in M.A1
            g[r] += w * trprod(sol.G, S)
        end
        for (r, s, S) in M.A2
            t = trprod(sol.G, S)
            t == 0.0 && continue
            if r == s
                g[r] += w * 2.0 * η[r] * t
                hess && (H[r, r] += w * 2.0 * t)
            else
                g[r] += w * η[s] * t
                g[s] += w * η[r] * t
                if hess
                    H[r, s] += w * t
                    H[s, r] += w * t
                end
            end
        end
    end

    for (c, con) in enumerate(cp.cons)
        is_param_dependent(con.expr.M) || continue
        accumulate!(con.expr.M, -sol.duals[c])
    end
    for (b, blk) in enumerate(cp.psd)   # LMI T ⪰ 0 contributes +Σ_ij Λ_ij ∂T_ij/∂η
        Λ = sol.psd_duals[b]
        for i in axes(blk.mat, 1), j in axes(blk.mat, 2)
            is_param_dependent(blk.mat[i, j].M) || continue
            accumulate!(blk.mat[i, j].M, Λ[i, j])
        end
    end
    for (k, piece) in enumerate(cp.obj)
        is_param_dependent(piece.M) || continue
        accumulate!(piece.M, sol.obj_duals[k])
    end
    return g, H
end

"""
    pullback(gη, Hη, J, Hpol) -> (gω, Hω)

Chain rule through a policy η(ω) with Jacobian `J` (p×κ) and per-coefficient
Hessians `Hpol` (`nothing` for linear policies): gω = J'gη,
Hω = J'HηJ + Σ_r gη[r]·Hpol[r].
"""
function pullback(gη::Vector{Float64}, Hη::Matrix{Float64},
                  J::AbstractMatrix{<:Real}, Hpol)
    gω = J' * gη
    Hω = isempty(Hη) ? zeros(size(J, 2), size(J, 2)) : Matrix(J' * Hη * J)
    if Hpol !== nothing
        for r in eachindex(gη)
            Hr = Hpol[r]
            (Hr === nothing || gη[r] == 0.0) && continue
            Hω .+= gη[r] .* Hr
        end
    end
    return gω, Hω
end
