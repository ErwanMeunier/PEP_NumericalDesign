# Matrix computations for gradients, Hessians, and PEP matrices (IGDM)
#
# Gram matrix layout (size 3N+5, 1-indexed):
#   u[1..N+1]       : x_1,…,x_{N+1}
#   u[N+2]          : x_*
#   u[N+3..2N+3]    : g_1,…,g_{N+1}   (g_n  → u[N+2+n])
#   u[2N+4]         : g_*              (g_*  → u[N+2+(N+2)] = u[2N+4])
#   u[2N+5..3N+5]   : d_1,…,d_{N+1}   (d_n  → u[2N+4+n])
#
# δ is fixed to the gradient-descent chain (δ_{n,n} = 1, else 0); only the
# step sizes β remain free, with the convention  x_{n+1} = x_n - ∑_k β_{n,k} d_k.
#
# Parameter layout for the flat ω vector (full β mapping, size N*(N+1)÷2):
#   β block:  positions flat_idx(n,k) = (n-1)*n÷2 + k   for k=1..n, n=1..N
#
# Key formula: v_n = u[n+1] - u[n] + ∑_{k=1}^n β_{n,k}·u[2N+4+k]
#              A2[n] = v_n * v_n'   (rank-1 PSD matrix encoding the update equality)

# ─────────────────────────────────────────────────────────────────────────────
# Update A2 and store current β in prob (all other matrices are constant).
# ─────────────────────────────────────────────────────────────────────────────
function compute_A_∇A_∇2A_IGDM!(β::Matrix{Float64}, prob::pep_problem_IGDM)
    N   = prob.N
    dim = 3N + 5

    for n in 1:N
        # Build v_n = e_{n+1} - e_n + ∑_{k=1}^n β[n,k]·e_{2N+4+k}
        v = zeros(Float64, dim)
        v[n+1] = 1.0
        v[n]  -= 1.0
        for k in 1:n
            v[2N+4+k] += β[n, k]
        end
        v_sp = sparse(v)
        prob.A2[n] = v_sp * v_sp'
    end

    prob.β .= β
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient and Hessian of σ*(β) using the dual variable λ2.
#
# Returns (grad_flat, hess_flat) in the flat β layout (size N*(N+1)÷2):
#   grad_flat[flat_idx(n,k)] = ∂σ*/∂β_{n,k}
#
# Derivation (envelope theorem, sign convention matches ITEM: flip_sign=-1 already applied):
#   ∂v_n/∂β_{n,k} = +u_{2N+4+k}
#   ∂A2_n/∂β_{n,k} = u_{2N+4+k} v_n' + v_n u_{2N+4+k}'
#   tr(G · ∂A2_n/∂β_{n,k}) = 2(G v_n)[2N+4+k]
#   ∂σ*/∂β_{n,k} = -λ2[n] · tr(G · ∂A2_n/∂β_{n,k}) = -2 λ2[n] (G v_n)[2N+4+k]
#
#   ∂²A2_n/∂β_{n,j}∂β_{n,l} = u_{2N+4+j} u_{2N+4+l}' + u_{2N+4+l} u_{2N+4+j}'
#   ∂²σ*/∂β_{n,j}∂β_{n,l} = -λ2[n] · 2 G[2N+4+j, 2N+4+l]   (cross-n terms are zero)
# ─────────────────────────────────────────────────────────────────────────────
function diff_w_IGDM(sol::sol_PEP_IGDM, prob::pep_problem_IGDM)
    N  = prob.N
    G  = sol.G
    λ2 = sol.lambda2
    β  = prob.β
    dim = 3N + 5

    κ = N * (N + 1) ÷ 2

    flat_idx(n, k) = (n - 1) * n ÷ 2 + k

    grad_flat = zeros(κ)
    hess_flat = zeros(κ, κ)

    for n in 1:N
        # Build v_n
        v = zeros(Float64, dim)
        v[n+1] = 1.0
        v[n]  -= 1.0
        for k in 1:n
            v[2N+4+k] += β[n, k]
        end

        # Gv_n = G * v_n
        Gv = G * v

        for k in 1:n
            fi = flat_idx(n, k)
            # Gradient
            grad_flat[fi] = -2.0 * λ2[n] * Gv[2N+4+k]

            # Hessian: diagonal block for row n
            for l in 1:n
                fl = flat_idx(n, l)
                hess_flat[fi, fl] += -2.0 * λ2[n] * G[2N+4+k, 2N+4+l]
            end
        end
    end

    return grad_flat, hess_flat
end

# ─────────────────────────────────────────────────────────────────────────────
# Chain-rule wrapper: gradient/Hessian of σ* w.r.t. ω when β = mapping(ω).
#
# compute_β_fn(ω, N) must return (β, ∇β, ∇2β) where
#   β       : N×N Float64 matrix (lower triangular)
#   ∇β[n,k] : Vector{Float64}(κ)  — Jacobian ∂β_{n,k}/∂ω
#   ∇2β[n,k]: Matrix{Float64}(κ,κ)— Hessian ∂²β_{n,k}/∂ω²  (zero for linear mappings)
# ─────────────────────────────────────────────────────────────────────────────
function diff_w_IGDM_wrt_params(ω::AbstractVector, compute_β_fn::Function,
                                  sol::sol_PEP_IGDM, prob::pep_problem_IGDM)
    N = prob.N
    _, ∇β, ∇2β = compute_β_fn(ω, N)

    grad_flat, hess_flat = diff_w_IGDM(sol, prob)

    κ      = length(ω)
    κ_flat = N * (N + 1) ÷ 2

    flat_idx(n, k) = (n - 1) * n ÷ 2 + k

    # Build Jacobian J: (κ × κ_flat)
    J = zeros(κ, κ_flat)
    for n in 1:N
        for k in 1:n
            fi = flat_idx(n, k)
            J[:, fi] = ∇β[n, k]
        end
    end

    grad_ω = J * grad_flat
    hess_ω = J * hess_flat * J'

    # Second-order correction from nonlinear mappings (zero for all linear mappings)
    for n in 1:N
        for k in 1:n
            fi = flat_idx(n, k)
            hess_ω .+= grad_flat[fi] .* ∇2β[n, k]
        end
    end

    return grad_ω, hess_ω
end
