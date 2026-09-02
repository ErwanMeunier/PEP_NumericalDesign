# ITEM-specific matrix computations: A, ∇A, ∇2A, diff_w

# ─────────────────────────────────────────────────────────────────────────────
# Compute / update A, ∇A, ∇2A for ITEM given the parameter vector.
#
#   params = [β_1,…,β_N,  δ_1,…,δ_N]   (length 2N)
#
# Index convention (1-based Julia arrays):
#   params[n]   = β_n        for n = 1..N
#   params[N+n] = δ_n        for n = 1..N
#
# A1[n]: encodes  y_n = (1-β_n)z_n + β_n(y_{n-1} - (1/L)g_{n-1})
# A2[n]: encodes  z_{n+1} = (1-qδ_n)z_n + qδ_n(y_n-(1/μ)g_n)
# ─────────────────────────────────────────────────────────────────────────────
function compute_A_∇A_∇2A_ITEM!(params, pep_prob::pep_problem_ITEM)
    N  = pep_prob.N
    L  = pep_prob.L
    mu = pep_prob.mu
    q  = mu / L
    dim = 3N + 5

    # ── canonical-basis helpers ───────────────────────────────────────────────
    uZ(i)    = sparsevec([i],      [1.0], dim)   # z_i,  i=1..N+1
    uY(j)    = sparsevec([N+j+2],  [1.0], dim)   # y_j,  j=0..N
    uG(k)    = sparsevec([2N+k+4], [1.0], dim)   # g_k,  k=0..N

    sym_outer(a, b) = (a * b' .+ b * a') ./ 2

    ZZ(i,j) = sym_outer(uZ(i), uZ(j))
    ZY(i,j) = sym_outer(uZ(i), uY(j))
    ZG(i,j) = sym_outer(uZ(i), uG(j))
    YY(i,j) = sym_outer(uY(i), uY(j))
    YG(i,j) = sym_outer(uY(i), uG(j))
    GG(i,j) = sym_outer(uG(i), uG(j))

    for n in 1:N
        β  = params[n]      # β_n
        δ  = params[N+n]    # δ_n

        # ── A_n^{(1)}: ‖y_n - ((1-β)z_n + β(y_{n-1} - (1/L)g_{n-1}))‖² = 0 ──
        pep_prob.A1[n] =
              (1-β)^2            .* ZZ(n,n)     .+
              2*(1-β)*β          .* ZY(n,n-1)   .-
              2*(1-β)*(β/L)      .* ZG(n,n-1)   .-
              2*(1-β)            .* ZY(n,n)     .+
              β^2                .* YY(n-1,n-1) .+
              (-2*β^2/L)         .* YG(n-1,n-1) .+
              (-2*β)             .* YY(n-1,n)   .+
              (β/L)^2            .* GG(n-1,n-1) .+
              (2*β/L)            .* YG(n,n-1)   .+
                                    YY(n,n)

        # ── A_n^{(2)}: ‖z_{n+1} - ((1-qδ)z_n + qδ(y_n-(1/μ)g_n))‖² = 0 ──────
        pep_prob.A2[n] =
              (1-q*δ)^2           .* ZZ(n,n)   .+
              2*(1-q*δ)*(q*δ)     .* ZY(n,n)   .+
              (-2*(1-q*δ)*q*δ/mu) .* ZG(n,n)   .+
              (-2*(1-q*δ))        .* ZZ(n,n+1) .+
              (q*δ)^2             .* YY(n,n)   .+
              (-2*(q*δ)^2/mu)     .* YG(n,n)   .+
              (q*δ/mu)^2          .* GG(n,n)   .+
              (-2*q*δ)            .* ZY(n+1,n) .+
              (2*q*δ/mu)          .* ZG(n+1,n) .+
                                     ZZ(n+1,n+1)

        # ── ∂A_n^{(1)}/∂β_n ──────────────────────────────────────────────────
        pep_prob.∇A1[n] =
              2*(β-1)         .* ZZ(n,n)      .+
              2*(1-2*β)       .* ZY(n,n-1)    .+
              (-2*(1-2*β)/L)  .* ZG(n,n-1)    .+
              2               .* ZY(n,n)      .+
              2*β             .* YY(n-1,n-1)  .+
              (-4*β/L)        .* YG(n-1,n-1)  .+
              (-2)            .* YY(n-1,n)    .+
              (2*β/L^2)       .* GG(n-1,n-1)  .+
              (2/L)           .* YG(n,n-1)

        # ── ∂A_n^{(2)}/∂δ_n ──────────────────────────────────────────────────
        pep_prob.∇A2[n] =
              (-2*q*(1-q*δ))      .* ZZ(n,n)   .+
              (2*q*(1-2*q*δ))     .* ZY(n,n)   .+
              (-2*q/mu*(1-2*q*δ)) .* ZG(n,n)   .+
              (2*q)               .* ZZ(n,n+1) .+
              (2*q^2*δ)           .* YY(n,n)   .+
              (-4*q^2*δ/mu)       .* YG(n,n)   .+
              (2*q^2*δ/mu^2)      .* GG(n,n)   .+
              (-2*q)              .* ZY(n+1,n) .+
              (2*q/mu)            .* ZG(n+1,n)

        # ── ∂²A_n^{(1)}/∂β_n² ────────────────────────────────────────────────
        pep_prob.∇2A1[n] =
              2        .* ZZ(n,n)      .+
              (-4)     .* ZY(n,n-1)    .+
              (4/L)    .* ZG(n,n-1)    .+
              2        .* YY(n-1,n-1)  .+
              (-4/L)   .* YG(n-1,n-1)  .+
              (2/L^2)  .* GG(n-1,n-1)

        # ── ∂²A_n^{(2)}/∂δ_n² ────────────────────────────────────────────────
        pep_prob.∇2A2[n] =
              (2*q^2)      .* ZZ(n,n) .+
              (-4*q^2)     .* ZY(n,n) .+
              (4*q^2/mu)   .* ZG(n,n) .+
              (2*q^2)      .* YY(n,n) .+
              (-4*q^2/mu)  .* YG(n,n) .+
              (2*q^2/mu^2) .* GG(n,n)
    end

    return pep_prob.A1, pep_prob.A2, pep_prob.∇A1, pep_prob.∇A2, pep_prob.∇2A1, pep_prob.∇2A2
end


# ─────────────────────────────────────────────────────────────────────────────
# ITEM: gradient and Hessian of w^{sdp}(params) using dual variables.
#
#   params = [β_1,…,β_N, δ_1,…,δ_N]  (length 2N)
#
#   ∂w/∂β_n   = -τ1[n] · tr(G · ∇A1[n])
#   ∂w/∂δ_n   = -τ2[n] · tr(G · ∇A2[n])
#   ∂²w/∂β_n² = -τ1[n] · tr(G · ∇2A1[n])   (Hessian is diagonal)
#   ∂²w/∂δ_n² = -τ2[n] · tr(G · ∇2A2[n])
# ─────────────────────────────────────────────────────────────────────────────
function diff_w_ITEM(params, pep_sol::sol_PEP_ITEM, pep_prob::pep_problem_ITEM)
    N = pep_prob.N
    G = pep_sol.G

    grad_w = zeros(2N)
    hess_w = zeros(2N, 2N)

    for n in 1:N
        τ1 = pep_sol.tau1[n]
        τ2 = pep_sol.tau2[n]

        grad_w[n]   = -τ1 * tr(G * Matrix(pep_prob.∇A1[n]))
        grad_w[N+n] = -τ2 * tr(G * Matrix(pep_prob.∇A2[n]))

        hess_w[n,   n]   = -τ1 * tr(G * Matrix(pep_prob.∇2A1[n]))
        hess_w[N+n, N+n] = -τ2 * tr(G * Matrix(pep_prob.∇2A2[n]))
    end

    return grad_w, hess_w
end

# Chain-rule wrapper: gradient/Hessian of w w.r.t. ω when params = α(ω).
function diff_w_ITEM_wrt_parameters(ω, compute_α_∇α_∇2α, pep_sol::sol_PEP_ITEM, pep_prob::pep_problem_ITEM)
    params, ∇α, ∇2α = compute_α_∇α_∇2α(ω, pep_prob.N)
    grad_p, hess_p  = diff_w_ITEM(params, pep_sol, pep_prob)

    κ = length(ω)
    grad_ω = zeros(κ)
    for k in 1:κ
        for n in eachindex(params)
            grad_ω[k] += grad_p[n] * ∇α[n][k]
        end
    end

    hess_ω = zeros(κ, κ)
    for k in 1:κ
        for s in 1:κ
            for n in eachindex(params)
                hess_ω[k,s] += grad_p[n] * ∇2α[n][k,s]
                for m in eachindex(params)
                    hess_ω[k,s] += hess_p[n,m] * ∇α[n][k] * ∇α[m][s]
                end
            end
        end
    end
    return grad_ω, hess_ω
end

# ─────────────────────────────────────────────────────────────────────────────
# ITEM policy combiner.
#
# Wraps two single-sequence policies (one for β, one for δ) into a single
# mapping  ω = [ω_β; ω_δ]  →  params = [β; δ]  (length 2N).
#
# Arguments:
#   F_β  : (ω_β, N) → (β, ∇β, ∇2β)   policy for the β sequence
#   κ_β  : length of ω_β  (number of β policy parameters)
#   F_δ  : (ω_δ, N) → (δ, ∇δ, ∇2δ)   policy for the δ sequence
#
# Returns a function  policy(ω, N) → (params, ∇params, ∇2params)
# with the same interface expected by diff_w_ITEM_wrt_parameters.
#
# Example (both sequences use sum-of-exponentials with k=2):
#   policy = make_ITEM_policy(compute_α_∇α_∇2α_sum_of_exp, 5,
#                             compute_α_∇α_∇2α_sum_of_exp)
#   ω_init = [ω_β_init; ω_δ_init]   # length 10
# ─────────────────────────────────────────────────────────────────────────────
function make_ITEM_policy(F_β::Function, κ_β::Int, F_δ::Function)
    return function(ω::AbstractVector, N::Int)
        ω_β = ω[1:κ_β]
        ω_δ = ω[κ_β+1:end]
        κ_δ = length(ω_δ)
        κ   = κ_β + κ_δ

        β,  ∇β,  ∇2β  = F_β(ω_β, N)
        δ,  ∇δ,  ∇2δ  = F_δ(ω_δ, N)

        params = vcat(β, δ)  # length 2N

        # ∇params[n][k]: ∂params[n]/∂ω[k]
        ∇params = Vector{Vector{Float64}}(undef, 2N)
        for n in 1:N
            ∇params[n]   = vcat(∇β[n],       zeros(κ_δ))
            ∇params[N+n] = vcat(zeros(κ_β),  ∇δ[n])
        end

        # ∇2params[n][k,s]: ∂²params[n]/∂ω[k]∂ω[s]
        ∇2params = Vector{Matrix{Float64}}(undef, 2N)
        for n in 1:N
            ∇2params[n]   = [∇2β[n]             zeros(κ_β, κ_δ);
                             zeros(κ_δ, κ_β)   zeros(κ_δ, κ_δ)]
            ∇2params[N+n] = [zeros(κ_β, κ_β)   zeros(κ_β, κ_δ);
                             zeros(κ_δ, κ_β)   ∇2δ[n]]
        end

        return params, ∇params, ∇2params
    end
end
