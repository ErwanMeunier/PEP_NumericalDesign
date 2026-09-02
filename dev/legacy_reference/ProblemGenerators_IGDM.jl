# Problem generator and oracle factory for the IGDM PEP instance.
#
# IGDM = Inexact Gradient Descent with Memory
#
# generate_IGDM(N, L, ε, β) builds all constant constraint matrices (A1, A3, A4, C)
# and calls compute_A_∇A_∇2A_IGDM! to initialise the parameter-dependent A2.
#
# δ is fixed to the gradient-descent chain (δ_{n,n} = 1, else 0); only the step
# sizes β are free, with the update  x_{n+1} = x_n - ∑_k β_{n,k} d_k.
#
# Gram matrix index convention (dim = 3N+5, 1-indexed):
#   x_i  (i=1..N+1)  → u[i]
#   x_*              → u[N+2]
#   g_j  (j=1..N+2)  → u[N+2+j]   (g_* = g_{N+2} → u[2N+4])
#   d_k  (k=1..N+1)  → u[2N+4+k]
#
# Shorthand symmetric outer products (match paper notation):
#   XX(i,j) = (u[i] u[j]' + u[j] u[i]') / 2      i,j ∈ {1..N+2}
#   XG(i,j) = (u[i] u[N+2+j]' + u[N+2+j] u[i]') / 2
#   GG(i,j) = (u[N+2+i] u[N+2+j]' + u[N+2+j] u[N+2+i]') / 2
#   GD(i,j) = (u[N+2+i] u[2N+4+j]' + u[2N+4+j] u[N+2+i]') / 2
#   DD(i,j) = (u[2N+4+i] u[2N+4+j]' + u[2N+4+j] u[2N+4+i]') / 2
#
# Constraint matrices (from Lemma 1 of the paper):
#   A1[i,j] = XG(i,j) - XG(j,j) + 1/(2L)·(GG(i,i) - 2·GG(i,j) + GG(j,j))
#   A2[n]   = v_n v_n'   where v_n = u[n+1] - u[n] + ∑_k β_{n,k} u[2N+4+k]
#   A3[n]   = DD(n,n) - 2·GD(n,n) + (1-ε²)·GG(n,n)
#   A4      = GG(N+2, N+2)   (forces ‖g_*‖² = 0)
#   C       = GG(N+1, N+1)   (objective ‖g_{N+1}‖²)

# ─────────────────────────────────────────────────────────────────────────────
# Build the full IGDM PEP instance.
# ─────────────────────────────────────────────────────────────────────────────
function generate_IGDM(N::Int, L::Float64, ε::Float64, β::Matrix{Float64})
    dim = 3N + 5

    # Canonical basis vectors (sparse)
    u = [sparsevec([i], [1.0], dim) for i in 1:dim]

    # Symmetric outer product helper
    sym_outer(a, b) = (a * b' .+ b * a') ./ 2

    # Index helpers (match Gram layout above)
    uX(i)   = u[i]           # x_i,    i ∈ {1..N+2}
    uG(j)   = u[N+2+j]       # g_j,    j ∈ {1..N+2}
    uD(k)   = u[2N+4+k]      # d_k,    k ∈ {1..N+1}

    XG(i,j) = sym_outer(uX(i), uG(j))
    GG(i,j) = sym_outer(uG(i), uG(j))
    GD(i,j) = sym_outer(uG(i), uD(j))
    DD(i,j) = sym_outer(uD(i), uD(j))

    # ── A1: smoothness (constant) ─────────────────────────────────────────────
    # A1[i,j] for i,j ∈ {1..N+2}, diagonal entries set to zero (unused)
    A1 = Matrix{SpMat}(undef, N+2, N+2)
    inv2L = 1.0 / (2.0 * L)
    for i in 1:N+2
        for j in 1:N+2
            if i != j
                A1[i,j] = XG(i,j) .- XG(j,j) .+
                           inv2L .* (GG(i,i) .- 2.0 .* GG(i,j) .+ GG(j,j))
            else
                A1[i,j] = spzeros(dim, dim)
            end
        end
    end

    # ── A3: noisy gradient bound (constant) ──────────────────────────────────
    # A3[n] for n=1..N+1: ‖d_n - g_n‖² ≤ ε² ‖g_n‖²
    ε2 = ε^2
    A3 = Vector{SpMat}(undef, N+1)
    for n in 1:N+1
        A3[n] = DD(n,n) .- 2.0 .* GD(n,n) .+ (1.0 - ε2) .* GG(n,n)
    end

    # ── A4: optimality g_* = 0 (constant) ────────────────────────────────────
    A4 = GG(N+2, N+2)   # u[2N+4] * u[2N+4]'

    # ── C: objective ‖g_{N+1}‖² (constant) ──────────────────────────────────
    C = GG(N+1, N+1)    # u[2N+3] * u[2N+3]'

    # ── A2: algorithm update (parameter-dependent, initialised to zero) ───────
    A2 = Vector{SpMat}(undef, N)
    for n in 1:N
        A2[n] = spzeros(dim, dim)
    end

    prob = pep_problem_IGDM(A1, A2, A3, A4, C, u,
                             copy(β), N, L, ε)
    compute_A_∇A_∇2A_IGDM!(β, prob)
    return prob
end

###############################################################################
# Oracle factory for IGDM.
#
# Returns a named tuple with fields:
#   generate(N, ω0)          -> pep_problem_IGDM
#   update!(ω, prob)         -> nothing  (updates A2 in-place)
#   solve(prob)              -> sol_PEP_IGDM
#   grad_hess(ω, sol, prob)  -> (grad_ω, hess_ω)
#   params(ω, N)             -> β  current step-size matrix
#
# Arguments:
#   L              : Lipschitz constant
#   ε              : inexactness level (ε=0 → exact gradients)
#   compute_β_fn   : policy function (ω, N) → (β, ∇β, ∇2β)
#                    defaults to the full-memory mapping
###############################################################################
function make_IGDM_oracle(L::Real, ε::Real,
                           compute_β_fn::Function=compute_β_∇β_∇2β_full)
    _generate = function(N, ω0)
        β0 = compute_β_fn(ω0, N)[1]
        return generate_IGDM(N, Float64(L), Float64(ε), β0)
    end

    _update! = function(ω, prob)
        β = compute_β_fn(ω, prob.N)[1]
        compute_A_∇A_∇2A_IGDM!(β, prob)
    end

    _solve     = prob -> sdp_pep_IGDM(prob)

    _grad_hess = (ω, sol, prob) -> diff_w_IGDM_wrt_params(ω, compute_β_fn, sol, prob)

    _params    = (ω, N) -> compute_β_fn(ω, N)[1]   # returns β

    return (generate=_generate, var"update!"=_update!, solve=_solve,
            grad_hess=_grad_hess, params=_params)
end
