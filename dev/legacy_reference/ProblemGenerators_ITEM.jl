# Problem generator for the ITEM PEP instance.

# ─────────────────────────────────────────────────────────────────────────────
# Generate the ITEM PEP instance.
#
# N    : number of iterations (y_1…y_N, z_1…z_N are computed by ITEM)
# L    : smoothness constant
# mu   : strong-convexity constant  (q = mu / L)
# D    : initial condition radius  ||z_0 - y_*|| = D
# params : [β_1,…,β_N, δ_0,…,δ_{N-1}]  (length 2N)
#
# Gram matrix layout (dimension 3N+5, 0-indexed variables):
#   z_0…z_N  (indices 1..N+1)
#   y_0…y_N  (indices N+2..2N+2)
#   y_*      (index  2N+3)
#   g_0…g_N  (indices 2N+4..3N+4)
#   g_*      (index  3N+5)
#
# F vector layout (length N+2, 1-indexed):
#   positions 1..N+1  → f(y_0)..f(y_N)
#   position  N+2     → f(y_*)
# ─────────────────────────────────────────────────────────────────────────────
function generate_ITEM(N, L, mu, D, params)
    # Gram matrix dimension
    dim = 3N + 5

    # ── canonical-basis helpers ──────────────────────────────────────────────
    # Gram matrix layout (paper notation, 1-indexed z):
    #   z_1…z_{N+1}  (positions 1…N+1)
    #   y_0…y_N      (positions N+2…2N+2)
    #   y_*          (position 2N+3)
    #   g_0…g_N      (positions 2N+4…3N+4)
    #   g_*          (position 3N+5)
    uZ(i)    = sparsevec([i],        [1.0], dim)   # z_i,  i=1..N+1
    uY(j)    = sparsevec([N+j+2],    [1.0], dim)   # y_j,  j=0..N
    uYstar() = sparsevec([2N+3],     [1.0], dim)   # y_*
    uG(k)    = sparsevec([2N+k+4],   [1.0], dim)   # g_k,  k=0..N
    uGstar() = sparsevec([3N+5],     [1.0], dim)   # g_*

    # symmetric outer product of two sparse vectors
    function sym_outer(a, b)
        (a * b' .+ b * a') ./ 2
    end

    # shorthand symmetric matrices
    ZZ(i,j)   = sym_outer(uZ(i),    uZ(j))
    ZY(i,j)   = sym_outer(uZ(i),    uY(j))
    ZYs(i)    = sym_outer(uZ(i),    uYstar())
    ZG(i,j)   = sym_outer(uZ(i),    uG(j))
    YY(i,j)   = sym_outer(uY(i),    uY(j))
    YYs(i)    = sym_outer(uY(i),    uYstar())
    YsYs()    = uYstar() * uYstar()'
    YG(i,j)   = sym_outer(uY(i),    uG(j))
    GG(i,j)   = sym_outer(uG(i),    uG(j))

    # ── constant matrices (do not depend on params) ─────────────────────────
    # Objective: ||z_{N+1} - y_*||^2
    C = ZZ(N+1,N+1) .- 2 .* ZYs(N+1) .+ YsYs()

    # Interpolation constraints A_{ij}^{(3)} for i,j ∈ {0..N, *}
    # Index convention: 1..N+1 → y_0..y_N,  N+2 → y_*
    # uY_idx(k): 1..N+1 → uY(k-1),  N+2 → uYstar()
    # uG_idx(k): 1..N+1 → uG(k-1),  N+2 → uGstar()
    function uY_idx(k)
        k <= N+1 ? uY(k-1) : uYstar()
    end
    function uG_idx(k)
        k <= N+1 ? uG(k-1) : uGstar()
    end

    A3 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N+2, N+2)
    c_mu = mu / (2 * (1 - mu/L))
    for k in 1:N+2
        for l in 1:N+2
            if k == l
                A3[k,l] = spzeros(dim, dim)
                continue
            end
            Yi = uY_idx(k);  Yj = uY_idx(l)
            Gi = uG_idx(k);  Gj = uG_idx(l)
            GjYi = sym_outer(Gj, Yi);  GjYj = sym_outer(Gj, Yj)
            GiGi = Gi * Gi';            GiGj = sym_outer(Gi, Gj)
            GjGj = Gj * Gj'
            YiYi = Yi * Yi';            YiYj = sym_outer(Yi, Yj)
            YiGi = sym_outer(Yi, Gi);   YiGj = sym_outer(Yi, Gj)
            YjGi = sym_outer(Yj, Gi);   YjGj = sym_outer(Yj, Gj)
            YjYj = Yj * Yj'
            A3[k,l] = GjYi .- GjYj .+
                      (1/(2L)) .* (GiGi .- 2 .* GiGj .+ GjGj) .+
                      c_mu .* (YiYi .- 2 .* YiYj .-
                               (2/L) .* YiGi .+ (2/L) .* YiGj .+
                               (2/L) .* YjGi .- (2/L) .* YjGj .+
                               YjYj .+ (1/L^2) .* GiGi .-
                               (2/L^2) .* GiGj .+ (1/L^2) .* GjGj)
        end
    end

    # Initial condition: ||z_1 - y_*||^2 = D^2
    A5 = ZZ(1,1) .- 2 .* ZYs(1) .+ YsYs()

    # g_* = 0: ||g_*||^2 = 0
    A6 = uGstar() * uGstar()'

    # Initial equality: z_1 = y_0  (algorithm sets z_1 = y_0 = x_1)
    A_init = ZZ(1,1) .- 2 .* ZY(1,0) .+ YY(0,0)

    # ── parameter-dependent matrices: initialise then fill ──────────────────
    A1    = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    A2    = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    ∇A1   = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    ∇A2   = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    ∇2A1  = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    ∇2A2  = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)

    pep = pep_problem_ITEM(A1, A2, A3, A5, A6, A_init, C, ∇A1, ∇A2, ∇2A1, ∇2A2, N, L, mu, D)
    compute_A_∇A_∇2A_ITEM!(params, pep)
    return pep
end


###############################################################################
# Oracle factory for ITEM.
#
# Returns a named tuple with fields:
#   generate(N, ω0)          -> pep_problem_ITEM
#   update!(ω, prob)         -> nothing  (updates A matrices in-place)
#   solve(prob)              -> sol_PEP_ITEM
#   grad_hess(ω, sol, prob)  -> (grad_ω, hess_ω)
#   params(ω, N)             -> algorithm parameters (e.g. [β; δ] for ITEM)
#
# compute_params_from_ω  : optional Function (ω, N) -> (params, ∇params, ∇2params)
#   If nothing, ω is used directly as params (identity mapping).
###############################################################################
function make_ITEM_oracle(L::Real, mu::Real, D::Real,
                          compute_params_from_ω::Union{Nothing,Function}=nothing)
    _map = compute_params_from_ω !== nothing ?
               compute_params_from_ω :
               (ω, N) -> (ω, nothing, nothing)

    _generate  = (N, ω0) -> generate_ITEM(N, Float64(L), Float64(mu), Float64(D),
                                           _map(ω0, N)[1])
    _update!   = (ω, prob) -> compute_A_∇A_∇2A_ITEM!(_map(ω, prob.N)[1], prob)
    _solve     = prob -> sdp_pep_ITEM(prob)
    _grad_hess = (ω, sol, prob) -> begin
        if compute_params_from_ω !== nothing
            diff_w_ITEM_wrt_parameters(ω, compute_params_from_ω, sol, prob)
        else
            diff_w_ITEM(_map(ω, prob.N)[1], sol, prob)
        end
    end
    _params = (ω, N) -> _map(ω, N)[1]
    return (generate=_generate, var"update!"=_update!, solve=_solve,
            grad_hess=_grad_hess, params=_params)
end
