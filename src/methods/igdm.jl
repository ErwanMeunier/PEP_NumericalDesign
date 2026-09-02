# IGDM (Inexact Gradient Descent with Memory) PEP via the DSL.
#
# Update: x_{n+1} = x_n − Σ_{k≤n} β_{n,k} d_k with ‖d_n − g_n‖ ≤ ε‖g_n‖
# (δ fixed to the gradient-descent chain; residual-style encoding).
# Setting: f L-smooth convex, f(x_1) − f(x_*) ≤ D.
#
# Coefficients: η = flattened lower-triangular β in row order,
#   η[n(n−1)/2 + k] = β_{n,k},  1 ≤ k ≤ n ≤ N   (p = N(N+1)/2).
# x_* = 0 and g_* = 0 are eliminated from the Gram basis (dim 3N+3).

"""Flat coefficient index of β_{n,k} (1 ≤ k ≤ n ≤ N)."""
igdm_flat_index(n::Int, k::Int) = (n * (n - 1)) ÷ 2 + k

"""Number of free IGDM coefficients at horizon N."""
igdm_dof(N::Int) = (N * (N + 1)) ÷ 2

"""Flat indices of the diagonal β_{n,n} (memoryless positions)."""
igdm_diag_indices(N::Int) = [igdm_flat_index(n, n) for n in 1:N]

"""
    igdm_pep(N, L, ε; D=1.0, objective=:last_grad) -> (PEPModel, β::Vector{PAff})

Build the parametric IGDM PEP. `objective = :last_grad` maximizes ‖g_{N+1}‖²
(legacy convention); `:min_grad` maximizes min_{k=2..N+1} ‖g_k‖² (epigraph).
"""
function igdm_pep(N::Int, L::Real, ε::Real; D::Real = 1.0,
                  objective::Symbol = :last_grad)
    m = PEPModel()
    β = coeffs!(m, igdm_dof(N))
    βnk(n, k) = β[igdm_flat_index(n, k)]

    fobj = SmoothConvexFunction(m; L = L, name = "f")
    x = [point!(m; name = "x$i") for i in 1:(N + 1)]
    xstar = PointExpr()   # x_* = 0 (translation pinning)
    g = Vector{PointExpr}(undef, N + 1)
    d = Vector{PointExpr}(undef, N + 1)
    f = Vector{QExpr}(undef, N + 1)
    for n in 1:(N + 1)    # relative inexact oracle: ‖d_n − g_n‖² ≤ ε²‖g_n‖²
        d[n], g[n], f[n] = inexact_gradient!(fobj, x[n], ε; notion = :relative)
    end
    _, fstar = add_oracle_point!(fobj, xstar, PointExpr())   # g_* = 0 (optimality)

    for n in 1:N   # update residual ‖x_{n+1} − x_n + Σ_k β_{n,k} d_k‖² = 0
        r = x[n + 1] - x[n] + sum(βnk(n, k) * d[k] for k in 1:n)
        add_eq!(m, sqnorm(r); name = "upd[$n]")
    end
    add_eq!(m, fstar; name = "fstar0")
    add_le!(m, f[1] - fstar - D; name = "init")

    if objective == :last_grad
        objective_max!(m, sqnorm(g[N + 1]))
    elseif objective == :min_grad
        objective_maxmin!(m, [sqnorm(g[k]) for k in 2:(N + 1)])
    else
        error("unknown IGDM objective :$objective")
    end
    return m, β
end

"""
    compile_igdm(N, L, ε; D=1.0, objective=:last_grad) :: CompiledPEP
"""
compile_igdm(N::Int, L::Real, ε::Real; D::Real = 1.0,
             objective::Symbol = :last_grad) =
    compile(igdm_pep(N, L, ε; D, objective)[1])

"""
    igdm_hmem(ε) 

Memoryless reference step size h(ε) (Vernimmen–Glineur): β = h/L.
"""
igdm_hmem(ε::Real) = (3ε + 2 - sqrt(4 - 3ε^2)) / (2ε * (ε + 1))

# ── structured IGDM policies (flat lower-triangular β coefficients) ──────────

"""Diagonal β with schedule `base` (memoryless): β_{n,n} = base(ω)[n]."""
igdm_diagonal_policy(base::AbstractPolicy = IdentityPolicy()) =
    MappedPolicy(base, igdm_dof, igdm_diag_indices)

"""
    IGDMKMemoryPolicy(K)

Free coefficients on the last K+1 lags only: β_{n,k} = 0 for k < n−K;
ω holds the Σ_n min(n, K+1) free entries (row-major).
"""
struct IGDMKMemoryPolicy <: AbstractPolicy
    K::Int
end
nparams(pol::IGDMKMemoryPolicy, N::Int) = sum(min(n, pol.K + 1) for n in 1:N)
function evaluate_policy(pol::IGDMKMemoryPolicy, ω::AbstractVector{<:Real}, N::Int)
    p = igdm_dof(N)
    idx = Int[]
    for n in 1:N, k in max(1, n - pol.K):n
        push!(idx, igdm_flat_index(n, k))
    end
    η = zeros(p)
    η[idx] = ω
    J = zeros(p, length(ω))
    for (i, r) in enumerate(idx)
        J[r, i] = 1.0
    end
    return η, J, nothing
end

"""
    IGDMStationaryPolicy(K)

Shift-invariant memory: β_{n,n−ℓ} = ω[ℓ+1] for lags ℓ = 0..K (κ = K+1).
"""
struct IGDMStationaryPolicy <: AbstractPolicy
    K::Int
end
nparams(pol::IGDMStationaryPolicy, N::Int) = pol.K + 1
function evaluate_policy(pol::IGDMStationaryPolicy, ω::AbstractVector{<:Real}, N::Int)
    p = igdm_dof(N)
    η = zeros(p)
    J = zeros(p, pol.K + 1)
    for n in 1:N, ℓ in 0:min(pol.K, n - 1)
        r = igdm_flat_index(n, n - ℓ)
        η[r] = ω[ℓ + 1]
        J[r, ℓ + 1] = 1.0
    end
    return η, J, nothing
end

"""
    IGDMLagPolicy(K, base)

One independent scalar-schedule policy per lag ℓ = 0..K:
β_{n,n−ℓ} = base(ω_ℓ, N−ℓ)[n−ℓ]; ω = [ω_0; …; ω_K] (κ = (K+1)·κ_base).
"""
struct IGDMLagPolicy{P<:AbstractPolicy} <: AbstractPolicy
    K::Int
    base::P
end
nparams(pol::IGDMLagPolicy, N::Int) = (pol.K + 1) * nparams(pol.base, N)
function evaluate_policy(pol::IGDMLagPolicy, ω::AbstractVector{<:Real}, N::Int)
    κb = nparams(pol.base, N)   # assumed N-independent for schedule families
    κ = length(ω)
    p = igdm_dof(N)
    η = zeros(p)
    J = zeros(p, κ)
    H = Vector{Union{Nothing,Matrix{Float64}}}(nothing, p)
    anyH = false
    for ℓ in 0:min(pol.K, N - 1)
        off = ℓ * κb
        ωℓ = ω[(off + 1):(off + κb)]
        s, Js, Hs = evaluate_policy(pol.base, ωℓ, N - ℓ)
        for i in 1:(N - ℓ)
            n = ℓ + i
            r = igdm_flat_index(n, n - ℓ)
            η[r] = s[i]
            J[r, (off + 1):(off + κb)] = Js[i, :]
            if Hs !== nothing && Hs[i] !== nothing
                Hr = zeros(κ, κ)
                Hr[(off + 1):(off + κb), (off + 1):(off + κb)] = Hs[i]
                H[r] = Hr
                anyH = true
            end
        end
    end
    return η, J, anyH ? H : nothing
end
