# ITEM (Information-Theoretic Exact Method) PEP via the DSL, residual-style.
#
# y_n = (1−β_n) z_n + β_n (y_{n−1} − g_{n−1}/L),        n = 1..N
# z_{n+1} = (1−qδ_n) z_n + qδ_n (y_n − g_n/μ),  q = μ/L, n = 1..N
# Objective: max ‖z_{N+1} − y_*‖²  s.t.  ‖z_1 − y_*‖² ≤ D², z_1 = y_0.
#
# Coefficients: η = [β_1..β_N, δ_1..δ_N] (code index n ↔ paper index n−1;
# the analytic optimum uses β*_{n-1}, δ*_{n-1}).
# Conditioning: z_1 = y_0 = 0 and g_* = 0 are eliminated from the Gram basis
# (translation/optimality pinning), giving dimension 3N+2 instead of 3N+5.

"""
    item_pep(N, L, μ, D) -> (PEPModel, θ::Vector{PAff})

Build the parametric ITEM PEP; `θ = [β; δ]` of length 2N.
"""
function item_pep(N::Int, L::Real, μ::Real, D::Real)
    q = μ / L
    m = PEPModel()
    θ = coeffs!(m, 2N)
    β = θ[1:N]
    δ = θ[(N + 1):(2N)]

    fobj = SmoothStronglyConvexFunction(m; μ = μ, L = L, name = "f")
    origin = PointExpr()
    z = [i == 1 ? origin : point!(m; name = "z$i") for i in 1:(N + 1)]   # z_1 = 0
    y = [j == 1 ? origin : point!(m; name = "y$(j - 1)") for j in 1:(N + 1)]  # y_0 = 0
    ystar, _, fstar = stationary_point!(fobj)                 # g_* ≡ 0
    g = Vector{PointExpr}(undef, N + 1)
    f = Vector{QExpr}(undef, N + 1)
    for j in 1:(N + 1)                                        # oracle at y_{j-1}
        g[j], f[j] = oracle!(fobj, y[j])
    end

    for n in 1:N   # y_n recurrence residual (y_{n-1}=y[n], g_{n-1}=g[n])
        r = (1.0 - β[n]) * z[n] + β[n] * y[n] - (1 / L) * β[n] * g[n] - y[n + 1]
        add_eq!(m, sqnorm(r); name = "yrec[$n]")
    end
    for n in 1:N   # z_{n+1} recurrence residual (y_n=y[n+1], g_n=g[n+1])
        r = (1.0 - q * δ[n]) * z[n] + q * δ[n] * y[n + 1] -
            (q / μ) * δ[n] * g[n + 1] - z[n + 1]
        add_eq!(m, sqnorm(r); name = "zrec[$n]")
    end

    add_eq!(m, fstar; name = "fstar0")                              # f_* = 0
    add_le!(m, sqnorm(ystar) - D^2; name = "init")                  # ‖z_1−y_*‖² ≤ D²

    objective_max!(m, sqnorm(z[N + 1] - ystar))
    return m, θ
end

"""
    compile_item(N, L, μ, D) :: CompiledPEP
"""
compile_item(N::Int, L::Real, μ::Real, D::Real) = compile(item_pep(N, L, μ, D)[1])

# ── fixed-step (substitution) ITEM encoding ──────────────────────────────────
#
# N oracle calls g_0..g_{N-1} at y_0..y_{N-1} (y_0 = z_0 = 0, pinned) and
#   y_n = −(1/L) Σ_{i<n} h_{n,i} g_i     (n = 1..N−1)
#   z_N = −(1/L) Σ_{i<N} c_i g_i
# Coefficients η = [h (row-major, n = 1..N−1, i = 0..n−1); c_0..c_{N−1}],
# length N(N+1)/2. Every constraint is affine or quadratic in η (the
# μ‖y_i − y_j‖² interpolation term), Gram basis [y_*, g_0..g_{N−1}].
# Taylor–Drori's lower bound makes 1/(1+qA_N) the exact optimum of this
# class, attained by ITEM.

"""Number of fixed-step ITEM coefficients, N(N+1)/2."""
item_fs_dof(N::Int) = N * (N + 1) ÷ 2

"""Flat index of h_{n,i} (1 ≤ n ≤ N−1, 0 ≤ i < n) in the fixed-step vector."""
item_fs_hindex(n::Int, i::Int) = (n - 1) * n ÷ 2 + i + 1

"""Flat index of c_i (0 ≤ i < N) in the fixed-step vector."""
item_fs_cindex(N::Int, i::Int) = (N - 1) * N ÷ 2 + i + 1

"""
    item_fs_pep(N, L, μ, D) -> (PEPModel, η::Vector{PAff})

Fixed-step (substitution) ITEM PEP: maximize ‖z_N − y_*‖² over L-smooth
μ-strongly convex f with ‖z_0 − y_*‖ ≤ D. See [`item_fs_dof`](@ref).
"""
function item_fs_pep(N::Int, L::Real, μ::Real, D::Real)
    N >= 1 || error("item_fs_pep: N ≥ 1 required")
    m = PEPModel()
    η = coeffs!(m, item_fs_dof(N))
    fobj = SmoothStronglyConvexFunction(m; μ = μ, L = L, name = "f")
    ystar, _, fstar = stationary_point!(fobj)
    y = Vector{PointExpr}(undef, N)
    g = Vector{PointExpr}(undef, N)
    y[1] = PointExpr()                                       # y_0 = z_0 = 0
    g[1], _ = oracle!(fobj, y[1])
    for n in 1:(N - 1)
        y[n + 1] = -(1 / L) * sum(η[item_fs_hindex(n, i)] * g[i + 1] for i in 0:(n - 1))
        g[n + 1], _ = oracle!(fobj, y[n + 1])
    end
    zN = -(1 / L) * sum(η[item_fs_cindex(N, i)] * g[i + 1] for i in 0:(N - 1))
    add_eq!(m, fstar; name = "fstar0")
    add_le!(m, sqnorm(ystar) - D^2; name = "init")
    objective_max!(m, sqnorm(zN - ystar))
    return m, η
end

"""
    compile_item_fs(N, L, μ, D) :: CompiledPEP
"""
compile_item_fs(N::Int, L::Real, μ::Real, D::Real) = compile(item_fs_pep(N, L, μ, D)[1])

"""
    item_fs_map(θ, N, L, μ) -> η

Map ITEM-form parameters θ = [β_0..β_{N−1}; δ_0..δ_{N−1}] to fixed-step
coefficients by running the ITEM recursion symbolically in the gradient
basis (β_0 is inert: y_0 = z_0). Generic in the element type (ForwardDiff).
"""
function item_fs_map(θ::AbstractVector{T}, N::Int, L::Real, μ::Real) where {T}
    q = μ / L
    β = view(θ, 1:N)
    δ = view(θ, (N + 1):(2N))
    z = zeros(T, N)
    yprev = zeros(T, N)
    η = zeros(T, item_fs_dof(N))
    for k in 0:(N - 1)
        if k == 0
            y = copy(z)
        else
            y = (1 - β[k + 1]) .* z .+ β[k + 1] .* yprev
            y[k] -= β[k + 1] / L                             # −β_k g_{k−1}/L
            for i in 0:(k - 1)
                η[item_fs_hindex(k, i)] = -L * y[i + 1]
            end
        end
        znew = (1 - q * δ[k + 1]) .* z .+ (q * δ[k + 1]) .* y
        znew[k + 1] -= q * δ[k + 1] / μ                      # −qδ_k g_k/μ
        z, yprev = znew, y
    end
    for i in 0:(N - 1)
        η[item_fs_cindex(N, i)] = -L * z[i + 1]
    end
    return η
end

"""
    item_reference(N, L, μ) -> (θ*, W*)

Exact ITEM parameters θ* = [β_0..β_{N−1}; δ_0..δ_{N−1}] (Taylor & Drori,
2021) and the tight rate W* = 1/(1+qA_N) of the fixed-step encoding (D = 1).
"""
function item_reference(N::Int, L::Real, μ::Real)
    q = μ / L
    A = zeros(N + 1)                                         # A[k+1] = A_k
    for k in 1:N
        Ak = A[k]
        A[k + 1] = ((1 + q) * Ak + 2 * (1 + sqrt((1 + Ak) * (1 + q * Ak)))) / (1 - q)^2
    end
    β = [A[k + 1] / ((1 - q) * A[k + 2]) for k in 0:(N - 1)]
    δ = [((1 - q)^2 * A[k + 2] - (1 + q) * A[k + 1]) / (2 * (1 + q + q * A[k + 1]))
         for k in 0:(N - 1)]
    return [β; δ], 1 / (1 + q * A[N + 1])
end

"""
    ITEMPolicy(inner, L, μ)

Structured prior on the fixed-step ITEM space: η = item_fs_map(inner(ω)),
where `inner` produces the 2N ITEM-form parameters [β; δ] (e.g.
`IdentityPolicy()` or `ProductPolicy(dct_policy(3), dct_policy(3))`).
"""
struct ITEMPolicy{P<:AbstractPolicy} <: AbstractPolicy
    inner::P
    L::Float64
    μ::Float64
end
ITEMPolicy(inner::AbstractPolicy, L::Real, μ::Real) = ITEMPolicy(inner, Float64(L), Float64(μ))
nparams(pol::ITEMPolicy, N::Int) = nparams(pol.inner, N)
policy_label(pol::ITEMPolicy) = "item(" * policy_label(pol.inner) * ")"
function evaluate_policy(pol::ITEMPolicy, ω::AbstractVector{<:Real}, N::Int)
    θ, Jθ, Hθ = evaluate_policy(pol.inner, ω, N)
    length(θ) == 2N || error("ITEMPolicy: inner policy must produce 2N values")
    F = t -> item_fs_map(t, N, pol.L, pol.μ)
    η = F(θ)
    Jm = ForwardDiff.jacobian(F, θ)
    J = Jm * Jθ
    p, m = length(η), length(θ)
    T3 = reshape(ForwardDiff.jacobian(t -> vec(ForwardDiff.jacobian(F, t)), θ), p, m, m)
    H = Vector{Matrix{Float64}}(undef, p)
    for r in 1:p
        Hr = Jθ' * T3[r, :, :] * Jθ
        if Hθ !== nothing
            for s in eachindex(θ)
                Hθ[s] === nothing || (Hr .+= Jm[r, s] .* Hθ[s])
            end
        end
        H[r] = Hr
    end
    return η, J, H
end

"""
    item_optimal_params(N, L, μ) -> (β*, δ*, W*_factor)

Reference ITEM coefficients and guarantee factor `1/(1+qA_N)`.

The compact two-sequence rewrite implemented here omits the paper's δ₀ step
(z₁ = y₀ is given), so slot n carries paper β_n but paper δ_{n−1}, and the
guarantee is a near-tight reference only (empirically within ~0.2%), not an
exact optimum of this parameterization.
"""
function item_optimal_params(N::Int, L::Real, μ::Real)
    q = μ / L
    A = zeros(N + 2)   # A[k] = A_{k-1}
    for k in 2:(N + 2)
        Ap = A[k - 1]
        A[k] = ((1 + q) * Ap + 2 * (1 + sqrt((1 + Ap) * (1 + q * Ap)))) / (1 - q)^2
    end
    β = [A[n + 1] / ((1 - q) * A[n + 2]) for n in 1:N]                 # paper β_n
    δ = [((1 - q)^2 * A[n + 1] - (1 + q) * A[n]) / (2 * (1 + q + q * A[n]))
         for n in 1:N]                                                 # paper δ_{n-1}
    return β, δ, 1 / (1 + q * A[N + 1])
end
