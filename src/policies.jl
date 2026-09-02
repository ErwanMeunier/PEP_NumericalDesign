# Step-size policy families: callable structs mapping (ω, N) → (η, J, Hpol).
#
# Contract: evaluate_policy(pol, ω, N) returns
#   η    :: Vector{Float64}          — coefficient vector (length = PEP's np)
#   J    :: Matrix{Float64} (p×κ)    — Jacobian ∂η/∂ω
#   Hpol :: Vector of κ×κ matrices (∂²η_r/∂ω²), or `nothing` when linear in ω.

abstract type AbstractPolicy end

"""Number of design parameters κ of a policy at horizon N."""
function nparams end

"""Free coefficients: η = ω (κ = N, Kamri-style unrestricted baseline)."""
struct IdentityPolicy <: AbstractPolicy end
nparams(::IdentityPolicy, N::Int) = N
function evaluate_policy(::IdentityPolicy, ω::AbstractVector{<:Real}, N::Int)
    p = length(ω)
    return collect(Float64, ω), Matrix{Float64}(LinearAlgebra.I, p, p), nothing
end

"""Single shared value: η_n = ω_1 for all n (κ = 1)."""
struct ConstantPolicy <: AbstractPolicy end
nparams(::ConstantPolicy, N::Int) = 1
function evaluate_policy(::ConstantPolicy, ω::AbstractVector{<:Real}, N::Int)
    return fill(Float64(ω[1]), N), ones(N, 1), nothing
end

"""
    FunctionPolicy(f, κ; name="custom")

Generic scalar-schedule policy `η_n = f(ω, n, N)`; derivatives via ForwardDiff.
Used for all catalog families without hand-coded derivatives.
"""
struct FunctionPolicy{F} <: AbstractPolicy
    f::F
    κ::Int
    name::String
end
FunctionPolicy(f, κ::Int; name::String = "custom") = FunctionPolicy(f, κ, name)
nparams(pol::FunctionPolicy, N::Int) = pol.κ
function evaluate_policy(pol::FunctionPolicy, ω::AbstractVector{<:Real}, N::Int)
    ωf = collect(Float64, ω)
    η = [pol.f(ωf, n, N) for n in 1:N]
    J = zeros(N, pol.κ)
    H = Vector{Matrix{Float64}}(undef, N)
    for n in 1:N
        J[n, :] = ForwardDiff.gradient(w -> pol.f(w, n, N), ωf)
        H[n] = ForwardDiff.hessian(w -> pol.f(w, n, N), ωf)
    end
    return η, J, H
end

"""Display label of a policy (for experiment tables and figures)."""
policy_label(pol::FunctionPolicy) = pol.name
policy_label(::IdentityPolicy) = "identity"
policy_label(::ConstantPolicy) = "constant"
policy_label(pol::AbstractPolicy) = string(nameof(typeof(pol)))

# ── catalog of scalar schedule families (paper §5) ───────────────────────────

"""Sum of exponentials η_n = Σᵢ aᵢ e^{−bᵢ n} + c; ω = [a₁..a_k, b₁..b_k, c]."""
sum_of_exp_policy(k::Int = 2) = FunctionPolicy(
    (ω, n, N) -> sum(ω[i] * exp(-ω[k + i] * n) for i in 1:k) + ω[2k + 1],
    2k + 1; name = "sum_of_exp$(k)")

"""Log-polynomial η_n = exp(Σⱼ cⱼ log(n+δ)ʲ); ω = [c₀..c_d]."""
log_poly_policy(d::Int = 2; δ::Float64 = 1.0) = FunctionPolicy(
    (ω, n, N) -> exp(sum(ω[j + 1] * log(n + δ)^j for j in 0:d)),
    d + 1; name = "log_poly$(d)")

"""DCT spectral η_n = Σₖ cₖ cos(πk(n−1)/(N−1)); ω = [c₀..c_K]. N-dependent."""
dct_policy(K::Int = 3) = FunctionPolicy(
    (ω, n, N) -> N == 1 ? ω[1] :
        sum(ω[k + 1] * cos(π * k * (n - 1) / (N - 1)) for k in 0:K),
    K + 1; name = "dct$(K)")

"""Cosine decay η_n = a(1+cos(πn/N))ᵖ + c; ω = [a, p, c]. N-dependent."""
cosine_decay_policy() = FunctionPolicy(
    (ω, n, N) -> ω[1] * (1 + cos(π * n / N))^ω[2] + ω[3],
    3; name = "cosine_decay")

"""Rational η_n = (a₀+a₁n)/(1+b₁n+b₂n²); ω = [a₀, a₁, b₁, b₂]."""
rational_policy() = FunctionPolicy(
    (ω, n, N) -> (ω[1] + ω[2] * n) / (1 + ω[3] * n + ω[4] * n^2),
    4; name = "rational")

"""Warped Chebyshev η_n = Σₖ aₖ Tₖ(2log(n+1)/log(N+1)−1); ω = [a₀..a_K]."""
function warped_chebyshev_policy(K::Int = 3)
    f = function (ω, n, N)
        v = 2 * log(n + 1) / log(N + 1) - 1
        Tkm1, Tk = one(v), v
        acc = ω[1] * Tkm1
        K >= 1 && (acc += ω[2] * Tk)
        for k in 2:K
            Tkm1, Tk = Tk, 2v * Tk - Tkm1
            acc += ω[k + 1] * Tk
        end
        acc
    end
    return FunctionPolicy(f, K + 1; name = "warped_chebyshev$(K)")
end

"""Piecewise exponential (split at n_split, default N÷2); ω = [a₁,b₁,a₂,b₂]."""
piecewise_exp_policy(; n_split::Union{Int,Nothing} = nothing) = FunctionPolicy(
    (ω, n, N) -> n <= (n_split === nothing ? N ÷ 2 : n_split) ?
        ω[1] * exp(-ω[2] * n) : ω[3] * exp(-ω[4] * n),
    4; name = "piecewise_exp")

"""
    ProductPolicy(p1, p2)

Stack two independent policies: η = [η₁(ω₁); η₂(ω₂)] with ω = [ω₁; ω₂]
(e.g. ITEM's β-schedule ⊗ δ-schedule).
"""
struct ProductPolicy{A<:AbstractPolicy,B<:AbstractPolicy} <: AbstractPolicy
    p1::A
    p2::B
end
nparams(pp::ProductPolicy, N::Int) = nparams(pp.p1, N) + nparams(pp.p2, N)
function evaluate_policy(pp::ProductPolicy, ω::AbstractVector{<:Real}, N::Int)
    κ1 = nparams(pp.p1, N)
    κ = length(ω)
    η1, J1, H1 = evaluate_policy(pp.p1, ω[1:κ1], N)
    η2, J2, H2 = evaluate_policy(pp.p2, ω[(κ1 + 1):end], N)
    p1, p2 = length(η1), length(η2)
    J = zeros(p1 + p2, κ)
    J[1:p1, 1:κ1] = J1
    J[(p1 + 1):end, (κ1 + 1):end] = J2
    H = nothing
    if H1 !== nothing || H2 !== nothing
        H = Vector{Union{Nothing,Matrix{Float64}}}(nothing, p1 + p2)
        if H1 !== nothing
            for r in 1:p1
                Hr = zeros(κ, κ)
                Hr[1:κ1, 1:κ1] = H1[r]
                H[r] = Hr
            end
        end
        if H2 !== nothing
            for r in 1:p2
                Hr = zeros(κ, κ)
                Hr[(κ1 + 1):end, (κ1 + 1):end] = H2[r]
                H[p1 + r] = Hr
            end
        end
    end
    return [η1; η2], J, H
end

"""
    MappedPolicy(base, dof, index)

Lift a base policy into a larger flat coefficient vector: the base output of
length `length(index(N))` is scattered to positions `index(N)` of a zero
vector of length `dof(N)` (e.g. schedule policies on IGDM's β diagonal).
"""
struct MappedPolicy{P<:AbstractPolicy} <: AbstractPolicy
    base::P
    dof::Function
    index::Function
end
nparams(mp::MappedPolicy, N::Int) = nparams(mp.base, N)
function evaluate_policy(mp::MappedPolicy, ω::AbstractVector{<:Real}, N::Int)
    ηb, Jb, Hb = evaluate_policy(mp.base, ω, N)
    p = mp.dof(N)
    idx = mp.index(N)
    length(idx) == length(ηb) ||
        error("MappedPolicy: base policy produced $(length(ηb)) values for $(length(idx)) slots")
    η = zeros(p)
    η[idx] = ηb
    J = zeros(p, length(ω))
    J[idx, :] = Jb
    H = nothing
    if Hb !== nothing
        H = Vector{Union{Nothing,Matrix{Float64}}}(nothing, p)
        for (i, r) in enumerate(idx)
            H[r] = Hb[i]
        end
    end
    return η, J, H
end

"""Power law η_n = a/(n^b + c), ω = [a, b, c] (closed-form derivatives)."""
struct PowerLawPolicy <: AbstractPolicy end
nparams(::PowerLawPolicy, N::Int) = 3
function evaluate_policy(::PowerLawPolicy, ω::AbstractVector{<:Real}, N::Int)
    a, b, c = Float64.(ω[1:3])
    η = zeros(N)
    J = zeros(N, 3)
    H = Vector{Matrix{Float64}}(undef, N)
    for n in 1:N
        nb = Float64(n)^b
        d = nb + c
        lg = log(n)
        η[n] = a / d
        J[n, 1] = 1 / d
        J[n, 2] = -a * nb * lg / d^2
        J[n, 3] = -a / d^2
        Hn = zeros(3, 3)
        Hn[1, 2] = Hn[2, 1] = -nb * lg / d^2
        Hn[1, 3] = Hn[3, 1] = -1 / d^2
        Hn[2, 2] = a * nb * lg^2 * (nb - c) / d^3
        Hn[2, 3] = Hn[3, 2] = 2a * nb * lg / d^3
        Hn[3, 3] = 2a / d^3
        H[n] = Hn
    end
    return η, J, H
end
