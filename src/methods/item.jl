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
