# OGD regret PEP via the high-level frontend (parity with the legacy
# FOM_SOM formulation).
#
# Setting: N convex L-Lipschitz losses f_n on a convex set of diameter D,
# x* = 0, projection encoded through indicator normal-cone oracles.
# Coefficients: η = α ∈ R^N (α_N is unused by the recursion but kept so that
# policies of length N apply unchanged; ∂W/∂α_N = 0).
#
# Each round n uses one ConvexLipschitzFunction sampled at (x_n, x*); the
# feasible set is one ConvexIndicatorFunction with diameter D sampled at
# x_1..x_N and x*, whose normal element at x* is pinned to −Σ g*_s
# (optimality of x* for the total loss). Objective: max Σ_n f_n − f*_n.

"""
    ogd_pep(N, L, D) -> (PEPModel, α::Vector{PAff})

Build the parametric regret PEP for Online Gradient Descent.
"""
function ogd_pep(N::Int, L::Real, D::Real)
    m = PEPModel()
    α = coeffs!(m, N)
    fs = [ConvexLipschitzFunction(m; M = L, name = "f$n") for n in 1:N]
    C = ConvexIndicatorFunction(m; D = D, name = "C")
    xstar = PointExpr()                             # x* = 0 (translation pinning)

    x = Vector{PointExpr}(undef, N)
    x[1] = point!(m; name = "x1")
    oracle!(C, x[1])                                # Ψ₁ ∈ N_C(x₁)
    g = Vector{PointExpr}(undef, N)
    fx = Vector{QExpr}(undef, N)                    # f_n(x_n)
    for n in 1:N
        g[n], fx[n] = oracle!(fs[n], x[n])
        if n < N                                    # projected step Π_C(x_n − α_n g_n)
            x[n + 1], _, _ = proximal_step!(x[n] - α[n] * g[n], C, 1.0)
        end
    end
    gstar = Vector{PointExpr}(undef, N)
    fxs = Vector{QExpr}(undef, N)                   # f_n(x*)
    for n in 1:N
        gstar[n], fxs[n] = oracle!(fs[n], xstar)
    end
    add_oracle_point!(C, xstar, -reduce(+, gstar))  # Ψ* = −Σ g*_s ∈ N_C(x*)

    objective_max!(m, sum(fx[n] - fxs[n] for n in 1:N))
    return m, α
end

"""
    compile_ogd(N, L, D) :: CompiledPEP
"""
compile_ogd(N::Int, L::Real, D::Real) = compile(ogd_pep(N, L, D)[1])
