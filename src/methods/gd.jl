# Gradient descent PEP (Kamri–Hendrickx–Glineur benchmark), substitution style.
#
#   x_{k+1} = x_k − γ_{k+1} ∇f(x_k),  k = 0..N−1,   f ∈ 𝓕_{0,L},  ‖x_0 − x*‖ ≤ R
#   Objective: max f(x_N) − f*.
#
# Coefficients: η = γ ∈ R^N. The iterates are affine in γ, so the interpolation
# constraints are quadratic in γ (Kamri's encoding). Pinning x* = 0, g* = 0,
# f* = 0 gives exactly Kamri's Gram basis [x_0, g_0..g_N] (dim N+2) and
# function-value vector [f_0..f_N].

"""
    gd_pep(N, L; R=1.0) -> (PEPModel, γ::Vector{PAff})

Build the parametric gradient-descent PEP with worst-case `f(x_N) − f*`.
"""
function gd_pep(N::Int, L::Real; R::Real = 1.0)
    m = PEPModel()
    γ = coeffs!(m, N)
    f = SmoothConvexFunction(m; L = L, name = "f")
    xstar = PointExpr()
    x0 = point!(m; name = "x0")
    x = x0
    for k in 1:N
        gx, _ = oracle!(f, x)
        x = x - γ[k] * gx
    end
    _, fN = oracle!(f, x)
    add_oracle_point!(f, xstar, PointExpr(), 0.0; stationary = true)
    add_le!(m, sqnorm(x0 - xstar) - R^2; name = "init")
    objective_max!(m, fN)
    return m, γ
end

"""
    compile_gd(N, L; R=1.0) :: CompiledPEP
"""
compile_gd(N::Int, L::Real; R::Real = 1.0) = compile(gd_pep(N, L; R)[1])
