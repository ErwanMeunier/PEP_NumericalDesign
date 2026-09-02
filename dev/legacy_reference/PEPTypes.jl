# Data structures for PEP problems and solutions

# Convenience alias used throughout the ITEM structs.
const SpMat = SparseMatrixCSC{Float64,Int}

# ─────────────────────────────────────────────────────────────────────────────
# Generic (OGD) PEP structures
# ─────────────────────────────────────────────────────────────────────────────

# Holds all problem data for the generic OGD PEP.
#
# A, ∇A, ∇2A:  nested tuple/vector structures built by compute_A_∇A_∇2A!
#              A  = ([A1,…,A7], [A5, A8])   where each element is Vector{SpMat}
#              ∇A = same shape as A, each inner matrix is now SpMat
#              ∇2A= same shape, but matrices indexed by [i, n, p] or [i, j, n, p]
mutable struct pep_problem_generic
    A::Any
    ∇A::Any
    ∇2A::Any
    A_R::SpMat                         # constraint on initial radius (zero for OGD)
    R::Float64
    C::SpMat                           # objective matrix (zero for OGD; b'F used instead)
    b::Vector{Float64}                 # linear objective on function values
    u::Vector{SparseVector{Float64,Int}} # canonical basis vectors for the Gram space
    N::Int64                           # number of iterations
    L::Float64
    D::Float64
end

mutable struct sol_PEP_generic
    obj_value::Float64
    G::Matrix{Float64}                 # primal Gram matrix
    F::Vector{Float64}                 # function values
    duals_cons::Any                    # nested dual multipliers (vector of vectors)
    dual_initial_cond::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# ITEM PEP structures
# ─────────────────────────────────────────────────────────────────────────────

# Holds all problem data for the ITEM PEP.
# params = [β_1, …, β_N, δ_1, …, δ_N]  (length 2N)
# Gram matrix dimension: 3N+5
#   Rows/cols:  z_1…z_{N+1}  |  y_0…y_N  |  y_*  |  g_0…g_N  |  g_*
#   Indices:    1…N+1         |  N+2…2N+2 |  2N+3 |  2N+4…3N+4|  3N+5
mutable struct pep_problem_ITEM
    A1::Vector{SpMat}     # A_n^{(1)} for n=1..N  (y_n recurrence)
    A2::Vector{SpMat}     # A_n^{(2)} for n=1..N  (z_{n+1} recurrence)
    A3::Matrix{SpMat}     # A_{ij}^{(3)} IC matrices, size (N+2)×(N+2)
    A5::SpMat             # initial condition  ‖z_1 - y_*‖² = D²
    A6::SpMat             # optimality  ‖g_*‖² = 0
    A_init::SpMat         # starting point equality  ‖z_1 - y_0‖² = 0
    C::SpMat              # objective  ‖z_{N+1} - y_*‖²
    ∇A1::Vector{SpMat}    # ∂A_n^{(1)}/∂β_n  for n=1..N
    ∇A2::Vector{SpMat}    # ∂A_n^{(2)}/∂δ_n  for n=1..N
    ∇2A1::Vector{SpMat}   # ∂²A_n^{(1)}/∂β_n²  for n=1..N
    ∇2A2::Vector{SpMat}   # ∂²A_n^{(2)}/∂δ_n²  for n=1..N
    N::Int64
    L::Float64
    mu::Float64           # strong-convexity constant (q = mu/L)
    D::Float64            # initial distance ‖z_1 - y_*‖ = D
end

# Holds the solution of the ITEM PEP primal SDP.
mutable struct sol_PEP_ITEM
    obj_value::Float64
    G::Matrix{Float64}         # primal Gram matrix
    F::Vector{Float64}         # function values [f_0,…,f_N, f_*]
    tau1::Vector{Float64}      # duals of A^{(1)} equalities  (n=1..N)
    tau2::Vector{Float64}      # duals of A^{(2)} equalities  (n=1..N)
    lambda::Matrix{Float64}    # duals of IC inequalities  (N+2)×(N+2)
    tau5::Float64              # dual of initial condition
    tau6::Float64              # dual of ‖g_*‖²=0
    tau_init::Float64          # dual of z_1 = y_0
end

# ─────────────────────────────────────────────────────────────────────────────
# IGDM PEP structures
# ─────────────────────────────────────────────────────────────────────────────

# Holds all problem data for the IGDM (Inexact Gradient Descent with Memory) PEP.
#
# Algorithm: x_{n+1} = x_n - ∑_{k=1}^{n} β_{n,k} d_k, n ∈ [N]
#            where d_n is an ε-relatively inexact gradient: ‖d_n - g_n‖ ≤ ε ‖g_n‖.
#            δ is fixed to the gradient-descent chain (δ_{n,n} = 1, else 0) and is
#            no longer a free parameter; β ≥ 0 are the (positive) step sizes.
#
# Gram matrix dimension: 3N+5
#   Rows/cols: x_1…x_{N+1} | x_*  | g_1…g_{N+1} | g_*   | d_1…d_{N+1}
#   Indices:   1…N+1        | N+2  | N+3…2N+3    | 2N+4  | 2N+5…3N+5
#
# Constraint families (from Lemma 1 in the paper):
#   A1[i,j]:  smoothness (L-smooth f), constant w.r.t. β
#   A2[n]:    algorithm update equality, parameter-dependent
#   A3[n]:    noisy gradient inequality, constant w.r.t. β
#   A4:       optimality condition g_* = 0, constant
#   C:        objective ‖g_{N+1}‖², constant
mutable struct pep_problem_IGDM
    A1::Matrix{SpMat}                      # (N+2)×(N+2) smoothness constraint matrices
    A2::Vector{SpMat}                      # length N, algorithm update (param-dependent)
    A3::Vector{SpMat}                      # length N+1, noisy gradient bound
    A4::SpMat                              # optimality: ‖g_*‖² = 0
    C::SpMat                               # objective: ‖g_{N+1}‖²
    u::Vector{SparseVector{Float64,Int}}   # 3N+5 canonical basis vectors
    β::Matrix{Float64}                     # N×N, lower-triangular (current step sizes)
    N::Int64
    L::Float64
    ε::Float64
end

# Holds the solution of the IGDM PEP primal SDP.
mutable struct sol_PEP_IGDM
    obj_value::Float64
    G::Matrix{Float64}         # primal Gram matrix (3N+5)×(3N+5)
    F::Vector{Float64}         # function values [f(x_1),…,f(x_{N+1}), f(x_*)]
    lambda1::Matrix{Float64}   # (N+2)×(N+2), smoothness duals (≥ 0)
    lambda2::Vector{Float64}   # length N, algorithm update duals (free sign)
    lambda3::Vector{Float64}   # length N+1, noisy gradient duals (≥ 0)
    lambda4::Float64           # optimality dual (free sign)
    tau::Float64               # initial condition dual (≥ 0)
end