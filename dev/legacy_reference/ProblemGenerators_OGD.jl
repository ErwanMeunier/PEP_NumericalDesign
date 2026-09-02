# Problem generator for the OGD PEP instance.

# N: Number of iterations
# L: Lipschitz constant
# D: Diameter of the domain
# α: step-sizes
# Build the OGD PEP instance and precompute A, ∇A and ∇2A.
function generate_OGD(N, L, D,  α)
    # Defining the PEP problem data for OGD
    # No initial condition for the first estimate
    A_R = spzeros(3N+1, 3N+1)
    R = 0
    #############################################
    C = spzeros(3N+1, 3N+1) # No dual information in the objective
    b = [ones(N); .- ones(N); zeros(N); [0]] # Objective: maximizing ∑_n f_n(x_n) - f_n(x_*)
    u = Vector{SparseArrays.SparseVector{Float64,Int64}}(undef, 3N+1)
    for i in 1:3N+1
        # Canonical basis vectors as sparse vectors
        u[i] = sparsevec([i], [1.0], 3N+1)
    end

    A, ∇A, ∇2A = nothing, nothing, nothing
    # Computing A, ∇A, ∇2A
    pep_problem_generic_instance = pep_problem_generic(A, ∇A, ∇2A, A_R, R, C, b, u, N, L, D)
    _, _, _ = compute_A_∇A_∇2A!(α, pep_problem_generic_instance)
    return pep_problem_generic_instance
end


###############################################################################
# Oracle factory for OGD.
#
# Returns a named tuple with fields:
#   generate(N, ω0)          -> pep_problem_generic
#   update!(ω, prob)         -> nothing  (updates A, ∇A, ∇2A in-place)
#   solve(prob)              -> sol_PEP_generic
#   grad_hess(ω, sol, prob)  -> (grad_ω, hess_ω)
#   params(ω, N)             -> α  (step-sizes)
###############################################################################
function make_OGD_oracle(L::Real, D::Real,
                         compute_α_∇α_∇2α::Function=compute_α_∇α_∇2α_identity)
    _generate  = (N, ω0) -> generate_OGD(N, Float64(L), Float64(D),
                                          compute_α_∇α_∇2α(ω0, N)[1])
    _update!   = (ω, prob) -> compute_A_∇A_∇2A!(compute_α_∇α_∇2α(ω, prob.N)[1], prob)
    _solve     = prob -> sdp_pep_generic(prob)
    _grad_hess = (ω, sol, prob) -> diff_w_wrt_stepsizes_parameters(
                                       ω, compute_α_∇α_∇2α, sol, prob)
    _params    = (ω, N) -> compute_α_∇α_∇2α(ω, N)[1]
    return (generate=_generate, var"update!"=_update!, solve=_solve,
            grad_hess=_grad_hess, params=_params)
end
    