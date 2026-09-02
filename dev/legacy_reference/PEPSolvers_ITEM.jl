# ITEM-specific SDP solvers (primal and dual)

# ─────────────────────────────────────────────────────────────────────────────
# Solve the ITEM PEP primal SDP.
#
# Primal variables: G ⪰ 0 (dim 3N+5),  F ∈ R^{N+2}
#
# Constraints:
#   tr(A1[n]  G) = 0        ∀ n=1..N   (y_n recurrence)
#   tr(A2[n]  G) = 0        ∀ n=1..N   (z_n recurrence)
#   tr(A3[k,l]G) ≤ F[k]-F[l]  ∀ k≠l,  k,l ∈ 1..N+2
#   tr(A5     G) = D^2          (initial condition)
#   tr(A6     G) = 0            (g_* = 0)
#   tr(A_init G) = 0            (z_1 = y_0)
#
# Objective: max tr(C G)
# ─────────────────────────────────────────────────────────────────────────────
function sdp_pep_ITEM(pep_prob::pep_problem_ITEM;verbose=0)
    N  = pep_prob.N
    D  = pep_prob.D
    dim = 3N + 5

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG", verbose > 0 ? 1 : 0)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", 1) # Prevent over-subscription when called from Julia @threads
    #set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
    #set_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS",   1e-9)
    #set_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS",   1e-9)

    @variable(model, G[1:dim, 1:dim], PSD)
    @variable(model, F[1:N+2])

    # Efficient tr(M·G) via sparse dot-product (valid for symmetric M, G)
    function tMG(M)
        rows, cols, vals = findnz(M)
        return @expression(model, sum(vals[k] * G[rows[k], cols[k]] for k in eachindex(vals)))
    end

    cons_A1     = @constraint(model, [n=1:N], tMG(pep_prob.A1[n]) == 0)
    cons_A2     = @constraint(model, [n=1:N], tMG(pep_prob.A2[n]) == 0)
    cons_A3     = @constraint(model, [k=1:N+2, l=1:N+2; k != l],
                      tMG(pep_prob.A3[k,l]) <= F[k] - F[l])
    @constraint(model, F[N+2] == 0)   # translational degree of freedom: f(y_*) = 0
    cons_A5     = @constraint(model, tMG(pep_prob.A5)     <= D^2)
    cons_A6     = @constraint(model, tMG(pep_prob.A6)     == 0)
    cons_A_init = @constraint(model, tMG(pep_prob.A_init) == 0)
    cons_scaling = @constraint(model, G[N+2,N+2] == 0)  # scaling constraint to avoid unboundedness
    # Breaking symmetries (not strictly necessary, but helps MOSEK's convergence): 
    @constraint(model, G[3N+5, :] .== 0) # g_* = 0 
    @constraint(model, G[:, 3N+5] .== 0) # g_* = 0 
    #
    @objective(model, Max, tMG(pep_prob.C))

    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        @warn("ITEM SDP did not solve to optimality. Status: $status")
    end
    # @show value(tMG(pep_prob.A5))
    
    tau1 = dual.(cons_A1)
    tau2 = dual.(cons_A2)

    # cons_A3 uses a filtered index (k != l) → JuMP returns a SparseAxisArray
    lambda_sparse = dual.(cons_A3)
    lambda = zeros(N+2, N+2)
    for k in 1:N+2, l in 1:N+2
        k != l && (lambda[k, l] = lambda_sparse[k, l])
    end

    flip_sign = -1. # The sign is flipped since MOSEK returns - dual variables
    return sol_PEP_ITEM(
        objective_value(model),
        value.(G),
        value.(F),
        flip_sign * tau1,
        flip_sign * tau2,
        flip_sign * lambda,
        flip_sign * dual(cons_A5),
        flip_sign * dual(cons_A6),
        flip_sign * dual(cons_A_init),
        #status
    )
end


# ─────────────────────────────────────────────────────────────────────────────
# Solve the ITEM PEP dual SDP.
#
#   min  D^2 τ^{(5)}
#   s.t. ∑_n τ_n^{(1)}A1[n] + ∑_n τ_n^{(2)}A2[n]
#          + ∑_{k≠l} λ_{kl}A3[k,l] + τ^{(5)}A5 + τ^{(6)}A6 + τ_init·A_init  ⪰  C
#        ∑_{k≠l} λ_{kl}(e_k - e_l) = 0
#        λ_{kl} ≥ 0
# ─────────────────────────────────────────────────────────────────────────────
function sdp_pep_dual_ITEM(pep_prob::pep_problem_ITEM)
    N  = pep_prob.N
    D  = pep_prob.D
    dim = 3N + 5

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG", 0)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS",   1e-9)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS",   1e-9)

    @variable(model, tau1[1:N])
    @variable(model, tau2[1:N])
    @variable(model, lambda[1:N+2, 1:N+2] >= 0)
    @variable(model, tau5)
    @variable(model, tau6)
    @variable(model, tau_init)

    dual_mat = @expression(model,
                   sum(tau1[n] * pep_prob.A1[n] for n in 1:N) +
                   sum(tau2[n] * pep_prob.A2[n] for n in 1:N) +
                   sum(lambda[k,l] * pep_prob.A3[k,l]
                       for k in 1:N+2 for l in 1:N+2 if k != l) +
                   tau5 * pep_prob.A5 + tau6 * pep_prob.A6 +
                   tau_init * pep_prob.A_init - pep_prob.C)
    @constraint(model, dual_mat in PSDCone())

    for k in 1:N+2
        @constraint(model,
            sum(lambda[k,l] - lambda[l,k] for l in 1:N+2 if l != k) == 0)
    end

    @objective(model, Min, D^2 * tau5)

    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        @warn("ITEM dual SDP did not solve to optimality. Status: $(termination_status(model))")
    end

    return (
        obj_value = objective_value(model),
        tau1      = value.(tau1),
        tau2      = value.(tau2),
        lambda    = value.(lambda),
        tau5      = value(tau5),
        tau6      = value(tau6),
        tau_init  = value(tau_init)
    )
end
