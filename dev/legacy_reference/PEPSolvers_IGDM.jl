# SDP solvers for IGDM PEP problems (primal and dual)
#
# Primal SDP (Lemma 1 in the paper):
#   max   tr(C G)
#   s.t.  tr(A1[i,j] G) ≤ F[i] - F[j]   ∀ i≠j ∈ [N+2]   (smoothness)
#         tr(A2[n]   G) = 0               ∀ n ∈ [N]       (algorithm update)
#         tr(A3[n]   G) ≤ 0               ∀ n ∈ [N+1]     (noisy gradient)
#         tr(A4      G) = 0                                 (optimality g_*=0)
#         F[1] - F[N+2] ≤ 1                                (initial condition)
#         F[N+2] = 0                                        (normalization w.l.o.g.)
#         G ≽ 0
#
# Dual SDP (Lemma 2 in the paper):
#   min   τ
#   s.t.  ∑_{i≠j} λ1[i,j]·A1[i,j] + ∑_n λ2[n]·A2[n] + ∑_n λ3[n]·A3[n] + λ4·A4 - C ≽ 0
#         ∑_{j≠m} (λ1[m,j] - λ1[j,m]) + τ·(1_{m=1} - 1_{m=N+2}) = 0  ∀m ∈ [N+2]
#         λ1[i,j] ≥ 0,  λ3[n] ≥ 0,  τ ≥ 0
#         λ2[n] free,   λ4 free

# ─────────────────────────────────────────────────────────────────────────────
# Solve the IGDM primal SDP.
# ─────────────────────────────────────────────────────────────────────────────
function sdp_pep_IGDM(prob::pep_problem_IGDM; verbose=0)
    N   = prob.N
    dim = 3N + 5

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG",                   verbose > 0 ? 1 : 0)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP",   1e-8)
    set_attribute(model, "MSK_IPAR_NUM_THREADS",           1)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS",   1e-9)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS",   1e-9)

    @variable(model, G[1:dim, 1:dim], PSD)
    @variable(model, F[1:N+2])

    # Normalization and symmetries breaking constraints:
    @constraint(model, F[N+2] == 0)   # normalization: f(x_*) = 0
    @constraint(model, G[N+2, N+2] == 0)  # normalization: x_* = 0
    @constraint(model, G[N+2, :] .== 0)  # symmetry breaking: g_{N+1} = 0
    @constraint(model, G[:, N+2] .== 0)  # symmetry breaking: g_{N+1} = 0
    @constraint(model, G[2N+4, :] .== 0)  # symmetry breaking: u_1 = 0
    @constraint(model, G[:, 2N+4] .== 0)
    # Efficient tr(M·G) via sparse dot-product
    function tMG(M)
        rows, cols, vals = findnz(M)
        return @expression(model,
            sum(vals[k] * G[rows[k], cols[k]] for k in eachindex(vals)))
    end

    # (a) Smoothness: tr(A1[i,j] G) ≤ F[i] - F[j]
    cons_A1 = @constraint(model, [i=1:N+2, j=1:N+2; i != j],
                  tMG(prob.A1[i, j]) <= F[i] - F[j])

    # (b) Algorithm update equality: tr(A2[n] G) = 0
    cons_A2 = @constraint(model, [n=1:N], tMG(prob.A2[n]) == 0)

    # (c) Noisy gradient: tr(A3[n] G) ≤ 0
    cons_A3 = @constraint(model, [n=1:N+1], tMG(prob.A3[n]) <= 0)

    # (d) Optimality: tr(A4 G) = 0  (g_* = 0)
    cons_A4 = @constraint(model, tMG(prob.A4) == 0)

    # (e) Initial condition: F[1] - F[N+2] ≤ 1
    cons_init = @constraint(model, F[1] - F[N+2] <= 1)

    
    @objective(model, Max, tMG(prob.C))

    optimize!(model)
    status = termination_status(model)
    if verbose > 0
        println("SDP solve status: $status")
    end
    if status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        @warn("IGDM primal SDP did not solve to optimality. Status: $status")
    end

    # MOSEK sign convention: multiply duals by -1 (consistent with ITEM solver)
    flip = -1.0

    lambda1_sparse = dual.(cons_A1)
    lambda1 = zeros(N+2, N+2)
    for i in 1:N+2, j in 1:N+2
        i != j && (lambda1[i, j] = flip * lambda1_sparse[i, j])
    end

    return sol_PEP_IGDM(
        objective_value(model),
        value.(G),
        value.(F),
        lambda1,
        flip .* dual.(cons_A2),
        flip .* dual.(cons_A3),
        flip  * dual(cons_A4),
        flip  * dual(cons_init),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve the IGDM dual SDP.
# Returns a named tuple with obj_value, lambda1, lambda2, lambda3, lambda4, tau.
# ─────────────────────────────────────────────────────────────────────────────
function sdp_pep_dual_IGDM(prob::pep_problem_IGDM)
    N   = prob.N
    dim = 3N + 5

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG",                  0)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP",  1e-8)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS",    1e-9)
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS",    1e-9)
    set_attribute(model, "MSK_IPAR_NUM_THREADS",          1)

    @variable(model, lambda1[1:N+2, 1:N+2] >= 0)  # smoothness (diagonal unused)
    @variable(model, lambda2[1:N])                  # update equality (free sign)
    @variable(model, lambda3[1:N+1] >= 0)           # noisy gradient
    @variable(model, lambda4)                        # optimality (free sign)
    @variable(model, tau >= 0)                       # initial condition

    # PSD constraint: weighted sum of constraint matrices minus C
    dual_mat = @expression(model,
        sum(lambda1[i,j] * prob.A1[i,j]
            for i in 1:N+2, j in 1:N+2 if i != j) +
        sum(lambda2[n]   * prob.A2[n] for n in 1:N) +
        sum(lambda3[n]   * prob.A3[n] for n in 1:N+1) +
        lambda4 * prob.A4 - prob.C)
    @constraint(model, dual_mat in PSDCone())

    # F-variable dual feasibility: for each m ∈ [N+2]
    #   ∑_{j≠m} (λ1[m,j] - λ1[j,m]) + τ·(1_{m=1} - 1_{m=N+2}) = 0
    for m in 1:N+2
        τ_coeff = (m == 1 ? 1.0 : 0.0) - (m == N+2 ? 1.0 : 0.0)
        @constraint(model,
            sum(lambda1[m, j] - lambda1[j, m] for j in 1:N+2 if j != m) +
            τ_coeff * tau == 0)
    end

    @objective(model, Min, tau)

    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        @warn("IGDM dual SDP did not solve to optimality. Status: $status")
    end

    return (
        obj_value = objective_value(model),
        lambda1   = value.(lambda1),
        lambda2   = value.(lambda2),
        lambda3   = value.(lambda3),
        lambda4   = value(lambda4),
        tau       = value(tau),
    )
end
