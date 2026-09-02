# SDP solvers for PEP problems (OGD / generic)

# Solve the primal SDP formulation and return objective, primal variables and dual multipliers.
function sdp_pep_generic(pep_problem::pep_problem_generic) # ok
    # Unpacking the PEP problem
    A = pep_problem.A
    A_R = pep_problem.A_R
    R = pep_problem.R
    C = pep_problem.C
    b = pep_problem.b
    u = pep_problem.u
    N = pep_problem.N
    L = pep_problem.L
    D = pep_problem.D
    # Creating the PEP model
    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG", 0) # Suppress Mosek output
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", 1) # Prevent over-subscription when called from Julia @threads
    # Variables
    @variable(model, G[1:size(A_R, 1), 1:size(A_R, 1)], PSD)
    @variable(model, F[1:size(A_R,1)])
    ########################### Parsing constraints #########################
    # Hard constraints for now
    # Convention for OGD: A = ([A1 A2 A3 A4 A6 A7 A9], [A5  A8])
    cons_1idx = Vector{Any}(undef, 7) # A1 
    cons_1idx[1] = @constraint(model, [n=1:N], F' * (u[n] - u[N+n]) >= tr(G * A[1][1][n] )) # A1
    cons_1idx[2] = @constraint(model, [n=1:N], F' * (u[N+n] - u[n]) >= tr(G* A[1][2][n] )) # A2
    cons_1idx[3] = @constraint(model, [n=1:N], L^2 >= tr(G * A[1][3][n])) # A3
    cons_1idx[4] = @constraint(model, [n=1:N], L^2 >= tr(G * A[1][4][n])) # A4
    cons_1idx[5] = @constraint(model, [n=1:N], 0 >= tr(G * A[1][5][n])) # A6 
    cons_1idx[6] = @constraint(model, [n=1:N], 0 >= tr(G * A[1][6][n])) # A7 
    cons_1idx[7] = @constraint(model, [n=1:N], D^2 >= tr(G * A[1][7][n])) # A9
    ################################################################################

    cons_2idc = Vector{Any}(undef, 2) # A5 and A8
    cons_2idc[1] = @constraint(model, [i=1:N, j=1:N], 0 >= tr(G * A[2][1][i,j])) # A5
    cons_2idc[2] = @constraint(model, [i=1:N, j=1:N], D^2 >= tr(G * A[2][2][i,j])) # A8

    @constraint(model, initial_cond, tr(A_R * G) <= R^2) # useless for OGD
    #########################################################################
    # Objective function 
    @objective(model, Max, tr(G * C) + b' * F)

    #### Solve the PEP
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.SLOW_PROGRESS
        # Extract the solution and dual variables
        @warn("The PEP did not solve to optimality. Status: $status")
    end
    return sol_PEP_generic(objective_value(model), value.(G), value.(F), [[dual.(cons_1idx[i]) for i in eachindex(cons_1idx)], [dual.(cons_2idc[i]) for i in eachindex(cons_2idc)]], dual(initial_cond))
end


# Solve the dual SDP formulation and return dual multipliers and objective value.
function sdp_pep_dual_generic(pep_problem::pep_problem_generic) # ok
    # Unpacking the PEP problem
    A = pep_problem.A
    A_R = pep_problem.A_R
    R = pep_problem.R
    C = pep_problem.C
    b = pep_problem.b
    u = pep_problem.u
    N = pep_problem.N
    L = pep_problem.L
    D = pep_problem.D
    # Creating the PEP model
    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_LOG", 0) # Suppress Mosek output
    set_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", 1) # Prevent over-subscription when called from Julia @threads
    # Variables
    @variable(model,τ1[[3,4,6,7,9],1:N] >= 0)
    @variable(model, τ2[[5,8],1:N,1:N] >= 0)
    @variable(model, λ[[1,2],1:N] >=0)
    @variable(model, S[1:size(A_R, 1), 1:size(A_R, 1)], PSD)
    ########################### Parsing constraints #########################
    Φ1 = Dict(3=>3, 4=>4, 6=>5, 7=>6, 9=>7) # since τ1 is only defined for these indices
    Φ2 = Dict(5=>1, 8=>2) # since τ2 is only defined for these indices
    S1 = @expression(model, sum(τ1[k,n] * pep_problem.A[1][Φ1[k]][n] for k in [3,4,6,7,9] for n in 1:N))
    S2 = @expression(model, sum(τ2[k,i,j] * pep_problem.A[2][Φ2[k]][i,j] for k in [5,8] for i in 1:N for j in 1:N))
    S3 = @expression(model, sum(λ[k,n] * pep_problem.A[1][k][n] for k in [1,2] for n in 1:N))
    @constraint(model, S .== S1 + S2 + S3)    
    @constraint(model, b .== sum((λ[1,n] - λ[2,n]) * (u[N+n] - u[n]) for n in 1:N))
    #########################################################################
    # Objective function
    @objective(model, Min, sum(L^2 * (τ1[3,n] + τ1[4,n]) + D^2 * τ1[9,n] for n in 1:N) + D^2 * sum(τ2[5,i,j] + τ2[8,i,j] for i in 1:N for j in 1:N))

    #### Solve the PEP
    optimize!(model)
    status = termination_status(model)
    if status == MOI.OPTIMAL
        a= [[value(λ[k,n]) for n in 1:N] for k in [1,2]]
        b= [[value(τ1[k,n]) for n in 1:N] for k in [3,4,6,7,9]]
        c= [[value(τ2[k,i,j]) for i in 1:N, j in 1:N] for k in [5,8]]
        return [[a ; b], c], objective_value(model)
    else
        error("The PEP did not solve to optimality. Status: $status")
    end
end
