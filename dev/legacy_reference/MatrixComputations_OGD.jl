# Matrix computations for gradients, Hessians, and PEP matrices (OGD)

# Compute gradient/Hessian of the SDP objective with respect to α.
function diff_w(α, pep_sol::sol_PEP_generic, pep_prob::pep_problem_generic)
    # Computing the (sub)gradient of w_sdp 
    d = size(pep_sol.G,1)
    acc = [zeros(d,d) for _ in eachindex(α)] # we define an accumulator for each α_n we derive w.r.t.
    grad_w = zeros(length(α))
    hess_w = zeros(length(α), length(α))
    N = pep_prob.N
    for n in eachindex(α)
        ################################ First-order information ################################
        # 1. Index constraints matrices
        K = size(pep_sol.duals_cons[1],1)
        for k = 1:K
            for i = 1:N
            acc[n] .+= pep_sol.duals_cons[1][k][i] .* pep_prob.∇A[1][k][i,n]
            end
        end
        # 2. Double indices constraints matrices
        K = size(pep_sol.duals_cons[2],1) 
        for k = 1:K
            for i = 1:N
                for j = 1:N
                    acc[n] .+= pep_sol.duals_cons[2][k][i,j] .* pep_prob.∇A[2][k][i,j,n] 
                end
            end
        end
        grad_w[n] = -tr(pep_sol.G * acc[n])
        ############################### Second-order information ###############################
        for p in eachindex(α)
            hess_contribution = zeros(d,d)
            for i in 1:N
                for k = 1:length(pep_sol.duals_cons[1])
                    # ∂^2 A_{i} / ∂ α_{n} ∂ α_{p} = pep_prob.∇2A[i][n,p]
                    hess_contribution .+= pep_sol.duals_cons[1][k][i] .* pep_prob.∇2A[1][k][i,n,p]
                end
                for j in 1:N
                    for k = 1:length(pep_sol.duals_cons[2])
                        # ∂^2 A_{ij} / ∂ α_{n} ∂ α_{p} = pep_prob.∇2A[i,j][n,p]
                        hess_contribution .+= pep_sol.duals_cons[2][k][i,j] .* pep_prob.∇2A[2][k][i,j,n,p]
                    end
                end
            end
            hess_w[n,p] = -tr(pep_sol.G * hess_contribution)
        end
    end    
    return grad_w, hess_w
end

# Compute gradient/Hessian with respect to ω using chain rule and α(ω).
function diff_w_wrt_stepsizes_parameters(ω, compute_α_∇α_∇2α, pep_sol::sol_PEP_generic, pep_prob::pep_problem_generic)
    α, ∇α, ∇2α = compute_α_∇α_∇2α(ω, pep_prob.N)
    grad_w_α, hess_w_α = diff_w(α, pep_sol, pep_prob)

    κ = length(ω)
    # Computing the gradient of wsdp w.r.t. the parameters ω
    grad_w_ω = zeros(κ)
    for k in 1:κ
        for n in eachindex(α)
            grad_w_ω[k] += grad_w_α[n] * ∇α[n][k]
        end
    end

    # Computing the Hessian of wsdp w.r.t. the parameters ω
    hess_w_ω = zeros(κ, κ)
    for k in 1:κ
        for s in 1:κ
            for n in eachindex(α)
                hess_w_ω[k,s] += grad_w_α[n] * ∇2α[n][k,s]
                for m in eachindex(α)
                    hess_w_ω[k,s] += hess_w_α[n,m] * ∇α[n][k] * ∇α[m][s]
                end
            end
        end
    end
    return grad_w_ω, hess_w_ω
end

# Build A, ∇A and ∇2A for the current step-size vector α.
function compute_A_∇A_∇2A!(α, pep_prob::pep_problem_generic)
    # Getting data from pep_problem
    u = pep_prob.u
    N = pep_prob.N
    # Computing h_n
    h = [sparse([.- α[1:n-1] ; zeros(2N-n+1); [0];.- ones(n-1); zeros(N-n); [1]]) for n in 1:N]

    # Computing matrices
    A1::Vector{SparseMatrixCSC{Float64,Int64}} = [(u[N+n] * (h[n]') .+ h[n] * (u[N+n]'))/2 for n in 1:N]
    A2::Vector{SparseMatrixCSC{Float64,Int64}} = [.-(u[n]*(h[n]') .+ h[n]*(u[n]'))/2 for n=1:N]
    A3::Vector{SparseMatrixCSC{Float64,Int64}} = [u[n]*(u[n]') for n=1:N]
    A4::Vector{SparseMatrixCSC{Float64,Int64}} = [u[N+n]*(u[N+n]') for n=1:N]
    A5 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for i in 1:N
        for j in 1:N
            A5[i,j] = (u[2N+i] * (h[j]') .+ h[j] * (u[2N+i]'))/2 .- (u[2N+i] * (h[i]') .+ h[i] * (u[2N+i]'))/2
        end               
    end

    A6 = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N) 
    for n in 1:N
        A6[n] = .- (u[2N+n] * (h[n]') + h[n] * (u[2N+n]'))/2
    end
    
    A7 = Vector{SparseMatrixCSC{Float64,Int64}}(undef, N)
    for n in 1:N
        A7[n] = spzeros(3N+1,3N+1)
        for s in 1:N
            A7[n] .-= (u[N+s]*(h[n]') + h[n]*(u[N+s]'))/2
        end
    end 

    A8 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for i in 1:N
        for j in 1:N
            A8[i,j] = h[i]*(h[i]') .+ h[j]*(h[j]') .- h[i]*(h[j]') .- h[j]*(h[i]')
        end
    end

    A9::Vector{SparseMatrixCSC{Float64,Int64}} = [h[n]*(h[n]') for n=1:N]

    ###############################################################################################################
    ################################ First-order derivatives ∇A  ##################################################
    ############################################################################################################### 

    # Computing ∂ A_{ij} / ∂ α_{n}
    # sparse(I,J,V,m,n) creates an m-by-n sparse matrix with entries V[k] at positions (I[k],J[k])
    function ∂uh(i, j, n)
        if (n <= j-1)
            return sparse([i],[n],[-1], 3N+1,3N+1)
        else
            return spzeros(3N+1,3N+1)
        end
    end

    # Returns ∂h_i*h_j / ∂ α_{n}
    function ∂hh(i, j, n)
        I1_i, J1_i, V1_i = [], [], []
        I2_i, J2_i, V2_i = [], [], []
        I3_i, J3_i, V3_i = [], [], []
        if n <= i-1
            J1_i = 1:j-1
            I1_i = fill(n, length(J1_i))
            V1_i = [α[k] for k in J1_i]
            J2_i = 2n+2:2n+j-1
            I2_i = fill(n, length(J2_i))
            V2_i = ones(length(J2_i))
            I3_i = [n]
            J3_i = [3N+1]
            V3_i = [-1]
        end
        I1_j, J1_j, V1_j = [], [], []
        I2_j, J2_j, V2_j = [], [], []
        I3_j, J3_j, V3_j = [], [], []
        if n <= j-1
            I1_j = 1:i-1
            J1_j = fill(n, length(I1_j))
            V1_j = [α[l] for l in I1_j]
            I2_j = 2*N+2:2*N+i-1
            J2_j = fill(n, length(I2_j))
            V2_j = ones(length(I2_j))
            J3_j = [3N+1]
            I3_j = [n]
            V3_j = [-1]
        end

        I::Vector{Int64} = vcat(I1_i,I2_i,I3_i,I1_j,I2_j,I3_j)
        J::Vector{Int64} = vcat(J1_i,J2_i,J3_i,J1_j,J2_j,J3_j)
        V::Vector{Float64} = vcat(V1_i,V2_i,V3_i,V1_j,V2_j,V3_j)

        if !isempty(I) # guarantees that the matrix will be not empty
            return sparse(I,J,V,3N+1,3N+1)
        else
            return spzeros(3N+1,3N+1)
        end     
    end

    # Each matrix will be differentiated w.r.t. each α_n
    # A1[n,p] = ∂ A_{n}^1 / ∂ α_{p}
    ∇A1 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A1[n,p] = (∂uh(N+n, n, p) + (∂uh(N+n, n, p)'))/2
        end
    end
    ∇A2 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A2[n,p] = .- (∂uh(n, n, p) + (∂uh(n, n, p))')/2
        end
    end
    ∇A3 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A3[n,p] = spzeros(3N+1,3N+1)
        end
    end
    ∇A4 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A4[n,p] = spzeros(3N+1,3N+1) # since A4 does not depend on α
        end
    end
    # A5[i,j,p] = ∂ A_{ij}^5 / ∂ α_{p}
    ∇A5 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for i in 1:N
        for j in 1:N
            for p in 1:N # TO BE CHECKED
                ∇A5[i,j,p] = ∂uh(2N+i, j, p) + (∂uh(2N+i, j, p)')/2 -  (∂uh(2N+i, i, p) + (∂uh(2N+i, i, p)'))/2# TO BE MODIFIED
            end
        end
    end

    ∇A6 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N # TO BE CHECKED
            ∇A6[n,p] =  -(∂uh(2N+n, n, p) + (∂uh(2N+n, n, p)'))/2
        end
    end

    ∇A7 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A7[n,p] = spzeros(3N+1,3N+1)
            for s in 1:N
                ∇A7[n,p] .-= (∂uh(N+s, n, p) .+ (∂uh(N+s, n, p)'))/2
            end
        end
    end
    ∇A8 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for i in 1:N
        for j in 1:N
            for p in 1:N
                ∇A8[i,j,p] = ∂hh(i, i, p) .+ ∂hh(j, j, p) .- ∂hh(i, j, p) .- ∂hh(j, i, p)
            end
        end
    end
    ∇A9 = Matrix{SparseMatrixCSC{Float64,Int64}}(undef, N, N)
    for n in 1:N
        for p in 1:N
            ∇A9[n,p] = ∂hh(n, n, p)
        end
    end
    ###############################################################################################################
    ################################ Second-order derivatives ∇2A  ################################################
    ############################################################################################################### 
    # Computing ∂^2 A_{ij} / ∂ α_{n} ∂ α_{p}
    function hess_hh(i, j, n, p) # TO BE REDEFINED
        a = spzeros(3N+1,3N+1)
        if n <= (i-1) && p <= (j-1)
            a .+= sparse([n], [p], [1], 3N+1,3N+1)
        end
        if p <= (i-1) && n <= (j-1)
            a .+= sparse([p], [n], [1], 3N+1,3N+1)
        end
        return a
    end

    function hess_αuh(i,j,n,p) # TO BE REDEFINED
        a = spzeros(3N+1,3N+1)
        if i == p 
            a .+= (∂uh(i,j,n) + ∂uh(i,j,n)')/2
        end
        if i == n
            a .+= (∂uh(i,j,p) + ∂uh(i,j,p)')/2
        end
        return a
    end

    HessA1 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    HessA2 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA1[i,n,p] = spzeros(3N+1,3N+1)
                HessA2[i,n,p] = spzeros(3N+1,3N+1)
            end
        end
    end

    HessA3 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA3[i,n,p] = spzeros(3N+1,3N+1)
            end
        end
    end

    HessA4 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA4[i,n,p] = spzeros(3N+1,3N+1) # since A4 does not depend on α
            end
        end
    end

    HessA5 = Array{SparseMatrixCSC{Float64,Int64},4}(undef, N, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                for j in 1:N
                    HessA5[i,j,n,p] = spzeros(3N+1,3N+1)
                end
            end
        end
    end

    HessA6 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA6[i,n,p] = spzeros(3N+1,3N+1)
            end
        end
    end

    HessA7 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA7[i,n,p] = spzeros(3N+1,3N+1)
                for s in 1:N
                    HessA7[i,n,p] .+= (∂uh(N+s, i, p) .+ (∂uh(N+s, i, p)'))/2
                end
            end
        end
    end

    HessA8 = Array{SparseMatrixCSC{Float64,Int64},4}(undef, N, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                for j in 1:N
                    HessA8[i,j,n,p] = hess_hh(i,i,n,p) .+ hess_hh(j,j,n,p) .- hess_hh(i,j,n,p) .- hess_hh(j,i,n,p)
                end
            end
        end
    end

    HessA9 = Array{SparseMatrixCSC{Float64,Int64},3}(undef, N, N, N)
    for n in 1:N
        for p in 1:N
            for i in 1:N
                HessA9[i,n,p] = hess_hh(i,i,n,p)
            end
        end
    end

    # Stacking all matrices
    A = ([A1, A2, A3, A4, A6, A7, A9], [A5, A8])
    ∇A = ([∇A1, ∇A2, ∇A3, ∇A4, ∇A6, ∇A7, ∇A9], [∇A5, ∇A8])
    ∇2A = ([HessA1, HessA2, HessA3, HessA4, HessA6, HessA7, HessA9], [HessA5, HessA8])

    # Updating the pep_problem
    pep_prob.A = A
    pep_prob.∇A = ∇A
    pep_prob.∇2A = ∇2A

    return pep_prob.A, pep_prob.∇A, pep_prob.∇2A
end
