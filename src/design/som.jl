# Second-order design methods: damped Newton on the frozen-certificate
# curvature, with ZG non-monotone Armijo or Strong Wolfe line search.
# (Ports of the legacy _run_damped_newton_{armijo,wolfe}, oracle-generic.)

# LM-regularized Newton direction (N&W §3.4); steepest-descent fallback.
function _newton_dir(H::Matrix{Float64}, g::Vector{Float64})
    λ = 0.0
    for _ in 1:40
        try
            F = cholesky(Symmetric(H .+ λ .* LinearAlgebra.I))
            d = F \ (-g)
            dot(g, d) < 0 && return d
        catch err
            err isa InterruptException && rethrow()
        end
        λ = λ == 0.0 ? 1e-6 * max(1.0, maximum(abs, diag(H))) : 10λ
    end
    return -g
end

"""
    design_som(dp, ω0; iters, linesearch=:ArmijoZG, verbose=false,
               sigma_met=-1.0, theta_met=2.0, M_met=10) :: DesignTrace

Damped Newton on the frozen-certificate curvature H_cert.
`linesearch = :ArmijoZG` (Aminifard–Grapiglia non-monotone; `sigma_met=0`
recovers plain Armijo) or `:Wolfe` (Strong Wolfe bracketing + zoom).
"""
function design_som(dp::AbstractDesignObjective, ω0::AbstractVector{<:Real};
                    iters::Int, linesearch::Symbol = :ArmijoZG,
                    verbose::Bool = false,
                    sigma_met::Float64 = -1.0, theta_met::Float64 = 2.0,
                    M_met::Int = 10)
    κ = length(ω0)
    ev = PEPEvaluator(dp, κ; hess = true)
    c1 = 1e-4

    ω = collect(Float64, ω0)
    ω_hist = [copy(ω)]
    refresh!(ev, ω)
    values = [ev.f]
    solves_hist = [ev.nsolves]

    σ_eff = sigma_met < 0.0 ? abs(ev.f) : sigma_met
    f_window = fill(ev.f, max(1, M_met))

    φ(ωk, d, α) = (refresh!(ev, ωk .+ α .* d); ev.f)
    dφ(ωk, d, α) = (refresh!(ev, ωk .+ α .* d); dot(ev.g, d))

    # ZG-relaxed Armijo backtracking (Aminifard & Grapiglia 2025, Alg. 1).
    function armijo_zg(ωk, d, fk, dφ0, f_lk, k)
        α = 1.0
        for _ in 1:60
            fα = φ(ωk, d, α)
            νk = 0.0
            if σ_eff > 0.0
                ratio = (f_lk - fα) / (c1 * α * dφ0)
                νk = σ_eff * exp(-max(theta_met, ratio) * log(k + 1))
            end
            fα <= fk + c1 * α * dφ0 + νk && return α
            α *= 0.5
            α < 1e-14 * (1 + norm(ωk)) && break
        end
        return α
    end

    # Strong Wolfe: N&W Algorithms 3.5 (bracketing) and 3.6 (zoom).
    c2 = 0.9
    function wolfe_zoom(ωk, d, f0, dφ0, α_lo, α_hi, f_lo)
        for _ in 1:60
            αj = (α_lo + α_hi) / 2
            fj = φ(ωk, d, αj)
            if fj > f0 + c1 * αj * dφ0 || fj >= f_lo
                α_hi = αj
            else
                dj = dφ(ωk, d, αj)
                abs(dj) <= -c2 * dφ0 && return αj
                dj * (α_hi - α_lo) >= 0 && (α_hi = α_lo)
                α_lo = αj
                f_lo = fj
            end
            abs(α_hi - α_lo) < 1e-14 && break
        end
        return (α_lo + α_hi) / 2
    end
    function wolfe_ls(ωk, d, f0, dφ0)
        α_prev, f_prev, α = 0.0, f0, 1.0
        for i in 1:60
            fi = φ(ωk, d, α)
            if fi > f0 + c1 * α * dφ0 || (i > 1 && fi >= f_prev)
                return wolfe_zoom(ωk, d, f0, dφ0, α_prev, α, f_prev)
            end
            di = dφ(ωk, d, α)
            abs(di) <= -c2 * dφ0 && return α
            di >= 0 && return wolfe_zoom(ωk, d, f0, dφ0, α, α_prev, fi)
            α_prev, f_prev = α, fi
            α = min(2α, 1.0)   # never expand beyond the full Newton step
        end
        return α
    end

    for k in 1:iters
        refresh!(ev, ω)
        fk = ev.f
        g = copy(ev.g)
        H = copy(ev.H)
        d = _newton_dir(H, g)
        dφ0 = dot(g, d)
        if abs(dφ0) < 1e-12 * (1 + abs(fk))
            verbose && @info "design_som: stationary" iter = k - 1
            break
        end
        α = linesearch == :ArmijoZG ?
            armijo_zg(ω, d, fk, dφ0, maximum(f_window), k) :
            linesearch == :Wolfe ? wolfe_ls(ω, d, fk, dφ0) :
            error("unknown linesearch :$linesearch (use :ArmijoZG or :Wolfe)")
        ω = ω .+ α .* d
        refresh!(ev, ω)
        f_window[mod1(k, length(f_window))] = ev.f
        push!(ω_hist, copy(ω))
        push!(values, ev.f)
        push!(solves_hist, ev.nsolves)
        verbose && @info "SOM iter $k" W = ev.f step = α
    end
    return DesignTrace(ω_hist, values, solves_hist, ev.nsolves)
end
