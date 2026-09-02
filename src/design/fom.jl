# First-order design methods: normalized steepest descent and Adam.

"""
    design_fom(dp, ω0; iters, steps, method=:SD, verbose=false,
               adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8) :: DesignTrace

Minimize W(η(ω)) with a first-order method. `steps` is a step-size schedule:
a vector of length ≥ `iters` or a function `t -> Float64`.
`:SD` uses the normalized gradient; `:Adam` uses `steps` as learning rate.
"""
function design_fom(dp::AbstractDesignObjective, ω0::AbstractVector{<:Real};
                    iters::Int, steps, method::Symbol = :SD,
                    verbose::Bool = false,
                    adam_beta1::Float64 = 0.9, adam_beta2::Float64 = 0.999,
                    adam_eps::Float64 = 1e-8)
    κ = length(ω0)
    step_at = steps isa AbstractVector ? (t -> Float64(steps[t])) :
                                         (t -> Float64(steps(t)))
    ev = PEPEvaluator(dp, κ; hess = false)

    ω = collect(Float64, ω0)
    ω_hist = [copy(ω)]
    refresh!(ev, ω)
    values = [ev.f]
    solves_hist = [ev.nsolves]

    m = zeros(κ)   # Adam first moment
    v = zeros(κ)   # Adam second moment

    for t in 1:iters
        g = copy(ev.g)
        ng = norm(g)
        if method == :SD
            ng > 0 && (ω = ω .- step_at(t) .* g ./ ng)
        elseif method == :Adam
            m .= adam_beta1 .* m .+ (1 - adam_beta1) .* g
            v .= adam_beta2 .* v .+ (1 - adam_beta2) .* g .^ 2
            m̂ = m ./ (1 - adam_beta1^t)
            v̂ = v ./ (1 - adam_beta2^t)
            ω = ω .- step_at(t) .* m̂ ./ (sqrt.(v̂) .+ adam_eps)
        else
            error("unknown FOM method :$method (use :SD or :Adam)")
        end
        refresh!(ev, ω)
        push!(ω_hist, copy(ω))
        push!(values, ev.f)
        push!(solves_hist, ev.nsolves)
        verbose && @info "FOM iter $t" W = ev.f norm_g = ng
    end
    return DesignTrace(ω_hist, values, solves_hist, ev.nsolves)
end
