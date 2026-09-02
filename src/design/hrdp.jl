# Multi-horizon robust design (HRDP): normalized objective σ(𝓗, ω), transfer
# metrics, and the prefix-exact / tail-sampled estimator (paper Alg. 3).

"""
    HRDPObjective(compile_fn, policy, H, Wstar; backend=MosekBackend())

Accumulated scheme-wise normalized worst-case
`σ(𝓗, ω) = 1/|𝓗| Σ_{N∈𝓗} W(η^(N)(ω), N) / W*_N`.
`compile_fn(N) :: CompiledPEP`; `Wstar[N] > 0` are the horizon-wise optima.
Horizon evaluations run in parallel under `Threads.@threads` (Mosek is pinned
to 1 thread by the default backend).
"""
struct HRDPObjective{P<:AbstractPolicy,B<:SDPBackend} <: AbstractDesignObjective
    dps::Vector{DesignProblem{P,B}}
    H::Vector{Int}
    Wstar::Vector{Float64}
end

function HRDPObjective(compile_fn::Function, policy::AbstractPolicy, H,
                       Wstar::AbstractDict; backend::SDPBackend = MosekBackend())
    Hs = collect(Int, H)
    dps = [DesignProblem(compile_fn(N), policy, N, backend) for N in Hs]
    ws = [Float64(Wstar[N]) for N in Hs]
    all(>(0), ws) || error("all W*_N must be positive")
    return HRDPObjective(dps, Hs, ws)
end

"""σ, gradient, frozen-certificate curvature, and solve count at ω."""
function eval_all(ho::HRDPObjective, ω::AbstractVector{<:Real}; hess::Bool = true)
    n = length(ho.H)
    κ = length(ω)
    fs = zeros(n)
    gs = [zeros(κ) for _ in 1:n]
    Hs = [zeros(κ, κ) for _ in 1:n]
    Threads.@threads for i in 1:n
        f, g, Hm, _ = eval_all(ho.dps[i], ω; hess)
        fs[i] = f / ho.Wstar[i]
        gs[i] = g ./ ho.Wstar[i]
        hess && (Hs[i] = Hm ./ ho.Wstar[i])
    end
    f = sum(fs) / n
    g = sum(gs) ./ n
    H = hess ? sum(Hs) ./ n : zeros(0, 0)
    return f, g, H, n
end

"""
    snw(ho, ω) :: Dict{Int,Float64}

Per-horizon normalized values W_N(ω)/W*_N (|𝓗| SDP solves, threaded).
"""
function snw(ho::HRDPObjective, ω::AbstractVector{<:Real})
    vals = zeros(length(ho.H))
    Threads.@threads for i in eachindex(ho.H)
        W, _, _ = pep_value(ho.dps[i], ω)
        vals[i] = W / ho.Wstar[i]
    end
    return Dict(zip(ho.H, vals))
end

"""Worst Generalization Compromise: max_N W_N(ω)/W*_N."""
wgc(ho::HRDPObjective, ω::AbstractVector{<:Real}) = maximum(values(snw(ho, ω)))

"""
    generalization_ratio(ho_full, ω_sub, ω_full)

GR = σ(𝓗_full, ω_sub) / σ(𝓗_full, ω_full) ≥ 1 (up to tolerances).
"""
function generalization_ratio(ho_full::HRDPObjective, ω_sub, ω_full)
    σ_sub, _, _, _ = eval_all(ho_full, ω_sub; hess = false)
    σ_full, _, _, _ = eval_all(ho_full, ω_full; hess = false)
    return σ_sub / σ_full
end

"""
    SampledHRDP(ho, r, m; rng=Random.default_rng(), weights=nothing)

Prefix-exact, tail-sampled estimator of σ (paper Alg. 3): horizons N ≤ r are
evaluated exactly; the tail is estimated with `m` importance samples drawn
with probabilities ∝ `weights` (uniform by default). Estimates are unbiased;
use with `design_fom` (line-search-free methods).
"""
struct SampledHRDP{R<:Random.AbstractRNG} <: AbstractDesignObjective
    ho::HRDPObjective
    prefix::Vector{Int}      # indices into ho.H with H[i] ≤ r
    tail::Vector{Int}
    p::Vector{Float64}       # tail sampling distribution
    m::Int
    rng::R
end

function SampledHRDP(ho::HRDPObjective, r::Int, m::Int;
                     rng::Random.AbstractRNG = Random.default_rng(),
                     weights = nothing)
    prefix = findall(N -> N <= r, ho.H)
    tail = findall(N -> N > r, ho.H)
    isempty(tail) && error("empty tail: increase 𝓗 or decrease r")
    w = weights === nothing ? ones(length(tail)) :
        [Float64(weights[ho.H[i]]) for i in tail]
    return SampledHRDP(ho, prefix, tail, w ./ sum(w), m, rng)
end

function eval_all(sh::SampledHRDP, ω::AbstractVector{<:Real}; hess::Bool = true)
    ho = sh.ho
    κ = length(ω)
    n = length(ho.H)
    f = 0.0
    g = zeros(κ)
    Hm = hess ? zeros(κ, κ) : zeros(0, 0)
    nsolves = 0
    for i in sh.prefix
        fi, gi, Hi, _ = eval_all(ho.dps[i], ω; hess)
        f += fi / ho.Wstar[i]
        g .+= gi ./ ho.Wstar[i]
        hess && (Hm .+= Hi ./ ho.Wstar[i])
        nsolves += 1
    end
    ks = [sh.tail[findfirst(>=(u), cumsum(sh.p))] for u in rand(sh.rng, sh.m)]
    for (j, i) in enumerate(ks)
        w = 1.0 / (sh.m * sh.p[findfirst(==(i), sh.tail)])
        fi, gi, Hi, _ = eval_all(ho.dps[i], ω; hess)
        f += w * fi / ho.Wstar[i]
        g .+= w .* gi ./ ho.Wstar[i]
        hess && (Hm .+= w .* Hi ./ ho.Wstar[i])
        nsolves += 1
    end
    return f / n, g ./ n, hess ? Hm ./ n : Hm, nsolves
end

"""
    compute_wstar(compile_fn, H; policy=IdentityPolicy(), ω0_fn, iters=50,
                  backend=MosekBackend(), method=:both) :: Dict{Int,Float64}

Horizon-wise optimal values W*_N by per-horizon design (threaded over N).
`ω0_fn(N)` returns one start (vector) or several (vector of vectors);
`method ∈ (:som, :fom, :both)` — the best value over starts×methods is kept.
"""
function compute_wstar(compile_fn::Function, H; policy = IdentityPolicy(),
                       ω0_fn::Function, iters::Int = 50,
                       backend::SDPBackend = MosekBackend(), method::Symbol = :both)
    Hs = collect(Int, H)
    vals = zeros(length(Hs))
    Threads.@threads for i in eachindex(Hs)
        N = Hs[i]
        dp = DesignProblem(compile_fn(N), policy, N, backend)
        starts = ω0_fn(N)
        starts isa AbstractVector{<:AbstractVector} || (starts = [starts])
        best = Inf
        for ω0 in starts
            if method in (:som, :both)
                best = min(best, best_point(design_som(dp, ω0; iters))[2])
            end
            if method in (:fom, :both)
                best = min(best, best_point(design_fom(dp, ω0; iters,
                                steps = t -> 0.1 / sqrt(t)))[2])
            end
        end
        vals[i] = best
    end
    return Dict(zip(Hs, vals))
end
