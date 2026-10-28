# IGDM: does a larger memory K improve anytime behavior?
# Prior: stationary-by-lag K (K+1 lag weights), min_grad, epsilon = 0.3.
# The families are nested (K+1 contains K with a zero last weight).
#
# Stage 1 (tailored): W^K_N = best stationary-K value at each horizon N,
#   by SLM chained over K (each K warm-started from the previous K's
#   optimum padded with zeros), at every N in 2..Nd and at the extra
#   horizons (default 40, 48).
# Stage 2 (designs): for each K, one weight vector is designed for all N in
#   the dense grid 2..Nd by minimizing the worst ratio
#       max_{N=2..Nd} W_N(alpha^K(omega)) / What_N
#   with two normalizers:
#     common - What_N = W^ref_N = min over all K of W^K_N (absolute quality);
#     within - What_N = W^K_N (price of not knowing N inside the prior).
#   The minimax is solved by a trust-region sequential LP on the epigraph
#   (certified PEP values and envelope gradients at every grid horizon).
#   Common designs are chained over K, so their worst ratio cannot increase
#   with K.
# Stage 3: every design is evaluated at the extra horizons.
#
# julia --project=. experiments\exp11_memory_anytime.jl --smoke
# julia --project=. experiments\exp11_memory_anytime.jl --workers 3 --seed <exp10 shape.toml>
# julia --project=. experiments\exp11_memory_anytime.jl --out <dir> --workers 3   (resume)
# Plot with experiments\plot_memory_anytime.jl <dir>\memory.toml.
#
# Memory: stages at N > Nd use at most --big-workers workers (default 2);
# tailored SLM at N = 48 needs about 7-9 GB per worker.

using Distributed
using TOML
using Printf
using Statistics

const EXP9 = joinpath(@__DIR__, "exp9_anytime.jl")
const PROJECT = dirname(@__DIR__)

function parse_options(args)
    opts = Dict{String,Any}("workers" => 3, "out" => "", "seed" => "", "smoke" => false,
                            "ks" => [0, 1, 2, 3, 4, 6, 8, 12], "nd" => 32,
                            "extra" => [40, 48], "big_workers" => 2)
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--smoke"
            opts["smoke"] = true
            i += 1
            continue
        end
        i < length(args) || throw(ArgumentError("missing value for $flag"))
        val = args[i + 1]
        if flag == "--workers"
            opts["workers"] = parse(Int, val)
        elseif flag == "--big-workers"
            opts["big_workers"] = parse(Int, val)
        elseif flag == "--out"
            opts["out"] = abspath(val)
        elseif flag == "--seed"
            opts["seed"] = abspath(val)
        elseif flag == "--ks"
            opts["ks"] = parse.(Int, split(val, ","))
        elseif flag == "--nd"
            opts["nd"] = parse(Int, val)
        elseif flag == "--extra"
            opts["extra"] = isempty(val) ? Int[] : parse.(Int, split(val, ","))
        else
            throw(ArgumentError("unknown option $flag"))
        end
        i += 2
    end
    opts["workers"] >= 1 || throw(ArgumentError("--workers must be positive"))
    issorted(opts["ks"]) && first(opts["ks"]) >= 0 ||
        throw(ArgumentError("--ks must be increasing and nonnegative"))
    if opts["smoke"]
        opts["ks"], opts["nd"], opts["extra"] = [0, 1, 2], 6, [8]
    end
    return opts
end

const RUN_AS_SCRIPT = abspath(PROGRAM_FILE) == (@__FILE__)
const OPTS = RUN_AS_SCRIPT ? parse_options(ARGS) : nothing
RUN_AS_SCRIPT && addprocs(OPTS["workers"]; exeflags = "--project=$(PROJECT)")

@everywhere include($EXP9)

@everywhere begin
    # Compiled PEPs do not depend on K. Models for N <= 32 are kept; at most
    # one larger model is held per process.
    const _CP = Dict{Int,Any}()
    function _cp(N)
        if !haskey(_CP, N)
            if N > 32
                foreach(k -> k > 32 && delete!(_CP, k), collect(keys(_CP)))
                GC.gc()
            end
            _CP[N] = compile_igdm(N, 1.0, 0.3; objective = :min_grad,
                                  encoding = :substitution)
        end
        return _CP[N]
    end
    release_models() = (empty!(_CP); GC.gc(); nothing)
    _dp(K, N) = DesignProblem(_cp(N), IGDMStationaryPolicy(K), N)

    function eval_task(K, ω, N)
        BLAS.set_num_threads(1)
        t = time()
        W = certified_value(_dp(K, N), ω)
        return (; W, seconds = time() - t)
    end

    function value_grad(K, ω, N)
        BLAS.set_num_threads(1)
        t = time()
        bad = (; W = Inf, g = fill(NaN, length(ω)), seconds = 0.0)
        try
            dp = _dp(K, N)
            W, sol, η = pep_value(dp, ω)
            sol.diagnostics.certified && isfinite(W) || return bad
            g = pep_grad_hess(dp, ω, η, sol; hess = false)[1]
            return (; W, g = collect(Float64, g), seconds = time() - t)
        catch err
            err isa InterruptException && rethrow()
            return bad
        end
    end

    # Tailored SLM chained over K at one horizon. Lags >= N are inactive, so
    # once the previous K reaches N - 1 the value is copied.
    function tailor_chain(N, Ks, seeds, iters, max_time)
        BLAS.set_num_threads(1)
        out = Dict{Int,Any}()
        prevK, prevω = -1, Float64[]
        for K in Ks
            t = time()
            if prevK >= N - 1
                ω = vcat(prevω, zeros(K - prevK))
                out[K] = (; W = out[prevK].W, omega = ω, solves = 0, seconds = 0.0)
                prevK, prevω = K, ω
                continue
            end
            starts = Tuple{Vector{Float64},Float64}[]
            if prevK < 0
                push!(starts, ([1.24; zeros(K)], 0.1))
                N <= 16 && push!(starts, ([1.0; zeros(K)], 1.0))
            else
                ω0 = vcat(prevω, zeros(K - prevK))
                push!(starts, (ω0, 0.1))
                N <= 16 && push!(starts, (ω0, 1.0))
            end
            haskey(seeds, K) && push!(starts, (seeds[K], 0.1))
            dp = _dp(K, N)
            bω, bW, solves = starts[1][1], Inf, 0
            for (ω0, Δ0) in starts
                tr = design_slm(dp, ω0; iters, Δ0, max_time, require_certified = true)
                ω, W = best_point(tr)
                solves += last(tr.solves_hist) + last(tr.tangent_hist)
                W < bW && ((bω, bW) = (ω, W))
            end
            out[K] = (; W = bW, omega = bω, solves, seconds = time() - t)
            prevK, prevω = K, bω
        end
        N > 32 && (delete!(_CP, N); GC.gc())
        return out
    end
end

using JuMP

function persist(path, data)
    open(path * ".tmp", "w") do io
        TOML.print(io, data)
    end
    mv(path * ".tmp", path; force = true)
end

# Dynamic task queue over a worker pool; results are returned in task order.
function pool_map(f, argsets, pool)
    out = Vector{Any}(undef, length(argsets))
    order = sortperm([a[end] for a in argsets]; rev = true)   # largest N first
    next = Ref(0)
    @sync for p in pool
        @async while true
            next[] += 1
            next[] > length(order) && break
            i = order[next[]]
            out[i] = remotecall_fetch(f, p, argsets[i]...)
        end
    end
    return out
end

# Trust-region sequential LP for min_ω max_i W_i(ω)/What_i. Each iteration
# solves min t s.t. r_i + (g_i/What_i)·d <= t, |d|_inf <= Δ, then accepts the
# trial if the actual decrease is a fraction of the predicted one.
function minimax_slp(K, ω0, Ns, What; iters = 60, Δ0 = 0.05, Δmin = 1e-7,
                     ptol = 1e-9, label = "")
    κ = K + 1
    # Explicit locals: assigning W or G here would rebind the outer variables.
    function evalH(x)
        local rs = pool_map(value_grad, [(K, x, N) for N in Ns], workers())
        return [q.W for q in rs], reduce(vcat, [permutedims(q.g) for q in rs])
    end
    t0 = time()
    ω = collect(Float64, ω0)
    W, G = evalH(ω)
    r = W ./ What
    f = maximum(r)
    isfinite(f) || error("minimax_slp: start is not certified at every horizon")
    Δ, solves, hist = Δ0, length(Ns), [f]
    it = 0
    for outer it in 1:iters
        m = PEPDesign._make_model(MosekBackend())
        @variable(m, -Δ <= d[1:κ] <= Δ)
        @variable(m, t)
        @variable(m, s[1:κ] >= 0)
        @constraint(m, s .>= d)
        @constraint(m, s .>= -d)
        for i in eachindex(Ns)
            @constraint(m, r[i] + sum(G[i, j] / What[i] * d[j] for j in 1:κ) <= t)
        end
        # Tiny l1 term: directions that do not change the max stay at zero.
        @objective(m, Min, t + 1e-6 * sum(s))
        optimize!(m)
        if !(termination_status(m) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) || !has_values(m)
            Δ /= 2
            Δ < Δmin && break
            continue
        end
        step = value.(d)
        pred = f - value(t)
        pred <= ptol * f && break
        Wt, Gt = evalH(ω .+ step)
        solves += length(Ns)
        rt = Wt ./ What
        ft = all(isfinite, rt) ? maximum(rt) : Inf
        ρ = (f - ft) / pred
        if ρ > 0.1
            ω, W, G, r, f = ω .+ step, Wt, Gt, rt, ft
            ρ > 0.75 && maximum(abs, step) > 0.99Δ && (Δ *= 2)
            ρ < 0.25 && (Δ /= 2)
        else
            Δ /= 2
        end
        push!(hist, f)
        Δ < Δmin && break
    end
    maximum(W ./ What) ≈ f || error("minimax_slp: stored values do not match f")
    @printf("  [slp %s] K=%d f=%.6f its=%d PEPs=%d Δ=%.1e (%.0fs)\n", label, K, f, it,
            solves, Δ, time() - t0)
    flush(stdout)
    return (; omega = ω, f, W, iterations = it, pep_solves = solves,
            seconds = time() - t0, history = hist)
end

pad(ω, K) = vcat(ω, zeros(K + 1 - length(ω)))

function main(opts)
    smoke = opts["smoke"]
    Ks = opts["ks"]
    Nd = opts["nd"]
    dense = collect(2:Nd)
    extra = opts["extra"]
    Nall = [dense; extra]
    out = opts["out"]
    if isempty(out)
        out = joinpath(PROJECT, "experiments", "results",
                       smoke ? "igdm_memory_smoke" : "igdm_memory_anytime")
    end
    mkpath(out)
    path = joinpath(out, "memory.toml")
    data = isfile(path) ? TOML.parsefile(path) :
        Dict{String,Any}("config" => Dict("problem" => "IGDM min_grad", "L" => 1.0,
            "D" => 1.0, "epsilon" => 0.3, "encoding" => "substitution",
            "policy" => "stationary-by-lag K", "Ks" => Ks, "dense" => dense,
            "extra" => extra),
            "tailored" => Dict{String,Any}(), "designs" => Dict{String,Any}())
    data["config"]["Ks"] == Ks && data["config"]["dense"] == dense ||
        error("existing study uses another K list or grid; choose a new --out")
    lk = ReentrantLock()
    save() = lock(() -> persist(path, data), lk)
    println("Study folder: ", out)
    t0 = time()
    big_pool = workers()[1:min(opts["big_workers"], nworkers())]

    seeds = Dict{Int,Dict{Int,Vector{Float64}}}()
    if !isempty(opts["seed"]) && 2 in Ks
        for (k, v) in TOML.parsefile(opts["seed"])["tailored"]
            seeds[parse(Int, k)] = Dict(2 => Float64.(v["omega"]))
        end
    end
    have(N) = haskey(data["tailored"], string(N)) &&
              all(K -> haskey(data["tailored"][string(N)], string(K)), Ks)
    function run_tailored(Ns, pool, max_time)
        local tasks = [(N, Ks, get(seeds, N, Dict{Int,Vector{Float64}}()),
                        smoke ? 10 : 60, max_time) for N in Ns if !have(N)]
        isempty(tasks) && return
        local next = Ref(0)
        local order = sortperm([t[1] for t in tasks]; rev = true)
        @sync for p in pool
            @async while true
                next[] += 1
                next[] > length(order) && break
                local task = tasks[order[next[]]]
                local Nt = task[1]
                local res = remotecall_fetch(tailor_chain, p, task...)
                lock(lk) do
                    data["tailored"][string(Nt)] = Dict(string(K) => Dict("W" => v.W,
                        "omega" => v.omega, "solves" => v.solves, "seconds" => v.seconds)
                        for (K, v) in res)
                end
                save()
                @printf("[tailored %6.0fs] N=%2d %s\n", time() - t0, Nt,
                        join([@sprintf("K%d=%.6g", K, res[K].W) for K in Ks], " "))
                flush(stdout)
            end
        end
    end

    # Stage 1: tailored values on the dense grid.
    run_tailored(dense, workers(), smoke ? 60.0 : 240.0)
    WK(N, K) = data["tailored"][string(N)][string(K)]["W"]
    ωK(N, K) = Float64.(data["tailored"][string(N)][string(K)]["omega"])
    Wref(N) = minimum(WK(N, K) for K in Ks)

    # Stage 2: designs on the dense grid.
    mid = dense[argmin(abs.(log.(dense) .- log(sqrt(first(dense) * last(dense)))))]
    dk(obj, K) = "$(obj)/K$(K)"
    for obj in ("common", "within"), (j, K) in enumerate(Ks)
        haskey(data["designs"], dk(obj, K)) && continue
        What = obj == "common" ? [Wref(N) for N in dense] : [WK(N, K) for N in dense]
        starts = [ωK(mid, K)]
        j > 1 && haskey(data["designs"], dk(obj, Ks[j - 1])) &&
            pushfirst!(starts, pad(Float64.(data["designs"][dk(obj, Ks[j - 1])]["omega"]), K))
        obj == "within" && haskey(data["designs"], dk("common", K)) &&
            pushfirst!(starts, Float64.(data["designs"][dk("common", K)]["omega"]))
        best = nothing
        solves, secs = 0, 0.0
        for (s, ω0) in enumerate(starts)
            res = minimax_slp(K, ω0, dense, What; iters = smoke ? 8 : 60,
                              label = "$(dk(obj, K)) start $s")
            solves += res.pep_solves
            secs += res.seconds
            (best === nothing || res.f < best.f) && (best = res)
        end
        data["designs"][dk(obj, K)] = Dict{String,Any}("K" => K, "objective" => obj,
            "omega" => best.omega, "f" => best.f, "starts" => length(starts),
            "pep_solves" => solves, "seconds" => secs, "iterations" => best.iterations,
            "history" => best.history,
            "values" => Dict{String,Any}(string(N) => w for (N, w) in zip(dense, best.W)))
        save()
        @printf("[design %6.0fs] %-10s f=%.6f omega=%s\n", time() - t0, dk(obj, K), best.f,
                string(round.(best.omega; digits = 4)))
        flush(stdout)
    end

    # Stage 3: tailored values and design evaluations at the extra horizons.
    if !isempty(extra)
        for p in workers()
            remotecall_fetch(release_models, p)
        end
        run_tailored(extra, big_pool, smoke ? 60.0 : 300.0)
        etasks = [(d, N) for d in values(data["designs"]) for N in extra
                  if !haskey(d["values"], string(N))]
        eres = pool_map(eval_task, [(d["K"], Float64.(d["omega"]), N) for (d, N) in etasks],
                        big_pool)
        for ((d, N), r) in zip(etasks, eres)
            d["values"][string(N)] = r.W
        end
        save()
    end
    write_outputs(out, data, Ks, Nall)
end

function write_outputs(out, data, Ks, Nall)
    dense = Int.(data["config"]["dense"])
    WK(N, K) = data["tailored"][string(N)][string(K)]["W"]
    # Final reference: best value found over all K, tailored or designed.
    Wref = Dict(N => minimum([[WK(N, K) for K in Ks];
                              [get(d["values"], string(N), Inf) for d in values(data["designs"])]])
                for N in Nall)
    improved = count(N -> Wref[N] < minimum(WK(N, K) for K in Ks) * (1 - 1e-7), Nall)
    data["Wref"] = Dict(string(N) => Wref[N] for N in Nall)
    persist(joinpath(out, "memory.toml"), data)

    open(joinpath(out, "memory_tailored.csv"), "w") do io
        println(io, "N,K,W,ratio_to_ref,omega")
        for N in Nall, K in Ks
            t = data["tailored"][string(N)][string(K)]
            println(io, "$N,$K,$(t["W"]),$(t["W"] / Wref[N]),\"",
                    join(t["omega"], ";"), "\"")
        end
    end
    keys_ = [k for obj in ("common", "within") for K in Ks
             for k in ["$(obj)/K$(K)"] if haskey(data["designs"], k)]
    open(joinpath(out, "memory_ratios.csv"), "w") do io
        println(io, "design,K,objective,N,W,ratio_common,ratio_within")
        for k in keys_, N in Nall
            d = data["designs"][k]
            haskey(d["values"], string(N)) || continue
            W = d["values"][string(N)]
            println(io, "$k,$(d["K"]),$(d["objective"]),$N,$W,$(W / Wref[N]),",
                    W / WK(N, d["K"]))
        end
    end
    rows = []
    for k in keys_
        d = data["designs"][k]
        K = d["K"]
        rc = Dict(N => d["values"][string(N)] / Wref[N] for N in Nall
                  if haskey(d["values"], string(N)))
        rw = Dict(N => d["values"][string(N)] / WK(N, K) for N in Nall
                  if haskey(d["values"], string(N)))
        wc, ac = findmax(N -> rc[N], dense)
        ww, aw = findmax(N -> rw[N], dense)
        tail_ = [get(rc, N, NaN) for N in Nall if !(N in dense)]
        push!(rows, (; k, K, obj = d["objective"], wc, Nc = dense[ac], ww, Nw = dense[aw],
                     mc = mean(rc[N] for N in dense), tail_, solves = d["pep_solves"],
                     secs = d["seconds"], omega = d["omega"]))
    end
    extra = [N for N in Nall if !(N in dense)]
    open(joinpath(out, "memory_summary.csv"), "w") do io
        println(io, "design,K,objective,worst_common_dense,argN_common,worst_within_dense,",
                "argN_within,mean_common_dense,", join(["common_at_$N" for N in extra], ","),
                isempty(extra) ? "" : ",", "pep_solves,seconds,omega")
        for r in rows
            println(io, "$(r.k),$(r.K),$(r.obj),$(r.wc),$(r.Nc),$(r.ww),$(r.Nw),$(r.mc),",
                    join(r.tail_, ","), isempty(extra) ? "" : ",",
                    "$(r.solves),$(r.secs),\"", join(r.omega, ";"), "\"")
        end
    end

    println("\nReference improved by a design at $improved of $(length(Nall)) horizons")
    println("\nTailored W^K_N / W^ref_N")
    @printf("%4s", "N")
    foreach(K -> @printf(" %7s", "K=$K"), Ks)
    println()
    for N in Nall
        (N in (2, 3, 4, 6, 8, 12, 16, 24, 32) || !(N in dense)) || continue
        @printf("%4d", N)
        foreach(K -> @printf(" %7.4f", WK(N, K) / Wref[N]), Ks)
        println()
    end
    println("\nDesigns (ratios on 2..$(last(dense)); common = vs best over K, within = vs W^K_N)")
    @printf("%-10s %9s %4s %9s %4s %9s %s %7s\n", "design", "worstC", "N", "worstW", "N",
            "meanC", join([@sprintf("%8s", "C@$N") for N in extra]), "PEPs")
    for r in rows
        @printf("%-10s %9.4f %4d %9.4f %4d %9.4f %s %7d\n", r.k, r.wc, r.Nc, r.ww, r.Nw, r.mc,
                join([@sprintf("%8.4f", x) for x in r.tail_]), r.solves)
    end
    println("Saved: ", out)
end

RUN_AS_SCRIPT && main(OPTS)
