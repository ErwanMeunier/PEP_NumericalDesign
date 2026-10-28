# IGDM: how the shape of the design horizon set H drives anytime behavior.
# Prior: stationary-by-lag K = 2 (3 parameters), min_grad, epsilon = 0.3.
# For every set H and objective (max or mean of W_N(ω)/W*_N over H), the
# design ω_H is optimized by the exp9 multistart Nelder–Mead on certified
# values, then evaluated at every N in 2..Nmax. W*_N is the tailored value in
# the same prior: per-N SLM, then the min with every evaluated design.
#
# Three factors are separated:
#   shape   - sets with the same extent [2, 32]: window, geometric, arithmetic,
#             fine geometric, three points, endpoints;
#   weights - mean vs max on the window, geometric and arithmetic sets;
#   extent  - geometric sets with different ends, windows {2..10}, {16..32},
#             and single-horizon reuse (H = {N}).
# Optimality cross-check: every design is scored on every set's criterion with
# the final W*. A design beaten on its own criterion is re-optimized from the
# better competitor ("polished"), at most twice.
# Exchange stage: from the geometric minimax design, repeatedly add the worst
# horizon of the dense grid {min H..max H} to H and re-solve, until the dense
# worst case equals the worst case on H (design "exchange_2_32/max").
#
# julia --project=. experiments\exp10_horizon_shape.jl --smoke
# julia --project=. experiments\exp10_horizon_shape.jl --workers 4 --seed <igdm anytime.toml>
# julia --project=. experiments\exp10_horizon_shape.jl --out <dir> --workers 2 --seed <igdm anytime.toml> --tailored-only
# julia --project=. experiments\exp10_horizon_shape.jl --out <dir> --workers 4   (resume)
# Plot with experiments\plot_horizon_shape.jl <dir>\shape.toml.
#
# Peak memory per worker: about 7 GB for tailored SLM at N = 48 (PEP and
# tangent SDP), 3.3 GB for a PEP evaluation at N = 48. Run the tailored stage
# with few workers (--tailored-only), then resume with more.

using Distributed
using TOML
using Printf
using Statistics

const EXP9 = joinpath(@__DIR__, "exp9_anytime.jl")
const PROJECT = dirname(@__DIR__)

function parse_options(args)
    opts = Dict{String,Any}("workers" => 4, "out" => "", "seed" => "", "smoke" => false,
                            "nmax" => 48, "tailored_only" => false)
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag in ("--smoke", "--tailored-only")
            opts[flag == "--smoke" ? "smoke" : "tailored_only"] = true
            i += 1
            continue
        end
        i < length(args) || throw(ArgumentError("missing value for $flag"))
        val = args[i + 1]
        if flag == "--workers"
            opts["workers"] = parse(Int, val)
        elseif flag == "--out"
            opts["out"] = abspath(val)
        elseif flag == "--seed"
            opts["seed"] = abspath(val)
        elseif flag == "--nmax"
            opts["nmax"] = parse(Int, val)
        else
            throw(ArgumentError("unknown option $flag"))
        end
        i += 2
    end
    opts["workers"] >= 1 || throw(ArgumentError("--workers must be positive"))
    opts["smoke"] && (opts["nmax"] = 12)
    return opts
end

const RUN_AS_SCRIPT = abspath(PROGRAM_FILE) == (@__FILE__)
const OPTS = RUN_AS_SCRIPT ? parse_options(ARGS) : nothing
RUN_AS_SCRIPT && addprocs(OPTS["workers"]; exeflags = "--project=$(PROJECT)")

@everywhere include($EXP9)

@everywhere begin
    const _S = setup("igdm", false)
    const _POL = _S.families["stationary_K2"][1]
    const _STARTS = _S.families["stationary_K2"][2]
    const _SCALE = _S.families["stationary_K2"][3]
    # Compiled PEPs for N <= 32 are small and kept; at most one larger model
    # is held per process.
    const _CP = Dict{Int,Any}()
    function _cp(N)
        if !haskey(_CP, N)
            if N > 32
                foreach(k -> k > 32 && delete!(_CP, k), collect(keys(_CP)))
                GC.gc()
            end
            _CP[N] = _S.compiler(N)
        end
        return _CP[N]
    end
    _dp(N) = DesignProblem(_cp(N), _POL, N)

    function eval_task(ω, N)
        BLAS.set_num_threads(1)
        t = time()
        W = certified_value(_dp(N), ω)
        return (; W, seconds = time() - t, worker = myid())
    end

    function tailor_task(N, starts)
        BLAS.set_num_threads(1)
        t = time()
        best_ω, best_W, solves = first(starts)[1], Inf, 0
        for (ω0, Δ0) in starts
            tr = design_slm(_dp(N), ω0; iters = 60, Δ0, max_time = 600.0,
                            require_certified = true)
            ω, W = best_point(tr)
            solves += last(tr.solves_hist) + last(tr.tangent_hist)
            W < best_W && ((best_ω, best_W) = (ω, W))
        end
        # One tailored task per N: release large models immediately.
        if N > 32
            delete!(_CP, N)
            GC.gc()
        end
        return (; omega = best_ω, W = best_W, solves, seconds = time() - t, worker = myid())
    end

    function design_task(H, WH, objective, starts, maxevals)
        BLAS.set_num_threads(1)
        dps = [_dp(N) for N in H]
        function σ(ω)
            r = [certified_value(d, ω) / w for (d, w) in zip(dps, WH)]
            all(isfinite, r) || return Inf
            return objective == "mean" ? mean(r) : maximum(r)
        end
        t = time()
        ω, f, e = minimize(σ, starts, _SCALE; maxevals)
        return (; omega = ω, f, evals = e, pep_solves = e * length(H),
                seconds = time() - t, worker = myid())
    end
end

function study_sets(smoke)
    if smoke
        return [("window_2_8", collect(2:8), ["max", "mean"]),
                ("geom_2_8", [2, 4, 8], ["max"]),
                ("ends_2_8", [2, 8], ["max"]),
                ("single_8", [8], ["reuse"])]
    end
    return [
        # shape at fixed extent [2, 32]
        ("window_2_32", collect(2:32), ["max", "mean"]),
        ("geom_2_32", [2, 4, 8, 16, 32], ["max", "mean"]),
        ("arith_2_32", [2, 9, 17, 24, 32], ["max", "mean"]),
        ("geomfine_2_32", [2, 3, 4, 6, 8, 11, 16, 23, 32], ["max"]),
        ("triple_2_32", [2, 8, 32], ["max"]),
        ("ends_2_32", [2, 32], ["max"]),
        # extent
        ("geom_2_8", [2, 4, 8], ["max"]),
        ("geom_2_16", [2, 4, 8, 16], ["max"]),
        ("geom_4_32", [4, 8, 16, 32], ["max"]),
        ("geom_8_32", [8, 16, 32], ["max"]),
        ("window_2_10", collect(2:10), ["max", "mean"]),
        ("window_16_32", collect(16:32), ["max"]),
        # single-horizon reuse
        ("single_10", [10], ["reuse"]),
        ("single_16", [16], ["reuse"]),
        ("single_32", [32], ["reuse"])]
end

# Piecewise-linear interpolation of seed tailored coefficients over N, with
# linear extrapolation from the last two seeds.
function interp_seed(seed, N)
    Ns = sort!(collect(keys(seed)))
    length(Ns) == 1 && return copy(seed[Ns[1]])
    N <= Ns[1] && return copy(seed[Ns[1]])
    k = searchsortedlast(Ns, N)
    k == length(Ns) && (k -= 1)
    a, b = Ns[k], Ns[k + 1]
    θ = (N - a) / (b - a)
    return (1 - θ) .* seed[a] .+ θ .* seed[b]
end

function persist(path, data)
    open(path * ".tmp", "w") do io
        TOML.print(io, data)
    end
    mv(path * ".tmp", path; force = true)
end

function run_pool!(tasks, f, record!)
    isempty(tasks) && return
    next = Ref(0)
    @sync for p in workers()
        @async while true
            next[] += 1
            i = next[]
            i > length(tasks) && break
            record!(tasks[i], remotecall_fetch(f, p, tasks[i].args...))
        end
    end
end

dkey(name, obj) = "$(name)/$(obj)"

function final_wstar(data, Ns)
    W = Dict{Int,Float64}()
    for N in Ns
        w = data["tailored"][string(N)]["W"]
        for d in data["designs"]
            v = get(d["values"], string(N), Inf)
            isfinite(v) && (w = min(w, v))
        end
        W[N] = w
    end
    return W
end

criterion(H, obj, d, W) = begin
    r = [d["values"][string(N)] / W[N] for N in H]
    obj == "mean" ? mean(r) : maximum(r)
end

function main(opts)
    smoke = opts["smoke"]
    nmax = opts["nmax"]
    Ns = collect(2:nmax)
    sets = study_sets(smoke)
    maxevals = smoke ? 20 : 150
    root = joinpath(@__DIR__, "results")
    out = opts["out"]
    if isempty(out)
        mkpath(root)
        out = mktempdir(root; prefix = smoke ? "shape_smoke_" : "igdm_horizon_shape_",
                        cleanup = false)
    end
    mkpath(out)
    path = joinpath(out, "shape.toml")
    data = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}(
        "config" => merge(_S.config, Dict{String,Any}(
            "family" => "stationary_K2", "eval_horizons" => Ns,
            "sets" => Dict(name => H for (name, H, _) in sets),
            "objectives" => Dict(name => objs for (name, _, objs) in sets),
            "optimizer" => "Nelder-Mead multistart + restart (exp9 minimize)",
            "maxevals_per_start" => maxevals,
            "tailored" => "SLM, 60 iterations, certified, from seed interpolation (and [1,0,0] for N <= 20)",
            "seed" => opts["seed"], "certified_values_only" => true,
            "julia_version" => string(VERSION), "BLAS_threads" => 1)),
        "tailored" => Dict{String,Any}(), "designs" => Dict{String,Any}[])
    lk = ReentrantLock()
    save() = lock(() -> persist(path, data), lk)
    println("Study folder: ", out)

    # Stage 1: tailored value in the prior at every N.
    seed = Dict{Int,Vector{Float64}}()
    if !isempty(opts["seed"])
        for (k, v) in TOML.parsefile(opts["seed"])["tailored"]
            seed[parse(Int, k)] = Float64.(v)
        end
    end
    tasks = []
    for N in Ns
        haskey(data["tailored"], string(N)) && continue
        starts = Tuple{Vector{Float64},Float64}[]
        isempty(seed) || push!(starts, (interp_seed(seed, N), 0.1))
        (isempty(seed) || N <= 20) && push!(starts, (Float64.(_STARTS[1]), 1.0))
        push!(tasks, (; N, args = (N, starts)))
    end
    sort!(tasks; by = t -> -t.N)
    t0 = time()
    run_pool!(tasks, tailor_task, (task, res) -> begin
        lock(lk) do
            data["tailored"][string(task.N)] = Dict("omega" => res.omega, "W" => res.W,
                "solves" => res.solves, "seconds" => res.seconds)
        end
        save()
        @printf("[tailored %6.0fs] N=%2d W=%.8f omega=%s (%.1fs, worker %d)\n", time() - t0,
                task.N, res.W, string(round.(res.omega; digits = 5)), res.seconds, res.worker)
        flush(stdout)
    end)
    tail(N) = Float64.(data["tailored"][string(N)]["omega"])
    if opts["tailored_only"]
        println("Tailored stage complete (--tailored-only)")
        return
    end
    Wt = Dict(N => data["tailored"][string(N)]["W"] for N in Ns)
    all(isfinite, values(Wt)) || error("a tailored value is not certified")

    # Stage 2: designs.
    have = Set(dkey(d["set"], d["objective"]) for d in data["designs"])
    for (name, H, objs) in sets, obj in objs
        dkey(name, obj) in have && continue
        obj == "reuse" || continue
        push!(data["designs"], Dict{String,Any}("set" => name, "H" => H, "objective" => obj,
            "omega" => tail(only(H)), "train_objective" => 1.0, "evaluations" => 0,
            "pep_solves" => 0, "seconds" => 0.0, "polished" => 0,
            "values" => Dict{String,Any}()))
    end
    save()
    function design_starts(H, extra = Vector{Float64}[])
        mid = H[argmin(abs.(log.(H) .- (log(H[1]) + log(H[end])) / 2))]
        return [[Float64.(_STARTS[1]), tail(mid)]; extra]
    end
    tasks = []
    for (name, H, objs) in sets, obj in objs
        (obj == "reuse" || dkey(name, obj) in have) && continue
        push!(tasks, (; key = dkey(name, obj), name, H, obj, cost = sum(float(N)^3 for N in H),
                      args = (H, [Wt[N] for N in H], obj, design_starts(H), maxevals)))
    end
    sort!(tasks; by = t -> -t.cost)
    run_pool!(tasks, design_task, (task, res) -> begin
        lock(lk) do
            push!(data["designs"], Dict{String,Any}("set" => task.name, "H" => task.H,
                "objective" => task.obj, "omega" => res.omega, "train_objective" => res.f,
                "evaluations" => res.evals, "pep_solves" => res.pep_solves,
                "seconds" => res.seconds, "polished" => 0, "values" => Dict{String,Any}()))
        end
        save()
        @printf("[design %6.0fs] %-22s f=%.6f omega=%s evals=%d (%.0fs, worker %d)\n",
                time() - t0, task.key, res.f, string(round.(res.omega; digits = 5)),
                res.evals, res.seconds, res.worker)
        flush(stdout)
    end)

    function evaluate_all!()
        tasks = []
        for d in data["designs"], N in Ns
            haskey(d["values"], string(N)) && continue
            push!(tasks, (; d, N, args = (Float64.(d["omega"]), N)))
        end
        sort!(tasks; by = t -> -t.N)
        isempty(tasks) || println("Evaluating ", length(tasks), " (design, N) pairs")
        run_pool!(tasks, eval_task, (task, res) -> begin
            lock(() -> (task.d["values"][string(task.N)] = res.W), lk)
            task.N >= 40 && @printf("[eval %6.0fs] N=%2d %-22s W=%.8f (%.1fs)\n",
                                    time() - t0, task.N,
                                    dkey(task.d["set"], task.d["objective"]), res.W,
                                    res.seconds)
            save()
        end)
        save()
    end

    # Stage 3: evaluate; Stage 4: cross-check and polish (at most twice).
    evaluate_all!()
    for round in 1:2
        W = final_wstar(data, Ns)
        tasks = []
        for d in data["designs"]
            d["objective"] == "reuse" && continue
            H = Int.(d["H"])
            own = criterion(H, d["objective"], d, W)
            vals = [criterion(H, d["objective"], e, W) for e in data["designs"]]
            b = argmin(vals)
            vals[b] < own * (1 - 1e-5) || continue
            better = Float64.(data["designs"][b]["omega"])
            @printf("polish %-22s own=%.6f beaten by %s (%.6f)\n",
                    dkey(d["set"], d["objective"]), own,
                    dkey(data["designs"][b]["set"], data["designs"][b]["objective"]), vals[b])
            push!(tasks, (; d, args = (H, [W[N] for N in H], d["objective"],
                                       [Float64.(d["omega"]), better], maxevals)))
        end
        isempty(tasks) && break
        run_pool!(tasks, design_task, (task, res) -> lock(lk) do
            d = task.d
            d["omega"] = res.omega
            d["train_objective"] = res.f
            d["evaluations"] += res.evals
            d["pep_solves"] += res.pep_solves
            d["seconds"] += res.seconds
            d["polished"] += 1
            d["values"] = Dict{String,Any}()
        end)
        save()
        evaluate_all!()
    end

    # Stage 5: exchange (Remez-type) method. Start from the geometric minimax
    # design; while its worst ratio on the dense grid exceeds its value on H,
    # add the worst horizon to H and re-solve from the current design.
    start = smoke ? "geom_2_8" : "geom_2_32"
    if !haskey(data, "exchange")
        W = final_wstar(data, Ns)
        d0 = only(filter(d -> d["set"] == start && d["objective"] == "max", data["designs"]))
        H = Int.(d0["H"])
        D = collect(first(H):last(H))
        ω = Float64.(d0["omega"])
        vals = Dict(N => d0["values"][string(N)] for N in D)
        steps = Dict{String,Any}[]
        solves, secs = d0["pep_solves"], d0["seconds"]
        for it in 0:4
            r = Dict(N => vals[N] / W[N] for N in D)
            own = maximum(r[N] for N in H)
            worst, Nw = findmax(r)
            push!(steps, Dict("iteration" => it, "H" => copy(H), "omega" => copy(ω),
                              "own" => own, "dense_worst" => worst, "argmax" => Nw,
                              "cumulative_pep_solves" => solves, "cumulative_seconds" => secs))
            @printf("[exchange] it=%d H=%s own=%.6f dense worst=%.6f at N=%d PEPs=%d\n", it,
                    string(H), own, worst, Nw, solves)
            flush(stdout)
            worst <= own + 2e-4 && break
            push!(H, Nw)
            sort!(H)
            res = remotecall_fetch(design_task, first(workers()), H, [W[N] for N in H], "max",
                                   [ω, Float64.(_STARTS[1])], maxevals)
            ω = res.omega
            solves += res.pep_solves
            secs += res.seconds
            # Dense check: one PEP per grid horizon, in parallel.
            check = [(; N, args = (ω, N)) for N in D]
            run_pool!(check, eval_task, (task, res2) -> begin
                vals[task.N] = res2.W
                secs += res2.seconds
            end)
            solves += length(D)
        end
        data["exchange"] = Dict("start" => start, "dense_grid" => D, "steps" => steps)
        last_step = steps[end]
        push!(data["designs"], Dict{String,Any}("set" => "exchange_$(first(D))_$(last(D))",
            "H" => last_step["H"], "objective" => "max", "omega" => last_step["omega"],
            "train_objective" => last_step["own"], "evaluations" => 0,
            "pep_solves" => last_step["cumulative_pep_solves"],
            "seconds" => last_step["cumulative_seconds"], "polished" => 0,
            "values" => Dict{String,Any}(string(N) => vals[N] for N in D)))
        save()
        evaluate_all!()
    end
    W = final_wstar(data, Ns)
    data["Wstar"] = Dict(string(N) => W[N] for N in Ns)
    save()
    write_outputs(data, out, Ns, W, smoke)
    println("Saved horizon-shape study: ", path)
end

function write_outputs(data, out, Ns, W, smoke)
    designs = data["designs"]
    ratio(d, N) = d["values"][string(N)] / W[N]
    ext = smoke ? 8 : 32
    inr = filter(<=(ext), Ns)
    beyond = filter(>(ext), Ns)

    rows = ["design,set,objective,N,in_H,value,ratio"]
    for d in designs, N in Ns
        push!(rows, join((dkey(d["set"], d["objective"]), d["set"], d["objective"], N,
                          N in d["H"], d["values"][string(N)], ratio(d, N)), ","))
    end
    write(joinpath(out, "shape_ratios.csv"), join(rows, "\n") * "\n")

    nimp = [(N, data["tailored"][string(N)]["W"] / W[N] - 1) for N in Ns]
    nimp = filter(x -> x[2] > 1e-9, nimp)
    @printf("\nNormalizer: tailored SLM improved by a design at %d of %d horizons (max %.2e relative)\n",
            length(nimp), length(Ns), isempty(nimp) ? 0.0 : maximum(last, nimp))

    head = "design,set,objective,H_size,H_min,H_max,omega,pep_solves,design_seconds,polished," *
           "own_criterion,best_competitor_on_own,worst_in_$(first(inr))_$(last(inr)),argmax_in," *
           "mean_in,worst_beyond_$(ext),argmax_beyond,worst_all,active_horizons"
    srows = [head]
    @printf("\n%-22s %4s %8s %7s %9s %5s %8s %9s %5s  %s\n", "design", "|H|", "PEPs", "own",
            "worst<=$(ext)", "argN", "mean<=$(ext)", "worst>$(ext)", "argN", "active")
    for d in designs
        H = Int.(d["H"])
        obj = d["objective"] == "mean" ? "mean" : "max"
        own = criterion(H, obj, d, W)
        comp = minimum(criterion(H, obj, e, W) for e in designs)
        rin = [ratio(d, N) for N in inr]
        rb = isempty(beyond) ? [NaN] : [ratio(d, N) for N in beyond]
        rH = [ratio(d, N) for N in H]
        active = obj == "max" ? H[rH .>= maximum(rH) - 2e-4] : Int[]
        push!(srows, join((dkey(d["set"], d["objective"]), d["set"], d["objective"], length(H),
                           first(H), last(H), "\"" * join(d["omega"], ";") * "\"",
                           d["pep_solves"], d["seconds"], d["polished"], own, comp,
                           maximum(rin), inr[argmax(rin)], mean(rin), maximum(rb),
                           isempty(beyond) ? 0 : beyond[argmax(rb)],
                           max(maximum(rin), maximum(rb)), "\"" * join(active, ";") * "\""), ","))
        @printf("%-22s %4d %8d %7.4f %9.4f %5d %8.4f %9.4f %5d  %s\n",
                dkey(d["set"], d["objective"]), length(H), d["pep_solves"], own, maximum(rin),
                inr[argmax(rin)], mean(rin), maximum(rb), isempty(beyond) ? 0 : beyond[argmax(rb)],
                string(active))
    end
    write(joinpath(out, "shape_summary.csv"), join(srows, "\n") * "\n")

    # Criterion (rows) × design (columns): value of each design on each set's objective.
    crit = [(d["set"], Int.(d["H"]), d["objective"]) for d in designs if d["objective"] != "reuse"]
    crows = ["criterion," * join([dkey(d["set"], d["objective"]) for d in designs], ",")]
    for (name, H, obj) in crit
        push!(crows, dkey(name, obj) * "," *
                     join([criterion(H, obj, d, W) for d in designs], ","))
    end
    write(joinpath(out, "shape_cross.csv"), join(crows, "\n") * "\n")

    if haskey(data, "exchange")
        ex = data["exchange"]
        erows = ["iteration,H,own,dense_worst,argmax,cumulative_pep_solves,cumulative_seconds"]
        println("\nExchange from $(ex["start"]) on dense grid $(first(ex["dense_grid"]))..$(last(ex["dense_grid"])):")
        for s in ex["steps"]
            push!(erows, join((s["iteration"], "\"" * join(s["H"], ";") * "\"", s["own"],
                               s["dense_worst"], s["argmax"], s["cumulative_pep_solves"],
                               s["cumulative_seconds"]), ","))
            @printf("  it=%d H=%s own=%.6f dense worst=%.6f (N=%d) PEPs=%d time=%.0fs\n",
                    s["iteration"], string(s["H"]), s["own"], s["dense_worst"], s["argmax"],
                    s["cumulative_pep_solves"], s["cumulative_seconds"])
        end
        write(joinpath(out, "shape_exchange.csv"), join(erows, "\n") * "\n")
    end
end

RUN_AS_SCRIPT && main(OPTS)
