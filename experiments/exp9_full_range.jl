# Full-range re-evaluation of an exp9 anytime study whose normalizer is exact
# (OGD regret: W*_N = sqrt(N) with L = D = 1). exp9 evaluates its designs on a
# sampled horizon set. This script evaluates every stored design at every
# requested horizon, adds the reuse of the schedule tailored at the largest
# geometric design horizon, checks the normalizer at every horizon
# (W_N(1/sqrt(N)) = sqrt(N)), and optionally runs a free-step SLM multistart
# to check that no unrestricted schedule beats sqrt(N).
#
# Results accumulate in <study>\full_range.toml, which has the anytime.toml
# schema. The run is resumable: only missing (design, N) pairs are computed,
# and certified values already present in anytime.toml are reused.
#
# julia --project=. experiments\exp9_full_range.jl <anytime.toml> --horizons 2:40 --workers 4
# julia --project=. experiments\exp9_full_range.jl <anytime.toml> --horizons 48 --workers 2 --only <keys>
# julia --project=. experiments\exp9_full_range.jl <anytime.toml> --free-check 2,3,4,6,8,12,16 --workers 4
# julia --project=. experiments\exp9_full_range.jl <anytime.toml>          (rewrite summaries only)
# Plot with experiments\plot_anytime.jl <study>\full_range.toml.
#
# Design keys are "family/kind/trainset", e.g. "power_law/hrdp_max/geometric".
# Peak memory per worker is about 3.3 GB at N = 48; choose --workers accordingly.

using Distributed
using TOML
using Printf

const EXP9 = joinpath(@__DIR__, "exp9_anytime.jl")
const PROJECT = dirname(@__DIR__)

function parse_horizons(s)
    out = Int[]
    for item in split(s, ",")
        if occursin(":", item)
            a, b = parse.(Int, split(item, ":"))
            append!(out, a:b)
        else
            push!(out, parse(Int, item))
        end
    end
    all(>=(1), out) || throw(ArgumentError("horizons must be positive"))
    return sort!(unique(out))
end

function parse_options(args)
    isempty(args) && throw(ArgumentError(
        "usage: exp9_full_range.jl <anytime.toml> [--horizons spec] [--workers k] " *
        "[--only key1,key2] [--free-check spec]"))
    opts = Dict{String,Any}("study" => abspath(args[1]), "horizons" => Int[],
                            "workers" => 2, "only" => String[], "free_check" => Int[])
    i = 2
    while i <= length(args)
        flag = args[i]
        i < length(args) || throw(ArgumentError("missing value for $flag"))
        val = args[i + 1]
        if flag == "--horizons"
            opts["horizons"] = parse_horizons(val)
        elseif flag == "--workers"
            opts["workers"] = parse(Int, val)
        elseif flag == "--only"
            opts["only"] = String.(split(val, ","))
        elseif flag == "--free-check"
            opts["free_check"] = parse_horizons(val)
        else
            throw(ArgumentError("unknown option $flag"))
        end
        i += 2
    end
    opts["workers"] >= 1 || throw(ArgumentError("--workers must be positive"))
    return opts
end

const RUN_AS_SCRIPT = abspath(PROGRAM_FILE) == (@__FILE__)
const OPTS = RUN_AS_SCRIPT ? parse_options(ARGS) : nothing
if RUN_AS_SCRIPT && (!isempty(OPTS["horizons"]) || !isempty(OPTS["free_check"]))
    addprocs(OPTS["workers"]; exeflags = "--project=$(PROJECT)")
end

@everywhere include($EXP9)

@everywhere begin
    # One compiled PEP per process; released when the horizon changes so that
    # a worker never holds two large models.
    const _COMPILED = Dict{Int,Any}()
    function _compiled(problem, N)
        if !haskey(_COMPILED, N)
            empty!(_COMPILED)
            GC.gc()
            _COMPILED[N] = setup(problem, false).compiler(N)
        end
        return _COMPILED[N]
    end

    function eval_design_task(problem, family, ω, N)
        BLAS.set_num_threads(1)
        pol = setup(problem, false).families[family][1]
        cp = _compiled(problem, N)
        t = time()
        W = certified_value(DesignProblem(cp, pol, N), ω)
        return (; W, seconds = time() - t, worker = myid())
    end

    function free_check_task(problem, N, ω0, Δ0, iters)
        BLAS.set_num_threads(1)
        cp = _compiled(problem, N)
        t = time()
        tr = design_slm(DesignProblem(cp, IdentityPolicy(), N), ω0; iters, Δ0,
                        max_time = 900.0, require_certified = true)
        ω, W = best_point(tr)
        return (; W, omega = ω, seconds = time() - t, pep_solves = last(tr.solves_hist),
                worker = myid())
    end
end

design_key(d) = "$(d["family"])/$(d["kind"])/$(d["trainset"])"

function initialize(src, S)
    cfg = copy(src["config"])
    cfg["source"] = "anytime.toml (same folder)"
    cfg["sampled_horizons"] = cfg["horizons"]
    cfg["normalizer_check"] = "W_N(constant 1/sqrt(N)) evaluated at every horizon"
    designs = Dict{String,Any}[]
    for d in src["designs"]
        e = copy(d)
        e["origin"] = "exp9"
        e["values"] = Dict{String,Any}(k => v for (k, v) in d["values"])
        e["eval_seconds"] = Dict{String,Any}()
        push!(designs, e)
    end
    Ng = maximum(S.trainsets["geometric"])
    push!(designs, Dict{String,Any}("family" => "constant", "kind" => "reuse",
        "trainset" => "N$(Ng)", "omega" => [1 / sqrt(Ng)], "train_objective" => NaN,
        "evaluations" => 0, "seconds" => 0.0, "origin" => "reference",
        "note" => "schedule tailored at the largest geometric design horizon",
        "values" => Dict{String,Any}(), "eval_seconds" => Dict{String,Any}()))
    return Dict{String,Any}("config" => cfg, "designs" => designs,
                            "Wstar" => Dict{String,Any}(),
                            "normalizer_check" => Dict{String,Any}(),
                            "free_check" => Dict{String,Any}())
end

function persist(path, data)
    open(path * ".tmp", "w") do io
        TOML.print(io, data)
    end
    mv(path * ".tmp", path; force = true)
end

# Longest-first dynamic scheduling over the worker pool; results are recorded
# and persisted on the master as soon as each task returns.
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

function evaluated_horizons(data)
    Ns = Set{Int}()
    for d in data["designs"]
        union!(Ns, parse.(Int, collect(keys(d["values"]))))
    end
    return sort!(collect(Ns))
end

function refresh!(data, S)
    H = evaluated_horizons(data)
    data["config"]["horizons"] = H
    data["Wstar"] = Dict(string(N) => S.exact(N) for N in H)
end

function write_summaries(data, S, outdir)
    H = evaluated_horizons(data)
    designs = data["designs"]
    Wstar(N) = S.exact(N)
    has(d, N) = haskey(d["values"], string(N))
    ratio(d, N) = d["values"][string(N)] / Wstar(N)
    # Largest block min(H)..M evaluated for every design.
    lo = minimum(H)
    M = lo - 1
    while all(d -> has(d, M + 1), designs)
        M += 1
    end
    extra = filter(N -> N > M && any(d -> has(d, N), designs), H)
    N0 = S.N0
    trainsets = data["config"]["trainsets"]
    reuse = only(filter(d -> d["kind"] == "reuse" && d["trainset"] == "N0", designs))
    sampled = Set(Int.(data["config"]["sampled_horizons"]))

    rows = ["design,family,kind,trainset,N,in_trainset,in_exp9_sample,value,ratio"]
    for d in designs, N in H
        has(d, N) || continue
        intrain = haskey(trainsets, d["trainset"]) && N in trainsets[d["trainset"]]
        push!(rows, join((design_key(d), d["family"], d["kind"], d["trainset"], N, intrain,
                          N in sampled, d["values"][string(N)], ratio(d, N)), ","))
    end
    write(joinpath(outdir, "full_range_ratios.csv"), join(rows, "\n") * "\n")

    beyond = M >= lo ? collect((N0 + 1):M) : Int[]
    head = "design,omega,block_lo,block_hi,worst_ratio_block,argmax_N,worst_ratio_trainset," *
           "wins_vs_reuse_N0_beyond_N0,horizons_beyond_N0" *
           join([",ratio_at_$(N)" for N in extra])
    srows = [head]
    @printf("\nComplete block N = %d..%d (%d horizons) for all %d designs; extra horizons: %s\n",
            lo, M, M - lo + 1, length(designs), string(extra))
    @printf("%-34s %9s %5s %9s %9s %s\n", "design", "worst", "argN", "trainset", "wins>N0",
            join([@sprintf("%9s", "N=$(N)") for N in extra]))
    for d in designs
        block = M >= lo ? [ratio(d, N) for N in lo:M] : Float64[]
        worst = isempty(block) ? NaN : maximum(block)
        argN = isempty(block) ? 0 : (lo:M)[argmax(block)]
        trs = haskey(trainsets, d["trainset"]) ?
              [ratio(d, N) for N in trainsets[d["trainset"]] if has(d, N)] : Float64[]
        tr = isempty(trs) ? NaN : maximum(trs)
        wins = count(N -> ratio(d, N) < ratio(reuse, N) - 1e-9, beyond)
        ex = [has(d, N) ? ratio(d, N) : NaN for N in extra]
        push!(srows, join((design_key(d), "\"" * join(d["omega"], ";") * "\"", lo, M, worst,
                           argN, tr, wins, length(beyond), ex...), ","))
        @printf("%-34s %9.4f %5d %9.4f %5d/%-3d %s\n", design_key(d), worst, argN, tr, wins,
                length(beyond), join([@sprintf("%9.4f", x) for x in ex]))
    end
    write(joinpath(outdir, "full_range_summary.csv"), join(srows, "\n") * "\n")

    nc = data["normalizer_check"]
    if !isempty(nc)
        dev = maximum(abs(nc[k] / Wstar(parse(Int, k)) - 1) for k in keys(nc))
        @printf("\nNormalizer check: %d horizons, max |W_N(1/sqrt N)/sqrt N - 1| = %.2e\n",
                length(nc), dev)
    end
    fc = data["free_check"]
    if !isempty(fc)
        frows = ["N,runs,best_free_value,sqrtN,best_ratio"]
        for k in sort!(collect(keys(fc)); by = x -> parse(Int, x))
            runs = fc[k]
            isempty(runs) && continue
            best = minimum(r["W"] for r in values(runs))
            N = parse(Int, k)
            push!(frows, join((N, length(runs), best, Wstar(N), best / Wstar(N)), ","))
            @printf("Free-step check N=%2d: %d SLM runs, best/sqrt(N) = %.8f\n", N,
                    length(runs), best / Wstar(N))
        end
        write(joinpath(outdir, "free_check.csv"), join(frows, "\n") * "\n")
    end
end

function main(opts)
    src_path = opts["study"]
    src = TOML.parsefile(src_path)
    problem = startswith(src["config"]["problem"], "OGD") ? "ogd" :
        throw(ArgumentError("full-range evaluation needs an exact normalizer (OGD study)"))
    S = setup(problem, false)
    outdir = dirname(src_path)
    path = joinpath(outdir, "full_range.toml")
    data = isfile(path) ? TOML.parsefile(path) : initialize(src, S)
    persist_lock = ReentrantLock()
    save() = lock(() -> (refresh!(data, S); persist(path, data)), persist_lock)

    keyset = Set(design_key(d) for d in data["designs"])
    for k in opts["only"]
        k in keyset || throw(ArgumentError("unknown design key '$k'; known: $(sort!(collect(keyset)))"))
    end
    selected = isempty(opts["only"]) ? data["designs"] :
               filter(d -> design_key(d) in opts["only"], data["designs"])

    tasks = []
    for N in opts["horizons"]
        for d in selected
            haskey(d["values"], string(N)) && continue
            push!(tasks, (; key = design_key(d), N,
                          args = (problem, d["family"], Float64.(d["omega"]), N)))
        end
        haskey(data["normalizer_check"], string(N)) ||
            push!(tasks, (; key = "normalizer", N,
                          args = (problem, "constant", [1 / sqrt(N)], N)))
    end
    sort!(tasks; by = t -> (-t.N, t.key))
    byname = Dict(design_key(d) => d for d in data["designs"])
    done = Ref(0)
    t0 = time()
    function record_design!(task, res)
        lock(persist_lock) do
            if task.key == "normalizer"
                data["normalizer_check"][string(task.N)] = res.W
            else
                d = byname[task.key]
                d["values"][string(task.N)] = res.W
                d["eval_seconds"][string(task.N)] = res.seconds
            end
        end
        save()
        done[] += 1
        @printf("[%3d/%3d %6.0fs] N=%2d %-34s W=%.8f ratio=%.5f (%.1fs, worker %d)\n",
                done[], length(tasks), time() - t0, task.N, task.key, res.W,
                res.W / S.exact(task.N), res.seconds, res.worker)
        flush(stdout)
    end
    isempty(tasks) || println("Evaluating ", length(tasks), " (design, horizon) pairs on ",
                              nworkers(), " workers; saving to ", path)
    run_pool!(tasks, eval_design_task, record_design!)

    free_tasks = []
    for N in opts["free_check"]
        runs = get!(data["free_check"], string(N), Dict{String,Any}())
        starts = [("constant", fill(1 / sqrt(N), N)),
                  ("anytime_0.7", [0.7 / sqrt(n) for n in 1:N]),
                  ("constant_0.3", fill(0.3, N))]
        for (label, ω0) in starts, Δ0 in (1.0, 0.1)
            name = "$(label)/Delta0=$(Δ0)"
            haskey(runs, name) && continue
            push!(free_tasks, (; key = name, N, args = (problem, N, ω0, Δ0, 80)))
        end
    end
    sort!(free_tasks; by = t -> (-t.N, t.key))
    function record_free!(task, res)
        lock(persist_lock) do
            data["free_check"][string(task.N)][task.key] = Dict(
                "W" => res.W, "omega" => res.omega, "seconds" => res.seconds,
                "pep_solves" => res.pep_solves, "iterations" => 80)
        end
        save()
        @printf("free N=%2d %-24s best=%.8f sqrtN=%.8f ratio=%.8f (%.1fs)\n", task.N,
                task.key, res.W, S.exact(task.N), res.W / S.exact(task.N), res.seconds)
        flush(stdout)
    end
    isempty(free_tasks) || println("Running ", length(free_tasks), " free-step SLM checks")
    run_pool!(free_tasks, free_check_task, record_free!)

    save()
    write_summaries(data, S, outdir)
    println("Saved full-range study: ", path)
end

RUN_AS_SCRIPT && main(OPTS)
