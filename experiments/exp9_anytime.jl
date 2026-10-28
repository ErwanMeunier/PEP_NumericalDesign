# Anytime / horizon-robust design check (paper, "Robustness to additional
# iterations"). For prefix-consistent priors (the N-step schedule is the
# prefix of the (N+1)-step one), compare normalized worst cases
# W_N(α(ω))/W*_N at every horizon, including horizons never used for design:
#   reuse      ω tailored to the single horizon N0;
#   hrdp_mean  argmin of the paper's σ(H, ω) = mean_{N∈H} W_N/W*_N;
#   hrdp_max   argmin of max_{N∈H} W_N/W*_N (anytime competitive ratio on H).
# H is either the window {2..N0} (paper) or a geometric grid. Priors have ≤ 3
# parameters and are optimized by multistart Nelder–Mead on certified PEP
# values only (no dependence on gradient-solver tuning).
#
# julia --project=. experiments\exp9_anytime.jl ogd
# julia --project=. experiments\exp9_anytime.jl igdm
# julia --project=. experiments\exp9_anytime.jl --smoke
# Plot with experiments\plot_anytime.jl <anytime.toml>.

using PEPDesign
using LinearAlgebra
using Statistics
using TOML
using Printf

function nelder_mead(f, x0, scale; maxevals = 200, ftol = 1e-9)
    n = length(x0)
    X = [collect(Float64, x0)]
    for i in 1:n
        x = copy(X[1])
        x[i] += scale[i]
        push!(X, x)
    end
    F = [f(x) for x in X]
    evals = n + 1
    while evals < maxevals
        p = sortperm(F)
        X, F = X[p], F[p]
        isfinite(F[end]) && F[end] - F[1] <= ftol * max(1.0, abs(F[1])) && break
        c = sum(X[1:n]) / n
        xr = c .+ (c .- X[end])
        fr = f(xr)
        evals += 1
        if fr < F[1]
            xe = c .+ 2 .* (c .- X[end])
            fe = f(xe)
            evals += 1
            X[end], F[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < F[n]
            X[end], F[end] = xr, fr
        else
            xc = fr < F[end] ? c .+ 0.5 .* (xr .- c) : c .+ 0.5 .* (X[end] .- c)
            fc = f(xc)
            evals += 1
            if fc < min(fr, F[end])
                X[end], F[end] = xc, fc
            else
                for i in 2:(n + 1)
                    X[i] = X[1] .+ 0.5 .* (X[i] .- X[1])
                    F[i] = f(X[i])
                    evals += 1
                end
            end
        end
    end
    i = argmin(F)
    return X[i], F[i], evals
end

# Multistart; each start is followed by one restart with a shrunken simplex.
function minimize(f, starts, scale; maxevals = 200)
    best_x, best_f, evals = collect(Float64, first(starts)), Inf, 0
    for x0 in starts
        x, _, e = nelder_mead(f, x0, scale; maxevals)
        x, fx, e2 = nelder_mead(f, x, scale ./ 4; maxevals = maxevals ÷ 2)
        evals += e + e2
        fx < best_f && ((best_x, best_f) = (x, fx))
    end
    return best_x, best_f, evals
end

function certified_value(dp, ω)
    try
        η, _, _ = evaluate_policy(dp.policy, ω, dp.N)
        all(isfinite, η) || return Inf
        W, sol, _ = pep_value(dp, ω)
        sol.diagnostics.certified && isfinite(W) || return Inf
        return W
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
end

inv_sqrt_policy() = FunctionPolicy((ω, n, N) -> ω[1] / sqrt(n), 1; name = "inv_sqrt")

function setup(problem, smoke)
    if problem == "ogd"
        N0 = smoke ? 4 : 8
        trainsets = smoke ?
            Dict("window" => collect(2:N0), "geometric" => [2, 4, 6]) :
            Dict("window" => collect(2:N0), "geometric" => [2, 4, 8, 16])
        extra = smoke ? [5, 6] : [10, 12, 16, 20, 24, 32, 40]
        compiler = N -> compile_ogd(N, 1.0, 1.0)
        a0 = 1 / sqrt(N0)
        families = Dict(
            "constant" => (ConstantPolicy(), [[a0]], [0.1a0]),
            "inv_sqrt" => (inv_sqrt_policy(), [[1.0], [0.7]], [0.1]),
            "power_law" => (PowerLawPolicy(), [[a0, 0.0, 0.0], [1.0, 0.5, 0.0]],
                            [0.1a0, 0.2, 0.2]),
            "log_poly2" => (log_poly_policy(2), [[log(a0), 0.0, 0.0], [0.0, -0.5, 0.0]],
                            [0.2, 0.2, 0.05]))
        plan = [[(f, "window", obj) for f in ("constant", "inv_sqrt", "power_law", "log_poly2")
                 for obj in (:mean, :max)];
                [(f, "geometric", :max) for f in ("constant", "inv_sqrt", "power_law")]]
        # Tailored optimum at N0 is the constant 1/sqrt(N0), with W*_N = sqrt(N).
        fixed = [("constant", "reuse", [a0]), ("inv_sqrt", "textbook", [1.0]),
                 ("inv_sqrt", "classical_anytime", [1 / sqrt(2)])]
        exact = N -> sqrt(N)
        config = Dict("problem" => "OGD regret", "L" => 1.0, "D" => 1.0,
                      "normalizer" => "exact W*_N = sqrt(N) (free optimum, constant 1/sqrt(N))")
    elseif problem == "igdm"
        N0, K = 10, 2
        trainsets = Dict("window" => collect(2:N0))
        extra = [15, 20, 25, 30]
        compiler = N -> compile_igdm(N, 1.0, 0.3; objective = :min_grad,
                                     encoding = :substitution)
        families = Dict("stationary_K2" =>
            (IGDMStationaryPolicy(K), [[1.0, 0.0, 0.0]], [0.1, 0.05, 0.05]))
        plan = [("stationary_K2", "window", obj) for obj in (:mean, :max)]
        fixed = Tuple{String,String,Vector{Float64}}[]   # reuse = tailored at N0 (SLM)
        exact = nothing
        config = Dict("problem" => "IGDM min_grad", "L" => 1.0, "D" => 1.0,
                      "epsilon" => 0.3, "encoding" => "substitution", "K" => K,
                      "normalizer" => "best known stationary-by-lag value per N " *
                                      "(SLM multistart, then min with every evaluated design)")
    else
        throw(ArgumentError("problem must be ogd or igdm"))
    end
    return (; N0, trainsets, extra, compiler, families, plan, fixed, exact, config)
end

function persist(path, data)
    open(path * ".tmp", "w") do io
        TOML.print(io, data)
    end
    mv(path * ".tmp", path; force = true)
end

function main(args)
    smoke = args == ["--smoke"]
    problem = smoke ? "ogd" : (length(args) == 1 ? lowercase(args[1]) :
        throw(ArgumentError("usage: exp9_anytime.jl ogd|igdm|--smoke")))
    BLAS.set_num_threads(1)
    S = setup(problem, smoke)
    maxevals = smoke ? 20 : 150
    horizons = sort!(unique([reduce(vcat, values(S.trainsets)); S.extra]))
    cps = Dict(N => S.compiler(N) for N in horizons)
    dp(pol, N) = DesignProblem(cps[N], pol, N)

    root = joinpath(@__DIR__, "results")
    mkpath(root)
    out = mktempdir(root; prefix = "anytime_$(problem)_", cleanup = false)
    path = joinpath(out, "anytime.toml")
    config = merge(S.config, Dict("N0" => S.N0, "horizons" => horizons,
        "trainsets" => S.trainsets, "optimizer" => "Nelder-Mead multistart + restart",
        "maxevals_per_start" => maxevals, "certified_values_only" => true,
        "julia_version" => string(VERSION), "BLAS_threads" => 1))
    data = Dict{String,Any}("config" => config, "designs" => Dict{String,Any}[])

    Wstar = Dict{Int,Float64}()
    tailored = Dict{Int,Vector{Float64}}()
    if S.exact !== nothing
        for N in horizons
            Wstar[N] = S.exact(N)
        end
    else
        pol, starts, _ = only(values(S.families))
        prev = nothing
        for N in horizons
            cands = prev === nothing ? starts : [starts; [prev]]
            best_ω, best_W = first(cands), Inf
            t = time()
            for ω0 in cands
                tr = design_slm(dp(pol, N), ω0; iters = 60, Δ0 = 1.0, max_time = 300.0,
                                require_certified = true)
                ω, W = best_point(tr)
                W < best_W && ((best_ω, best_W) = (ω, W))
            end
            Wstar[N], tailored[N], prev = best_W, best_ω, best_ω
            @printf("tailored N=%2d W=%.8f omega=%s  %.1fs\n", N, best_W,
                    string(round.(best_ω; digits = 5)), time() - t)
            flush(stdout)
        end
        data["tailored"] = Dict(string(N) => tailored[N] for N in horizons)
    end

    function add_design!(family, kind, trainset, ω, train_obj, evals, seconds)
        pol = S.families[family][1]
        vals = Dict(string(N) => certified_value(dp(pol, N), ω) for N in horizons)
        push!(data["designs"], Dict("family" => family, "kind" => kind,
            "trainset" => trainset, "omega" => ω, "train_objective" => train_obj,
            "evaluations" => evals, "seconds" => seconds, "values" => vals))
        persist(path, data)
        @printf("%-10s %-18s %-10s omega=%s  obj=%.6f  evals=%d  %.1fs\n", family, kind,
                trainset, string(round.(ω; digits = 5)), train_obj, evals, seconds)
        flush(stdout)
    end

    if S.exact === nothing
        family = only(keys(S.families))
        add_design!(family, "reuse", "N0", tailored[S.N0], 1.0, 0, 0.0)
    end
    for (family, kind, ω) in S.fixed
        add_design!(family, kind, kind == "reuse" ? "N0" : "none", ω, NaN, 0, 0.0)
    end

    for (family, setname, obj) in S.plan
        pol, starts, scale = S.families[family]
        H = S.trainsets[setname]
        train_dps = [dp(pol, N) for N in H]
        function σ(ω)
            r = [certified_value(d, ω) / Wstar[d.N] for d in train_dps]
            all(isfinite, r) || return Inf
            return obj == :mean ? mean(r) : maximum(r)
        end
        cands = S.exact === nothing ? [starts; [tailored[S.N0]]] : starts
        prior = filter(d -> d["family"] == family && d["trainset"] == setname,
                       data["designs"])
        cands = [cands; [Vector{Float64}(d["omega"]) for d in prior]]
        t = time()
        ω, f, e = minimize(σ, cands, scale; maxevals)
        add_design!(family, "hrdp_$(obj)", setname, ω, f, e, time() - t)
    end

    if S.exact === nothing
        for d in data["designs"], N in horizons
            W = d["values"][string(N)]
            isfinite(W) && W < Wstar[N] - 1e-9 &&
                @printf("note: %s/%s improves W*_%d from %.8f to %.8f\n",
                        d["family"], d["kind"], N, Wstar[N], W)
            Wstar[N] = min(Wstar[N], W)
        end
    end
    data["Wstar"] = Dict(string(N) => Wstar[N] for N in horizons)
    persist(path, data)

    istrain(d, N) = d["trainset"] in keys(S.trainsets) && N in S.trainsets[d["trainset"]]
    rows = ["family,kind,trainset,N,in_trainset,value,ratio"]
    for d in data["designs"], N in horizons
        W = d["values"][string(N)]
        push!(rows, join((d["family"], d["kind"], d["trainset"], N, istrain(d, N), W,
                          W / Wstar[N]), ","))
    end
    write(joinpath(out, "anytime_ratios.csv"), join(rows, "\n") * "\n")

    ratio(d, N) = d["values"][string(N)] / Wstar[N]
    reuse_d = only(filter(d -> d["kind"] == "reuse", data["designs"]))
    beyond = filter(>(S.N0), horizons)
    println()
    @printf("%-10s %-18s %-9s | %s | %8s %8s  beats reuse (N>N0)\n", "family", "design",
            "trainset", join([@sprintf("%6d", N) for N in horizons], " "), "max_all",
            "max_N>N0")
    for d in data["designs"]
        r = [ratio(d, N) for N in horizons]
        wins = count(N -> ratio(d, N) < ratio(reuse_d, N) - 1e-9, beyond)
        @printf("%-10s %-18s %-9s | %s | %8.4f %8.4f  %d/%d\n", d["family"], d["kind"],
                d["trainset"], join([@sprintf("%6.4f", x) for x in r], " "),
                maximum(r), maximum(ratio(d, N) for N in beyond), wins, length(beyond))
    end
    println("Saved anytime study: ", path)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
