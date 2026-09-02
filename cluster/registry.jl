# Cluster experiment registry: grid definition, stable task numbering, and
# manifest generation for the SLURM array pipeline.
#
# TASK-ID STABILITY RULE: task ids index into the grids below. Never reorder
# methods, policies (see policy_suite in experiments/common.jl), Ns, or starts
# of an already-submitted experiment tag — append only.
#
# Environment overrides (set in the .slurm files or before local runs):
#   PEP_TAG        output tag              (default "default")
#   PEP_SMOKE      "1" → tiny local grid   (default off)
#   PEP_IGDM_EPS   IGDM ε                  (default 0.3)
#   PEP_IGDM_K     reserved for K-grids    (default -1 = suite as-is)
#
# Print the array sizes for submission scripts:
#   julia --project=. PEPDesign/cluster/registry.jl counts

include(joinpath(@__DIR__, "..", "experiments", "common.jl"))

const SMOKE_GRID = get(ENV, "PEP_SMOKE", "0") == "1"
const TAG = get(ENV, "PEP_TAG", SMOKE_GRID ? "smoke" : "default")
const IGDM_EPS = parse(Float64, get(ENV, "PEP_IGDM_EPS", "0.3"))

const GRID = (
    methods = SMOKE_GRID ? [:OGD] : [:OGD, :ITEM, :IGDM],
    Ns = SMOKE_GRID ? [3, 4] : [10, 20, 30, 40],
    H_train_max = SMOKE_GRID ? 4 : 20,
    iota_max = SMOKE_GRID ? 1 : 10,
    nstarts = SMOKE_GRID ? 1 : 3,
    iters_fom = SMOKE_GRID ? 8 : 120,
    iters_som = SMOKE_GRID ? 5 : 40,
    wstar_iters = SMOKE_GRID ? 8 : 60,
    L = 1.0, D = 1.0, μ = 0.5, ε = IGDM_EPS,
    seed = 42,
)

const OUT_DIR = joinpath(@__DIR__, "out", TAG)
mkpath(OUT_DIR)
out_path(parts...) = joinpath(OUT_DIR, parts...)

n_policies(method) = length(policy_suite(method, maximum(GRID.Ns);
                                         L = GRID.L, D = GRID.D, μ = GRID.μ,
                                         ε = GRID.ε, nstarts = 1))

# ── phase 0: W*_N per (method, N over train∪test ∪ Ns) ────────────────────────
function phase0_tasks()
    horizons(method) = sort(unique(vcat(collect(2:(GRID.H_train_max + GRID.iota_max)),
                                        GRID.Ns)))
    return [(method, N) for method in GRID.methods for N in horizons(method)]
end

# ── phase 1: single-horizon designs per (method, policy, N) ───────────────────
# (starts are looped inside one task: they share the compiled PEP)
function phase1_tasks()
    return [(method, p, N) for method in GRID.methods
            for p in 1:n_policies(method) for N in GRID.Ns]
end

# ── phase 2: HRDP + anytime per (method, policy) ──────────────────────────────
function phase2_tasks()
    return [(method, p) for method in GRID.methods
            for p in 1:n_policies(method)]
end

compiler_for(method) = method_compiler(method; L = GRID.L, D = GRID.D,
                                       μ = GRID.μ, ε = GRID.ε)

suite_for(method, N) = policy_suite(method, N; L = GRID.L, D = GRID.D,
                                    μ = GRID.μ, ε = GRID.ε,
                                    nstarts = GRID.nstarts,
                                    rng = Random.Xoshiro(GRID.seed))

"""Task id from ARGS[1] or SLURM_ARRAY_TASK_ID; 0 → run all sequentially."""
function task_id()
    for a in ARGS
        a == "--local-all" && return 0
        tryparse(Int, a) !== nothing && return parse(Int, a)
    end
    return parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
end

run_tasks(f, tasks) = begin
    id = task_id()
    if id == 0
        for (i, t) in enumerate(tasks)
            @info "local-all: task $i/$(length(tasks))" t
            f(t)
        end
    else
        1 <= id <= length(tasks) || error("task id $id out of 1:$(length(tasks))")
        f(tasks[id])
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__) && !isempty(ARGS) && ARGS[1] == "counts"
    println("phase0=", length(phase0_tasks()))
    println("phase1=", length(phase1_tasks()))
    println("phase2=", length(phase2_tasks()))
end
