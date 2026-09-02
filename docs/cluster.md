# Cluster Guide — SLURM pipeline on the CECI clusters

Run the full experiment grid (all methods × policies × horizons × starts,
plus HRDP/anytime) as four dependency-chained SLURM array jobs.
See [manual.md](manual.md) for the API and [experiments.md](experiments.md)
for what each experiment computes.

---

## 1. Pipeline overview

```
 phase 0  (array over (method, N))          phase 1  (array over (method, policy, N))
 W*_N via free-baseline multistart          budgeted policy designs (exp1 data)
        │                                            │
        └────────────► phase 2 ◄─── needs W*_N       │
                (array over (method, policy))        │
                HRDP ω_H + anytime curves (exp2)     │
                       │                             │
                       └──────────► phase 3 ◄────────┘
                              single job: aggregate → summary_<method>.jld2
```

| Phase | Script | One array task = | Needs |
|---|---|---|---|
| 0 | `cluster/phase0_wstar.jl` | (method, N): W*_N by free-baseline multistart | — |
| 1 | `cluster/phase1_designs.jl` | (method, policy, N): all starts, FOM+SOM, budget curves | — |
| 2 | `cluster/phase2_hrdp.jl` | (method, policy): HRDP training + tailored/transfer/reuse | phase 0 |
| 3 | `cluster/phase3_aggregate.jl` | aggregation into `summary_<method>.jld2` | phases 1–2 |

All tasks are **idempotent**: each writes exactly one JLD2 file under
`PEPDesign/cluster/out/<tag>/` and skips itself if that file exists. Re-run a
failed array index at any time; delete an output file to force recomputation.

## 2. One-time setup on a CECI cluster

References: <https://www.ceci-hpc.be/clusters.html> and the CECI
documentation for your cluster (Lemaitre4, NIC5, Hercules2, …).

### 2.1 Code

```bash
ssh <cluster>
git clone <your-repo-url> ~/TANGO && cd ~/TANGO/WP1/Opt_Methods
```

### 2.2 Julia

Check what the cluster provides and adapt the `module load` line in the
`.slurm` files (they try `module load Julia` then
`module load releases/2023b Julia`):

```bash
module spider julia        # or: module avail 2>&1 | grep -i julia
```

If no module exists, install via juliaup in your home and replace the
`module load` lines with `export PATH=$HOME/.juliaup/bin:$PATH`.

### 2.3 Julia depot on scratch

Package precompilation caches must NOT live on the (slow, quota-limited)
home. The `.slurm` templates set

```bash
export JULIA_DEPOT_PATH=${GLOBALSCRATCH:-$HOME}/julia_depot
```

`$GLOBALSCRATCH` is defined on all CECI clusters. Create it once:
`mkdir -p $GLOBALSCRATCH/julia_depot`.

### 2.4 Mosek license

Copy your license to the cluster (`~/mosek/mosek.lic` by default) — for
floating/server licenses export the address instead:

```bash
# in ~/.bashrc, or leave the template default
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
```

Compute nodes must be able to read the file (home is shared — fine). Verify
before submitting anything big:

```bash
julia --project=. -e 'using Mosek; println("Mosek OK")'
```

### 2.5 Instantiate + precompile on the login node

**Important:** do this once before submitting, otherwise dozens of array
tasks race to precompile into the same depot:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
```

(`submit_all.sh` runs `Pkg.instantiate()` for you, but running it manually
first lets you catch license/module problems interactively.) If you still
see precompile contention in job logs, add
`export JULIA_PKG_PRECOMPILE_AUTO=0` to the templates.

## 3. Configuration: the registry

Everything is defined once in [`cluster/registry.jl`](../cluster/registry.jl):

```julia
GRID = (methods = [:OGD, :ITEM, :IGDM],
        Ns = [10, 20, 30, 40],          # phase-1 horizons
        H_train_max = 20, iota_max = 10, # HRDP train ≤ 20, test 21..30
        nstarts = 3,
        iters_fom = 120, iters_som = 40, wstar_iters = 60,
        L = 1.0, D = 1.0, μ = 0.5, ε = PEP_IGDM_EPS, seed = 42)
```

Environment variables (export before `sbatch`, or put in the templates):

| Variable | Default | Effect |
|---|---|---|
| `PEP_TAG` | `default` | output directory `cluster/out/<tag>/` — one tag per grid configuration |
| `PEP_SMOKE` | `0` | `1` → tiny grid for local testing |
| `PEP_IGDM_EPS` | `0.3` | IGDM inexactness ε for this run |

**Task-id stability rule.** Array task ids index the grids and the
`policy_suite` list (`experiments/common.jl`). For a tag that has already
produced outputs, **never reorder or remove** methods, policies, horizons —
append only, or start a fresh tag.

Print the array sizes (used by the submit script):

```bash
julia --project=. PEPDesign/cluster/registry.jl counts
# phase0=…  phase1=…  phase2=…
```

Full default grid: phase0 = 3 methods × 33 horizons ≈ 99 tasks (the union of
2..30 and {10,20,30,40}); phase1 = (10+5+6 policies) × 4 horizons = 84 tasks;
phase2 = 21 tasks.

## 4. Submitting

### 4.1 Everything at once

```bash
bash PEPDesign/cluster/slurm/submit_all.sh            # tag "default"
bash PEPDesign/cluster/slurm/submit_all.sh mytag      # custom tag
```

The script instantiates the environment, queries the registry for array
sizes, patches them into the templates, and chains the jobs:

- phase 0 and phase 1 start immediately (independent);
- phase 2 waits on phase 0 (`--dependency=afterok`);
- phase 3 waits on phases 1 and 2.

### 4.2 IGDM ε-grid

One tag per ε (results land in separate output directories):

```bash
PEP_IGDM_EPS=0.1 bash PEPDesign/cluster/slurm/submit_all.sh eps01
PEP_IGDM_EPS=0.3 bash PEPDesign/cluster/slurm/submit_all.sh eps03
PEP_IGDM_EPS=0.5 bash PEPDesign/cluster/slurm/submit_all.sh eps05
```

### 4.3 Manual submission / partial reruns

```bash
# rerun only phase-1 task 17 of tag "default"
PEP_TAG=default sbatch --array=17 <(sed 's/__NTASKS1__/84/' PEPDesign/cluster/slurm/phase1.slurm)

# or interactively on a compute node (srun) / the login node for tiny tasks:
PEP_TAG=default julia --project=. --threads=4 PEPDesign/cluster/phase1_designs.jl 17
```

Any phase script also accepts `--local-all` to loop over all its tasks
sequentially in one process.

## 5. Resource requests

Defaults in the templates (adjust per cluster limits):

| Phase | array | cpus/task | mem/cpu | time |
|---|---|---|---|---|
| 0 | ~99 | 4 | 4 GB | 02:00:00 |
| 1 | ~84 | 4 | 8 GB | 12:00:00 |
| 2 | ~21 | 8 | 4 GB | 12:00:00 |
| 3 | 1 | 1 | 4 GB | 01:00:00 |

Sizing guidance (laptop-measured, Mosek 1 thread):

- one OGD solve: N=20 ≈ 0.6 s, N=30 ≈ 4.5 s, N=40 ≈ 19 s;
- a phase-1 task ≈ nstarts × (iters_fom + ~3·iters_som) solves at one N —
  the N=40 tasks dominate (~3–8 h with defaults);
- phase-2 tasks solve across all horizons 2..30 (threaded — give them more
  CPUs);
- memory is modest (< 2 GB per solve at N=40); 4–8 GB/cpu is generous.

`--cpus-per-task` is forwarded to `julia --threads`; Mosek stays at 1 thread
per solve (`MosekBackend` default), so threads scale across horizons inside
HRDP evaluations and across FD probes.

## 6. Monitoring and results

```bash
squeue --me                               # queue state
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
ls PEPDesign/cluster/logs/                # pep_phase1_<taskid>.out
grep -l "ERROR" PEPDesign/cluster/logs/*.out   # find failed tasks
ls PEPDesign/cluster/out/<tag>/           # per-task JLD2 outputs
```

A finished tag contains `wstar_*.jld2`, `design_*.jld2`, `hrdp_*.jld2` and
the aggregated `summary_{OGD,ITEM,IGDM}.jld2`. Fetch to your machine:

```bash
rsync -av <cluster>:~/TANGO/WP1/Opt_Methods/PEPDesign/cluster/out/ PEPDesign/cluster/out/
```

then plot/tabulate locally (schemas match the experiment scripts — see
[experiments.md §5](experiments.md#5-using-cluster-outputs)).

### Resuming after failures

Because tasks skip existing outputs, the cheapest resume is to resubmit the
whole array — completed tasks exit in seconds:

```bash
PEP_TAG=default sbatch <(sed 's/__NTASKS1__/84/' PEPDesign/cluster/slurm/phase1.slurm)
```

then rerun phase 3.

## 7. Local end-to-end test

Before burning cluster hours, validate the whole pipeline on your machine
(~1 minute):

```powershell
$env:PEP_SMOKE = "1"
julia --project=. PEPDesign/cluster/registry.jl counts
julia --project=. --threads=4 PEPDesign/cluster/phase0_wstar.jl  --local-all
julia --project=. --threads=4 PEPDesign/cluster/phase1_designs.jl --local-all
julia --project=. --threads=4 PEPDesign/cluster/phase2_hrdp.jl    --local-all
julia --project=.             PEPDesign/cluster/phase3_aggregate.jl
Remove-Item Env:PEP_SMOKE
```

Outputs land in `cluster/out/smoke/`.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `MOSEK error 1008` / license failure in job logs | `MOSEKLM_LICENSE_FILE` wrong or unreadable from compute nodes; for IP-locked licenses request a CECI-server-tied academic license. Test with `srun --pty julia --project=. -e 'using Mosek'`. |
| `module: command not found` / no Julia module | Adapt the `module load` line (§2.2) or use a juliaup install in `$HOME`. |
| Many tasks stuck "precompiling" | Precompile on the login node first (§2.5); optionally `JULIA_PKG_PRECOMPILE_AUTO=0`. |
| `task id out of 1:…` | Array size doesn't match the registry (grid changed?). Re-generate with `registry.jl counts`; remember the append-only rule. |
| Phase 2 `missing phase0 output` | Phase 0 incomplete for the horizons 2..H_train_max+ι_max — check its logs, resubmit, keep the `afterok` chain. |
| TIMEOUT on N=40 phase-1 tasks | Raise `--time`, or split budgets (lower `iters_*` in the registry under a new tag). |
| OOM | Raise `--mem-per-cpu`; check with `sacct … --format=MaxRSS`. |
| Different results across tags | Expected if the grid changed — configs are embedded in every JLD2 (`grid` field) for auditing. |
| Want Slurm emails | Add `#SBATCH --mail-user=…` and `--mail-type=END,FAIL` to the templates. |
