# Experiments Guide

How to reproduce every numerical artifact of the paper with
`PEPDesign/experiments/`. See [manual.md](manual.md) for the API and
[cluster.md](cluster.md) for running the heavy grids on CECI.

---

## 1. Running

All scripts run from the repository root under the root environment:

```powershell
julia --project=. --threads=4 PEPDesign/experiments/exp1_policy_vs_free.jl [--smoke]
julia --project=. --threads=4 PEPDesign/experiments/exp2_hrdp_anytime.jl  [--smoke]
julia --project=. --threads=4 PEPDesign/experiments/exp3_ogd_sN.jl        [--smoke]
julia --project=. --threads=4 PEPDesign/experiments/exp4_ssdp_diagnostics.jl [--smoke]
```

- `--smoke`: tiny configuration (minutes, laptop) to check the pipeline.
- Without `--smoke`: the full paper configuration (N up to 40). exp1/exp2 at
  full size take hours locally — prefer the [cluster pipeline](cluster.md)
  and use the local scripts for plotting/tables.
- `--threads` matters: HRDP objectives and `compute_wstar` parallelize over
  horizons (Mosek itself is pinned to 1 thread).

## 2. Outputs

| Location | Content |
|---|---|
| `experiments/results/<name>_<cfghash>.jld2` | raw data + the config that produced it |
| `experiments/figures/<name>.{pdf,png}` | journal-style figures (Computer Modern, 300 dpi) |
| `experiments/tables/<name>.tex` | booktabs tables, ready for `\input` in `main.tex` |

Every artifact is tagged with `cfg_hash(CFG)` — an 8-hex-digit hash of the
experiment configuration. Changing any config field produces new file names;
stale artifacts are never silently overwritten by different configs.

## 3. The experiments

### exp1 — Parameterized policies vs free step-sizes (equal budget)

**Claim tested:** policies reach a good worst-case *much faster* (in SDP
solves) than the unrestricted coefficient array (Kamri-style baseline).

For each method (OGD, ITEM, IGDM) and horizon N ∈ {10, 20, 30, 40}: every
policy in `policy_suite` and the `free` baseline are run with identical
multistart budgets (FOM + SOM per start, sequentially); the **best-so-far W
vs cumulative SDP solves** envelope is recorded via
`DesignTrace.solves_hist`.

Outputs per (method, N): budget-curve figure (`free` dashed), JLD2 with all
curves, and a LaTeX table (policy, κ, best W, per-optimizer W, solves, wall
time). The paper `\input`s e.g. `tables/exp1_OGD_N40.tex`.

Notes: ITEM at N = 40 is skipped locally (cluster covers it); the config
block at the top of the script controls horizons, budgets and starts.

### exp2 — HRDP anytime improvement

**Claim tested:** a multi-horizon policy ω_H transfers to *unseen* horizons
better than reusing a single-horizon design.

For each method and each **horizon-consistent** policy family:

1. `W*_N` over train ∪ test horizons via the free baseline (multistart);
2. ω_H = HRDP minimizer over 𝓗_train = {2..N_train};
3. ω_N = tailored minimizer at N_train;
4. For M = N_train+1 … N_train+ι_max, compare normalized values
   `tailored W̄_M(ω_M) ≤ transfer W̄_M(ω_H) vs reuse W̄_M(ω_N)`;
5. Report GR (vs the full-horizon design) and WGC.

Full config: N_train = 20, ι_max = 10. Output: per-method anytime figure
(transfer/reuse/tailored per family) and the metrics table.

### exp3 — OGD 1/√N conjecture

For N ∈ {5, 10, …, 40}: optimize the constant policy (best of SOM/FOM),
report `s_N = (L√N/D)·c_N`, the constant-policy value, the free-baseline
value under the same budget, and their relative gap. The conjecture predicts
s_N → 1 and gap → 0. Outputs: `exp3_ogd_sN.{pdf,tex}`; smoke run already
gives s₅ ≈ 0.985 with a 1e-4 relative gap.

### exp4 — Second-order model diagnostics (paper §Second-Order Diagnostics)

On a fixed OGD instance:

1. directional FD errors of `dmat`/`d2mat` (exact, ~1e-15 — reported);
2. ∇W vs central FD away from kinks;
3. solver comparison SD / damped Newton / SSDP from identical starts
   (`tables/exp4_solver_comparison.tex`);
4. SSDP per-iteration diagnostics: predicted vs actual merit reduction,
   ζ_t, γ_t, KKT residual (`figures/exp4_ssdp_diagnostics.pdf`);
5. timing split PEP-solve / derivative assembly / tangent subproblem
   (`tables/exp4_timing.tex`).

## 4. Shared infrastructure (`common.jl`)

| Symbol | Purpose |
|---|---|
| `method_compiler(method; L, D, μ, ε)` | `N → CompiledPEP` factory per method symbol |
| `policy_suite(method, N; …)` | ordered `(label, policy, starts)` list; **order frozen** (cluster ids) — append only |
| `budget_run(dp; starts, iters_fom, iters_som)` | sequential multistart FOM+SOM; returns the best-so-far envelope vs cumulative solves |
| `perturbed_starts(base, n, rng)` | deterministic multistart generation (seeded) |
| `save_result`, `save_figure`, `write_latex_table` | artifact emission |
| `setup_plots!`, `PALETTE` | journal figure style |
| `is_smoke()` | `--smoke` CLI flag |

Reproducibility: all randomness flows through seeded `Random.Xoshiro`
generators in the configs; identical configs ⇒ identical artifacts (up to
solver nondeterminism at the 1e-8 level).

## 5. Using cluster outputs

The [cluster pipeline](cluster.md) writes per-task JLD2 files and
`summary_<method>.jld2` aggregates under `PEPDesign/cluster/out/<tag>/`.
Their schemas match the local experiment outputs (`bests`/`solves` curves,
`tailored`/`transfer`/`reuse` vectors, `Wstar` dicts), so the plotting and
table code in the experiment scripts can be pointed at them — load with

```julia
using JLD2
s = load("PEPDesign/cluster/out/default/summary_OGD.jld2", "summary")
s["Wstar"], s["designs"], s["hrdp"]
```
