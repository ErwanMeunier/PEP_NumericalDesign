# Experiments Guide

How to reproduce every numerical artifact of the paper with
`experiments/`. See [manual.md](manual.md) for the API and
[cluster.md](cluster.md) for running the heavy grids on CECI.

---

## 1. Running

All scripts run from the repository root under the root environment:

```powershell
julia --project=. --threads=4 experiments/exp1_policy_vs_free.jl [--smoke]
julia --project=. --threads=4 experiments/exp2_hrdp_anytime.jl  [--smoke]
julia --project=. --threads=4 experiments/exp3_ogd_sN.jl        [--smoke]
julia --project=. --threads=4 experiments/exp4_ssdp_diagnostics.jl [--smoke]
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

The transfer-beats-reuse claim does not hold in general. See exp9. For OGD,
the window designs in the constant, power-law and log-poly2 families are
worse than reuse at every N in 9..40. The c/√n window designs beat reuse
only for N ≥ 35. For IGDM, both window designs are worse than reuse at all
four sampled test horizons.

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

### exp5 — Reuse a learned mapping at larger horizons

This standalone, compute-only experiment uses the root project and standard
library TOML output; it does not load the plotting stack in `common.jl`.

```powershell
julia --project=. experiments\exp5_horizon_reuse.jl --smoke
julia --project=. experiments\exp5_horizon_reuse.jl 10 11,15 4 identity
julia --project=. experiments\exp5_horizon_reuse.jl 10 11,15 4 power_law
```

Positional arguments are `N_train`, comma-separated target horizons, fixed
memory K, and scalar base family (defaults: `10 11,15 4 identity`).
Families: `identity`, `constant`, `power_law`, `sum_of_exp`, `log_poly`,
`dct`, `cosine_decay`, `rational`, `warped_chebyshev`, `piecewise_exp`.
The family parameterizes **lag**, using `IGDMStationaryPolicy(K, base)`.
The public wrapper also accepts custom `FunctionPolicy` mappings.

The experiment:

1. Designs the stationary mapping at N_train using SLM.
2. Keeps omega and K unchanged at N_train and every requested target.
3. Assesses `W(alpha(omega_N, M), M)` with one PEP per horizon.
4. Independently optimizes free coefficients in the **same K-memory band**
   from two diagonal starts (steps 1 and `igdm_hmem(0.3)`).
5. Reports reused/free-best values, their ratio, separate assessment and
   free-design costs, and the one-off source-training cost.

IGDM uses L=D=1, epsilon=0.3, `:min_grad`, and substitution encoding.
Default budget: 100 SLM iterations / 300 seconds **per start** (soft cap).
Smoke: N_train=3, targets=4,5, K=1, identity base, 8 iterations / 60 seconds.
These are local diagnostic runs, not a statistically replicated HPC study.

Each execution creates a unique `experiments/results/reuse_IGDM_*/`
directory containing `assessment.toml`: configuration, trained omega,
training/free traces, reused coefficient arrays, numerical certificate
diagnostics, timings, counts, and comparison ratios. This permits later
plotting without rerunning optimization. No existing results are replaced.

The free values are **best found**, not proven global optima. Ratios below
1 are possible and retained. Good transfer is the hypothesis being tested,
not a guarantee. Training is a one-off investment; generating steps for
deployment needs no SDP, whereas *certifying their quality* costs one PEP
per target. To assess other methods or externally learned weights directly,
use `assess_transfer` as documented in the
[manual](manual.md#reusing-a-single-horizon-design-without-redesign).

The legacy `policy_suite(:IGDM, ...)` also offers stationary constant,
power-law, and DCT bases (appended after its existing entries). Use a new
cluster tag when changing the policy grid of an already submitted study.

### exp7 — Amortized transfer with optional free refinement

```powershell
julia --project=. experiments\exp7_amortized_reuse.jl --smoke
julia --project=. experiments\exp7_amortized_reuse.jl 5 6,8,10 2 8
julia --project=. experiments\plot_amortized_reuse.jl <path-to-amortized.toml>
```

This diagnostic trains a stationary-by-lag IGDM mapping once, transfers its
unchanged weights to multiple longer horizons, and compares:

1. reused weights without redesign (one PEP for offline certification);
2. those weights as the initialization of free SLM within the same K-memory
   band; and
3. cold-start free SLM in that same band, from the matched diagonal default.

The cumulative amortization calculation charges source compilation/design
once, then target transfer assessment plus warm refinement per horizon.
Cold free SLM is charged per target. Target compilation is shared and
reported separately. `amortized.toml` contains all values, traces,
certificates, timing, compilation costs, and cumulative budgets. The plotter
creates per-target objective ratios to cold-free quality and cumulative
offline-cost/break-even plots, plus a CSV summary.

Only the no-redesign transfer manufactures steps at essentially zero
deployment compute cost; PEP assessment is an *offline verification cost*.
An optional refinement uses free-design time and is charged separately.
Source training is amortized across the requested target horizons but is not
hidden. Cold and warm designs use one start and the same iteration cap.
These local SLM values are best-found, not global-optimality certificates.

The smoke run also includes a full-free cold baseline; its higher dimension
is intentionally excluded from the K-matched cost ratio. The regular run
focuses on fixed K to keep the diagnostic bounded. This experiment is
evidence about reuse/refinement economics, distinct from exp6's
matched-iteration prior-versus-free curves.

### exp8 — Sensitivity to the source design initialization

```powershell
julia --project=. experiments\exp8_transfer_repeatability.jl --smoke
julia --project=. experiments\exp8_transfer_repeatability.jl 10 15,20 2 8
julia --project=. experiments\plot_transfer_repeatability.jl <path-to-repeatability.toml>
```

Repeat stationary K-band source training from diagonal starts h in
{0.9,1.0,1.1}, reuse each trained mapping at the specified targets, and
free-refine each transfer for the stated update cap. A single cold-start
free run per target (diagonal h=1) is paired with all source fits. This
tests whether observed warm-start gains depend on one specially selected
source initialization. Reports include per-fit and aggregated
source-sensitivity ratios, complete traces, and certificate diagnostics.
Compilation is shared and reported separately; optimizer timings include
final value verification. See `paired_summary.csv` for per-target mean,
range, and number of positive warm-start advantages over the source fits.

The same caution applies: these are deterministic perturbation sensitivity
checks, not random-sample confidence intervals. Values are local
best-found SLM outputs. Cold-start ratios are for one specified
initialization, not the best of an arbitrarily large multistart campaign.

### exp9 — Anytime robustness

```powershell
julia --project=. experiments\exp9_anytime.jl --smoke
julia --project=. experiments\exp9_anytime.jl ogd
julia --project=. experiments\exp9_anytime.jl igdm
julia --project=. experiments\plot_anytime.jl <path-to-anytime.toml>
# OGD only (exact normalizer): every design at every horizon, plus checks
julia --project=. experiments\exp9_full_range.jl <ogd-anytime.toml> --horizons 2:40 --free-check 2,3,4,6,8,12,16 --workers 4
julia --project=. experiments\exp9_full_range.jl <ogd-anytime.toml> --horizons 48 --workers 3 --only <key1,key2,...>
julia --project=. experiments\plot_anytime.jl <study>\full_range.toml [--only <key1,key2,...>]
```

Checks the HRDP claim on prefix-consistent priors by plotting
`W_N(α(ω))/W*_N` against N, both on design horizons and on unseen ones.
It compares three kinds of design:

- **reuse**: the schedule tailored to N0.
- **hrdp_mean**: minimizes the paper's σ, the mean ratio over the design set.
- **hrdp_max**: minimizes the worst ratio over the design set, i.e. the
  anytime competitive ratio on that set.

The design set is either the window {2..N0} or a geometric grid.

- **OGD** uses the normalizer `W*_N = √N`.
  - N0 = 8. Families: constant, c/√n, power law, log-poly2.
  - Fixed references: the textbook 1/√n and the classical anytime step
    1/√(2n).
  - Design sets: the window {2..8} and the geometric grid {2,4,8,16}.
  - exp9 evaluates 14 sampled horizons: 2..8, 10, 12, 16, 20, 24, 32, 40.
- **IGDM** uses the stationary K=2 prior.
  - The normalizer is the best known stationary value per N: an SLM
    multistart, then the minimum with every evaluated design.
  - N0 = 10. Design horizons 2..10; sampled test horizons 15, 20, 25, 30.

Priors have at most three parameters. They are optimized by multistart
Nelder–Mead on certified PEP values only, so results do not depend on any
gradient-solver tuning. Outputs are `anytime.toml`, `anytime_ratios.csv`
(with an `in_trainset` flag), and `anytime_ratios.{pdf,png}`.

`exp9_full_range.jl` re-evaluates an OGD study at every requested horizon
on a pool of Distributed workers.

- It adds one reference design: reuse of the tailored constant at N = 16,
  the largest grid horizon.
- It checks the normalizer at every horizon by evaluating `W_N(1/√N)`.
- `--free-check` runs six free-step SLM runs per horizon: three starts ×
  Δ0 ∈ {1, 0.1}, 80 iterations each, certified values only.
- It is resumable. Results accumulate in `full_range.toml`, which uses the
  `anytime.toml` schema; certified exp9 values are reused, not recomputed.
- Tables: `full_range_ratios.csv`, `full_range_summary.csv` and
  `free_check.csv`. The summary gives the worst ratio over the largest
  complete block and the wins against reuse beyond N0.
- Design keys are `family/kind/trainset`, e.g.
  `power_law/hrdp_max/geometric`.
- Cost: one certified OGD PEP takes about 105 s at N = 40 and 275 s at
  N = 48 (single-threaded Mosek). A worker peaks near 3.3 GB at N = 48.

Observed results (local runs, Julia 1.12, Mosek). OGD numbers come from
`experiments/results/anytime_ogd_zPmohW/full_range_*`: all 15 designs at
every N in 2..40 (585 certified values), plus N = 48 for five of them. IGDM
numbers come from `experiments/results/anytime_igdm_I0QK9N` and are on the
sampled horizons only.

- **Normalizer.** `W_N(1/√N) = √N` to within 2.3e-8 relative at all 40
  evaluated horizons, so √N is attained.
  - The free-step SLM check never went below √N at N ∈ {2,3,4,6,8,12,16}:
    best/√N ≥ 1 − 1e-8 over 6 runs per horizon.
  - Elsewhere √N is the value of the tailored constant, an upper bound on
    W*_N. Comparisons between designs at the same N do not depend on the
    normalizer.
- **Window HRDP does not transfer beyond N0.** This holds for both the mean
  and the max objective.
  - OGD: the constant, power-law and log-poly2 window designs are worse than
    reuse at all 32 horizons in 9..40.
  - OGD: the c/√n window designs settle at c ≈ 0.68–0.70, i.e. the classical
    1/√(2n). They beat reuse only for N ≥ 35 (6/32), as does the classical
    step itself. This is the slow growth of a decaying family, not transfer
    from the window.
  - IGDM: both window designs are worse than reuse at 15, 20, 25 and 30.
- **Why: the tailored optimum drifts monotonically with N.** For OGD it is
  1/√N; for IGDM the lag weights grow with N. A window objective therefore
  pulls the design toward short horizons.
- **Flexible families collapse to a constant on the window.** With the
  mean objective, power-law and log-poly2 return exactly the constant step
  0.462.
  - With the max objective they can extrapolate even worse; log-poly2
    reaches 1.96 at N = 40, versus 1.34 for reuse.
- **What HRDP does deliver is uniformity inside the design range.**
  - OGD worst ratio on {2..8}: 1.164 for reuse; for hrdp_max, 1.061
    (constant), 1.053 (power law) and 1.044 (log-poly2). The best c/√n
    design found reaches only 1.166.
  - IGDM worst ratio on {2..10}: 1.114 for reuse, 1.015 for hrdp_max.
- **Minimax on the geometric grid with a decaying family gives the best
  worst-case ratio over 2..40.** This is OGD with the power law
  η_n = a/(n^b + c) fitted on {2,4,8,16}:

  | Design | Worst ratio, N = 2..40 (at N) | Worst ratio, N = 2..8 | Beats reuse@8 on 9..40 | Ratio at N = 48 |
  |---|---|---|---|---|
  | power law, minimax on {2,4,8,16} | **1.195** (40) | 1.087 | 21/32 (N ≥ 20) | 1.230 |
  | classical 1/√(2n) | 1.291 (40) | 1.167 | 6/32 (N ≥ 35) | 1.301 |
  | c/√n, any HRDP design | 1.291 (40) | 1.166–1.167 | 6/32 (N ≥ 35) | — |
  | reuse of the N = 16 schedule | 1.299 (3) | 1.299 | 29/32 (N ≥ 12) | 1.155 |
  | reuse of the N = 8 schedule | 1.342 (40) | 1.164 | — | 1.429 |
  | textbook 1/√n | 1.368 (40) | 1.232 | 0/32 | — |
  | constant, minimax on {2,4,8,16} | 1.489 (40) | 1.125 | 0/32 | 1.595 |

  No design dominates. The power law is the only one within 20% of the
  tailored value at every N in 2..40. Reusing the N = 16 schedule beats the
  power law at every N ≥ 8, including N = 48, but costs up to 30% at small
  N (1.299 at N = 3).
- **No asymptotic guarantee.** The fitted power law (a, b, c) = (0.0371,
  0.0213, −0.920) decays roughly like 1/log n. Its ratio rises by 0.035
  from N = 40 to 48, against 0.010 for the classical step. A grid that stops
  at 16 cannot enforce behavior far beyond it.

### exp10 — IGDM: shape of the horizon set

```powershell
julia --project=. experiments\exp10_horizon_shape.jl --smoke
# tailored normalizer first, with few workers (memory), then resume
julia --project=. experiments\exp10_horizon_shape.jl --out <dir> --workers 2 --seed <igdm anytime.toml> --tailored-only
julia --project=. experiments\exp10_horizon_shape.jl --out <dir> --workers 3
julia --project=. experiments\plot_horizon_shape.jl <dir>\shape.toml
```

This study keeps exp9's IGDM prior and optimizer fixed and varies only the
design horizon set H.

- **Setup.** Stationary-by-lag K = 2 (3 parameters), min_grad, ε = 0.3.
  - Each design ω_H minimizes the max (or mean) of `W_N(ω)/W*_N` over H.
    It is found by exp9's multistart Nelder–Mead on certified values.
  - Each design is then evaluated at every N in 2..48. Designs use
    H ⊂ [2, 32]; N = 33..48 measures extrapolation.
- **Normalizer.** W*_N is the tailored value within the same prior.
  - Per-N SLM, warm-started from an interpolation of exp9's tailored ω.
  - Then the minimum with every design's value. No design improved it
    (0/47 horizons).
- **Three factors.**
  - Shape at a fixed extent [2, 32]: window, geometric {2,4,8,16,32},
    arithmetic, fine geometric {2,3,4,6,8,11,16,23,32}, three points, ends.
  - Objective: mean vs max.
  - Extent: [2,8], [2,16], [4,32], [8,32], {2..10}, {16..32}, and
    single-horizon reuse.
- **Optimality cross-check.** Every design is scored on every other set's
  criterion. A design beaten by more than 1e-5 relative would be
  re-optimized; this never happened.
- **Exchange stage (Remez-type).**
  - Start from the geometric minimax design.
  - Add the worst horizon of the dense grid 2..32 to H, then re-solve.
  - Stop when the dense worst case equals the worst case on H.
- **Memory.** A tailored SLM at N ≈ 45 needs about 7–9 GB per worker, and a
  PEP evaluation at N = 48 about 3–4 GB. Hence the 2-worker
  `--tailored-only` pass.
- **Outputs** (in `experiments/results/igdm_horizon_shape/`):
  - `shape.toml`
  - `shape_ratios.csv`, `shape_summary.csv`, `shape_cross.csv`,
    `shape_exchange.csv`
  - `shape_panels.{pdf,png}`

Observed results. "In" means N ∈ 2..32 and "beyond" means N ∈ 33..48; the
worst ratio beyond is always reached at N = 48. "Active" lists the horizons
of H within 2e-4 of the design's worst ratio on H.

| Design (objective) | H | PEP solves | Worst on H | Worst in (at N) | Mean in | Worst beyond | Active |
|---|---|---|---|---|---|---|---|
| window (max) | {2..32} | 14012 | 1.0477 | **1.0477** (3) | 1.0238 | 1.0660 | {2,3,32} |
| fine geometric (max) | {2,3,4,6,8,11,16,23,32} | 4068 | 1.0477 | **1.0477** (3) | 1.0238 | 1.0660 | {2,3,32} |
| exchange (max) | {2,3,4,8,16,32} | 5009 (cumulative) | 1.0477 | **1.0477** (32) | 1.0238 | 1.0660 | {2,3,32} |
| geometric (max) | {2,4,8,16,32} | 2260 | 1.0402 | 1.0854 (3) | 1.0233 | 1.0568 | {2,4,32} |
| geometric (mean) | {2,4,8,16,32} | 2155 | 1.0245 | 1.0558 (32) | 1.0278 | 1.0751 | — |
| window (mean) | {2..32} | 13609 | 1.0199 | 1.1064 (2) | **1.0199** | 1.0457 | — |
| arithmetic (max) | {2,9,17,24,32} | 2265 | 1.0326 | 1.1744 (3) | 1.0378 | 1.0447 | {2,9,32} |
| arithmetic (mean) | {2,9,17,24,32} | 2260 | 1.0241 | 1.1206 (3) | 1.0256 | 1.0503 | — |
| three points (max) | {2,8,32} | 1356 | 1.0328 | 1.1627 (3) | 1.0344 | 1.0456 | {2,8,32} |
| ends (max) | {2,32} | 906 | 1.0326 | 1.1801 (3) | 1.0397 | 1.0443 | {2,32} |
| geometric (max) | {2,4,8,16} | 1808 | 1.0209 | 1.0552 (32) | 1.0284 | 1.0743 | {2,4,16} |
| geometric (max) | {2,4,8} | 1359 | 1.0104 | 1.0734 (32) | 1.0379 | 1.0945 | {2,4,8} |
| window (max) | {2..10} | 4068 | 1.0152 | 1.0752 (32) | 1.0377 | 1.0967 | {2,3,10} |
| geometric (max) | {4,8,16,32} | 1828 | 1.0354 | 1.1081 (2) | 1.0206 | 1.0519 | {4,32} |
| geometric (max) | {8,16,32} | 1356 | 1.0168 | 1.1670 (2) | 1.0234 | 1.0291 | {8,32} |
| window (max) | {16..32} | 7769 | 1.0037 | 1.2827 (2) | 1.0416 | 1.0102 | {16,32} |
| reuse of N = 10 | {10} | 0 | 1 | 1.1141 (2) | 1.0210 | 1.0531 | — |
| reuse of N = 16 | {16} | 0 | 1 | 1.1896 (2) | 1.0257 | 1.0253 | — |
| reuse of N = 32 | {32} | 0 | 1 | 1.4249 (4) | 1.0684 | 1.0018 | — |

Findings:

- **Shape acts through the active horizons, not through density.** The
  dense minimax over {2..32} is pinned by only three horizons, {2,3,32}: as
  many as the prior has parameters.
  - The tested sets containing {2,3,32} give the same design (not a general
    guarantee: the minimax over {2,3,32} need not be unique, and the
    optimizer is local). The fine geometric
    set reproduces it exactly using 29% of the PEP solves. The exchange
    design matches it to within 1.7e-5 in ratio at every N in 2..48.
- **The small-N end is where the tailored optimum moves fastest.** The third
  lag weight goes 0 → 0.005 → 0.042 over N = 2..4.
  - Sets without N = 3 underestimate their worst case on 2..32: geometric
    1.040 on H vs 1.085 at N = 3; arithmetic, three points and ends
    1.16–1.18 at N = 3.
  - Mid-range horizons matter much less. Over 6..24 every [2,32] design
    stays below 1.07, while at N = 3 the designs range from 1.048 to 1.180.
- **A coarse set is cheap but optimistic.** The value of the criterion on H
  is a valid certificate only on H. It must be checked on the dense grid.
  - The exchange stage fixes this in one step. It adds N = 3 and stops,
    at 5009 cumulative PEP solves versus 14012 for the dense window.
- **Mean vs max trades the worst case for the average.**
  - On the window: worst in 1.106 vs 1.048, mean in 1.020 vs 1.024.
  - The geometric mean design has a better worst case on 2..32 than the
    geometric max design (1.056 vs 1.085). This is only because the max
    design ignores N = 3.
- **Extent sets the two ends.**
  - The lower end of H controls the small-N worst case: 1.11–1.28 at N = 2
    when H starts at 4 or more.
  - The upper end controls extrapolation: 1.07–1.10 at N = 48 when H stops
    at 8–16, and 1.01–1.08 when it stops at 32.
  - Single-horizon reuse is the extreme case: near-optimal around its
    horizon, and up to 1.42 at small N for N0 = 32.
- **Best full-range worst case.** Over 2..48 it is 1.066, from the dense,
  fine-geometric and exchange minimax designs, attained at N = 48. Designs
  with a better worst case beyond 32 (ends 1.044, reuse@32 1.002) pay
  1.18–1.42 at small N.
- **Caveats.**
  - Ratios are relative to the best value within the prior, not to the
    unrestricted (free-step) optimum.
  - Nelder–Mead is local. The cross-check rules out any design beating
    another on its own criterion, but not better unseen designs.
  - Recorded design wall-times are not comparable (parallel runs on a shared
    machine); use the PEP-solve counts.

### exp11 — IGDM: memory size K and anytime

```powershell
julia --project=. experiments\exp11_memory_anytime.jl --smoke
julia --project=. experiments\exp11_memory_anytime.jl --workers 3 --seed <exp10 shape.toml>
julia --project=. experiments\exp11_memory_anytime.jl --out <dir> --workers 3   # resume
julia --project=. experiments\plot_memory_anytime.jl <dir>\memory.toml
```

Question: does a larger memory K improve the anytime behavior of IGDM?

- **Setup.** As in exp10: min_grad, ε = 0.3, stationary-by-lag prior
  β_{n,n−ℓ} = ω_ℓ for ℓ = 0..K. K ∈ {0,1,2,3,4,6,8,12}, design grid
  2..32, extrapolation at N = 40 and 48. The families are nested (K ⊂ K+1).
- **Tailored values W^K_N** (stage 1).
  - Per-N SLM, chained over K: each K starts from the previous K's optimum
    padded with zeros. The exp10 K = 2 optimum is an extra seed.
  - W^ref_N is the minimum over K of W^K_N and of all design values. No
    design improved it (0/33 horizons), so W^ref_N = W^{12}_N.
- **Anytime designs** (stage 2). There is one ω per K and per normalizer:
  min_ω max_{N=2..32} W_N(α^K(ω)) / Ŵ_N.
  - *common*: Ŵ_N = W^ref_N. This is the absolute quality, comparable
    across K. The designs are chained over K, so the worst ratio cannot
    increase with K.
  - *within*: Ŵ_N = W^K_N. This is the price of not knowing N inside the
    prior K.
- **Solver.** A trust-region sequential LP on the epigraph replaces
  Nelder–Mead. It uses certified PEP values and envelope gradients at all
  31 grid horizons, with 2–3 starts per design.
  - It reproduces exp10's window_2_32/max design: f = 1.047726 and
    ω = [1.2595, 0.1636, 0.0943], versus 1.04773 and
    [1.2603, 0.1632, 0.0942].
  - It needs about 1–2 × 10³ PEP solves per start, versus 14012 for
    Nelder–Mead in exp10.
- **Memory.** The tailored SLM at N = 48 needs about 10 GB per worker. The
  extra-horizon stage uses `--big-workers` (default 2). The full run took
  about 6 h with 3 workers on a shared machine.
- **Outputs** (in `experiments/results/igdm_memory_anytime/`):
  - `memory.toml`
  - `memory_tailored.csv`, `memory_ratios.csv`, `memory_summary.csv`
  - `memory_panels.{pdf,png}`

Observed tailored ratios W^K_N / W^ref_N:

| N | K=0 | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|---|---|
| 4 | 1.074 | 1.006 | 1.001 | 1 | 1 | 1 | 1 |
| 8 | 1.161 | 1.043 | 1.014 | 1.008 | 1.008 | 1.004 | 1 |
| 16 | 1.270 | 1.104 | 1.048 | 1.026 | 1.014 | 1.009 | 1.009 |
| 32 | 1.468 | 1.248 | 1.145 | 1.083 | 1.045 | 1.010 | 1.001 |
| 48 | 1.631 | 1.375 | 1.240 | 1.153 | 1.096 | 1.037 | 1.011 |

Observed anytime designs. "Worst common" and "mean common" are taken over
2..32 against W^ref; "worst within" is taken over 2..32 against W^K; the
last two columns are extrapolation against W^ref.

| K | worst common (N) | worst within (N) | mean common | at N = 40 | at N = 48 | tailored worst common |
|---|---|---|---|---|---|---|
| 0 | 1.468 (32) | 1.000 (2) | 1.276 | 1.554 | 1.631 | 1.468 |
| 1 | 1.248 (32) | 1.025 (32) | 1.140 | 1.315 | 1.376 | 1.248 |
| 2 | 1.163 (2) | 1.048 (3) | 1.086 | 1.221 | 1.274 | 1.145 |
| 3 | 1.130 (32) | 1.066 (32) | 1.067 | 1.184 | 1.234 | 1.083 |
| 4 | 1.111 (4) | 1.078 (32) | 1.057 | 1.162 | 1.209 | 1.045 |
| 6 | 1.089 (5) | 1.082 (6) | 1.052 | 1.134 | 1.177 | 1.030 |
| 8 | 1.077 (7) | 1.076 (8) | 1.051 | 1.118 | 1.157 | 1.027 |
| 12 | 1.068 (13) | 1.068 (13) | 1.051 | 1.104 | 1.140 | 1 |

Columns: worst common, mean common, N = 40 and N = 48 come from the
*common* design; worst within comes from the *within* design. The last
column is the tailored family, which knows N.

Findings:

- **Yes: more memory improves anytime, with diminishing returns.**
  - Against a fixed reference, the worst case on 2..32 falls monotonically:
    1.468 → 1.248 → 1.163 → 1.130 → 1.111 → 1.089 → 1.077 → 1.068.
  - Most of the gain comes from K ≤ 2. Beyond K = 4, each doubling of K
    gains about 2–3 points.
  - Extrapolation to N = 40 and 48 also improves monotonically (at 48:
    1.63 → 1.14). It stays worse than inside the grid.
- **One horizon-free design with more memory beats tailoring with less
  memory.**
  - The K = 3 anytime design is better than the K = 2 tailored family in
    worst case (1.130 vs 1.145). It is better pointwise at 22/31 horizons,
    losing only at small N.
  - The K = 6 anytime design is roughly as good as K = 3 tailored
    (1.089 vs 1.083).
- **The price of anytime within a prior grows with K, then saturates.**
  It goes 1.000 (K = 0: ω₀ ≈ 1.238 is almost horizon-free), 1.025, 1.048,
  1.066, 1.078, 1.082, 1.076, 1.068.
  - Larger families let the tailored optimum adapt more to N.
- **The bottleneck moves.**
  - For K ≤ 3 the binding horizon of the common design is N = 32: there is
    not enough memory to be good at large N.
  - For K ≥ 4 it moves to N ≈ K+1 (N = 4, 5, 7, 13).
  - When N ≤ K+1, the prior covers every lag. The tailored weights then put
    a negative correction on the largest lags (e.g. −0.08 on lag 12 at
    N = 13, K = 12), whereas large N wants positive, decaying lag weights.
  - One ω cannot do both. This sign conflict produces the 7–8% plateaus in
    panel (c), and the kink of W^ref at N = 13 = K_max + 1 in panels (a)
    and (b).
- **Shape of the anytime weights.** The designs are stable across K:
  ω₀ ≈ 1.245, then lag weights 0.164, 0.11, 0.07, 0.05, and a slowly
  decaying tail of about 0.04–0.01 (K = 12).
- **Caveats.**
  - The common minimax is degenerate for K ≤ 1. Its value equals the
    tailored worst case at N = 32, which no ω can improve, so it does not
    determine ω at other N. Use the within design there.
  - SLM and SLP are local optimizers, and W^ref is a best-known value.
    Free (time-varying) memory is far better still: in the pilot, free
    full-memory SLM at N = 12 reaches 0.59 W^0_N, versus 0.83 W^0_N for
    stationary K = 6.
  - The common/K8 second start took 4301 s because of paging on the shared
    machine; use the PEP-solve counts.

### exp6 — Targeted memory-size exploration

The paired screening grid is N in {10,20}, K in {0,1,2,4,N-1},
epsilon=0.3, with free K-band SLM, stationary-by-lag SLM, and lag-DCT1/3
SLM. All four start from **exactly the same coefficient array** (diagonal
step 1, other entries zero). Screening uses 16 outer updates, with a
120-second soft safety cap per run.

```powershell
julia --project=. experiments\exp6_memory_study.jl --smoke
julia --project=. experiments\exp6_memory_study.jl --screen
julia --project=. experiments\plot_memory_study.jl <path-to-study.toml>
julia --project=. experiments\summarize_memory_study.jl <path-to-study.toml>
# Follow a promising screen with three common initial steps and a K=0 control:
julia --project=. experiments\exp6_memory_study.jl --confirm 20 4 lag_dct1
```

Confirmation uses initial steps 1, 0.9, 1.1, 40 updates / 120 seconds per
run, comparing the requested prior with free coefficients both at the
selected K and at K=0. These are initialization-sensitivity checks, not
independent samples of a population. Warmup is done separately for every
policy type; PEP compilation is excluded and reused. Run order is shuffled
deterministically; solves run sequentially with BLAS/Mosek each on one
thread. Avoid running other CPU-intensive jobs during timing measurements.

Data are atomically saved after each completed run in a unique
`results/memory_<stage>_*/study.toml`, with all traces, statuses, final
certificates, initial steps, nominal/effective parameter dimensions and
timings. `require_certified=true` makes SLM reject uncertified PEP values.
A final verification solve per run is reported separately and is not
included in the optimization budget.

The plotting step writes PDF/PNG figures for both requested budget axes:

```
A_N,K(b) = (best_free_N,K(b) - best_prior_N,K(b)) / best_free_N,K(b)
b = elapsed wall time tau, or outer design updates T
```

Positive means a prior advantage, negative a free-step advantage. Initial
evaluation is T=0; rejected SLM steps still consume updates. Every curve
uses the best incumbent available at the budget, never future information.
No trace is extrapolated past its measured endpoint. The plot panels share
vertical scales, show the zero line, and retain negative cases.
`matched_curves.csv` contains the actual aligned values used in the plots.
Use `relative_advantage(free_trace, prior_trace; axis=:time)` or
`axis=:iterations` directly to build other plots.

Fresh-design and horizon-reuse studies answer different questions and are
kept separate. Do not interpret these empirical best-so-far values as
global optima. A promising screen is a candidate for confirmation, not a
license to omit losses. Consider time-to-target, terminal quality, and
parameterization-dependent trust-region scaling before concluding why a
prior helps.

The summary ranks cases by mean signed advantage over design budgets
T=4:16 and also reports T=4,8,12,16 and time-integrated signed advantage
over each pair's common observed interval. `observed_T_budgets` records
censoring when a run ends before T=16; missing budgets are never invented.
Treat these rankings as exploratory selection and use separate confirmation
runs, including the K=0 control. Do not pool the best of many tried priors
without accounting for the search/tuning budget.

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
`summary_<method>.jld2` aggregates under `cluster/out/<tag>/`.
Their schemas match the local experiment outputs (`bests`/`solves` curves,
`tailored`/`transfer`/`reuse` vectors, `Wstar` dicts), so the plotting and
table code in the experiment scripts can be pointed at them — load with

```julia
using JLD2
s = load("cluster/out/default/summary_OGD.jld2", "summary")
s["Wstar"], s["designs"], s["hrdp"]
```
