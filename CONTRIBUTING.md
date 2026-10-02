# Contributing to PEPDesign

PEPDesign is a public numerical toolbox. Changes must preserve mathematical
meaning, public API compatibility where practical, and reproducible solver
diagnostics. Please keep pull requests focused and explain any change to a
PEP formulation, sign convention, or numerical tolerance.

## Development setup

Use Julia 1.10 or newer and a working Mosek license:

```powershell
git clone <repository-url> PEP_NumericalDesign
cd PEP_NumericalDesign
julia --project=. -e "import Pkg; Pkg.instantiate()"
julia --project=. --threads=4 dev/run_all_tests.jl
```

The repository root is the Julia environment. Development scripts must
activate `joinpath(@__DIR__, "..")`; do not assume that this repository is
nested in a larger project.

## Architecture and conventions

- `src/params.jl`, `gram.jl`, and `compile.jl` define the symbolic and
  compiled PEP representation. Keep point coefficients affine and encode
  nested parameter products with explicit residual equalities.
- `src/facial_reduction.jl` may only remove a Gram-cone face when an existing
  pure-Gram equality provides a PSD exposing matrix. It must remain opt-in.
- `src/solve.jl` owns backend behavior, dual sign normalization, primal
  reconstruction, and certification. A solver termination status alone is
  not a numerical certificate.
- `src/sensitivity.jl` and `src/design/` consume the original constraint
  ordering. Never silently manufacture dual multipliers for eliminated rows.
- Built-in method parameters and problem constants must be explicit; compile
  once per method/horizon and reuse the `CompiledPEP`.
- Mosek defaults to one thread. Parallelism belongs at the Julia level over
  independent solves.

Keep exported identifiers in `src/PEPDesign.jl`. Add docstrings to public
APIs and update both `README.md` and `docs/manual.md` when behavior changes.

## Validation

Run the complete compatibility suite before submitting:

```powershell
julia --project=. --threads=4 dev/run_all_tests.jl
```

For a focused facial-reduction change, run this first:

```powershell
julia --project=. test/runtests.jl
# or: julia --project=. dev/test_facial_reduction.jl
```

New numerical code needs tests for the ordinary case, boundary/zero-rank
case, invalid or indefinite input, reconstruction in the original space,
and compatibility with the unreduced path. Compare values and invariants;
avoid assertions on solver-specific iteration counts.

When changing sensitivities, validate assembled first and second derivatives
and compare gradients with finite differences away from nonsmooth points.
When changing a formulation, retain parity checks against a known tight rate
or the frozen legacy implementation.

## Facial-reduction safety rules

The `:explicit` reducer implements the implication
`tr(A G)=0`, `A ⪧ 0`, `G ⪧ 0` ⇒ `range(A) ⊆ ker(G)`. It is deliberately a
partial reducer: it uses only exposing matrices already present in compiled
equalities and does not solve an auxiliary facial-reduction problem.

Any extension must:

1. leave `facial_reduction=:none` as the default;
2. use scale-aware PSD/rank tolerances and reject indefinite exposing rows;
3. reconstruct `G` and certify every original constraint and cone;
4. expose reduction metadata and never accept `SLOW_PROGRESS` by itself;
5. use finite differences or a mathematically justified reduced sensitivity
   map instead of assigning arbitrary duals to removed equalities; and
6. add focused tests plus the full compatibility suite.

The current method follows Borwein and Wolkowicz (1981), *Regularizing the
abstract convex program*, and Permenter and Parrilo (2018), *Partial facial
reduction: simplified, equivalent SDPs via approximations of the PSD cone*.

## Vendored copy

PEP_Control contains a vendored copy under `Julia/PEP_NumericalDesign`.
Develop and validate toolbox changes in this standalone repository first,
then synchronize the complete changed file set into that directory and run
the PEP_Control tests and demo. Do not edit generated results or cluster
outputs as part of a toolbox synchronization.
