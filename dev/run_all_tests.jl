# Run the complete PEPDesign validation suite (parity, sensitivity, design,
# SSDP, HRDP, policies). Requires a Mosek license.
# Run from repo root:  julia --project=. --threads=4 PEPDesign/dev/run_all_tests.jl

const SCRIPTS = [
    "test_policies.jl",
    "test_frontend_classes.jl",
    "parity_ogd.jl",
    "parity_item.jl",
    "parity_igdm.jl",
    "smoke_design.jl",
    "smoke_ssdp.jl",
    "smoke_hrdp.jl",
]

failures = String[]
for s in SCRIPTS
    println("\n══════════ $s ══════════")
    try
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..", "..")) --threads=$(Threads.nthreads()) $(joinpath(@__DIR__, s))`)
    catch
        push!(failures, s)
    end
end

if isempty(failures)
    println("\nALL SUITES PASSED")
else
    println("\nFAILED: ", join(failures, ", "))
    exit(1)
end
