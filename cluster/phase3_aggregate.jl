# Phase 3 — aggregate phase 0–2 outputs into per-method summary files
# (consumed by the experiments/ plotting scripts). Single job.
# Usage: julia --project=. PEPDesign/cluster/phase3_aggregate.jl

include(joinpath(@__DIR__, "registry.jl"))

for method in GRID.methods
    summary = Dict{String,Any}("grid" => repr(GRID), "tag" => TAG)

    wstar = Dict{Int,Float64}()
    for f in filter(f -> startswith(f, "wstar_$(method)_"), readdir(OUT_DIR))
        d = load(out_path(f))
        wstar[d["N"]] = d["Wstar"]
    end
    summary["Wstar"] = wstar

    designs = Dict{String,Any}()
    for f in filter(f -> startswith(f, "design_$(method)_"), readdir(OUT_DIR))
        d = load(out_path(f))
        designs["$(d["policy"])_N$(d["N"])"] = d
    end
    summary["designs"] = designs

    hrdp = Dict{String,Any}()
    for f in filter(f -> startswith(f, "hrdp_$(method)_"), readdir(OUT_DIR))
        d = load(out_path(f))
        hrdp[d["policy"]] = d
    end
    summary["hrdp"] = hrdp

    dest = out_path("summary_$(method).jld2")
    jldsave(dest; summary)
    @info "phase3: wrote" dest n_wstar = length(wstar) n_designs = length(designs) n_hrdp = length(hrdp)
end
println("phase3: complete")
