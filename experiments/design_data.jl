trace_data(tr) = Dict(
    "omega" => tr.ω_hist, "values" => tr.values,
    "pep_solves" => tr.solves_hist, "tangent_solves" => tr.tangent_hist,
    "seconds" => tr.time_hist)

function trace_from_data(data)
    return DesignTrace([Float64.(w) for w in data["omega"]],
                       Float64.(data["values"]), Int.(data["pep_solves"]),
                       Int(last(data["pep_solves"])), Int.(data["tangent_solves"]),
                       Float64.(data["seconds"]))
end

function diagnostic_data(sol)
    d = sol.diagnostics
    return Dict("status" => string(sol.status), "certified" => d.certified,
                "primal_status" => string(d.primal_status),
                "dual_status" => string(d.dual_status),
                "relative_gap" => d.relative_gap,
                "scalar_violation" => d.max_scalar_violation,
                "cone_violation" => d.max_cone_violation)
end
