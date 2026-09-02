# Reusable function-class interpolation constraint generators.

"""
    add_interpolation_fmuL!(m, ys, gs, fs; L, μ=0.0, name="ic")

Add the 𝓕_{μ,L} interpolation constraints (Taylor–Hendrickx–Glineur, Thm 4)
for the sampled triples `(ys[k], gs[k], fs[k])`, all ordered pairs k ≠ l:

    f_l − f_k + ⟨g_l, y_k − y_l⟩ + 1/(2L)‖g_k − g_l‖²
        + μ/(2(1−μ/L))‖y_k − y_l − (g_k − g_l)/L‖²  ≤ 0.

`μ = 0` gives plain L-smooth convex interpolation.
"""
function add_interpolation_fmuL!(m::PEPModel, ys::AbstractVector{PointExpr},
                                 gs::AbstractVector{PointExpr}, fs::AbstractVector;
                                 L::Real, μ::Real = 0.0, name::String = "ic")
    length(ys) == length(gs) == length(fs) ||
        error("ys, gs, fs must have equal length")
    cμ = μ > 0 ? μ / (2 * (1 - μ / L)) : 0.0
    for k in eachindex(ys), l in eachindex(ys)
        k == l && continue
        e = fs[l] - fs[k] + inner(gs[l], ys[k] - ys[l]) +
            (1 / (2L)) * sqnorm(gs[k] - gs[l])
        if cμ > 0
            e = e + cμ * sqnorm(ys[k] - ys[l] - (1 / L) * (gs[k] - gs[l]))
        end
        add_le!(m, e; name = "$name[$k,$l]")
    end
    return nothing
end
