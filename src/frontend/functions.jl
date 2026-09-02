# Function classes: interpolation-constraint generators over recorded oracles.
#
# Ported from PEPit.jl src/functions/ (MIT License, Copyright (c) 2025
# Shuvomoy Das Gupta and contributors). Each class stores oracle triples in a
# PEPFunc and emits its (necessary or interpolation) constraints at compile
# time. Classes marked [LMI] add PSD blocks (supported by solve_pep /
# grad_hess_eta; not yet by design_ssdp).

_pairs(n) = ((i, j) for i in 1:n, j in 1:n if i != j)

# ── 𝓕₀: convex CCP ───────────────────────────────────────────────────────────

"""
    ConvexFunction(m; name)

Closed convex proper functions: `f_i − f_j ≥ ⟨g_j, x_i − x_j⟩` for all i ≠ j.
"""
struct ConvexFunction <: AbstractPEPFunction
    core::PEPFunc
end
ConvexFunction(m::PEPModel; name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, ConvexFunction(PEPFunc(m, name, false)))

function add_class_constraints!(f::ConvexFunction)
    c = f.core
    for (i, j) in _pairs(length(c.xs))
        add_le!(c.model, c.fs[j] - c.fs[i] + inner(c.gs[j], c.xs[i] - c.xs[j]);
                name = "$(c.name):cvx[$i,$j]")
    end
end

# ── μ-strongly convex (possibly nonsmooth) ───────────────────────────────────

"""
    StronglyConvexFunction(m; μ, name)

μ-strongly convex CCP functions:
`f_i − f_j ≥ ⟨g_j, x_i − x_j⟩ + μ/2‖x_i − x_j‖²` for all i ≠ j.
"""
struct StronglyConvexFunction <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
end
StronglyConvexFunction(m::PEPModel; μ::Real,
                       name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, StronglyConvexFunction(PEPFunc(m, name, false), μ))

function add_class_constraints!(f::StronglyConvexFunction)
    c = f.core
    for (i, j) in _pairs(length(c.xs))
        d = c.xs[i] - c.xs[j]
        add_le!(c.model, c.fs[j] - c.fs[i] + inner(c.gs[j], d) +
                         (f.μ / 2) * sqnorm(d);
                name = "$(c.name):scvx[$i,$j]")
    end
end

# ── L-smooth (nonconvex) ─────────────────────────────────────────────────────

"""
    SmoothFunction(m; L, name)

L-smooth (not necessarily convex) functions:
`f_i − f_j ≥ ½⟨g_i + g_j, x_i − x_j⟩ + 1/(4L)‖g_i − g_j‖² − L/4‖x_i − x_j‖²`.
"""
struct SmoothFunction <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
end
SmoothFunction(m::PEPModel; L::Real, name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, SmoothFunction(PEPFunc(m, name, true), L))

function add_class_constraints!(f::SmoothFunction)
    c = f.core
    L = f.L
    for (i, j) in _pairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        dg = c.gs[i] - c.gs[j]
        add_le!(c.model,
                c.fs[j] - c.fs[i] + 0.5 * inner(c.gs[i] + c.gs[j], dx) +
                (1 / (4L)) * sqnorm(dg) - (L / 4) * sqnorm(dx);
                name = "$(c.name):sm[$i,$j]")
    end
end

# ── 𝓕_{0,L} and 𝓕_{μ,L} (delegate to classes.jl) ─────────────────────────────

"""
    SmoothConvexFunction(m; L, name)

L-smooth convex functions (𝓕_{0,L} interpolation).
"""
struct SmoothConvexFunction <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
end
SmoothConvexFunction(m::PEPModel; L::Real,
                     name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, SmoothConvexFunction(PEPFunc(m, name, true), L))

add_class_constraints!(f::SmoothConvexFunction) =
    add_interpolation_fmuL!(f.core.model, f.core.xs, f.core.gs, f.core.fs;
                            L = f.L, μ = 0.0, name = "$(f.core.name):ic")

"""
    SmoothStronglyConvexFunction(m; μ, L, name)

L-smooth μ-strongly convex functions (𝓕_{μ,L} interpolation,
Taylor–Hendrickx–Glineur Thm. 4).
"""
struct SmoothStronglyConvexFunction <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
SmoothStronglyConvexFunction(m::PEPModel; μ::Real, L::Real,
                             name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, SmoothStronglyConvexFunction(PEPFunc(m, name, true), μ, L))

add_class_constraints!(f::SmoothStronglyConvexFunction) =
    add_interpolation_fmuL!(f.core.model, f.core.xs, f.core.gs, f.core.fs;
                            L = f.L, μ = f.μ, name = "$(f.core.name):ic")

# ── convex + M-Lipschitz ─────────────────────────────────────────────────────

"""
    ConvexLipschitzFunction(m; M, name)

Convex CCP functions with `‖∂f‖ ≤ M`: convex interpolation + `‖g_i‖² ≤ M²`.
"""
struct ConvexLipschitzFunction <: AbstractPEPFunction
    core::PEPFunc
    M::Float64
end
ConvexLipschitzFunction(m::PEPModel; M::Real,
                        name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, ConvexLipschitzFunction(PEPFunc(m, name, false), M))

function add_class_constraints!(f::ConvexLipschitzFunction)
    c = f.core
    if isfinite(f.M)
        for i in eachindex(c.xs)
            add_le!(c.model, sqnorm(c.gs[i]) - f.M^2; name = "$(c.name):lip[$i]")
        end
    end
    for (i, j) in _pairs(length(c.xs))
        add_le!(c.model, c.fs[j] - c.fs[i] + inner(c.gs[j], c.xs[i] - c.xs[j]);
                name = "$(c.name):cvx[$i,$j]")
    end
end

"""
    SmoothConvexLipschitzFunction(m; L, M, name)

L-smooth convex M-Lipschitz functions (𝓕_{0,L} + `‖g_i‖² ≤ M²`).
"""
struct SmoothConvexLipschitzFunction <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
    M::Float64
end
SmoothConvexLipschitzFunction(m::PEPModel; L::Real, M::Real,
                              name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, SmoothConvexLipschitzFunction(PEPFunc(m, name, true), L, M))

function add_class_constraints!(f::SmoothConvexLipschitzFunction)
    c = f.core
    add_interpolation_fmuL!(c.model, c.xs, c.gs, c.fs;
                            L = f.L, μ = 0.0, name = "$(c.name):ic")
    if isfinite(f.M)
        for i in eachindex(c.xs)
            add_le!(c.model, sqnorm(c.gs[i]) - f.M^2; name = "$(c.name):lip[$i]")
        end
    end
end

# ── convex indicator (with optional diameter/radius) ─────────────────────────

"""
    ConvexIndicatorFunction(m; D=Inf, R=Inf, center=nothing, name)

Closed convex indicator functions. Oracle "gradients" are normal-cone
elements; values are pinned to 0. `D` bounds the domain diameter, `R` its
radius around `center` (a fresh point when not provided).
"""
struct ConvexIndicatorFunction <: AbstractPEPFunction
    core::PEPFunc
    D::Float64
    R::Float64
    center::Union{PointExpr,Nothing}
end
function ConvexIndicatorFunction(m::PEPModel; D::Real = Inf, R::Real = Inf,
                                 center = nothing,
                                 name::String = "ind$(length(m.funcs) + 1)")
    c = center
    if c === nothing && isfinite(R)
        c = point!(m; name = "center_$name")
    end
    _register!(m, ConvexIndicatorFunction(PEPFunc(m, name, false), D, R, c))
end

function add_class_constraints!(f::ConvexIndicatorFunction)
    c = f.core
    m = c.model
    for i in eachindex(c.xs)
        add_eq!(m, c.fs[i]; name = "$(c.name):f0[$i]")
    end
    for (i, j) in _pairs(length(c.xs))
        add_le!(m, inner(c.gs[j], c.xs[i] - c.xs[j]); name = "$(c.name):nc[$i,$j]")
    end
    if isfinite(f.D)
        for (i, j) in _pairs(length(c.xs))
            add_le!(m, sqnorm(c.xs[i] - c.xs[j]) - f.D^2;
                    name = "$(c.name):diam[$i,$j]")
        end
    end
    if isfinite(f.R)
        for i in eachindex(c.xs)
            add_le!(m, sqnorm(c.xs[i] - f.center) - f.R^2;
                    name = "$(c.name):rad[$i]")
        end
    end
end

# ── convex support function ──────────────────────────────────────────────────

"""
    ConvexSupportFunction(m; M=Inf, name)

Closed convex support functions (of a set within a ball of radius `M`):
`⟨g_i, x_i⟩ = f_i`, `⟨x_j, g_i − g_j⟩ ≤ 0`, and `‖g_i‖² ≤ M²` when finite.
"""
struct ConvexSupportFunction <: AbstractPEPFunction
    core::PEPFunc
    M::Float64
end
ConvexSupportFunction(m::PEPModel; M::Real = Inf,
                      name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, ConvexSupportFunction(PEPFunc(m, name, false), M))

function add_class_constraints!(f::ConvexSupportFunction)
    c = f.core
    m = c.model
    for i in eachindex(c.xs)
        add_eq!(m, inner(c.gs[i], c.xs[i]) - c.fs[i]; name = "$(c.name):sup[$i]")
        isfinite(f.M) &&
            add_le!(m, sqnorm(c.gs[i]) - f.M^2; name = "$(c.name):lip[$i]")
    end
    for (i, j) in _pairs(length(c.xs))
        add_le!(m, inner(c.xs[j], c.gs[i] - c.gs[j]); name = "$(c.name):mono[$i,$j]")
    end
end

# ── convex + quadratically upper bounded (QG⁺) ───────────────────────────────

"""
    ConvexQGFunction(m; L, name)

Convex functions with `f(x) − f_* ≤ L/2 ‖x − x_*‖²`. A stationary point is
auto-created if none was registered.
"""
struct ConvexQGFunction <: AbstractPEPFunction
    core::PEPFunc
    L::Float64
end
ConvexQGFunction(m::PEPModel; L::Real, name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, ConvexQGFunction(PEPFunc(m, name, false), L))

function add_class_constraints!(f::ConvexQGFunction)
    c = f.core
    m = c.model
    isempty(c.stationary) && stationary_point!(f)
    for s in c.stationary, j in eachindex(c.xs)
        c.xs[s] == c.xs[j] && continue
        add_le!(m, c.fs[j] - c.fs[s] + inner(c.gs[j], c.xs[s] - c.xs[j]) +
                   (1 / (2 * f.L)) * sqnorm(c.gs[j]);
                name = "$(c.name):qg[$s,$j]")
    end
    for (i, j) in _pairs(length(c.xs))
        add_le!(m, c.fs[j] - c.fs[i] + inner(c.gs[j], c.xs[i] - c.xs[j]);
                name = "$(c.name):cvx[$i,$j]")
    end
end

# ── RSI⁻ / EB⁺ ───────────────────────────────────────────────────────────────

"""
    RsiEbFunction(m; μ, L, name)

Functions with lower restricted secant inequality (μ) and upper error bound
(L) w.r.t. a stationary point (auto-created if absent).
"""
struct RsiEbFunction <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
RsiEbFunction(m::PEPModel; μ::Real, L::Real,
              name::String = "f$(length(m.funcs) + 1)") =
    _register!(m, RsiEbFunction(PEPFunc(m, name, false), μ, L))

function add_class_constraints!(f::RsiEbFunction)
    c = f.core
    m = c.model
    isempty(c.stationary) && stationary_point!(f)
    for s in c.stationary, j in eachindex(c.xs)
        c.xs[s] == c.xs[j] && continue
        dx = c.xs[s] - c.xs[j]
        dg = c.gs[s] - c.gs[j]
        add_le!(m, f.μ * sqnorm(dx) - inner(dg, dx); name = "$(c.name):rsi[$s,$j]")
        add_le!(m, sqnorm(dg) - f.L^2 * sqnorm(dx); name = "$(c.name):eb[$s,$j]")
    end
end

# ── smooth strongly convex quadratic [LMI] ───────────────────────────────────

"""
    SmoothStronglyConvexQuadraticFunction(m; μ, L, name)

Quadratics `f(x) = ½⟨x − x_*, H(x − x_*)⟩ + f_*` with `μI ⪯ H ⪯ LI`.
A stationary point is created on construction. Adds one LMI block.
"""
struct SmoothStronglyConvexQuadraticFunction <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
function SmoothStronglyConvexQuadraticFunction(m::PEPModel; μ::Real, L::Real,
        name::String = "f$(length(m.funcs) + 1)")
    f = _register!(m, SmoothStronglyConvexQuadraticFunction(
        PEPFunc(m, name, true), μ, L))
    stationary_point!(f)
    return f
end

function add_class_constraints!(f::SmoothStronglyConvexQuadraticFunction)
    c = f.core
    m = c.model
    s = c.stationary[1]
    xs = c.xs[s]
    fs = c.fs[s]
    n = length(c.xs)
    for i in 1:n   # f_i − f_* = ½⟨x_i − x_*, g_i⟩
        add_eq!(m, c.fs[i] - fs - 0.5 * inner(c.xs[i] - xs, c.gs[i]);
                name = "$(c.name):val[$i]")
    end
    for i in 1:n, j in (i + 1):n   # symmetry ⟨x_i − x_*, g_j⟩ = ⟨x_j − x_*, g_i⟩
        add_eq!(m, inner(c.xs[i] - xs, c.gs[j]) - inner(c.xs[j] - xs, c.gs[i]);
                name = "$(c.name):sym[$i,$j]")
    end
    T = Matrix{QExpr}(undef, n, n)   # Gram form of (LI − H)(H − μI) ⪰ 0
    for i in 1:n, j in 1:n
        T[i, j] = (f.L + f.μ) * inner(c.gs[i], c.xs[j] - xs) -
                  inner(c.gs[i], c.gs[j]) -
                  (f.μ * f.L) * inner(c.xs[i] - xs, c.xs[j] - xs)
    end
    add_psd!(m, T; name = "$(c.name):quad")
end

# ── smooth + quadratic Łojasiewicz (cheap / expensive) ───────────────────────

"""
    SmoothQuadraticLojasiewiczFunctionCheap(m; μ, L, α=nothing, name)

L-smooth functions with quadratic Łojasiewicz (PL) parameter μ — cheap
necessary conditions; optional strengthening parameter `α`.
"""
struct SmoothQuadraticLojasiewiczFunctionCheap <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
    α::Union{Float64,Nothing}
end
function SmoothQuadraticLojasiewiczFunctionCheap(m::PEPModel; μ::Real, L::Real,
        α = nothing, name::String = "f$(length(m.funcs) + 1)")
    0 <= μ <= L || error("need 0 ≤ μ ≤ L")
    α === nothing || 0 <= α <= 2μ / (2L + μ) ||
        error("need 0 ≤ α ≤ 2μ/(2L+μ)")
    _register!(m, SmoothQuadraticLojasiewiczFunctionCheap(
        PEPFunc(m, name, true), μ, L, α === nothing ? nothing : Float64(α)))
end

function add_class_constraints!(f::SmoothQuadraticLojasiewiczFunctionCheap)
    c = f.core
    m = c.model
    L, μ = f.L, f.μ
    isempty(c.stationary) && stationary_point!(f)
    for s in c.stationary, i in eachindex(c.xs)
        c.xs[s] == c.xs[i] && continue
        add_le!(m, c.fs[i] - c.fs[s] - (1 / (2μ)) * sqnorm(c.gs[i]);
                name = "$(c.name):pl[$i]")
        add_le!(m, c.fs[s] - c.fs[i] + (1 / (2L)) * sqnorm(c.gs[i]);
                name = "$(c.name):lb[$i]")
    end
    for (i, j) in _pairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        dg = c.gs[i] - c.gs[j]
        add_le!(m, c.fs[j] - c.fs[i] + 0.5 * inner(c.gs[i] + c.gs[j], dx) +
                   (1 / (4L)) * sqnorm(dg) - (L / 4) * sqnorm(dx);
                name = "$(c.name):sm[$i,$j]")
    end
    if f.α !== nothing
        α = f.α
        fs = c.fs[c.stationary[1]]
        cα = α / (1 - α) / (2μ - (L + μ) * α)
        for (i, j) in _pairs(length(c.xs))
            dx = c.xs[i] - c.xs[j]
            dg = c.gs[i] - c.gs[j]
            add_le!(m,
                c.fs[j] - c.fs[i] + 0.5 * inner(c.gs[i] + c.gs[j], dx) +
                (1 / (4L)) * sqnorm(dg) - (L / 4) * sqnorm(dx) +
                cα * ((1 - α)^2 * (L + μ) *
                      (c.fs[i] - fs - (1 / (2L)) * sqnorm(c.gs[i])) -
                      (L - μ) * (c.fs[j] - fs + (1 / (2L)) * sqnorm(c.gs[j])));
                name = "$(c.name):sma[$i,$j]")
        end
    end
end

"""
    SmoothQuadraticLojasiewiczFunctionExpensive(m; μ, L, name)

L-smooth quadratic-Łojasiewicz(μ) functions — strengthened conditions with
two 2×2 LMI blocks (and two scalar slacks) per ordered pair of points.
"""
struct SmoothQuadraticLojasiewiczFunctionExpensive <: AbstractPEPFunction
    core::PEPFunc
    μ::Float64
    L::Float64
end
function SmoothQuadraticLojasiewiczFunctionExpensive(m::PEPModel; μ::Real,
        L::Real, name::String = "f$(length(m.funcs) + 1)")
    0 <= μ <= L || error("need 0 ≤ μ ≤ L")
    _register!(m, SmoothQuadraticLojasiewiczFunctionExpensive(
        PEPFunc(m, name, true), μ, L))
end

function add_class_constraints!(f::SmoothQuadraticLojasiewiczFunctionExpensive)
    c = f.core
    m = c.model
    L, μ = f.L, f.μ
    isempty(c.stationary) && stationary_point!(f)
    for s in c.stationary, i in eachindex(c.xs)
        c.xs[s] == c.xs[i] && continue
        add_le!(m, c.fs[i] - c.fs[s] - (1 / (2μ)) * sqnorm(c.gs[i]);
                name = "$(c.name):pl[$i]")
        add_le!(m, c.fs[s] - c.fs[i] + (1 / (2L)) * sqnorm(c.gs[i]);
                name = "$(c.name):lb[$i]")
    end
    fs = c.fs[c.stationary[1]]
    for (i, j) in _pairs(length(c.xs))
        dx = c.xs[i] - c.xs[j]
        dg = c.gs[i] - c.gs[j]
        A = -c.fs[i] + c.fs[j] + 0.5 * inner(c.gs[i] + c.gs[j], dx) +
            (1 / (4L)) * sqnorm(dg) - (L / 4) * sqnorm(dx)
        B = (L + μ) * (c.fs[i] - fs - (1 / (2L)) * sqnorm(c.gs[i]))
        C = (L - μ) * (c.fs[j] - fs + (1 / (2L)) * sqnorm(c.gs[j]))
        Mt11 = -(2L + μ) * A
        Mt12 = qexpr(fval!(m))    # free slack
        Mt22 = qexpr(fval!(m))    # free slack
        D = B - C - (L + 3μ) * A
        M11 = Mt11 - (4μ / (2L + μ)) * Mt12 - D
        M12 = Mt12 - (μ / (2L + μ)) * Mt22 - ((L + μ) / 2) * A + B
        M22 = Mt22 - B
        add_psd!(m, [M11 M12; M12 M22]; name = "$(c.name):loj1[$i,$j]")
        add_psd!(m, [Mt11 Mt12; Mt12 Mt22]; name = "$(c.name):loj2[$i,$j]")
    end
end
