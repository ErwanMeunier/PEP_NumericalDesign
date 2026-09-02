# Primitive algorithmic steps on frontend functions.
#
# Ported from PEPit.jl src/primitive_steps/ (MIT License, Copyright (c) 2025
# Shuvomoy Das Gupta and contributors), generalized to symbolic step sizes
# γ ∈ Union{Real, PAff} wherever the degree-2 algebra allows it. A symbolic γ
# multiplying an η-dependent point raises the affmul error — encode such
# recurrences residual-style (see the manual).

const StepSize = Union{Real,PAff}

"""
    proximal_step!(x0, f, γ) -> (x, gx, fx)

Proximal step `x = prox_{γf}(x0)`, encoded as `x = x0 − γ·gx` with
`gx ∈ ∂f(x)`. `γ` may be a symbolic coefficient.
"""
function proximal_step!(x0::PointExpr, f::AbstractPEPFunction, γ::StepSize)
    m = model_of(f)
    gx = point!(m; name = "gprox_$(_name_of(f))")
    x = x0 - γ * gx
    _, fx = add_oracle_point!(f, x, gx)
    return x, gx, fx
end

"""
    inexact_gradient!(f, x, ε; notion=:absolute) -> (d, g, fx)

Approximate gradient `d ≈ ∇f(x)` with `‖d − g‖² ≤ ε²` (`:absolute`) or
`‖d − g‖² ≤ ε²‖g‖²` (`:relative`).
"""
function inexact_gradient!(f::AbstractPEPFunction, x::PointExpr, ε::Real;
                           notion::Symbol = :absolute)
    m = model_of(f)
    g, fx = oracle!(f, x)
    d = point!(m; name = "d_$(_name_of(f))")
    if notion === :absolute
        add_le!(m, sqnorm(g - d) - ε^2; name = "inexg_abs")
    elseif notion === :relative
        add_le!(m, sqnorm(g - d) - ε^2 * sqnorm(g); name = "inexg_rel")
    else
        error("unknown notion :$notion (use :absolute or :relative)")
    end
    return d, g, fx
end

"""
    inexact_gradient_step!(x0, f, γ, ε; notion=:absolute) -> (x, dx0, fx0)

Step `x = x0 − γ·d` along an ε-inexact gradient `d` of `f` at `x0`.
"""
function inexact_gradient_step!(x0::PointExpr, f::AbstractPEPFunction,
                                γ::StepSize, ε::Real; notion::Symbol = :absolute)
    d, _, fx0 = inexact_gradient!(f, x0, ε; notion)
    return x0 - γ * d, d, fx0
end

"""
    exact_linesearch_step!(x0, f, directions) -> (x, gx, fx)

Exact line/span-search surrogate: fresh `x` with `∇f(x) ⟂ x − x0` and
`∇f(x) ⟂ d` for every `d ∈ directions` (relaxation of span search).
"""
function exact_linesearch_step!(x0::PointExpr, f::AbstractPEPFunction, directions)
    m = model_of(f)
    x = point!(m; name = "xls_$(_name_of(f))")
    gx, fx = oracle!(f, x)
    add_eq!(m, inner(x - x0, gx); name = "els_dx")
    for (t, d) in enumerate(directions)
        add_eq!(m, inner(d, gx); name = "els_d$t")
    end
    return x, gx, fx
end

"""
    linear_optimization_step!(dir, ind) -> (x, gx, fx)

Linear minimization oracle over the domain of the indicator `ind`:
`x ∈ argmin ⟨dir, ·⟩`, encoded as `−dir ∈ ∂ind(x)`.
"""
function linear_optimization_step!(dir::PointExpr, ind::AbstractPEPFunction)
    m = model_of(ind)
    x = point!(m; name = "xlmo_$(_name_of(ind))")
    gx = -dir
    _, fx = add_oracle_point!(ind, x, gx)
    return x, gx, fx
end

"""
    shifted_optimization_step!(dir, f) -> (x, gx, fx)

Stationary point of `f − ⟨dir, ·⟩`, encoded as `dir ∈ ∂f(x)`.
"""
function shifted_optimization_step!(dir::PointExpr, f::AbstractPEPFunction)
    m = model_of(f)
    x = point!(m; name = "xshift_$(_name_of(f))")
    _, fx = add_oracle_point!(f, x, dir)
    return x, dir, fx
end

"""
    bregman_gradient_step!(gx0, sx0, mirror_map, γ) -> (x, sx, hx)

Mirror-descent step: `∇h(x) = sx = sx0 − γ·gx0` for a fresh iterate `x`
registered on the mirror map `h`.
"""
function bregman_gradient_step!(gx0::PointExpr, sx0::PointExpr,
                                mirror_map::AbstractPEPFunction, γ::StepSize)
    m = model_of(mirror_map)
    x = point!(m; name = "xmd_$(_name_of(mirror_map))")
    sx = sx0 - γ * gx0
    _, hx = add_oracle_point!(mirror_map, x, sx)
    return x, sx, hx
end

"""
    bregman_proximal_step!(sx0, mirror_map, min_function, γ) -> (x, sx, hx, gx, fx)

Proximal mirror step: fresh `x` with `gx ∈ ∂f(x)` registered on
`min_function` and `∇h(x) = sx0 − γ·gx` registered on the mirror map.
"""
function bregman_proximal_step!(sx0::PointExpr, mirror_map::AbstractPEPFunction,
                                min_function::AbstractPEPFunction, γ::StepSize)
    m = model_of(min_function)
    x = point!(m; name = "xmp_$(_name_of(min_function))")
    gx = point!(m; name = "gmp_$(_name_of(min_function))")
    _, fx = add_oracle_point!(min_function, x, gx)
    sx = sx0 - γ * gx
    _, hx = add_oracle_point!(mirror_map, x, sx)
    return x, sx, hx, gx, fx
end

"""
    epsilon_subgradient_step!(x0, f, γ) -> (x, g0, f0, ε)

Step along an ε-subgradient `g0 ∈ ∂_ε f(x0)`, characterized through the
Fenchel conjugate: `f(x0) + f*(g0) − ⟨g0, x0⟩ ≤ ε` with
`f*(g0) = ⟨g0, y⟩ − f(y)` at a fresh conjugate point `y`. `ε` is a free
scalar (QExpr) usable in objectives/constraints.
"""
function epsilon_subgradient_step!(x0::PointExpr, f::AbstractPEPFunction,
                                   γ::StepSize)
    m = model_of(f)
    g0 = point!(m; name = "geps_$(_name_of(f))")
    f0 = value!(f, x0)
    ε = qexpr(fval!(m))
    x = x0 - γ * g0
    y = point!(m; name = "yeps_$(_name_of(f))")
    _, fy = add_oracle_point!(f, y, g0)
    add_le!(m, f0 + inner(g0, y) - fy - inner(g0, x0) - ε; name = "epssub")
    return x, g0, f0, ε
end

"""
    inexact_proximal_step!(x0, f, γ; opt=:PD_gapII) -> (x, gx, fx, w, v, fw, ε)

Inexact proximal operation with primal-dual gap criterion `opt` ∈
{`:PD_gapI`, `:PD_gapII`, `:PD_gapIII`} (PEPit conventions). `ε` is the free
accuracy expression. Symbolic `γ` is supported for `:PD_gapII` only.
"""
function inexact_proximal_step!(x0::PointExpr, f::AbstractPEPFunction,
                                γ::StepSize; opt::Symbol = :PD_gapII)
    m = model_of(f)
    nm = _name_of(f)
    ε = qexpr(fval!(m))
    if opt === :PD_gapI
        γ isa Real || error("inexact_proximal_step!(:PD_gapI) needs numeric γ")
        v = point!(m; name = "vip_$nm")
        w = point!(m; name = "wip_$nm")
        _, fw = add_oracle_point!(f, w, v)
        x = point!(m; name = "xip_$nm")
        gx = point!(m; name = "gip_$nm")
        _, fx = add_oracle_point!(f, x, gx)
        e = x - x0 + γ * v
        eps_sub = fx - fw - inner(v, x - w)
        add_le!(m, 0.5 * sqnorm(e) + γ * eps_sub - ε; name = "iprox")
        return x, gx, fx, w, v, fw, ε
    elseif opt === :PD_gapII
        e = point!(m; name = "eip_$nm")
        gx = point!(m; name = "gip_$nm")
        x = x0 - γ * gx + e
        _, fx = add_oracle_point!(f, x, gx)
        add_le!(m, 0.5 * sqnorm(e) - ε; name = "iprox")
        return x, gx, fx, x, gx, fx, ε
    elseif opt === :PD_gapIII
        γ isa Real || error("inexact_proximal_step!(:PD_gapIII) needs numeric γ")
        x = point!(m; name = "xip_$nm")
        gx = point!(m; name = "gip_$nm")
        _, fx = add_oracle_point!(f, x, gx)
        w = point!(m; name = "wip_$nm")
        v = (1.0 / γ) * (x0 - x)
        _, fw = add_oracle_point!(f, w, v)
        eps_sub = fx - fw - inner(v, x - w)
        add_le!(m, γ * eps_sub - ε; name = "iprox")
        return x, gx, fx, w, v, fw, ε
    end
    error("unknown inexact proximal criterion :$opt")
end
