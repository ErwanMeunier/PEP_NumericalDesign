# Degree-≤2 polynomial algebra in the method coefficients η.
#
# This is the "symbolic" layer of PEPDesign: PEP constraint matrices are at most
# quadratic in the coefficients (updates encoded as ‖residual‖² = 0, or iterates
# substituted as affine combinations of basis vectors), so every scalar quantity
# is represented exactly as   c0 + Σ_r c_r η_r + Σ_{r≤s} c_{rs} η_r η_s.
# Extraction of constant/linear/quadratic parts is then exact and allocation-light —
# no computer-algebra system and no runtime AD are involved.

"""
    PAff

Affine scalar `c0 + Σ_r lin[r]*η_r` in the coefficient vector η.
"""
struct PAff
    c0::Float64
    lin::Dict{Int,Float64}
end

PAff() = PAff(0.0, Dict{Int,Float64}())
PAff(c::Real) = PAff(Float64(c), Dict{Int,Float64}())

"""Return the affine symbol for coefficient η_r."""
coeff(r::Int) = PAff(0.0, Dict(r => 1.0))

isconst(a::PAff) = isempty(a.lin)
Base.iszero(a::PAff) = a.c0 == 0.0 && isempty(a.lin)
Base.zero(::Type{PAff}) = PAff()
Base.convert(::Type{PAff}, x::Real) = PAff(x)
Base.convert(::Type{PAff}, a::PAff) = a
# Content equality (frontend oracle-point reuse relies on it).
Base.:(==)(a::PAff, b::PAff) = a.c0 == b.c0 && a.lin == b.lin
Base.hash(a::PAff, h::UInt) = hash((a.c0, a.lin), h)

# Merge `c .* src` into `dst`, dropping exact zeros.
function _madd!(dst::Dict{K,Float64}, src::Dict{K,Float64}, c::Float64) where {K}
    c == 0.0 && return dst
    for (k, v) in src
        nv = get(dst, k, 0.0) + c * v
        if nv == 0.0
            delete!(dst, k)
        else
            dst[k] = nv
        end
    end
    return dst
end

Base.:+(a::PAff, b::PAff) = PAff(a.c0 + b.c0, _madd!(copy(a.lin), b.lin, 1.0))
Base.:-(a::PAff) = PAff(-a.c0, Dict(k => -v for (k, v) in a.lin))
Base.:-(a::PAff, b::PAff) = PAff(a.c0 - b.c0, _madd!(copy(a.lin), b.lin, -1.0))
Base.:+(a::PAff, x::Real) = PAff(a.c0 + x, copy(a.lin))
Base.:+(x::Real, a::PAff) = a + x
Base.:-(a::PAff, x::Real) = a + (-x)
Base.:-(x::Real, a::PAff) = (-a) + x
Base.:*(x::Real, a::PAff) =
    x == 0.0 ? PAff() : PAff(x * a.c0, Dict(k => x * v for (k, v) in a.lin))
Base.:*(a::PAff, x::Real) = x * a
Base.:/(a::PAff, x::Real) = (1.0 / x) * a

"""
    affmul(a, b) :: PAff

Product of two affine scalars when at least one is constant. Errors otherwise:
point coefficients must remain affine in η — encode nested parameter-dependent
recurrences residual-style (declare the intermediate iterate as a Gram point and
add `‖residual‖² = 0` as a constraint).
"""
function affmul(a::PAff, b::PAff)
    isconst(a) && return a.c0 * b
    isconst(b) && return b.c0 * a
    error("point coefficients must remain affine in η: declare the intermediate " *
          "iterate as a Gram point and encode the update as ‖residual‖² = 0")
end

evaluate(a::PAff, η::AbstractVector{<:Real}) =
    a.c0 + sum(v * η[k] for (k, v) in a.lin; init = 0.0)

# ──────────────────────────────────────────────────────────────────────────────

# Monomial key: η_r η_s stored once with r ≤ s.
quadkey(r::Int, s::Int) = r <= s ? (r, s) : (s, r)

"""
    PQuad

Quadratic scalar `c0 + Σ_r lin[r]*η_r + Σ_{r≤s} quad[(r,s)]*η_r*η_s`.
"""
struct PQuad
    c0::Float64
    lin::Dict{Int,Float64}
    quad::Dict{Tuple{Int,Int},Float64}
end

PQuad() = PQuad(0.0, Dict{Int,Float64}(), Dict{Tuple{Int,Int},Float64}())
PQuad(x::Real) = PQuad(Float64(x), Dict{Int,Float64}(), Dict{Tuple{Int,Int},Float64}())
PQuad(a::PAff) = PQuad(a.c0, copy(a.lin), Dict{Tuple{Int,Int},Float64}())

isconst(q::PQuad) = isempty(q.lin) && isempty(q.quad)
Base.iszero(q::PQuad) = q.c0 == 0.0 && isconst(q)

Base.:+(a::PQuad, b::PQuad) = PQuad(a.c0 + b.c0,
    _madd!(copy(a.lin), b.lin, 1.0), _madd!(copy(a.quad), b.quad, 1.0))
Base.:-(a::PQuad) = -1.0 * a
Base.:-(a::PQuad, b::PQuad) = PQuad(a.c0 - b.c0,
    _madd!(copy(a.lin), b.lin, -1.0), _madd!(copy(a.quad), b.quad, -1.0))
Base.:*(x::Real, q::PQuad) =
    x == 0.0 ? PQuad() : PQuad(x * q.c0,
        Dict(k => x * v for (k, v) in q.lin), Dict(k => x * v for (k, v) in q.quad))
Base.:*(q::PQuad, x::Real) = x * q

"""
    quadmul(a, b) :: PQuad

Exact product of two affine scalars.
"""
function quadmul(a::PAff, b::PAff)
    c0 = a.c0 * b.c0
    lin = Dict{Int,Float64}()
    _madd!(lin, b.lin, a.c0)
    _madd!(lin, a.lin, b.c0)
    quad = Dict{Tuple{Int,Int},Float64}()
    for (r, ar) in a.lin, (s, bs) in b.lin
        k = quadkey(r, s)
        nv = get(quad, k, 0.0) + ar * bs
        if nv == 0.0
            delete!(quad, k)
        else
            quad[k] = nv
        end
    end
    return PQuad(c0, lin, quad)
end

function evaluate(q::PQuad, η::AbstractVector{<:Real})
    v = q.c0
    for (k, c) in q.lin
        v += c * η[k]
    end
    for ((r, s), c) in q.quad
        v += c * η[r] * η[s]
    end
    return v
end
