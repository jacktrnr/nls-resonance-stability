###############################################################
# wells_fullline.jl
#
# Pre-defined compactly-supported potentials for full-line NLS analysis.
#
# ALL potentials satisfy:
#   • V ≡ 0 outside (a, b)
#   • V(a) = V(b) = 0          (continuous at endpoints)
#   • Continuous on all of ℝ
#
# The potential argument to run_complete.jl is any callable V : ℝ → ℝ.
# Every constructor here returns such a callable.
#
# Usage in run_complete.jl (replace the Vfun = ... line):
#
#   Vfun = square_well(a, b, V0)
#   Vfun = cosine_arch(a, b, V0)
#   Vfun = sine_arch(a, b, V0)
#   Vfun = sech2_arch(a, b, V0; κ=3.0)
#   Vfun = smooth_double_arch(a, b, V0; sep=0.5)
#   Vfun = smooth_tapered_well(a, b, V0; center_frac=0.5, width_frac=0.5)
#   Vfun = antisymmetric_sine(a, b, A)
#   Vfun = asymmetric_arch(a, b, V0; skew=0.3)
#   Vfun = boundary_dipole(a, b, A; ε_frac=0.15)
#   Vfun = interior_dipole(a, b, A; x0_frac=0.35, ε_frac=0.1)
#   Vfun = smooth_step(a, b, A; sharpness=8.0)
#
###############################################################

# ──────────────────────────────────────────────────────────────────────────────
# Utility: ξ = (x-a)/(b-a) ∈ (0,1), zero outside
# ──────────────────────────────────────────────────────────────────────────────
@inline _xi(x, a, b) = (x - a) / (b - a)
@inline _in(x, a, b) = (a < x < b)


# ──────────────────────────────────────────────────────────────────────────────
# Basic arch / harmonic shapes
# ──────────────────────────────────────────────────────────────────────────────

"""
    square_well(a, b, V0)

Piecewise-constant square well/barrier:

    V(x) = V0,   a < x < b
         = 0,    otherwise

Negative `V0` gives an attractive square well, positive `V0` a barrier.
Unlike the smooth profiles below, this has jump discontinuities at `x=a,b`.
"""
function square_well(a::Real, b::Real, V0::Real)
    return x -> _in(x, a, b) ? float(V0) : 0.0
end

"""
    cosine_arch(a, b, V0)

Half-cosine arch: V(x) = V₀ cos(π(x - m) / (b-a)) where m = (a+b)/2.

Symmetric about midpoint. V(a) = V(b) = 0. Peak V₀ at the centre.
Negative V₀ gives a single attractive well (the "continuous square well").
"""
function cosine_arch(a::Real, b::Real, V0::Real)
    m = 0.5 * (a + b)
    L = b - a
    return x -> _in(x, a, b) ? V0 * cos(π * (x - m) / L) : 0.0
end


"""
    sine_arch(a, b, V0)

Half-sine arch: V(x) = V₀ sin(π ξ), ξ = (x-a)/(b-a).

Symmetric about midpoint. V(a) = V(b) = 0. Non-negative if V₀ > 0.
Compared to cosine_arch, the profile is flatter at the top (sin vs cos).
"""
function sine_arch(a::Real, b::Real, V0::Real)
    return x -> _in(x, a, b) ? V0 * sin(π * _xi(x, a, b)) : 0.0
end


"""
    antisymmetric_sine(a, b, A)

Full sine wave: V(x) = A sin(2π ξ), ξ = (x-a)/(b-a).

Antisymmetric about midpoint: V(-x) = -V(x) when [a,b] = [-b₀, b₀].
Integral ∫V = 0. Useful for testing non-symmetric scattering.
"""
function antisymmetric_sine(a::Real, b::Real, A::Real)
    return x -> _in(x, a, b) ? -A * sin(2π * _xi(x, a, b)) : 0.0
end


# ──────────────────────────────────────────────────────────────────────────────
# Smooth localized wells
# ──────────────────────────────────────────────────────────────────────────────

"""
    sech2_arch(a, b, V0; κ=3.0, center_frac=0.5)

Truncated sech² well with a cosine taper at the boundaries so V(a) = V(b) = 0.

V(x) = V₀ · sech²(κ (ξ - c)) · sin²(πξ)   where ξ = (x-a)/(b-a), c = center_frac.

The sin² taper is C^∞ at the endpoints. Increasing κ sharpens the central peak.
"""
function sech2_arch(a::Real, b::Real, V0::Real; κ::Real=3.0, center_frac::Real=0.5)
    c = center_frac
    return x -> begin
        _in(x, a, b) || return 0.0
        ξ = _xi(x, a, b)
        s = 1.0 / cosh(κ * (ξ - c))
        V0 * s^2 * sin(π * ξ)^2
    end
end


"""
    smooth_tapered_well(a, b, V0; center_frac=0.5, width_frac=0.5)

Smooth single well: Gaussian bell multiplied by sin²(πξ) compact taper.

V(x) = V₀ · exp(−((ξ−c)/(σ))²) · sin²(πξ)
where σ = width_frac / (2√(2ln2))  (FWHM = width_frac × (b−a)).

Negative V₀ gives an attractive well. center_frac ∈ (0,1) shifts the centre.
"""
function smooth_tapered_well(a::Real, b::Real, V0::Real;
                              center_frac::Real=0.5, width_frac::Real=0.5)
    c  = center_frac
    σ  = width_frac / (2 * sqrt(2 * log(2)))   # FWHM → σ
    σ  = max(σ, 1e-6)
    return x -> begin
        _in(x, a, b) || return 0.0
        ξ = _xi(x, a, b)
        V0 * exp(-((ξ - c) / σ)^2) * sin(π * ξ)^2
    end
end


"""
    smooth_double_arch(a, b, V0; sep=0.5)

Two equal smooth arches, symmetric about midpoint, each occupying one half.

sep ∈ (0,1): fractional separation between arch centres (centres at
m ± sep·(b−a)/2 in x-coordinates). Larger sep pushes wells toward boundaries.
"""
function smooth_double_arch(a::Real, b::Real, V0::Real; sep::Real=0.5)
    c1 = 0.5 - sep / 4   # ξ-centre of left arch
    c2 = 0.5 + sep / 4   # ξ-centre of right arch
    σ  = 0.14             # width in ξ-units (each arch occupies ≈ half the interval)
    return x -> begin
        _in(x, a, b) || return 0.0
        ξ = _xi(x, a, b)
        taper = sin(π * ξ)^2
        bump  = exp(-((ξ - c1) / σ)^2) + exp(-((ξ - c2) / σ)^2)
        V0 * bump * taper
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Asymmetric / ramp shapes
# ──────────────────────────────────────────────────────────────────────────────

"""
    asymmetric_arch(a, b, V0; skew=0.3)

Single smooth arch with its peak shifted off-centre.

V(x) = V₀ · sin^(2p)(πξ) · sin^(2q)(πξ) envelope, achieved by offsetting
the Gaussian centre: equivalent to smooth_tapered_well with center_frac = skew.
"""
asymmetric_arch(a::Real, b::Real, V0::Real; skew::Real=0.3) =
    smooth_tapered_well(a, b, V0; center_frac=skew, width_frac=0.6)


"""
    smooth_step(a, b, A; sharpness=8.0)

Smooth antisymmetric step: tanh-like transition multiplied by compact taper.

V(x) = A · tanh(s(ξ − ½)) · sin²(πξ),  where s = sharpness.

Antisymmetric about midpoint when [a,b] = [−b₀, b₀].
Approximates V = −A on left half, V = +A on right half.
Integral ∫V = 0. Useful for asymmetric scattering / Berry-phase experiments.
"""
function smooth_step(a::Real, b::Real, A::Real; sharpness::Real=8.0)
    return x -> begin
        _in(x, a, b) || return 0.0
        ξ = _xi(x, a, b)
        A * tanh(sharpness * (ξ - 0.5)) * sin(π * ξ)^2
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Delta-function approximations
# ──────────────────────────────────────────────────────────────────────────────

"""
    boundary_dipole(a, b, A; ε_frac=0.15)

Approximates  V ∝ −δ(x − a) + δ(x − b).

Two narrow half-sine bumps concentrated at each boundary endpoint:
  • Negative concentration of mass near x = a  (left boundary)
  • Positive concentration of mass near x = b  (right boundary)

As ε_frac → 0 the mass concentrates at the boundary, approximating the
boundary dipole.  The potential is continuous (zero at x=a and x=b),
C^∞ except at the joints x = a+ε and x = b−ε where it is C^1.

ε_frac : width of each boundary layer as a fraction of (b−a).
A      : amplitude; the integral of each lobe is ≈ 2A·ε/π.

Used for studying how boundary-localized potentials create transmission
resonances near k = 0.
"""
function boundary_dipole(a::Real, b::Real, A::Real; ε_frac::Real=0.15)
    ε = ε_frac * (b - a)
    return x -> begin
        _in(x, a, b) || return 0.0
        if x < a + ε
            ξ = (x - a) / ε          # ξ ∈ (0,1)
            return -A * sin(π * ξ)   # negative lobe at left boundary
        elseif x > b - ε
            ξ = (b - x) / ε          # ξ ∈ (0,1), measures distance from b
            return +A * sin(π * ξ)   # positive lobe at right boundary
        else
            return 0.0
        end
    end
end


"""
    interior_dipole(a, b, A; x0_frac=0.35, ε_frac=0.1)

Approximates  V ∝ −δ(x − x_L) + δ(x − x_R),  with symmetric placement.

Useful for studying interior transmission resonances in asymmetric geometries.
x0_frac : fractional position of left lobe from a (right lobe placed symmetrically).
ε_frac  : width fraction of each lobe.
"""
function interior_dipole(a::Real, b::Real, A::Real;
                          x0_frac::Real=0.35, ε_frac::Real=0.1)
    L  = b - a
    ε  = ε_frac * L
    xL = a + x0_frac * L              # left lobe centre
    xR = b - x0_frac * L              # right lobe centre (symmetric)
    return x -> begin
        _in(x, a, b) || return 0.0
        v = 0.0
        # Negative half-sine bell centred at xL: peak −A at xL, zero at xL ± ε
        if abs(x - xL) < ε
            ξ = (x - xL + ε) / (2ε)   # ξ ∈ (0,1), peak at ξ=0.5
            v += -A * sin(π * ξ)
        end
        # Positive half-sine bell centred at xR: peak +A at xR, zero at xR ± ε
        if abs(x - xR) < ε
            ξ = (x - xR + ε) / (2ε)
            v +=  A * sin(π * ξ)
        end
        v
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Quick-reference table (printed when this file is included)
# ──────────────────────────────────────────────────────────────────────────────

function _print_wells_menu()
    println()
    println("  wells_fullline.jl — available potentials (all continuous, V(a)=V(b)=0)")
    println("  " * "─"^66)
    println("  cosine_arch(a,b,V0)                   — cos(πξ) arch, symmetric")
    println("  sine_arch(a,b,V0)                     — sin(πξ) arch, symmetric")
    println("  antisymmetric_sine(a,b,A)             — sin(2πξ), ∫V=0, antisymm.")
    println("  sech2_arch(a,b,V0; κ,center_frac)     — sech² × sin² taper")
    println("  smooth_tapered_well(a,b,V0; c,w)      — Gaussian × sin² taper")
    println("  smooth_double_arch(a,b,V0; sep)       — two symmetric arches")
    println("  asymmetric_arch(a,b,V0; skew)         — off-centre single arch")
    println("  smooth_step(a,b,A; sharpness)         — tanh step × sin² taper, ∫V=0")
    println("  boundary_dipole(a,b,A; ε_frac)        — ≈ −δ(x−a) + δ(x−b)")
    println("  interior_dipole(a,b,A; x0_frac,ε)     — ≈ −δ(x−xL) + δ(x−xR)")
    println("  " * "─"^66)
    println()
end

_print_wells_menu()
