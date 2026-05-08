###############
# globals.jl  #
###############

"""
    κ(E)

Return κ = √(-E) for E < 0.
"""
κ(E) = sqrt(-E)

"""
    c_from_ζ(ζ, E)

Amplitude parameterization:

    c = √(-2E) * tanh(ζ),   E < 0

so that 0 < c < √(-2E) as ζ ∈ (0,∞).
"""
c_from_ζ(ζ, E) = sqrt(-2E) * tanh(ζ)

"""
    q_from_ζ(ζ, E)

Auxiliary quantity

    q = κ(E)^2 * sech(ζ)^2 = -E - 0.5 c^2

used for consistent homoclinic initial data.
"""
q_from_ζ(ζ, E) = κ(E)^2 * sech(ζ)^2

"""
    clamp1(x; eps=1e-12)

Clamp real x into (-1+eps, 1-eps) to avoid atanh/acos blowups.
"""
function clamp1(x; eps=1e-12)
    return max(min(real(x), 1 - eps), -1 + eps)
end

"""
    safe_div(v, u; eps=1e-14)

Compute v/u, but if |u| is very small, use ũ = eps*sign(u) instead.
"""
function safe_div(v, u; eps=1e-14)
    if abs(u) > eps
        return v / u
    else
        s = (u == 0 ? 1 : sign(real(u)))
        return v / (eps * s)
    end
end

"""
    sign_real(z; eps=1e-14)

Return +1.0 or -1.0 based on the sign of real(z),
or +1.0 if |Re z| < eps.
"""
function sign_real(z; eps=1e-14)
    r = real(z)
    if abs(r) > eps
        return r > 0 ? 1.0 : -1.0
    else
        return 1.0
    end
end

"""
    bisect_zero(f, a, b; tol=1e-10, maxit=80)

Simple robust bisection for scalar f on [a,b] where f(a)*f(b) < 0.
Returns approximate root.
"""
function bisect_zero(f, a, b; tol=1e-10, maxit=80)
    fa = f(a); fb = f(b)
    if !isfinite(fa) || !isfinite(fb)
        error("Non-finite residual in bracket")
    end
    if fa == 0.0; return a end
    if fb == 0.0; return b end
    @assert fa * fb < 0 "No sign change in [a,b]."

    lo, hi, flo, fhi = a, b, fa, fb
    for _ in 1:maxit
        mid = 0.5 * (lo + hi)
        fm  = f(mid)
        if abs(fm) ≤ tol || 0.5 * (hi - lo) ≤ tol
            return mid
        end
        if flo * fm < 0
            hi, fhi = mid, fm
        else
            lo, flo = mid, fm
        end
    end
    return 0.5 * (lo + hi)
end
