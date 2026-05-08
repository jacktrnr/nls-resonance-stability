###############################################################
# JL_evans.jl — Evans-function eigenvalue search for J𝓛 on ℝ.
#
# The eigenvalue problem  J𝓛 ξ = ν ξ  for ξ = (w, v) reduces to
# the 4-dim first-order ODE
#     Y' = A(x; ν) Y,    Y = (w, v, w', v')ᵀ
# with A given below. As |x| → ∞ (V → 0, ψ → 0), A → A_∞(ν), a
# constant matrix whose eigenvalues κ satisfy κ² ∈ {±√(|E|²+ν²)}.
# The four modes split into a 2-dim decay-at-(−∞) subspace
# 𝒥_−(ν) := span{eigenvectors with Re κ > 0} and similarly 𝒥_+
# (Re κ < 0).
#
# Evans function:
#   D(ν) := det[ Φ(X,−X;ν)·𝒥_−(ν) ; 𝒥_+(ν) ]  ∈ ℂ
# (a 4×4 determinant). Zeros of D are point eigenvalues of J𝓛
# whose eigenfunction lies in L²(ℝ).
#
# Workflow per case:
#   1. Load ψ, V, E from saved jld2 (X_BOX=40).
#   2. Wrap ψ as a function via interpolation; extend by 0 outside.
#   3. Implement Y'=A(x;ν)Y as ODEProblem.
#   4. Build D(ν), seed Newton-Raphson at known targets.
#   5. Verify eigenfunction L² decay.
###############################################################

using JLD2, LinearAlgebra, Printf
using OrdinaryDiffEq

# BASEDIR is consumed by some downstream callers; keep it relative to the package root.
const BASEDIR = joinpath(@__DIR__, "..", "data")

###############################################################
# Build A(x; ν) for the 4-dim ODE Y' = A Y, Y = (w, v, w', v')
#   w'' = (V - ψ² - E) w − ν v          ← from L_- w = ν v
#   v'' = −(V - 3ψ² - E) v − ν w        ← from −L_+ v = ν w
###############################################################
function build_A_func(Vfun, ψfun, E)
    function A(x, ν)
        Vx = Vfun(x); ψx = ψfun(x)
        a_ww = Vx - ψx^2 - E
        a_vv = -(Vx - 3ψx^2 - E)
        # block form: Y' = [[0 I];[M 0]] Y, where M is 2×2
        # row 0 / col 0..3 (Julia 1-based: rows 1..4)
        # Y = (w, v, w', v') →
        # Y'(1) = w' = Y[3]
        # Y'(2) = v' = Y[4]
        # Y'(3) = w'' = a_ww · w − ν v
        # Y'(4) = v'' = a_vv · v − ν w
        return @inline (Y) -> (
            (Y[3], Y[4], a_ww*Y[1] - ν*Y[2], a_vv*Y[2] - ν*Y[1])
        )
    end
    A
end

# Eigendecomposition of asymptotic matrix A_∞(ν) (V=0, ψ=0)
function asymptotic_modes(E, ν)
    # 4×4 system at infinity:
    # M_inf = [[0 0 1 0]
    #          [0 0 0 1]
    #          [|E|  −ν 0 0]
    #          [−ν −|E| 0 0]]
    Eabs = abs(E)
    M = ComplexF64[
        0  0  1  0;
        0  0  0  1;
        Eabs    -ν  0  0;
        -ν     -Eabs  0  0
    ]
    F = eigen(M)
    F.values, F.vectors
end

# Split 4 asymptotic modes into decay-at-(−∞) (Re κ > 0) and
# decay-at-(+∞) (Re κ < 0). Returns (J_minus, J_plus) as 4×2.
function jost_subspaces(E, ν)
    κ, V = asymptotic_modes(E, ν)
    p_minus = sortperm(κ; by = z -> -real(z))   # largest Re first
    p_plus  = sortperm(κ; by = z ->  real(z))   # smallest Re first
    Jm = V[:, p_minus[1:2]]
    Jp = V[:, p_plus[1:2]]
    return Jm, Jp, κ
end

###############################################################
# Propagator: integrate Y' = A(x;ν) Y from x = x0 to x = x1,
# with initial condition Y0 ∈ ℂ⁴.
###############################################################
function propagate(A, ν, Y0, x0, x1; reltol=1e-9, abstol=1e-11)
    function f(Y, p, x)
        a = A(x, ν)(Y)
        return [a[1], a[2], a[3], a[4]]
    end
    prob = ODEProblem(f, ComplexF64.(Y0), (x0, x1))
    sol = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false));
                reltol=reltol, abstol=abstol)
    sol.u[end]
end

# Evans function D(ν) at the given pair of cutoffs
function evans(A, E, ν, X_far; reltol=1e-9)
    Jm, Jp, κ = jost_subspaces(E, ν)
    # Propagate columns of Jm individually from -X_far to +X_far.
    cols = ComplexF64[]
    for j in 1:2
        Y_end = propagate(A, ν, Jm[:, j], -X_far, +X_far; reltol=reltol)
        append!(cols, Y_end)
    end
    Φ_Jm = reshape(cols, 4, 2)
    # Match: form 4×4 [Φ·Jm | Jp]; det = 0 ⇔ eigenvalue.
    Mmatch = hcat(Φ_Jm, Jp)
    return det(Mmatch), κ
end

###############################################################
# Newton-Raphson on D(ν) (complex). Numerical Jacobian.
###############################################################
function newton_evans(A, E, ν0, X_far; max_iter=30, tol=1e-9, h=1e-6)
    ν = ComplexF64(ν0)
    history = Tuple{ComplexF64,ComplexF64}[]
    for it in 1:max_iter
        D, _ = evans(A, E, ν, X_far)
        push!(history, (ν, D))
        @printf("    iter %2d  ν = %+.6f%+.6fi   |D|=%.3e\n",
                it, real(ν), imag(ν), abs(D))
        abs(D) < tol && return ν, D, history
        Dr, _ = evans(A, E, ν+h,    X_far)
        Di, _ = evans(A, E, ν+im*h, X_far)
        Dν_re = (Dr - D)/h
        Dν_im = (Di - D)/h
        # complex deriv: assume D analytic in ν, so dD/dν = (Dr-D)/h
        Dprime = Dν_re   # approximate via real-step
        abs(Dprime) < 1e-30 && return ν, D, history
        ν = ν - D / Dprime
    end
    return ν, evans(A, E, ν, X_far)[1], history
end

###############################################################
# Build interpolant of ψ from saved jld2 (linear, extend by 0)
###############################################################
function load_state(tag, suffix)
    p = joinpath(BASEDIR, "fl-cos3half-A5", "fl-cos3half-A5-prof3-JL-spectrum-$suffix.jld2")
    d = load(p)
    xu = d["xu"]; ψ = d["ψ"]; V_diag = d["V_diag"]; E = d["E_branch"]
    @printf("Loaded %s: E=%.4f, grid [%g,%g] N=%d\n",
            suffix, E, xu[1], xu[end], length(xu))
    # Linear interpolation, extend by 0 outside grid
    x0, xN = xu[1], xu[end]
    function ψfun(x)
        x ≤ x0 && return 0.0
        x ≥ xN && return 0.0
        # find bin
        i = searchsortedlast(xu, x)
        i == 0 && return ψ[1]
        i == length(xu) && return ψ[end]
        t = (x - xu[i]) / (xu[i+1] - xu[i])
        (1-t)*ψ[i] + t*ψ[i+1]
    end
    function Vfun(x)
        # The cos potential support is (-1, 1)
        (-1 < x < 1) ? 5.0*cos(1.5π*x) : 0.0
    end
    (ψfun=ψfun, Vfun=Vfun, E=E, xu=xu, ψ=ψ)
end

###############################################################
# Main: search for eigenvalues for cos TR and cos E1.
###############################################################
function search_case(suffix, seeds_re_im; X_far = 30.0)
    println("\n" * "="^60)
    println("Evans search for case $suffix")
    println("="^60)
    s = load_state("fl-cos3half-A5", suffix)
    A = build_A_func(s.Vfun, s.ψfun, s.E)
    @printf("X_far = %.1f  (interpolant supported on [%.1f, %.1f])\n",
            X_far, s.xu[1], s.xu[end])

    results = NamedTuple[]
    for (re0, im0) in seeds_re_im
        @printf("\n  Seed ν₀ = %+.4f%+.4fi\n", re0, im0)
        ν, D, hist = newton_evans(A, s.E, complex(re0, im0), X_far)
        push!(results, (seed=(re0, im0), ν=ν, D=D, converged=abs(D)<1e-7))
        @printf("  → final ν = %+.6f%+.6fi   |D|=%.3e   %s\n",
                real(ν), imag(ν), abs(D),
                abs(D)<1e-7 ? "CONVERGED" : "FAILED")
    end
    results
end


###############################################################
# (script-style "main" block — running searches against shipped data —
#  removed for library use; see paper_figures/ for analogous drivers.)
###############################################################
