################
# shooting.jl  #
################

"""
    integrate_support(a, b, E, Vfun; N=1000, c=1e-3, slope_sign=+1)

Integrate the stationary NLS/GP ODE

    u' = v
    v' = (V(x) - E) u - u^3

from x=a to x=b with initial data approximating a homoclinic tail:

    u(a) = c
    v(a) = slope_sign * c * sqrt(q),

where q = -E - 0.5 c^2. If q ≤ 0 or integration fails, returns empty arrays.
"""
function integrate_support(a, b, E, Vfun; N=1000, c=1e-3, slope_sign::Int=+1)
    q = -E - 0.5 * c^2
    q ≤ 0 && return ComplexF64[], ComplexF64[], ComplexF64[]

    u0 = [c, slope_sign * c * sqrt(Complex(q))]
    f!(du, u, p, x) = begin
        du[1] = u[2]
        du[2] = (Vfun(x) - E) * u[1] - u[1]^3
    end
    prob = ODEProblem(f!, u0, (a, b))
    sol  = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false));
                 reltol=1e-7, abstol=1e-9,
                 saveat=range(a, b; length=N+1))

    sol.retcode != ReturnCode.Success && return ComplexF64[], ComplexF64[], ComplexF64[]
    x = sol.t
    U = reduce(hcat, sol.u)
    return x, U[1,:], U[2,:]
end

"""
    H_residual_ζ(a, b, E, Vfun; ζ, slope_sign=+1, N=1000)

Hamiltonian residual at x=b using ζ-parametrization:

    c = √(-2E) tanh(ζ),  q = κ(E)^2 sech(ζ)^2.

Returns a scalar H/(c^2 + 1e-3) with a small penalty near c=0
to avoid very tiny amplitudes.
"""
function H_residual_ζ(a, b, E, Vfun; ζ, slope_sign::Int=+1, N::Int=1000)
    E = -abs(E)
    @assert E < 0
    c = c_from_ζ(ζ, E)
    q = q_from_ζ(ζ, E)   # guaranteed real > 0
    xs, us, vs = integrate_support(a, b, E, Vfun; N=N, c=c, slope_sign=slope_sign)
    if isempty(xs)
        return NaN
    end

    ub, vb = us[end], vs[end]
    # 1D energy-like quantity
    H = 0.5 * real(vb)^2 + 0.5 * E * real(ub)^2 + 0.25 * real(ub)^4

    factor = (c < 1e-4) ? abs(c - 1e-4) * 1e3 : 0.0
    return H /c
end

"""
    tail_residual_ζ(a, b, E, Vfun; ζ, slope_sign=+1, right_sign=+1, N=1000)

Direct right-tail matching residual at x=b:

    ψ'(b) - s_R ψ(b) sqrt(-E - ψ(b)^2/2),

where `s_R = right_sign ∈ {+1,-1}` picks one of the two homoclinic
tail branches. This distinguishes the two boundary-matching families
that are folded together by the Hamiltonian condition `H(b)=0`.
"""
function tail_residual_ζ(a, b, E, Vfun; ζ, slope_sign::Int=+1, right_sign::Int=+1, N::Int=1000)
    E = -abs(E)
    @assert E < 0
    c = c_from_ζ(ζ, E)
    xs, us, vs = integrate_support(a, b, E, Vfun; N=N, c=c, slope_sign=slope_sign)
    isempty(xs) && return NaN

    ub = real(us[end])
    vb = real(vs[end])
    q_b = -E - 0.5 * ub^2
    q_b < -1e-12 && return NaN
    q_b = max(q_b, 0.0)
    target = right_sign * ub * sqrt(q_b)
    scale = max(abs(vb), abs(target), abs(ub), 1e-8)
    return (vb - target) / scale
end
