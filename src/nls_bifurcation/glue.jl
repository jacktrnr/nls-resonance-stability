#############
# glue.jl   #
#############

"""
    tail_shifts_from_ends(a, b, E, ua, va, ub, vb; eps=1e-13)

Compute shifts x0L, x0R so that the endpoints (a,ua,va), (b,ub,vb)
lie on homoclinic tails of S_E(x) = ±A sech(κ(x - x0)),
where A = √(-2E), κ = √(-E).

Method: Match value |u| = A·sech(...), then pick x0 from two candidates
by matching derivative sign.

Returns (x0L, x0R, sL, sR), where sL/sR are the chosen signs.
"""
function tail_shifts_from_ends(a, b, E, ua, va, ub, vb; eps=1e-13)
    E = -abs(E)
    @assert E < 0 "Tail matching requires E < 0"
    κv = κ(E)
    A = sqrt(-2E)

    # Left endpoint matching
    sL = sign_real(ua; eps=eps)
    ratio_L = abs(ua) / A

    if ratio_L > 1 + eps
        return NaN, NaN, sL, NaN
    end
    ratio_L = clamp(ratio_L, 0.0, 1.0)

    # Two candidate shifts from value matching
    acosh_arg_L = A / max(abs(ua), eps)
    if acosh_arg_L < 1.0
        return NaN, NaN, sL, NaN
    end

    shift_L = acosh(acosh_arg_L) / κv
    x0L_option1 = a - shift_L  # x0L < a
    x0L_option2 = a + shift_L  # x0L > a

    # Pick x0L by derivative sign matching
    # For x0L < a: tanh(κ(a - x0L)) > 0, so S'(a) has sign -sL
    # For x0L > a: tanh(κ(a - x0L)) < 0, so S'(a) has sign +sL
    deriv_sign_option1 = -sL  # if x0L < a
    deriv_sign_option2 = +sL  # if x0L > a

    ua_deriv_sign = sign_real(va; eps=eps)

    if ua_deriv_sign == deriv_sign_option1
        x0L = x0L_option1
    else
        x0L = x0L_option2
    end

    # Right endpoint matching
    sR = sign_real(ub; eps=eps)
    ratio_R = abs(ub) / A

    if ratio_R > 1 + eps
        return x0L, NaN, sL, sR
    end
    ratio_R = clamp(ratio_R, 0.0, 1.0)

    acosh_arg_R = A / max(abs(ub), eps)
    if acosh_arg_R < 1.0
        return x0L, NaN, sL, sR
    end

    shift_R = acosh(acosh_arg_R) / κv
    x0R_option1 = b - shift_R  # x0R < b
    x0R_option2 = b + shift_R  # x0R > b

    # Pick x0R by derivative sign matching
    deriv_sign_option1 = -sR  # if x0R < b
    deriv_sign_option2 = +sR  # if x0R > b

    ub_deriv_sign = sign_real(vb; eps=eps)

    if ub_deriv_sign == deriv_sign_option1
        x0R = x0R_option1
    else
        x0R = x0R_option2
    end

    return x0L, x0R, sL, sR
end

"""
    glue_full_solution(a, b, E, x, u, v; Xmax=10.0, x_left=nothing, x_right=nothing,
                       NL=1200, NR=1200,
                       return_deriv=false, eps=1e-13)

Given interior solution (x,u,v) on [a,b], glue left/right homoclinic
tails S_E(x) = ±A sech(κ(x - x0)) with A=√(-2E), κ=√(-E).

Returns:
- if `return_deriv=false`: (xfull, ψfull, (x0L,x0R))
- if `return_deriv=true`:  (xfull, ψfull, vfull)
"""
function glue_full_solution(a, b, E, x, u, v;
                            Xmax=10.0, x_left=nothing, x_right=nothing,
                            NL=1200, NR=1200,
                            return_deriv=false, eps=1e-13)
    E = -abs(E)
    @assert E < 0 "Gluing requires E < 0"
    if isempty(x)
        return Float64[], Float64[], (NaN, NaN)
    end

    κv = κ(E)
    A  = sqrt(-2E)

    ua, va = u[1],  v[1]
    ub, vb = u[end], v[end]

    # guard: endpoints must be in admissible homoclinic range
    if !(abs(ua) ≤ A + eps && abs(ub) ≤ A + eps)
        return Float64[], Float64[], (NaN, NaN)
    end

    # shifts from matching formula
    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ua, va, ub, vb; eps=eps)

    if isnan(x0L) || isnan(x0R)
        return Float64[], Float64[], (NaN, NaN)
    end

    x_left = something(x_left, min(-Xmax, a - Xmax))
    x_right = something(x_right, max(Xmax, b + Xmax))

    # left tail grid
    xL = range(x_left, a; length=NL)
    ξL = κv .* (xL .- x0L)
    ψL = sL .* A .* sech.(ξL)

    # right tail grid
    xR = range(b, x_right; length=NR)
    ξR = κv .* (xR .- x0R)
    ψR = sR .* A .* sech.(ξR)

    xfull = vcat(xL, x, xR)
    ψfull = vcat(ψL, u, ψR)

    if return_deriv
        vL = sL .* (-A * κv) .* tanh.(ξL) .* sech.(ξL)
        vR = sR .* (-A * κv) .* tanh.(ξR) .* sech.(ξR)
        vfull = vcat(vL, v, vR)
        return xfull, ψfull, vfull
    else
        return xfull, ψfull, (x0L, x0R)
    end
end

"""
    compute_norm(a, b, E, x, u, v)

Compute the L² norm of the glued homoclinic:

    N = ∫ |ψ(x)|² dx  (bulk + analytic tails)

If the endpoint data are not admissible, returns NaN.
"""
function compute_norm(a, b, E, x, u, v)
    κv, A = κ(E), sqrt(-2E)
    ua, va, ub, vb = u[1], v[1], u[end], v[end]

    if abs(ua) > A + 1e-13 || abs(ub) > A + 1e-13
        return NaN
    end

    dx = (b - a) / (length(x) - 1)
    N_bulk = dx * (sum(abs2, u) - 0.5 * (u[1]^2 + u[end]^2))

    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ua, va, ub, vb)

    if isnan(x0L) || isnan(x0R)
        return NaN
    end

    # Compute tail contributions using tanh at endpoints
    tanhL = tanh(κv * (a - x0L))
    tanhR = tanh(κv * (b - x0R))

    N_left  = (A^2 / κv) * (tanhL + 1)
    N_right = (A^2 / κv) * (1 - tanhR)

    return real(N_bulk + N_left + N_right)
end

"""
    compute_H1_norm(a, b, E, x, u, v)

Compute the H¹-seminorm + L² norm:

    ∫(|ψ|² + |ψ_x|²) dx

using numerical bulk + analytic tail corrections.
"""
function compute_H1_norm(a, b, E, x, u, v)
    κv, A = κ(E), sqrt(-2E)
    ua, va, ub, vb = u[1], v[1], u[end], v[end]

    if abs(ua) > A + 1e-13 || abs(ub) > A + 1e-13
        return NaN
    end

    dx = (b - a) / (length(x) - 1)
    N_bulk = dx * (sum(abs2, u) - 0.5 * (u[1]^2 + u[end]^2))
    D_bulk = dx * (sum(abs2, v) - 0.5 * (v[1]^2 + v[end]^2))

    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ua, va, ub, vb)

    if isnan(x0L) || isnan(x0R)
        return NaN
    end

    tanhL = tanh(κv * (a - x0L))
    tanhR = tanh(κv * (b - x0R))

    N_left  = (A^2 / κv) * (tanhL + 1)
    N_right = (A^2 / κv) * (1 - tanhR)

    D_left  = (A^2 * κv / 3) * (1 + tanhL^3)
    D_right = (A^2 * κv / 3) * (1 - tanhR^3)

    N_total = real(N_bulk + N_left + N_right)
    D_total = real(D_bulk + D_left + D_right)

    return N_total + D_total
end

"""
    can_glue(E, u_b, v_b; tol=1e-13)

Check whether right endpoint (u_b, v_b) lies in admissible
homoclinic range:

    |u_b| ≤ √(-2E)

Returns true/false.
"""
function can_glue(E, u_b, v_b; tol=1e-13)
    A = sqrt(-2E)
    abs(u_b) > A + tol && return false
    return true
end
