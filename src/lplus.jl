###############################################
# lplus_analytic.jl
###############################################

using LinearAlgebra
using OrdinaryDiffEq

function _linear_interp_scalar(xsrc::AbstractVector, ysrc::AbstractVector, x::Real)
    x <= xsrc[1] && return ysrc[1]
    x >= xsrc[end] && return ysrc[end]
    j = searchsortedlast(xsrc, x)
    j = clamp(j, 1, length(xsrc) - 1)
    t = (x - xsrc[j]) / (xsrc[j + 1] - xsrc[j])
    return (1 - t) * ysrc[j] + t * ysrc[j + 1]
end

function _sample_sorted_indices(vals::AbstractVector, n_pts::Int)
    isempty(vals) && return Int[]
    n_pts <= 0 && return Int[]
    ord = sortperm(vals)
    if n_pts >= length(ord)
        return ord
    end
    raw = round.(Int, range(1, length(ord); length=n_pts))
    return unique(ord[raw])
end

_pt_plus_den(m, u) = m^2 - 1.0 + 3.0 * m * u + 3.0 * u^2
_pt_plus_num(κv, m, u) = κv * (-m * _pt_plus_den(m, u) + (1.0 - u^2) * (3.0 * m + 6.0 * u))

function _build_Lplus_full_problem(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=3000)
    E < 0 || error("E must be negative")
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return nothing

    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ui[1], vi[1], ui[end], vi[end])
    (isfinite(x0L) && isfinite(x0R)) || return nothing

    κv = sqrt(-E)
    uL = tanh(κv * (a - x0L))
    uR = tanh(κv * (b - x0R))
    ψ_at(x::Float64) = _linear_interp_scalar(xi, ui, x)

    return (;
        a, b, E, Vfun, c, slope_sign,
        xi, ui, vi,
        x0L, x0R, sL, sR,
        κv, uL, uR, ψ_at,
    )
end

function _left_jost_data_full(prob, μ::Real)
    κv = prob.κv
    μ >= κv^2 && return (NaN, NaN, NaN, NaN)
    λ = sqrt(κv^2 - μ)
    m = λ / κv
    tL = prob.uL
    NL = m^2 - 1.0 - 3.0 * m * tL + 3.0 * tL^2
    numL = (-3.0 * m + 6.0 * tL) * κv * (1.0 - tL^2)
    wL = λ + numL / NL
    return (wL, NL, numL, m)
end

function _right_jost_data_full(prob, μ::Real)
    κv = prob.κv
    μ >= κv^2 && return (NaN, NaN, NaN, NaN)
    λ = sqrt(κv^2 - μ)
    m = λ / κv
    tR = prob.uR
    NR = m^2 - 1.0 + 3.0 * m * tR + 3.0 * tR^2
    numR = (3.0 * m + 6.0 * tR) * κv * (1.0 - tR^2)
    wR = -λ + numR / NR
    return (wR, NR, numR, m)
end

function _shoot_inner_lplus_full(prob, μ::Real;
                                 ode_atol=1e-12,
                                 ode_rtol=1e-10)
    wL, NL, numL, m = _left_jost_data_full(prob, μ)
    isfinite(wL) || return (NaN, NaN, NaN, NaN, NaN, NaN, NaN)

    function ode!(du, u, p, t)
        du[1] = u[2]
        du[2] = (prob.Vfun(t) - 3.0 * prob.ψ_at(t)^2 - prob.E - μ) * u[1]
    end
    prob_ode = ODEProblem(ode!, ComplexF64[1.0, wL], (prob.a, prob.b))
    sol = solve(prob_ode, Tsit5(); abstol=ode_atol, reltol=ode_rtol, save_everystep=false)
    isempty(sol.u) && return (NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    φb, φb′ = sol.u[end]
    win = φb′ / φb
    return (φb, φb′, win, wL, NL, numL, m)
end

function build_Lplus_evans_full(a, b, E, Vfun, c, slope_sign;
                                N_shoot=3000,
                                ode_atol=1e-12,
                                ode_rtol=1e-10)
    prob = _build_Lplus_full_problem(a, b, E, Vfun, c, slope_sign; N_shoot=N_shoot)
    prob === nothing && return nothing

    function eval_at(μ::Float64)
        φb, φb′, win, wL, NL, numL, m = _shoot_inner_lplus_full(prob, μ; ode_atol=ode_atol, ode_rtol=ode_rtol)
        isfinite(real(φb)) || return (;
            μ, D=NaN, H=NaN, win=NaN, woutL=NaN, woutR=NaN,
            φb=NaN, φb′=NaN, NL=NaN, NR=NaN, m=NaN
        )
        wR, NR, numR, _ = _right_jost_data_full(prob, μ)
        H = φb′ * NR - numR * φb
        D = (abs(φb) > 1e-12 && abs(NR) > 1e-12) ? (win - wR) : NaN
        return (;
            μ,
            D = real(D),
            H = real(H),
            win = real(win),
            woutL = real(wL),
            woutR = real(wR),
            φb = real(φb),
            φb′ = real(φb′),
            NL = real(NL),
            NR = real(NR),
            m = real(m),
        )
    end

    return (; prob, eval_at)
end

function _extract_roots_from_evans_grid(μ_grid, H_grid;
                                        nev=6,
                                        bisect_tol=1e-10,
                                        max_bisect=80,
                                        local_tol=1e-6,
                                        evaluator=nothing)
    found = Float64[]

    _push_unique_root!(roots, μ; tol=1e-6) = begin
        any(abs(μ - r) ≤ tol for r in roots) || push!(roots, μ)
        return roots
    end

    _golden_section_min_abs(f, lo, hi; maxit=40) = begin
        ϕ = (sqrt(5) - 1) / 2
        a0, b0 = lo, hi
        c0 = b0 - ϕ * (b0 - a0)
        d0 = a0 + ϕ * (b0 - a0)
        fc = abs(f(c0))
        fd = abs(f(d0))
        for _ in 1:maxit
            if fc < fd
                b0 = d0
                d0 = c0
                fd = fc
                c0 = b0 - ϕ * (b0 - a0)
                fc = abs(f(c0))
            else
                a0 = c0
                c0 = d0
                fc = fd
                d0 = a0 + ϕ * (b0 - a0)
                fd = abs(f(d0))
            end
        end
        return fc < fd ? c0 : d0
    end

    _newton_refine_μ(f, μ0, μlo, μhi; maxit=20, tol=bisect_tol) = begin
        μ = clamp(μ0, μlo, μhi)
        for _ in 1:maxit
            fμ = f(μ)
            isfinite(fμ) || break
            abs(fμ) ≤ tol && return μ, fμ, true
            δμ = max(1e-8, 1e-5 * max(abs(μ), 1.0))
            μp = min(μ + δμ, μhi)
            μm = max(μ - δμ, μlo)
            fp = f(μp)
            fm = f(μm)
            (isfinite(fp) && isfinite(fm)) || break
            df = (fp - fm) / (μp - μm)
            abs(df) ≤ 1e-14 && break
            μnew = clamp(μ - fμ / df, μlo, μhi)
            abs(μnew - μ) ≤ tol * max(abs(μ), 1.0) && return μnew, f(μnew), true
            μ = μnew
        end
        fμ = f(μ)
        return μ, fμ, isfinite(fμ) && abs(fμ) ≤ 100tol
    end

    for i in 1:length(μ_grid)-1
        Hlo, Hhi = H_grid[i], H_grid[i + 1]
        isfinite(Hlo) && isfinite(Hhi) || continue
        Hlo * Hhi < 0 || continue
        lo, hi, Hlob = μ_grid[i], μ_grid[i + 1], Hlo
        broken = false
        for _ in 1:max_bisect
            mid = 0.5 * (lo + hi)
            Hmid = evaluator === nothing ? NaN : evaluator(mid)
            if !isfinite(Hmid)
                broken = true
                break
            end
            Hlob * Hmid < 0 ? (hi = mid) : (lo = mid; Hlob = Hmid)
            hi - lo < bisect_tol && break
        end
        !broken && _push_unique_root!(found, 0.5 * (lo + hi))
        length(found) == nev && break
    end

    if length(found) < nev && evaluator !== nothing
        for i in 2:length(μ_grid)-1
            isfinite(H_grid[i]) || continue
            abs(H_grid[i]) ≤ abs(H_grid[i-1]) || continue
            abs(H_grid[i]) ≤ abs(H_grid[i+1]) || continue
            abs(H_grid[i]) ≤ local_tol || continue
            fμ = evaluator
            μcand = _golden_section_min_abs(fμ, μ_grid[i-1], μ_grid[i+1])
            μref, href, ok = _newton_refine_μ(fμ, μcand, μ_grid[i-1], μ_grid[i+1])
            ok || continue
            abs(href) ≤ local_tol || continue
            _push_unique_root!(found, μref)
            length(found) == nev && break
        end
    end

    sort!(found)
    return found
end

function compute_Lplus_spectrum_fd_full(a, b, E, Vfun, c, slope_sign;
                                        Xmax=8.0,
                                        N_shoot=2000,
                                        N_grid=900,
                                        nev=6)
    E < 0 || error("E must be negative")
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return (Float64[], Float64[], Float64[])

    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ui[1], vi[1], ui[end], vi[end])
    (isfinite(x0L) && isfinite(x0R)) || return (Float64[], Float64[], Float64[])

    κv = sqrt(-E)
    A = sqrt(-2E)
    x_lo = min(-Xmax, a - Xmax)
    x_hi = max(Xmax, b + Xmax)
    x = collect(range(x_lo, x_hi; length=N_grid))
    h = x[2] - x[1]
    xi_r = real.(xi)
    ui_r = real.(ui)

    ψ = similar(x)
    for (j, xx) in enumerate(x)
        ψ[j] = if xx < a
            sL * A * sech(κv * (xx - x0L))
        elseif xx > b
            sR * A * sech(κv * (xx - x0R))
        else
            _linear_interp_scalar(xi_r, ui_r, xx)
        end
    end

    Vv = Vfun.(x)
    main = 2 / h^2 .+ Vv .- E .- 3 .* ψ.^2
    off = fill(-1 / h^2, length(x) - 1)
    vals = eigen(Symmetric(Matrix(SymTridiagonal(main, off)))).values
    λcut = κv^2
    discrete = [λ for λ in vals if λ < λcut - 1e-6]
    return x, ψ, discrete[1:min(nev, length(discrete))]
end

function compute_Lplus_spectrum_evans_full(a, b, E, Vfun, c, slope_sign;
                                           nev=6,
                                           N_shoot=3000,
                                           μ_min=-50.0,
                                           N_scan=1200,
                                           bisect_tol=1e-10,
                                           max_bisect=80,
                                           ode_atol=1e-12,
                                           ode_rtol=1e-10,
                                           fd_Xmax=8.0,
                                           fd_N_grid=900,
                                           fd_nev=max(nev, 6))
    evans = build_Lplus_evans_full(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=N_shoot, ode_atol=ode_atol, ode_rtol=ode_rtol)
    evans === nothing && return nothing

    κv = evans.prob.κv
    μ_max = κv^2 - 1e-10
    μ_grid = collect(range(μ_min, μ_max; length=N_scan))
    scan = [evans.eval_at(μ) for μ in μ_grid]
    D_grid = [item.D for item in scan]
    H_grid = [item.H for item in scan]
    win_grid = [item.win for item in scan]
    wL_grid = [item.woutL for item in scan]
    wR_grid = [item.woutR for item in scan]

    roots = _extract_roots_from_evans_grid(μ_grid, H_grid;
                                           nev=nev,
                                           bisect_tol=bisect_tol,
                                           max_bisect=max_bisect,
                                           local_tol=1e-6,
                                           evaluator = μ -> evans.eval_at(μ).H)

    x_fd, ψ_fd, fd_vals = compute_Lplus_spectrum_fd_full(a, b, E, Vfun, c, slope_sign;
                                                         Xmax=fd_Xmax,
                                                         N_shoot=N_shoot,
                                                         N_grid=fd_N_grid,
                                                         nev=fd_nev)

    _push_unique_root!(roots_vec, μ; tol=1e-6) = begin
        any(abs(μ - r) ≤ tol for r in roots_vec) || push!(roots_vec, μ)
        return roots_vec
    end
    for μfd in fd_vals
        μref = refine_Lplus_root_full(a, b, E, Vfun, c, slope_sign;
                                      μ_guess=μfd,
                                      μ_window=max(0.05, 0.5 * abs(μfd)),
                                      accept_radius=max(0.1, abs(μfd)))
        isfinite(μref) && _push_unique_root!(roots, μref)
    end
    sort!(roots)

    return (;
        μ_grid,
        D_grid,
        H_grid,
        win_grid,
        wL_grid,
        wR_grid,
        eigenvalues = roots,
        fd_eigenvalues = fd_vals,
        λ_edge = κv^2,
        x_fd,
        ψ_fd,
        problem = evans.prob,
    )
end

function compute_rescaled_evans_scan_full(a, b, E, Vfun, c, slope_sign;
                                          ν_center,
                                          ν_halfspan,
                                          N_shoot=3000,
                                          N_scan=500,
                                          ode_atol=1e-12,
                                          ode_rtol=1e-10)
    evans = build_Lplus_evans_full(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=N_shoot, ode_atol=ode_atol, ode_rtol=ode_rtol)
    evans === nothing && return nothing
    ε = c^2
    ε > 0 || return nothing

    ν_grid = collect(range(ν_center - ν_halfspan, ν_center + ν_halfspan; length=N_scan))
    Dtilde_grid = Float64[]
    μ_grid = Float64[]

    for ν in ν_grid
        μ = ε^2 * ν
        push!(μ_grid, μ)
        item = evans.eval_at(μ)
        Dval = item.D
        push!(Dtilde_grid, isfinite(Dval) ? Dval / ε : NaN)
    end

    return (;
        ν_grid,
        μ_grid,
        Dtilde_grid,
        ε,
        λ_edge = evans.prob.κv^2 / max(ε^2, 1e-30),
        problem = evans.prob,
    )
end

function _right_jost_profile_full(prob, μ::Real, x::Real)
    κv = prob.κv
    λ = sqrt(κv^2 - μ)
    m = λ / κv
    ξ = x - prob.x0R
    t = tanh(κv * ξ)
    return exp(-λ * ξ) * (m^2 - 1.0 + 3.0 * m * t + 3.0 * t^2) / ((m + 1.0) * (m + 2.0))
end

function _left_jost_profile_full(prob, μ::Real, x::Real)
    κv = prob.κv
    λ = sqrt(κv^2 - μ)
    m = λ / κv
    ξ = x - prob.x0L
    t = tanh(κv * ξ)
    return exp(λ * ξ) * (m^2 - 1.0 - 3.0 * m * t + 3.0 * t^2) / ((m + 1.0) * (m + 2.0))
end

function compute_Lplus_mode_jost_full(a, b, E, Vfun, c, slope_sign, μ;
                                      Xmax=8.0,
                                      N_shoot=2500,
                                      N_tail=500,
                                      ode_atol=1e-12,
                                      ode_rtol=1e-10)
    evans = build_Lplus_evans_full(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=N_shoot, ode_atol=ode_atol, ode_rtol=ode_rtol)
    evans === nothing && return Float64[], Float64[], NaN
    prob = evans.prob
    μ >= prob.κv^2 && return Float64[], Float64[], NaN

    wL, _, _, _ = _left_jost_data_full(prob, μ)
    isfinite(wL) || return Float64[], Float64[], NaN

    function ode!(du, u, p, t)
        du[1] = u[2]
        du[2] = (prob.Vfun(t) - 3.0 * prob.ψ_at(t)^2 - prob.E - μ) * u[1]
    end
    x_inner = collect(range(a, b; length=N_shoot))
    sol = solve(ODEProblem(ode!, ComplexF64[1.0, wL], (a, b)), Tsit5();
                abstol=ode_atol, reltol=ode_rtol, saveat=x_inner)
    isempty(sol.u) && return Float64[], Float64[], NaN
    ϕ_inner = ComplexF64[u[1] for u in sol.u]
    ϕb = ϕ_inner[end]
    x_lo = min(-Xmax, a - Xmax)
    x_hi = max(Xmax, b + Xmax)
    x_left = collect(range(x_lo, a; length=N_tail))
    x_right = collect(range(b, x_hi; length=N_tail))
    fLa = _left_jost_profile_full(prob, μ, a)
    fRb = _right_jost_profile_full(prob, μ, b)
    (abs(fLa) > 1e-14 && abs(fRb) > 1e-14) || return Float64[], Float64[], NaN

    left_scale = inv(fLa)
    right_scale = ϕb / fRb
    ϕ_left = left_scale .* ComplexF64[_left_jost_profile_full(prob, μ, x) for x in x_left]
    ϕ_right = right_scale .* ComplexF64[_right_jost_profile_full(prob, μ, x) for x in x_right]

    x_full = vcat(x_left[1:end-1], x_inner, x_right[2:end])
    ϕ_full = vcat(real.(ϕ_left[1:end-1]), real.(ϕ_inner), real.(ϕ_right[2:end]))
    normϕ = sqrt(sum(abs2, ϕ_full) * (x_full[2] - x_full[1]))
    normϕ > 0 && (ϕ_full ./= normϕ)
    imax = argmax(abs.(ϕ_full))
    ϕ_full[imax] < 0 && (ϕ_full .*= -1)
    return x_full, ϕ_full, μ
end

function _shoot_from_origin_local(b, E, Vfun, β; N=2000, ode_rtol=1e-10, ode_atol=1e-12)
    u0 = [0.0, abs(β)]
    function f!(du, u, p, x)
        du[1] = u[2]
        du[2] = (Vfun(x) - E) * u[1] - u[1]^3
    end
    prob = ODEProblem(f!, u0, (0.0, b))
    sol = solve(prob, Tsit5(); reltol=ode_rtol, abstol=ode_atol,
                saveat=range(0.0, b; length=N + 1))
    isempty(sol.u) && return Float64[], Float64[], Float64[]
    U = reduce(hcat, sol.u)
    return sol.t, U[1, :], U[2, :]
end

function _sech_tail_params_local(b, E, ub, vb)
    κv = sqrt(-E)
    A = sqrt(-2E)
    s = sign(ub)
    s == 0 && (s = sign(vb))
    s == 0.0 && (s = 1.0)
    if abs(ub) < 1e-14
        x_shift = b + (vb > 0 ? 10.0 / κv : -10.0 / κv)
    else
        ratio = max(A / abs(ub), 1.0 + 1e-10)
        y0 = acosh(ratio)
        dp = -s * κv * A * sech(y0) * tanh(y0)
        dn = -s * κv * A * sech(-y0) * tanh(-y0)
        x_shift = abs(dp - vb) < abs(dn - vb) ? b - y0 / κv : b + y0 / κv
    end
    return s, κv, A, x_shift
end

function compute_Lplus_eigenvalues_refined_dirichlet(a, b, E, Vfun, β;
                                                     nev=3,
                                                     N_shoot=2000,
                                                     μ_min=-50.0,
                                                     N_scan=500,
                                                     bisect_tol=1e-10,
                                                     max_bisect=60,
                                                     ode_atol=1e-12,
                                                     ode_rtol=1e-10)
    E < 0 || error("E must be negative")
    b0 = b - a
    Vshift(y) = Vfun(y + a)

    xi, ui, vi = _shoot_from_origin_local(b0, E, Vshift, β;
                                          N=N_shoot, ode_rtol=ode_rtol, ode_atol=ode_atol)
    isempty(xi) && return fill(NaN, nev)

    h_i = xi[2] - xi[1]
    s_t, κv, A_t, x_shift = _sech_tail_params_local(b0, E, ui[end], vi[end])
    !isfinite(x_shift) && return fill(NaN, nev)
    u_b = tanh(κv * (b0 - x_shift))

    function ψ_at(x::Float64)
        x ≤ 0.0 && return 0.0
        x ≥ b0 && return s_t * A_t * sech(κv * (x - x_shift))
        idx = clamp(floor(Int, x / h_i) + 1, 1, length(xi) - 1)
        return ui[idx] + (x - xi[idx]) / h_i * (ui[idx + 1] - ui[idx])
    end

    function inner_shoot(μ::Float64)
        function ode!(du, u, p, t)
            du[1] = u[2]
            du[2] = (Vshift(t) - 3.0 * ψ_at(t)^2 - E - μ) * u[1]
        end
        prob = ODEProblem(ode!, [0.0, 1.0], (0.0, b0))
        sol = solve(prob, Tsit5(); abstol=ode_atol, reltol=ode_rtol, save_everystep=false)
        isempty(sol.u) && return (NaN, NaN)
        return sol.u[end][1], sol.u[end][2]
    end

    function HD(μ::Float64)
        μ >= κv^2 && return (NaN, NaN)
        m = sqrt(κv^2 - μ) / κv
        φ, φ′ = inner_shoot(μ)
        (isnan(φ) || isnan(φ′)) && return (NaN, NaN)
        D = _pt_plus_den(m, u_b)
        w_num = _pt_plus_num(κv, m, u_b)
        return (φ′ * D - w_num * φ, D)
    end

    μ_max = κv^2 - 1e-10
    μ_grid = range(μ_min, μ_max; length=N_scan)
    Hv = [HD(μ)[1] for μ in μ_grid]
    found = Float64[]

    _push_unique_root!(roots, μ; tol=1e-6) = begin
        any(abs(μ - r) ≤ tol for r in roots) || push!(roots, μ)
        return roots
    end

    _golden_section_min_abs(f, lo, hi; maxit=40) = begin
        ϕ = (sqrt(5) - 1) / 2
        a0, b0 = lo, hi
        c0 = b0 - ϕ * (b0 - a0)
        d0 = a0 + ϕ * (b0 - a0)
        fc = abs(f(c0))
        fd = abs(f(d0))
        for _ in 1:maxit
            if fc < fd
                b0 = d0
                d0 = c0
                fd = fc
                c0 = b0 - ϕ * (b0 - a0)
                fc = abs(f(c0))
            else
                a0 = c0
                c0 = d0
                fc = fd
                d0 = a0 + ϕ * (b0 - a0)
                fd = abs(f(d0))
            end
        end
        return fc < fd ? c0 : d0
    end

    _newton_refine_μ(f, μ0, μlo, μhi; maxit=20, tol=bisect_tol) = begin
        μ = clamp(μ0, μlo, μhi)
        for _ in 1:maxit
            fμ = f(μ)
            isfinite(fμ) || break
            abs(fμ) ≤ tol && return μ, fμ, true
            δμ = max(1e-8, 1e-5 * max(abs(μ), 1.0))
            μp = min(μ + δμ, μhi)
            μm = max(μ - δμ, μlo)
            fp = f(μp)
            fm = f(μm)
            (isfinite(fp) && isfinite(fm)) || break
            df = (fp - fm) / (μp - μm)
            abs(df) ≤ 1e-14 && break
            μnew = clamp(μ - fμ / df, μlo, μhi)
            abs(μnew - μ) ≤ tol * max(abs(μ), 1.0) && return μnew, f(μnew), true
            μ = μnew
        end
        fμ = f(μ)
        return μ, fμ, isfinite(fμ) && abs(fμ) ≤ 100tol
    end

    for i in 1:length(μ_grid)-1
        Hlo, Hhi = Hv[i], Hv[i + 1]
        isfinite(Hlo) && isfinite(Hhi) || continue
        Hlo * Hhi < 0 || continue

        lo, hi, Hlob = μ_grid[i], μ_grid[i + 1], Hlo
        broken = false
        for _ in 1:max_bisect
            mid = 0.5 * (lo + hi)
            Hmid = HD(mid)[1]
            if !isfinite(Hmid)
                broken = true
                break
            end
            Hlob * Hmid < 0 ? (hi = mid) : (lo = mid; Hlob = Hmid)
            hi - lo < bisect_tol && break
        end
        !broken && _push_unique_root!(found, 0.5 * (lo + hi))
        length(found) == nev && break
    end

    local_tol = 1e-6
    if length(found) < nev
        for i in 2:length(μ_grid)-1
            isfinite(Hv[i]) || continue
            abs(Hv[i]) ≤ abs(Hv[i-1]) || continue
            abs(Hv[i]) ≤ abs(Hv[i+1]) || continue
            abs(Hv[i]) ≤ local_tol || continue
            fμ = μ -> HD(μ)[1]
            μcand = _golden_section_min_abs(fμ, μ_grid[i-1], μ_grid[i+1])
            μref, href, ok = _newton_refine_μ(fμ, μcand, μ_grid[i-1], μ_grid[i+1])
            ok || continue
            abs(href) ≤ local_tol || continue
            _push_unique_root!(found, μref)
            length(found) == nev && break
        end
    end

    sort!(found)

    while length(found) < nev
        push!(found, NaN)
    end
    return found[1:nev]
end

function compute_Lplus_eigenvalues_refined_full(a, b, E, Vfun, c, slope_sign;
                                                nev=3,
                                                N_shoot=3000,
                                                μ_min=-50.0,
                                                N_scan=500,
                                                bisect_tol=1e-10,
                                                max_bisect=60,
                                                ode_atol=1e-12,
                                                ode_rtol=1e-10)
    data = compute_Lplus_spectrum_evans_full(a, b, E, Vfun, c, slope_sign;
                                             nev=nev,
                                             N_shoot=N_shoot,
                                             μ_min=μ_min,
                                             N_scan=N_scan,
                                             bisect_tol=bisect_tol,
                                             max_bisect=max_bisect,
                                             ode_atol=ode_atol,
                                             ode_rtol=ode_rtol,
                                             fd_nev=max(nev, 6))
    data === nothing && return fill(NaN, nev)
    found = collect(data.eigenvalues)
    while length(found) < nev
        push!(found, NaN)
    end
    return found[1:nev]
end

function compute_Lplus_root_near_full(a, b, E, Vfun, c, slope_sign;
                                      μ_target=0.0,
                                      μ_window=max(1e-9, 10 * abs(μ_target), 5 * c^4),
                                      accept_radius=Inf,
                                      N_shoot=3000,
                                      N_scan_local=400,
                                      bisect_tol=1e-12,
                                      max_bisect=80,
                                      ode_atol=1e-12,
                                      ode_rtol=1e-10)
    E < 0 || error("E must be negative")
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return NaN

    x0L, x0R, _, _ = tail_shifts_from_ends(a, b, E, ui[1], vi[1], ui[end], vi[end])
    (isfinite(x0L) && isfinite(x0R)) || return NaN

    κv = sqrt(-E)
    uL = tanh(κv * (a - x0L))
    uR = tanh(κv * (b - x0R))
    ψ_at(x::Float64) = _linear_interp_scalar(xi, ui, x)

    function H(μ::Float64)
        μ >= κv^2 && return NaN
        m = sqrt(κv^2 - μ) / κv
        DL = _pt_plus_den(m, uL)
        wL_num = _pt_plus_num(κv, m, uL)
        DR = _pt_plus_den(m, uR)
        wR_num = _pt_plus_num(κv, m, uR)

        function ode!(du, u, p, t)
            du[1] = u[2]
            du[2] = (Vfun(t) - 3.0 * ψ_at(t)^2 - E - μ) * u[1]
        end
        prob = ODEProblem(ode!, ComplexF64[DL, wL_num], (a, b))
        sol = solve(prob, Tsit5(); abstol=ode_atol, reltol=ode_rtol, save_everystep=false)
        isempty(sol.u) && return NaN
        φb, φb′ = sol.u[end]
        return real(φb′ * DR - wR_num * φb)
    end

    _golden_section_min_abs(f, lo, hi; maxit=50) = begin
        ϕ = (sqrt(5) - 1) / 2
        a0, b0 = lo, hi
        c0 = b0 - ϕ * (b0 - a0)
        d0 = a0 + ϕ * (b0 - a0)
        fc = abs(f(c0))
        fd = abs(f(d0))
        for _ in 1:maxit
            if fc < fd
                b0 = d0
                d0 = c0
                fd = fc
                c0 = b0 - ϕ * (b0 - a0)
                fc = abs(f(c0))
            else
                a0 = c0
                c0 = d0
                fc = fd
                d0 = a0 + ϕ * (b0 - a0)
                fd = abs(f(d0))
            end
        end
        return fc < fd ? c0 : d0
    end

    _newton_refine_μ(f, μ0, μlo, μhi; maxit=30, tol=bisect_tol) = begin
        μ = clamp(μ0, μlo, μhi)
        for _ in 1:maxit
            fμ = f(μ)
            isfinite(fμ) || break
            abs(fμ) ≤ tol && return μ, fμ, true
            δμ = max(1e-12, 1e-5 * max(abs(μ), 1.0), 0.05 * μ_window)
            μp = min(μ + δμ, μhi)
            μm = max(μ - δμ, μlo)
            fp = f(μp)
            fm = f(μm)
            (isfinite(fp) && isfinite(fm)) || break
            df = (fp - fm) / (μp - μm)
            abs(df) ≤ 1e-14 && break
            μnew = clamp(μ - fμ / df, μlo, μhi)
            abs(μnew - μ) ≤ tol * max(abs(μ), 1.0) && return μnew, f(μnew), true
            μ = μnew
        end
        fμ = f(μ)
        return μ, fμ, isfinite(fμ) && abs(fμ) ≤ 100tol
    end

    _accept(μ) = isfinite(μ) && abs(μ - μ_target) ≤ accept_radius

    for scale in (1.0, 2.0, 4.0, 8.0)
        w = scale * μ_window
        lo = max(μ_target - w, -10.0)
        hi = min(μ_target + w, κv^2 - 1e-10)
        hi > lo || continue
        μ_grid = range(lo, hi; length=N_scan_local)
        Hv = [H(μ) for μ in μ_grid]

        for i in 1:length(μ_grid)-1
            Hlo, Hhi = Hv[i], Hv[i + 1]
            isfinite(Hlo) && isfinite(Hhi) || continue
            Hlo * Hhi < 0 || continue
            lo_b, hi_b, Hlob = μ_grid[i], μ_grid[i + 1], Hlo
            broken = false
            for _ in 1:max_bisect
                mid = 0.5 * (lo_b + hi_b)
                Hmid = H(mid)
                if !isfinite(Hmid)
                    broken = true
                    break
                end
                Hlob * Hmid < 0 ? (hi_b = mid) : (lo_b = mid; Hlob = Hmid)
                hi_b - lo_b < bisect_tol && break
            end
            if !broken
                μroot = 0.5 * (lo_b + hi_b)
                _accept(μroot) && return μroot
            end
        end

        absHv = abs.(Hv)
        imin = argmin(absHv)
        if isfinite(absHv[imin]) && absHv[imin] ≤ 1e-6
            iL = max(imin - 1, 1)
            iR = min(imin + 1, length(μ_grid))
            μcand = _golden_section_min_abs(H, μ_grid[iL], μ_grid[iR])
            μref, href, ok = _newton_refine_μ(H, μcand, μ_grid[iL], μ_grid[iR])
            ok && abs(href) ≤ 1e-6 && _accept(μref) && return μref
        end
    end

    return NaN
end

function build_Lplus_residual_full(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=3000,
                                   ode_atol=1e-12,
                                   ode_rtol=1e-10)
    evans = build_Lplus_evans_full(a, b, E, Vfun, c, slope_sign;
                                   N_shoot=N_shoot, ode_atol=ode_atol, ode_rtol=ode_rtol)
    evans === nothing && return nothing
    return μ -> evans.eval_at(μ).H
end

function refine_Lplus_root_full(a, b, E, Vfun, c, slope_sign;
                                μ_guess=0.0,
                                μ_window=max(1e-9, 10 * abs(μ_guess), 5 * c^4),
                                accept_radius=Inf,
                                N_shoot=3000,
                                ode_atol=1e-12,
                                ode_rtol=1e-10,
                                newton_tol=1e-12,
                                newton_maxit=30)
    H = build_Lplus_residual_full(a, b, E, Vfun, c, slope_sign;
                                  N_shoot=N_shoot, ode_atol=ode_atol, ode_rtol=ode_rtol)
    H === nothing && return NaN

    κv = sqrt(-E)
    μlo = max(μ_guess - μ_window, -10.0)
    μhi = min(μ_guess + μ_window, κv^2 - 1e-10)
    μhi > μlo || return NaN

    _accept(μ) = isfinite(μ) && abs(μ - μ_guess) ≤ accept_radius

    μ = clamp(μ_guess, μlo, μhi)
    for _ in 1:newton_maxit
        Hμ = H(μ)
        isfinite(Hμ) || break
        abs(Hμ) ≤ newton_tol && return _accept(μ) ? μ : NaN
        δμ = max(1e-12, 1e-6 * max(abs(μ), 1.0), 0.02 * μ_window)
        μp = min(μ + δμ, μhi)
        μm = max(μ - δμ, μlo)
        Hp = H(μp)
        Hm = H(μm)
        (isfinite(Hp) && isfinite(Hm)) || break
        dH = (Hp - Hm) / (μp - μm)
        abs(dH) ≤ 1e-14 && break
        μnew = clamp(μ - Hμ / dH, μlo, μhi)
        abs(μnew - μ) ≤ newton_tol * max(abs(μ), 1.0) && return _accept(μnew) ? μnew : NaN
        μ = μnew
    end

    return compute_Lplus_root_near_full(a, b, E, Vfun, c, slope_sign;
                                        μ_target=μ_guess,
                                        μ_window=μ_window,
                                        accept_radius=accept_radius,
                                        N_shoot=N_shoot,
                                        ode_atol=ode_atol,
                                        ode_rtol=ode_rtol)
end

function refine_Lplus_nu_full(a, b, E, Vfun, c, slope_sign;
                              ν_guess=0.0,
                              ν_window=max(0.5, 2 * abs(ν_guess)),
                              accept_radius=Inf,
                              N_shoot=3000,
                              ode_atol=1e-12,
                              ode_rtol=1e-10,
                              newton_tol=1e-12,
                              newton_maxit=30,
                              N_scan_local=400)
    ε = c^2
    ε > 0 || return 0.0
    μ_guess = ε^2 * ν_guess
    μ_window = max(1e-12, ε^2 * ν_window)
    μ_accept = isfinite(accept_radius) ? ε^2 * accept_radius : Inf

    μ_ref = compute_Lplus_root_near_full(
        a, b, E, Vfun, c, slope_sign;
        μ_target=μ_guess,
        μ_window=μ_window,
        accept_radius=μ_accept,
        N_shoot=N_shoot,
        N_scan_local=N_scan_local,
        bisect_tol=newton_tol,
        max_bisect=newton_maxit,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
    )

    return isfinite(μ_ref) ? μ_ref / ε^2 : NaN
end

function compute_Lplus_mode_fd_full(a, b, E, Vfun, c, slope_sign;
                                    mode_index=2,
                                    Xmax=8.0,
                                    N_shoot=2000,
                                    N_grid=900)
    E < 0 || error("E must be negative")
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return Float64[], Float64[], NaN

    x0L, x0R, sL, sR = tail_shifts_from_ends(a, b, E, ui[1], vi[1], ui[end], vi[end])
    (isfinite(x0L) && isfinite(x0R)) || return Float64[], Float64[], NaN

    κv = sqrt(-E)
    A = sqrt(-2E)
    x_lo = min(-Xmax, a - Xmax)
    x_hi = max(Xmax, b + Xmax)
    x = collect(range(x_lo, x_hi; length=N_grid))
    h = x[2] - x[1]
    xi_r = real.(xi)
    ui_r = real.(ui)

    ψ = similar(x)
    for (j, xx) in enumerate(x)
        ψ[j] = if xx < a
            sL * A * sech(κv * (xx - x0L))
        elseif xx > b
            sR * A * sech(κv * (xx - x0R))
        else
            _linear_interp_scalar(xi_r, ui_r, xx)
        end
    end

    Vv = Vfun.(x)
    main = 2 / h^2 .+ Vv .- E .- 3 .* ψ.^2
    off = fill(-1 / h^2, length(x) - 1)
    F = eigen(Symmetric(Matrix(SymTridiagonal(main, off))))
    vals = F.values
    vecs = F.vectors
    idx = clamp(mode_index, 1, length(vals))
    ϕ = vecs[:, idx]
    normϕ = sqrt(sum(abs2, ϕ) * h)
    normϕ > 0 && (ϕ ./= normϕ)
    imax = argmax(abs.(ϕ))
    ϕ[imax] < 0 && (ϕ .*= -1)
    return x, ϕ, vals[idx]
end

function branch_index_near_bifurcation(branches, E_bif; param_getter, E_tol=0.05)
    candidates = Tuple{Int,Float64}[]
    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue
        Es = [sol.param for sol in br.branch if param_getter(sol) > 1e-12]
        isempty(Es) && continue
        push!(candidates, (i, maximum(Es)))
    end
    isempty(candidates) && return nothing

    close = [item for item in candidates if abs(item[2] - E_bif) <= E_tol]
    if !isempty(close)
        _, idx = findmin(abs.(last.(close) .- E_bif))
        return close[idx][1]
    end

    _, idx = findmin(abs.(last.(candidates) .- E_bif))
    return candidates[idx][1]
end
