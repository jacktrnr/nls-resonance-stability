###############################################################
# generate_paper_figures.jl
#
# Generates all figures for main-Apr7.tex (Stability paper).
#
# Outputs (saved to Figures/):
#   smooth-potential.png       — Fig 1(a): V(x) on [0,1]
#   smooth-Ustar.pdf           — Fig 1(b): Resonance function U★(x)
#   smooth-NvsE.png            — Fig 1(c): N vs E bifurcation diagram
#   smooth-profiles.png        — Fig 1(d): Representative ψ_ε profiles
#   smooth-mu.pdf              — Fig 2(a): μ(ε) vs prediction
#   smooth-dNdE.pdf            — Fig 2(b): dN/dE vs prediction
#   transmission-potential.png     — V(x) = 3sin(πx) on [-1,1]
#   transmission-Ustar.pdf        — Transmission resonance U★(x)
#   transmission-NvsE.png         — N vs E bifurcation diagram
#   transmission-profiles.png     — Representative ψ_ε profiles
#   transmission-mu.pdf           — μ(ε) vs prediction
#   transmission-dNdE.pdf         — dN/dE vs prediction
#   transmission-neg-potential.png — -V(x) = -3sin(πx) on [-1,1]
#   transmission-neg-Ustar.pdf    — Transmission resonance U★(x) for -V
#   transmission-neg-NvsE.png     — N vs E bifurcation diagram for -V
#   transmission-neg-profiles.png — Representative ψ_ε profiles for -V
#   transmission-neg-mu.pdf       — μ(ε) vs prediction for -V
#   transmission-neg-dNdE.pdf     — dN/dE vs prediction for -V
#
# Half-line (Figs 1–2): V₀ = -7 smooth well on [0,1], Dirichlet at x = 0
# Full-line (Figs 3–4): V(x) = 3sin(πx) on [-1,1], transmission resonance
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "nls_bifurcation", "NLSBifurcation.jl"))
using .NLSBifurcation
include(joinpath(@__DIR__, "..", "src", "wells.jl"))
include(joinpath(@__DIR__, "..", "src", "lplus.jl"))

using OrdinaryDiffEq
using LinearAlgebra
using Plots, LaTeXStrings
using Printf
using Colors
using JLD2

ENV["GKSwstype"] = "100"   # headless plotting
gr()

# ── Global plot style (journal quality) ──────────────────────
default(
    fontfamily   = "Computer Modern",
    tickfontsize = 11,
    guidefontsize = 14,
    legendfontsize = 10,
    titlefontsize = 14,
    framestyle   = :box,
    grid         = false,
    linewidth    = 2.5,
    markersize   = 5,
    dpi          = 300,
    size         = (520, 380),
    margin       = 5Plots.mm,
    foreground_color_legend = nothing,
    background_color_legend = :white,
)

figdir = joinpath(@__DIR__, "Figures")
mkpath(figdir)

# ── Rescaled matching D̃(ν) for transmission eigenvalue ──────
# (replaces degenerate Evans function for full-line case)
function _compute_Dtilde_free(a, b, E, Vfun, c, ν, slope_sign; N_ode=3000)
    ε = c^2;  ε < 1e-30 && return NaN
    μ = ε^2 * ν
    κv = sqrt(-E);  μ >= κv^2 && return NaN
    λ = sqrt(κv^2 - μ);  A = sqrt(-2E)

    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_ode, c=c, slope_sign=slope_sign)
    isempty(xi) && return NaN

    ub, vb = real(ui[end]), real(vi[end])
    (abs(ub) > A + 1e-10 || abs(ub) < 1e-15) && return NaN
    acosh_R = acosh(A / abs(ub))
    x0R = (vb * ub > 0) ? (b + acosh_R / κv) : (b - acosh_R / κv)
    uR  = tanh(κv * (b - x0R))

    m  = λ / κv
    DR = m^2 - 1 + 3m * uR + 3uR^2
    abs(DR) < 1e-30 && return NaN
    wR_num = κv * (-m * DR + (1 - uR^2) * (3m + 6uR))
    w_out  = wR_num / DR

    ψ_at(x) = _linear_interp_scalar(xi, ui, x)
    function ode!(du, u, p, t)
        du[1] = u[2]
        du[2] = (Vfun(t) - 3.0 * ψ_at(t)^2 - E - μ) * u[1]
    end
    prob = ODEProblem(ode!, [1.0, λ], (Float64(a), Float64(b)))
    sol  = solve(prob, Tsit5(); abstol=1e-13, reltol=1e-11, save_everystep=false)
    isempty(sol.u) && return NaN
    φb, φb′ = sol.u[end]
    abs(φb) < 1e-30 && return NaN
    w_in = φb′ / φb

    return (w_in - w_out) / ε
end

function _find_nu_root_dtilde(a, b, E, Vfun, c, ν_guess, slope_sign;
                               ν_window=8.0, N_pts=500, N_ode=3000)
    ν_lo = ν_guess - ν_window
    ν_hi = ν_guess + ν_window
    ν_range = range(ν_lo, ν_hi; length=N_pts)
    D_vals = [_compute_Dtilde_free(a, b, E, Vfun, c, ν, slope_sign; N_ode=N_ode)
              for ν in ν_range]

    roots = Float64[]
    for i in 1:length(ν_range)-1
        D1, D2 = D_vals[i], D_vals[i+1]
        (isfinite(D1) && isfinite(D2) && D1 * D2 < 0) || continue
        lo, hi = ν_range[i], ν_range[i+1];  Dlo = D1
        for _ in 1:60
            mid = 0.5(lo + hi)
            Dm  = _compute_Dtilde_free(a, b, E, Vfun, c, mid, slope_sign; N_ode=N_ode)
            isfinite(Dm) || break
            if Dlo * Dm < 0;  hi = mid  else  lo = mid;  Dlo = Dm  end
        end
        push!(roots, 0.5(lo + hi))
    end

    isempty(roots) && return NaN
    _, best = findmin(abs.(roots .- ν_guess))
    return roots[best]
end

# ═══════════════════════════════════════════════════════════════
# PART A:  Half-line smooth well  (Figures 1 and 2)
# ═══════════════════════════════════════════════════════════════

println("="^70)
println("  PART A: Half-line smooth well  V₀ = -7, b = 1")
println("="^70)

# ── Potential ────────────────────────────────────────────────
const b_hl  = 1.0
const V0_hl = -7.0

function smooth_well_hl(x)
    (0 < x < b_hl) || return 0.0
    t = (x / b_hl)^2
    t >= 1.0 ? 0.0 : V0_hl * exp(-1.0 / (1.0 - t))
end

# ── Find scattering resonance by direct ODE shooting ────────
# Resonance condition: L₀ U★ = 0 on [0,b], U★(0)=0, U★'(b) = γ U★(b)
# Shoot (-U'' + V U = -γ² U) from x=0 with [U(0)=0, U'(0)=1], find γ s.t. G(γ)=0
function find_halfline_resonance_γ(b, Vfun; γ_min=0.01, γ_max=8.0, N_γ=2000)
    function G(γ)
        sol = solve(ODEProblem(
            (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) + γ^2)*u[1]),
            [0.0, 1.0], (0.0, b)),
            Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false)
        Ub, Upb = sol.u[end]
        return Upb - γ * Ub
    end

    γ_grid = collect(range(γ_min, γ_max; length=N_γ))
    G_vals = G.(γ_grid)

    roots = Float64[]
    for j in 1:length(G_vals)-1
        G_vals[j] * G_vals[j+1] > 0 && continue
        lo, hi = γ_grid[j], γ_grid[j+1]
        for _ in 1:80
            mid = 0.5(lo + hi)
            Gm = G(mid)
            abs(Gm) < 1e-14 && break
            abs(hi - lo) < 1e-14 && break
            sign(Gm) == sign(G_vals[j]) ? (lo = mid) : (hi = mid)
        end
        push!(roots, 0.5(lo + hi))
    end
    return sort(roots)
end

println("  Finding half-line resonance...")
γ_all = find_halfline_resonance_γ(b_hl, smooth_well_hl)
@printf("  Found %d resonance(s): %s\n", length(γ_all),
        join([@sprintf("%.6f", g) for g in γ_all], ", "))

γ_hl = γ_all[1]   # smallest γ = closest to threshold
E_bif_hl = -γ_hl^2
@printf("  Using γ = %.8f,  E_bif = %.8f\n", γ_hl, E_bif_hl)

# Compute U★ on [0,b] with U★(0)=0, U★'(0)=1
x_us = collect(range(0, b_hl; length=2001))
_sol_us = solve(ODEProblem(
    (du, u, p, x) -> (du[1] = u[2]; du[2] = (smooth_well_hl(x) + γ_hl^2)*u[1]),
    [0.0, 1.0], (0.0, b_hl)),
    Tsit5(); reltol=1e-13, abstol=1e-15, saveat=x_us)
U_star = [u[1] for u in _sol_us.u]
h_us = x_us[2] - x_us[1]
U_b = U_star[end]

# Compute integrals and formulas
I₂ = h_us * (sum(U_star[1:end-1].^2) + 0.5 * U_star[end]^2)
I₄ = h_us * (sum(U_star[1:end-1].^4) + 0.5 * U_star[end]^4)

# Half-line: U_a = 0 (Dirichlet)
Ω_hl = U_b^4 / (2γ_hl) - 2I₄
ℬ_hl = I₂ - U_b^2 / (2γ_hl)
ν₀_hl = 3Ω_hl / (4γ_hl)
dNdE_hl = -2/γ_hl + 2ℬ_hl^2 / Ω_hl
N_bif_hl = 4γ_hl    # half-line: one soliton tail

@printf("  U_b = %.8f,  I₂ = %.8f,  I₄ = %.8f\n", U_b, I₂, I₄)
@printf("  Ω = %.8f,  ℬ = %.8f,  ν₀ = %.8f\n", Ω_hl, ℬ_hl, ν₀_hl)
@printf("  dN/dE = %.8f\n", dNdE_hl)

# ── Figure 1(a): Potential ───────────────────────────────────
println("  Plotting Fig 1(a): potential...")
xV = range(-0.15, 1.15; length=800)
plt_V = plot(xV, smooth_well_hl.(xV);
    xlabel  = L"x",
    ylabel  = L"V(x)",
    color   = :black,
    lw      = 2.5,
    legend  = false,
    title   = "",
)
savefig(plt_V, joinpath(figdir, "smooth-potential.png"))
println("    → smooth-potential.png")

# ── Figure 1(b): Resonance function U★ ──────────────────────
println("  Plotting Fig 1(b): U★...")
# Show U★ on [0,b] plus decaying tail beyond b
x_tail = range(b_hl, 2.5; length=300)
U_tail = U_b .* exp.(γ_hl .* (x_tail .- b_hl))

plt_U = plot(x_us, U_star;
    xlabel = L"x",
    ylabel = L"U_\star(x)",
    color  = :steelblue,
    lw     = 2.5,
    label  = L"U_\star",
    legend = :topleft,
    title  = "",
)
plot!(plt_U, x_tail, U_tail;
    color = :steelblue, lw = 2.5, ls = :dash, label = "")
vline!(plt_U, [b_hl]; color = :gray, lw = 1.0, ls = :dot, label = L"x = b")
savefig(plt_U, joinpath(figdir, "smooth-Ustar.pdf"))
println("    → smooth-Ustar.pdf")

# ── Run half-line continuation ───────────────────────────────
println("  Running half-line continuation...")
a_hl = 0.0
N_ode = 3000

E_seed_hl = E_bif_hl - 0.2     # scan just BELOW bifurcation
p_min_hl  = E_bif_hl - 3.0    # ensure E_bif is well within bounds
seeds_hl = find_all_seeds_dirichlet(a_hl, b_hl, smooth_well_hl;
    E_list = [E_seed_hl],
    N      = N_ode,
    βmax   = 10.0,
    nscan  = 3400,
    tolH   = 1e-8)
seeds_hl = deduplicate_dirichlet_seeds(seeds_hl)
@printf("  Found %d seed(s) near E = %.4f\n", length(seeds_hl), E_seed_hl)

global br_hl = nothing
for seed in seeds_hl
    br = continue_single_seed_dirichlet(seed, a_hl, b_hl, smooth_well_hl;
        N        = N_ode,
        p_min    = p_min_hl,
        p_max    = -1e-6,
        ds       = 0.0005,
        dsmin    = 1e-12,
        dsmax    = 0.001,
        max_steps = 5000)
    if !isempty(br.branch)
        global br_hl = br
        break
    end
end

if br_hl === nothing || isempty(br_hl.branch)
    @warn "Half-line continuation failed — skipping Figs 1c/d and 2."
else
    npts = length(br_hl.branch)
    @printf("  Branch has %d points\n", npts)

    # Extract E, N, β along the branch
    Es_hl  = Float64[]
    Ns_hl  = Float64[]
    βs_hl  = Float64[]
    for sol in br_hl.branch
        sol.β < 1e-12 && continue
        sol.param > 0 && continue
        xs, us, vs = integrate_support_dirichlet(a_hl, b_hl, sol.param, smooth_well_hl;
                                                  N=N_ode, β=sol.β)
        isempty(xs) && continue
        N_val = compute_norm_dirichlet(a_hl, b_hl, sol.param, xs, us, vs)
        isfinite(N_val) || continue
        push!(Es_hl, sol.param)
        push!(Ns_hl, N_val)
        push!(βs_hl, sol.β)
    end

    # ── Figure 1(c): N vs E ──────────────────────────────────
    println("  Plotting Fig 1(c): N vs E...")
    Emin_plot = E_bif_hl - 1.5   # focus near bifurcation
    E_range = range(Emin_plot, -0.01; length=500)
    N_sol   = 4 .* sqrt.(abs.(E_range))
    N0_ref = 4γ_hl

    plt_NE  = plot(E_range, N_sol;
        xlabel  = L"E",
        ylabel  = L"\mathcal{N}",
        color   = :darkorange,
        ls      = :dot,
        lw      = 1.5,
        label   = L"4\sqrt{|E|}",
        legend  = :topright,
        title   = "",
    )
    # Shaded stability region: between 4√|E| and 4γ, on BOTH sides of bifurcation
    E_shade = range(Emin_plot, -0.01; length=500)
    N_shade_lo = 4 .* sqrt.(abs.(E_shade))
    plot!(plt_NE, E_shade, N_shade_lo;
        fillrange = N0_ref,
        fillalpha = 0.15,
        fillcolor = :darkorange,
        color     = :darkorange, lw = 0, label = "")
    hline!(plt_NE, [N0_ref]; color = :darkorange, ls = :dash, lw = 1.5,
           label = L"\mathcal{N}_\star = 4\gamma")
    # Bifurcation branch
    plot!(plt_NE, Es_hl, Ns_hl;
        color = :steelblue, lw = 2.5, label = "branch")
    # Bifurcation point
    scatter!(plt_NE, [E_bif_hl], [N0_ref];
        marker = :circle, ms = 8,
        color  = :limegreen, markerstrokewidth = 2, markerstrokecolor = :black,
        label  = "")
    # Pick 3 marked points near bifurcation
    # Sort by distance from E_bif so index 1 = closest to bifurcation
    sp_bif = sortperm(abs.(Es_hl .- E_bif_hl))
    n_branch = length(Es_hl)
    if n_branch >= 3
        prof_idxs = [sp_bif[max(1, round(Int, 0.02n_branch))],
                     sp_bif[round(Int, 0.15n_branch)],
                     sp_bif[round(Int, 0.40n_branch)]]
    else
        prof_idxs = collect(1:n_branch)
    end
    prof_colors = [RGB(0.10, 0.35, 0.70), RGB(0.00, 0.60, 0.45), RGB(0.80, 0.20, 0.20)]
    for (i, idx) in enumerate(prof_idxs)
        scatter!(plt_NE, [Es_hl[idx]], [Ns_hl[idx]];
            marker = :circle, ms = 7,
            color  = prof_colors[min(i, length(prof_colors))],
            markerstrokewidth = 1.5, markerstrokecolor = :black,
            label  = "")
    end
    xlims!(plt_NE, (Emin_plot, 0.0))
    savefig(plt_NE, joinpath(figdir, "smooth-NvsE.png"))
    println("    → smooth-NvsE.png")

    # ── Figure 1(d): Representative profiles ─────────────────
    println("  Plotting Fig 1(d): profiles...")
    prof_xs_all = Vector{Vector{Float64}}()
    prof_us_all = Vector{Vector{Float64}}()
    prof_εs = Float64[]
    plt_prof = plot(;
        xlabel = L"x",
        ylabel = L"\psi_\varepsilon(x)",
        legend = :topright,
        title  = "",
    )
    for (i, idx) in enumerate(prof_idxs)
        # Find closest branch point
        dists = [abs(s.param - Es_hl[idx]) + abs(s.β - βs_hl[idx]) for s in br_hl.branch]
        sol = br_hl.branch[argmin(dists)]
        xs, us, vs = integrate_support_dirichlet(a_hl, b_hl, sol.param, smooth_well_hl; N=N_ode, β=sol.β)
        isempty(xs) && continue
        xg, ug = glue_dirichlet_solution(a_hl, b_hl, sol.param, xs, us, vs; Xmax=10.0)
        ε_val = sol.β^2
        push!(prof_xs_all, real.(xg))
        push!(prof_us_all, real.(ug))
        push!(prof_εs, ε_val)
        plot!(plt_prof, real.(xg), real.(ug);
            color = prof_colors[min(i, length(prof_colors))],
            lw    = 2.2,
            label = latexstring(@sprintf("\\varepsilon = %.3f", ε_val)))
    end
    savefig(plt_prof, joinpath(figdir, "smooth-profiles.png"))
    println("    → smooth-profiles.png")
    # Save node data for replotting
    node_Es = [Es_hl[idx] for idx in prof_idxs]
    node_Ns = [Ns_hl[idx] for idx in prof_idxs]

    # ── Compute μ(ε) along branch (targeted Dirichlet Evans) ──
    println("  Computing μ(ε) along branch (Dirichlet Evans)...")
    ε_mu  = Float64[]
    μ_num = Float64[]
    μ_ana = Float64[]
    E_mu  = Float64[]
    N_mu  = Float64[]

    # Local Evans function for Dirichlet half-line L₊ eigenvalue near μ_target
    function find_dirichlet_mu_local(b, E, Vfun, β; μ_target=0.0, μ_window=0.5,
                                     N_shoot=3000, N_scan_local=600)
        E < 0 || return NaN
        # Shoot NLS with Dirichlet BC: ψ(0)=0, ψ'(0)=β
        xi_s, ui_s, vi_s = _shoot_from_origin_local(
            b, E, Vfun, β; N=N_shoot, ode_rtol=1e-13, ode_atol=1e-15)
        isempty(xi_s) && return NaN
        h_i = xi_s[2] - xi_s[1]
        s_t, κv, A_t, x_shift = _sech_tail_params_local(
            b, E, ui_s[end], vi_s[end])
        !isfinite(x_shift) && return NaN
        u_b = tanh(κv * (b - x_shift))

        ψ_at(x) = begin
            x ≤ 0.0 && return 0.0
            x ≥ b && return s_t * A_t * sech(κv * (x - x_shift))
            idx = clamp(floor(Int, x / h_i) + 1, 1, length(xi_s) - 1)
            return ui_s[idx] + (x - xi_s[idx]) / h_i * (ui_s[idx+1] - ui_s[idx])
        end

        function HD(μ)
            μ >= κv^2 && return NaN
            m = sqrt(κv^2 - μ) / κv
            sol = solve(ODEProblem(
                (du, u, p, t) -> (du[1] = u[2]; du[2] = (Vfun(t) - 3.0*ψ_at(t)^2 - E - μ)*u[1]),
                [0.0, 1.0], (0.0, b)),
                Tsit5(); abstol=1e-14, reltol=1e-12, save_everystep=false)
            isempty(sol.u) && return NaN
            φb, φb′ = sol.u[end]
            D  = _pt_plus_den(m, u_b)
            wn = _pt_plus_num(κv, m, u_b)
            return φb′ * D - wn * φb
        end

        μ_lo = μ_target - μ_window
        μ_hi = min(μ_target + μ_window, κv^2 - 1e-10)
        μ_grid = range(μ_lo, μ_hi; length=N_scan_local)
        Hv = HD.(μ_grid)

        # Find sign change closest to μ_target
        best_root = NaN
        best_dist = Inf
        for j in 1:length(Hv)-1
            (isfinite(Hv[j]) && isfinite(Hv[j+1])) || continue
            Hv[j] * Hv[j+1] > 0 && continue
            lo, hi, Hlo = μ_grid[j], μ_grid[j+1], Hv[j]
            for _ in 1:80
                mid = 0.5(lo + hi)
                Hm = HD(mid)
                !isfinite(Hm) && break
                abs(hi - lo) < 1e-14 && break
                Hlo * Hm < 0 ? (hi = mid) : (lo = mid; Hlo = Hm)
            end
            root = 0.5(lo + hi)
            d = abs(root - μ_target)
            if d < best_dist
                best_dist = d
                best_root = root
            end
        end
        return best_root
    end

    # Sample branch points (focused near bifurcation = small β)
    valid = [(i, sol) for (i, sol) in enumerate(br_hl.branch)
             if sol.β > 1e-12 && sol.param < 0]
    sort!(valid; by = t -> t[2].β)
    n_sample = min(80, length(valid))
    sample_idxs = unique(round.(Int, range(1, length(valid); length=n_sample)))

    for (cnt, si) in enumerate(sample_idxs)
        i, sol = valid[si]
        ε = sol.β^2
        μ_target = ν₀_hl * ε^2
        μ_window = max(0.01, 5.0 * abs(μ_target), 0.5 * ε)
        μ_val = find_dirichlet_mu_local(b_hl, sol.param, smooth_well_hl, sol.β;
                                         μ_target=μ_target, μ_window=μ_window,
                                         N_shoot=N_ode, N_scan_local=600)
        isfinite(μ_val) || continue

        xs, us, vs = integrate_support_dirichlet(a_hl, b_hl, sol.param, smooth_well_hl; N=N_ode, β=sol.β)
        isempty(xs) && continue
        N_val = compute_norm_dirichlet(a_hl, b_hl, sol.param, xs, us, vs)
        isfinite(N_val) || continue

        push!(ε_mu, ε)
        push!(μ_num, μ_val)
        push!(μ_ana, ν₀_hl * ε^2)
        push!(E_mu, sol.param)
        push!(N_mu, N_val)

        if cnt % 10 == 0
            @printf("    [%d/%d] β=%.5e, ε=%.5e, μ=%.5e (pred=%.5e)\n",
                    cnt, n_sample, sol.β, ε, μ_val, μ_target)
        end
    end

    if !isempty(ε_mu)
        sp = sortperm(ε_mu)
        ε_p  = ε_mu[sp]
        μ_p  = μ_num[sp]
        μ_a  = μ_ana[sp]
        E_p  = E_mu[sp]
        N_p  = N_mu[sp]

        # ── Figure 2(a): μ(ε) ────────────────────────────────
        println("  Plotting Fig 2(a): μ(ε)...")
        # Zoom to small ε for clean asymptotic comparison
        ε_max_plot = min(1.0, maximum(ε_p))
        mask_mu = ε_p .<= ε_max_plot
        ε_z = ε_p[mask_mu]; μ_z = μ_p[mask_mu]; μ_az = μ_a[mask_mu]
        plt_mu = plot(ε_z, μ_z;
            xlabel = L"\varepsilon = \beta^2",
            ylabel = L"\mu(\varepsilon)",
            color  = :steelblue,
            lw     = 0,
            markershape = :circle,
            ms     = 4,
            markerstrokewidth = 0,
            label  = L"\mu\ \mathrm{(computed)}",
            legend = :topleft,
            title  = "",
        )
        plot!(plt_mu, ε_z, μ_az;
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = L"\frac{3\varepsilon^2}{4\gamma}\Omega")
        savefig(plt_mu, joinpath(figdir, "smooth-mu.pdf"))
        println("    → smooth-mu.pdf")

        # ── Figure 2(b): dN/dE ───────────────────────────────
        println("  Plotting Fig 2(b): dN/dE...")
        # Numerical dN/dE by finite differences along the branch
        dNdE_num = Float64[]
        ε_dN     = Float64[]
        for j in 2:length(E_p)-1
            dE = E_p[j+1] - E_p[j-1]
            abs(dE) < 1e-14 && continue
            dN = N_p[j+1] - N_p[j-1]
            push!(dNdE_num, dN / dE)
            push!(ε_dN, ε_p[j])
        end

        plt_dN = plot(ε_dN, dNdE_num;
            xlabel = L"\varepsilon = \beta^2",
            ylabel = L"d\mathcal{N}/dE",
            color  = :steelblue,
            lw     = 0,
            markershape = :circle,
            ms     = 4,
            markerstrokewidth = 0,
            label  = L"d\mathcal{N}/dE\ \mathrm{(computed)}",
            legend = :topright,
            title  = "",
        )
        hline!(plt_dN, [dNdE_hl];
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("%.3f\\ \\mathrm{(predicted)}", dNdE_hl)))
        savefig(plt_dN, joinpath(figdir, "smooth-dNdE.pdf"))
        println("    → smooth-dNdE.pdf")

        # ── Save half-line data to JLD2 ──────────────────────────
        hl_datafile = joinpath(figdir, "smooth-data.jld2")
        hl_ν_p = μ_p ./ max.(ε_p.^2, 1e-30)
        hl_U_a = 0.0
        hl_Ė = Ω_hl / (2ℬ_hl)
        jldsave(hl_datafile;
            ε_p=ε_p, μ_p=μ_p, E_p=E_p, N_p=N_p, ν_p=hl_ν_p,
            Es=Es_hl, Ns=Ns_hl,
            ν₀=ν₀_hl, N_bif=N_bif_hl, γ=γ_hl, E_bif=E_bif_hl,
            U_a=hl_U_a, U_b=U_b, I₂=I₂, I₄=I₄,
            Ω=Ω_hl, ℬ=ℬ_hl, x_us=x_us, U_us=U_star,
            Ė_correct=hl_Ė, dNdE_pred=dNdE_hl,
            node_Es=node_Es, node_Ns=node_Ns, prof_εs=prof_εs,
            prof_xs=prof_xs_all, prof_us=prof_us_all)
        println("    → smooth-data.jld2")
    else
        @warn "No μ values computed for half-line — skipping Fig 2."
    end
end

# ── Early exit: set HALF_LINE_ONLY=true to skip Part B ───────
if get(ENV, "HALF_LINE_ONLY", "") != ""
    println("\n" * "="^70)
    println("  DONE (half-line only) — figures saved to: $figdir")
    println("="^70)
    exit(0)
end

# ═══════════════════════════════════════════════════════════════
# Transmission resonance figure generator (reusable)
# ═══════════════════════════════════════════════════════════════

"""
    find_transmission_resonance(a, b, Vfun; γ_min=0.01, γ_max=8.0, N_γ=2000, search_negative=false)

Find transmission resonances: γ s.t. U★'(b) = γ U★(b) with IC [U★(a), U★'(a)] = [1, γ].
When `search_negative=true`, also search γ < 0 (returns signed γ; use |γ| in formulas).
"""
function find_transmission_resonance(a, b, Vfun; γ_min=0.01, γ_max=8.0, N_γ=2000,
                                      search_negative=false)
    function _scan_γ_range(γ_lo, γ_hi, _Vfun, _a, _b, _N_γ)
        γ_grid = range(γ_lo, γ_hi; length=_N_γ)
        G_vals = Float64[]
        for γ_try in γ_grid
            sol = solve(ODEProblem(
                (du, u, p, x) -> (du[1] = u[2]; du[2] = (_Vfun(x) + γ_try^2)*u[1]),
                [1.0, γ_try], (_a, _b)),
                Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false)
            Ub, Upb = sol.u[end]
            push!(G_vals, Upb - γ_try * Ub)
        end
        found = Float64[]
        for j in 1:length(G_vals)-1
            G_vals[j] * G_vals[j+1] > 0 && continue
            lo, hi = γ_grid[j], γ_grid[j+1]
            for _ in 1:80
                mid = 0.5(lo + hi)
                sol = solve(ODEProblem(
                    (du, u, p, x) -> (du[1] = u[2]; du[2] = (_Vfun(x) + mid^2)*u[1]),
                    [1.0, mid], (_a, _b)),
                    Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false)
                Gm = sol.u[end][2] - mid * sol.u[end][1]
                abs(Gm) < 1e-14 && break
                abs(hi - lo) < 1e-14 && break
                sign(Gm) == sign(G_vals[j]) ? (lo = mid) : (hi = mid)
            end
            push!(found, 0.5(lo + hi))
        end
        return found
    end

    roots = _scan_γ_range(γ_min, γ_max, Vfun, a, b, N_γ)
    @printf("    Positive scan: %d root(s): %s\n", length(roots),
            join([@sprintf("%.6f", r) for r in roots], ", "))
    if search_negative
        neg_roots = _scan_γ_range(-γ_max, -γ_min, Vfun, a, b, N_γ)
        @printf("    Negative scan: %d root(s): %s\n", length(neg_roots),
                join([@sprintf("%.6f", r) for r in neg_roots], ", "))
        append!(roots, neg_roots)
    end
    # Deduplicate roots within 1e-6
    unique_roots = Float64[]
    for r in sort(roots; by=abs)
        if all(abs(r - u) > 1e-6 for u in unique_roots)
            push!(unique_roots, r)
        end
    end
    return unique_roots
end

# Targeted full-line Evans function for L₊ eigenvalue near μ_target
function find_fullline_mu_local(a, b, E, Vfun, c, slope_sign;
                                μ_target=0.0, μ_window=0.5,
                                N_shoot=3000, N_scan_local=600)
    E < 0 || return NaN
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return NaN

    x0L, x0R, _, _ = tail_shifts_from_ends(a, b, E, ui[1], vi[1], ui[end], vi[end])
    (isfinite(x0L) && isfinite(x0R)) || return NaN

    κv = sqrt(-E)
    uL = tanh(κv * (a - x0L))
    uR = tanh(κv * (b - x0R))
    ψ_at(x) = _linear_interp_scalar(xi, ui, x)

    function H(μ)
        μ >= κv^2 && return NaN
        m = sqrt(κv^2 - μ) / κv
        DL = _pt_plus_den(m, uL)
        wL = _pt_plus_num(κv, m, uL)
        DR = _pt_plus_den(m, uR)
        wR = _pt_plus_num(κv, m, uR)
        sol_ev = solve(ODEProblem(
            (du, u, p, t) -> (du[1] = u[2]; du[2] = (Vfun(t) - 3.0*ψ_at(t)^2 - E - μ)*u[1]),
            ComplexF64[DL, wL], (a, b)),
            Tsit5(); abstol=1e-14, reltol=1e-12, save_everystep=false)
        isempty(sol_ev.u) && return NaN
        φb, φb′ = sol_ev.u[end]
        return real(φb′ * DR - wR * φb)
    end

    μ_lo = max(μ_target - μ_window, -10.0)
    μ_hi = min(μ_target + μ_window, κv^2 - 1e-10)
    μ_grid = range(μ_lo, μ_hi; length=N_scan_local)
    Hv = H.(μ_grid)

    # Find ALL sign-change roots, return the one closest to μ_target
    roots = Float64[]
    for j in 1:length(Hv)-1
        (isfinite(Hv[j]) && isfinite(Hv[j+1])) || continue
        Hv[j] * Hv[j+1] > 0 && continue
        lo, hi, Hlo = μ_grid[j], μ_grid[j+1], Hv[j]
        for _ in 1:80
            mid = 0.5(lo + hi)
            Hm = H(mid)
            !isfinite(Hm) && break
            abs(hi - lo) < 1e-14 && break
            Hlo * Hm < 0 ? (hi = mid) : (lo = mid; Hlo = Hm)
        end
        push!(roots, 0.5(lo + hi))
    end
    isempty(roots) && return NaN
    _, best = findmin(abs.(roots .- μ_target))
    return roots[best]
end

"""
    run_transmission_figures(Vfun, a, b, prefix, Vlabel; figdir, N_ode)

Generate all transmission resonance figures (potential, U★, N vs E, profiles, μ(ε), dN/dE)
for a given potential Vfun on [a,b]. Files are saved with the given `prefix`.
Uses |γ| in all formulas (per Remark 4.6: reduce to γ > 0 by reflecting V).
When the natural γ is negative, reflects V(x) → V(-x) for continuation (Remark 4.6).
"""
function run_transmission_figures(Vfun, a, b, prefix, Vlabel;
                                   figdir, N_ode, prof_colors)
    println("\n" * "="^70)
    println("  Transmission resonance:  $Vlabel")
    println("="^70)

    # ── Find transmission resonance ─────────────────────────────
    println("  Finding transmission resonance (searching both signs of γ)...")
    γ_roots = find_transmission_resonance(a, b, Vfun; search_negative=true)
    @printf("  Found %d transmission resonance(s): γ = %s\n",
            length(γ_roots), join([@sprintf("%.6f", g) for g in γ_roots], ", "))
    isempty(γ_roots) && (@warn "No transmission resonances found — skipping $prefix."; return)

    γ_raw = γ_roots[1]   # smallest |γ| = closest to threshold
    γ = abs(γ_raw)        # always use |γ| (Remark 4.6)
    @printf("  Using γ_raw = %.8f → |γ| = %.8f\n", γ_raw, γ)
    E_bif = -γ^2

    # Remark 4.6: If γ_raw < 0, the soliton tail matching doesn't work directly.
    # Reduce to γ > 0 by reflecting: V(x) → V(-x). Bound states satisfy
    # ψ_{-V}(x) = ψ_{V}(-x) with identical E, N, μ(ε).
    reflected = γ_raw < 0
    if reflected
        println("  γ < 0 → reflecting V(x) → V(-x) for continuation (Remark 4.6)")
        Vfun_cont = x -> Vfun(-x)   # reflected potential for continuation
        # The reflected potential has γ > 0 at the SAME |γ|
        γ_cont = γ  # positive
    else
        Vfun_cont = Vfun
        γ_cont = γ_raw
    end

    # Compute U★ on [a,b] for the ORIGINAL Vfun (for plotting and integrals)
    # U★(a) = 1, U★'(a) = γ_raw (signed BC)
    sol_us = solve(ODEProblem(
        (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) + γ^2)*u[1]),
        [1.0, γ_raw], (a, b)),
        Tsit5(); reltol=1e-13, abstol=1e-15,
        saveat=range(a, b; length=2001))
    x_us = sol_us.t
    U_us = [u[1] for u in sol_us.u]
    U_b  = U_us[end]
    h    = x_us[2] - x_us[1]

    I₂ = h * (0.5*U_us[1]^2 + sum(U_us[2:end-1].^2) + 0.5*U_us[end]^2)
    I₄ = h * (0.5*U_us[1]^4 + sum(U_us[2:end-1].^4) + 0.5*U_us[end]^4)

    # Paper formulas (unified April 9, 2026 — new Ω absorbs Ua⁴/(4γ))
    U_a = 1.0
    Ω   = (2U_b^4 + U_a^4) / (4γ) - 2I₄          # eq 4.2 (new)
    ℬ   = I₂ - (U_b^2 - U_a^2) / (2γ)             # paper ℬ (MINUS sign)
    Ω̃   = Ω - 3U_a^4 / (4γ)                        # convenience
    ν₀  = 3Ω / (4γ)                                 # μ = ν₀ε²
    Ė_tex = Ω̃ / (2ℬ)                                # paper Ė
    dNdE_pred = 2ℬ^2 / Ω̃ - 2 / γ                   # paper dN/dE

    # Cross-check: direct existence matching (should agree)
    Ė_correct = -(I₄ + (U_a^4 - U_b^4) / (4γ)) / (I₂ + (U_a^2 - U_b^2) / (2γ))

    N_bif = 4γ    # one effective soliton tail (left tail negligible)

    @printf("  |γ| = %.8f,  E_bif = %.8f\n", γ, E_bif)
    @printf("  U_a = %.8f,  U_b = %.8f\n", U_a, U_b)
    @printf("  I₂ = %.8f,  I₄ = %.8f\n", I₂, I₄)
    @printf("  Ω = %.8f,  ℬ = %.8f,  Ω̃ = %.8f\n", Ω, ℬ, Ω̃)
    @printf("  ν₀ = %.8f\n", ν₀)
    @printf("  Ė = %.8f  (cross-check: %.8f)\n", Ė_tex, Ė_correct)
    @printf("  dN/dE = %.8f\n", dNdE_pred)
    @printf("  Stability: Ω %s 0, dN/dE %s 0 → %s\n",
            Ω > 0 ? ">" : "<", dNdE_pred < 0 ? "<" : ">",
            (Ω > 0 && dNdE_pred < 0) ? "STABLE" : "UNSTABLE")
    @printf("  U★ range: [%.6f, %.6f]  (positive = %s)\n",
            minimum(U_us), maximum(U_us), minimum(U_us) > 0 ? "yes" : "NO")

    # ── Potential plot ──────────────────────────────────────
    println("  Plotting potential...")
    L = b - a
    xV = range(a - 0.15L, b + 0.15L; length=800)
    plt_V = plot(xV, Vfun.(xV);
        xlabel = L"x",
        ylabel = L"V(x)",
        color  = :black,
        lw     = 2.5,
        legend = false,
        title  = "",
    )
    savefig(plt_V, joinpath(figdir, "$(prefix)-potential.png"))
    println("    → $(prefix)-potential.png")

    # ── Resonance function U★ ──────────────────────────────
    println("  Plotting U★...")
    # Tails: decaying side uses |γ|, growing side uses γ_raw
    x_tail_L = range(a - 1.5, a; length=200)
    U_tail_L = U_a .* exp.(γ_raw .* (x_tail_L .- a))
    x_tail_R = range(b, b + 1.5; length=200)
    U_tail_R = U_b .* exp.(γ_raw .* (x_tail_R .- b))

    plt_U = plot(x_us, U_us;
        xlabel = L"x",
        ylabel = L"U_\star(x)",
        color  = :steelblue,
        lw     = 2.5,
        label  = L"U_\star",
        legend = :topleft,
        title  = "",
    )
    plot!(plt_U, x_tail_L, U_tail_L;
        color = :steelblue, lw = 2.5, ls = :dash, label = "")
    plot!(plt_U, x_tail_R, U_tail_R;
        color = :steelblue, lw = 2.5, ls = :dash, label = "")
    vline!(plt_U, [a, b]; color = :gray, lw = 1.0, ls = :dot, label = "")
    ylims!(plt_U, (0, Inf))   # show U★ > 0
    savefig(plt_U, joinpath(figdir, "$(prefix)-Ustar.pdf"))
    println("    → $(prefix)-Ustar.pdf")

    # ── Run full-line continuation ──────────────────────────
    # Use Vfun_cont (possibly reflected) for continuation.
    # Seed BELOW E_bif where resonance branch exists.
    println("  Running full-line continuation...")
    E_seed = E_bif - 0.5
    p_min  = E_bif - 2.0
    seeds = find_all_seeds(a, b, Vfun_cont;
        E_list   = [E_seed],
        N        = N_ode,
        ζmax     = 8.0,
        nscan    = 5000,
        tolH     = 1e-10,
        slope_set = (+1, -1))
    seeds = deduplicate_seeds(seeds)
    @printf("  Found %d seed(s) near E = %.4f\n", length(seeds), E_seed)
    for (si, s) in enumerate(seeds)
        @printf("    Seed %d: c=%.5f, slope=%+d, right=%+d\n",
                si, s.c, s.slope_sign, get(s, :right_sign, 0))
    end

    # Continue ALL seeds — need to find the resonance branch
    branches_all = Any[]
    for (si, seed) in enumerate(seeds)
        br_try = continue_single_seed(seed, a, b, Vfun_cont;
            N         = N_ode,
            p_min     = p_min,
            p_max     = -1e-6,
            ds        = 0.001,
            dsmin     = 1e-7,
            dsmax     = 0.001,
            max_steps = 2000,
            ζ_min     = 1e-4)
        push!(branches_all, br_try)
        npts = isempty(br_try.branch) ? 0 : length(br_try.branch)
        @printf("    Seed %d/%d: %d points\n", si, length(seeds), npts)
    end

    # Select resonance branch: check N → 4γ near E_bif
    # The resonance branch has N → 4γ (one effective soliton tail).
    br_idx = nothing
    best_N_dist = Inf
    for (bi, br_try) in enumerate(branches_all)
        isempty(br_try.branch) && continue
        # Find the point closest to E_bif
        near_bif = sort(br_try.branch; by = sol -> abs(sol.param - E_bif))
        sol_near = near_bif[1]
        abs(sol_near.param - E_bif) > 0.1 && continue
        ss = get(sol_near, :slope_sign, +1)
        xs, us, vs = integrate_support(a, b, sol_near.param, Vfun_cont;
                                        N=N_ode, c=sol_near.c, slope_sign=ss)
        isempty(xs) && continue
        N_near = compute_norm(a, b, sol_near.param, xs, us, vs)
        isfinite(N_near) || continue
        d = abs(N_near - N_bif)
        @printf("    Branch %d: N_near_bif = %.4f (target %.4f, dist %.4f)\n",
                bi, N_near, N_bif, d)
        if d < best_N_dist
            best_N_dist = d
            br_idx = bi
        end
    end

    br = isnothing(br_idx) ? nothing : branches_all[br_idx]
    if br === nothing || isempty(br.branch)
        @warn "No resonance branch found for $prefix — no branch with N ≈ 4γ near E_bif."
        return
    end
    @printf("  Selected branch %d (of %d) — N ≈ 4|γ| near E_bif\n", br_idx, length(branches_all))
    @printf("  Branch has %d points\n", length(br.branch))

    # Extract E, N, c along the branch
    Es = Float64[]
    Ns = Float64[]
    cs = Float64[]
    for sol in br.branch
        sol.c < 1e-12 && continue
        sol.param > 0 && continue
        slope_s = get(sol, :slope_sign, +1)
        xs, us, vs = integrate_support(a, b, sol.param, Vfun_cont;
                                       N=N_ode, c=sol.c, slope_sign=slope_s)
        isempty(xs) && continue
        N_val = compute_norm(a, b, sol.param, xs, us, vs)
        isfinite(N_val) || continue
        push!(Es, sol.param)
        push!(Ns, N_val)
        push!(cs, sol.c)
    end

    # ── N vs E ──────────────────────────────────────────────
    println("  Plotting N vs E...")
    Emin_plot = E_bif - 1.5
    E_range = range(Emin_plot, -0.01; length=500)
    N_sol   = 4 .* sqrt.(abs.(E_range))

    plt_NE = plot(E_range, N_sol;
        xlabel  = L"E",
        ylabel  = L"\mathcal{N}",
        color   = :darkorange,
        ls      = :dot,
        lw      = 1.5,
        label   = L"4\sqrt{|E|}",
        legend  = :topright,
        title   = "",
    )
    # Shaded stability region on BOTH sides of bifurcation
    E_shade = range(Emin_plot, -0.01; length=500)
    N_shade_lo = 4 .* sqrt.(abs.(E_shade))
    plot!(plt_NE, E_shade, N_shade_lo;
        fillrange = N_bif,
        fillalpha = 0.15,
        fillcolor = :darkorange,
        color     = :darkorange, lw = 0, label = "")
    hline!(plt_NE, [N_bif]; color = :darkorange, ls = :dash, lw = 1.5,
           label = L"\mathcal{N}_\star = 4|\gamma|")
    # Window 1 shaded region (opposite side of soliton from sandwich)
    if U_a > 1e-8   # only for transmission
        slope_crit = -2/γ - 8γ * ℬ^2 / (3U_a^4)
        E_w1 = range(Emin_plot, E_bif; length=500)
        N_crit_line = N_bif .+ slope_crit .* (E_w1 .- E_bif)
        if Ė_tex < 0
            N_upper = max.(N_crit_line, 4 .* sqrt.(abs.(E_w1))) .+ 0.5
            plot!(plt_NE, E_w1, N_crit_line;
                fillrange = N_upper, fillalpha = 0.10, fillcolor = :steelblue,
                color = :steelblue, lw = 0, label = "")
        else
            N_lower = max.(N_crit_line .- 0.5, 0.0)
            plot!(plt_NE, E_w1, N_crit_line;
                fillrange = N_lower, fillalpha = 0.10, fillcolor = :steelblue,
                color = :steelblue, lw = 0, label = "")
        end
        plot!(plt_NE, E_w1, N_crit_line;
            color = :steelblue, ls = :dashdot, lw = 1.0, label = "")
    end
    # Branch
    plot!(plt_NE, Es, Ns;
        color = :steelblue, lw = 2.5, label = "branch")
    # Bifurcation point
    scatter!(plt_NE, [E_bif], [N_bif];
        marker = :circle, ms = 8,
        color  = :limegreen, markerstrokewidth = 2, markerstrokecolor = :black,
        label  = "")

    # Pick 3 marked points near bifurcation
    sp_bif = sortperm(abs.(Es .- E_bif))
    n_br = length(Es)
    if n_br >= 3
        prof_idxs = [sp_bif[max(1, round(Int, 0.02n_br))],
                     sp_bif[round(Int, 0.15n_br)],
                     sp_bif[round(Int, 0.40n_br)]]
    else
        prof_idxs = collect(1:n_br)
    end
    for (i, idx) in enumerate(prof_idxs)
        scatter!(plt_NE, [Es[idx]], [Ns[idx]];
            marker = :circle, ms = 7,
            color  = prof_colors[min(i, length(prof_colors))],
            markerstrokewidth = 1.5, markerstrokecolor = :black,
            label  = "")
    end
    xlims!(plt_NE, (Emin_plot, 0.0))
    savefig(plt_NE, joinpath(figdir, "$(prefix)-NvsE.png"))
    println("    → $(prefix)-NvsE.png")

    # ── Representative profiles ─────────────────────────────
    println("  Plotting profiles...")
    prof_xs_all = Vector{Vector{Float64}}()
    prof_us_all = Vector{Vector{Float64}}()
    prof_εs = Float64[]
    node_Es = [Es[idx] for idx in prof_idxs]
    node_Ns = [Ns[idx] for idx in prof_idxs]
    plt_prof = plot(;
        xlabel = L"x",
        ylabel = L"\psi_\varepsilon(x)",
        legend = :topright,
        title  = "",
    )
    for (i, idx) in enumerate(prof_idxs)
        dists = [abs(s.param - Es[idx]) + abs(s.c - cs[idx]) for s in br.branch]
        sol = br.branch[argmin(dists)]
        slope_s = get(sol, :slope_sign, +1)
        xs, us, vs = integrate_support(a, b, sol.param, Vfun_cont;
                                       N=N_ode, c=sol.c, slope_sign=slope_s)
        isempty(xs) && continue
        xg, ug = glue_full_solution(a, b, sol.param, xs, us, vs; Xmax=10.0)
        ε_val = sol.c^2
        # If reflected, mirror the profile: ψ_original(x) = ψ_reflected(-x)
        xg_plot = reflected ? -real.(xg) : real.(xg)
        sp_x = sortperm(xg_plot)
        push!(prof_xs_all, xg_plot[sp_x])
        push!(prof_us_all, real.(ug[sp_x]))
        push!(prof_εs, ε_val)
        plot!(plt_prof, xg_plot[sp_x], real.(ug[sp_x]);
            color = prof_colors[min(i, length(prof_colors))],
            lw    = 2.2,
            label = latexstring(@sprintf("\\varepsilon = %.3f", ε_val)))
    end
    savefig(plt_prof, joinpath(figdir, "$(prefix)-profiles.png"))
    println("    → $(prefix)-profiles.png")

    # ── Compute μ(ε) along branch (rescaled matching D̃) ─────
    # Uses D̃(ν) = (w_in - w_out)/ε with free-decay left BC
    # (replaces degenerate Evans function — see CLAUDE.md)
    println("  Computing μ(ε) along branch (D̃ scan)...")
    ε_mu  = Float64[]
    μ_num = Float64[]
    μ_ana = Float64[]
    E_mu  = Float64[]
    N_mu  = Float64[]

    valid = [(i, sol) for (i, sol) in enumerate(br.branch)
             if sol.c > 1e-12 && sol.param < 0]
    sort!(valid; by = t -> t[2].c)   # sort by increasing c (= increasing ε)
    n_sample = min(80, length(valid))
    sample_idxs = unique(round.(Int, range(1, length(valid); length=n_sample)))

    prev_ν = Float64[]
    prev_μ = Float64[]

    for (cnt, si) in enumerate(sample_idxs)
        i, sol = valid[si]
        ε = sol.c^2
        slope_sign_val = get(sol, :slope_sign, +1)

        # ν-tracking: extrapolate from previous values, start at ν₀
        ν_guess = if isempty(prev_ν)
            ν₀
        elseif length(prev_ν) == 1
            prev_ν[end]
        else
            prev_ν[end] + (prev_ν[end] - prev_ν[end-1])
        end
        ν_window = max(8.0, 1.5 * abs(ν_guess - ν₀) + 5.0)

        # D̃ scan: find ν root using rescaled matching function
        ν_ref = _find_nu_root_dtilde(a, b, sol.param, Vfun_cont, sol.c,
                    ν_guess, slope_sign_val; ν_window=ν_window, N_ode=max(N_ode, 3000))
        μ_val = isfinite(ν_ref) ? ε^2 * ν_ref : NaN
        isfinite(μ_val) || continue

        # Sign-matching and branch-jump detection
        μ_asym = ν₀ * ε^2
        if !isempty(prev_μ)
            jump_scale = max(abs(prev_μ[end]), abs(μ_asym), 1e-8)
            if sign(μ_asym) != 0 && μ_val * μ_asym < 0
                @printf("    Stopping μ-trace: sign switched at ε=%.5e\n", ε)
                break
            end
            if abs(μ_val - prev_μ[end]) > 25 * jump_scale
                @printf("    Stopping μ-trace: branch jump at ε=%.5e\n", ε)
                break
            end
        end

        xs_n, us_n, vs_n = integrate_support(a, b, sol.param, Vfun_cont;
                                              N=N_ode, c=sol.c, slope_sign=slope_sign_val)
        isempty(xs_n) && continue
        N_val = compute_norm(a, b, sol.param, xs_n, us_n, vs_n)
        isfinite(N_val) || continue

        push!(ε_mu, ε)
        push!(μ_num, μ_val)
        push!(μ_ana, ν₀ * ε^2)
        push!(E_mu, sol.param)
        push!(N_mu, N_val)
        push!(prev_μ, μ_val)
        push!(prev_ν, μ_val / max(ε^2, 1e-30))

        if cnt % 10 == 0 || cnt == 1
            @printf("    [%d/%d] c=%.5e, ε=%.5e, μ=%.5e, ν=%.4f (pred=%.4f), N=%.4f\n",
                    cnt, n_sample, sol.c, ε, μ_val, ν_ref, ν₀, N_val)
        end
    end

    if !isempty(ε_mu)
        sp = sortperm(ε_mu)
        ε_p = ε_mu[sp]
        μ_p = μ_num[sp]
        μ_a = μ_ana[sp]
        E_p = E_mu[sp]
        N_p = N_mu[sp]

        # ── Save computed data to JLD2 ────────────────────────
        datafile = joinpath(figdir, "$(prefix)-data.jld2")
        ν_p = μ_p ./ max.(ε_p.^2, 1e-30)
        ν₀_tex = 3Ω / (4γ)
        Ė_tex = Ω̃ / (2ℬ)
        dNdE_tex = -2/γ + 2ℬ^2/Ω̃
        prof_xs = prof_xs_all
        prof_us = prof_us_all
        @save datafile ε_p μ_p E_p N_p ν_p Es Ns ν₀ ν₀_tex Ė_correct Ė_tex dNdE_pred dNdE_tex N_bif γ E_bif U_a U_b I₂ I₄ Ω ℬ x_us U_us node_Es node_Ns prof_εs prof_xs prof_us
        println("  → Saved data to $(datafile)")

        # ── μ(ε) plot ────────────────────────────────────────
        println("  Plotting μ(ε)...")
        ε_max_pl = min(1.0, maximum(ε_p))
        mask = ε_p .<= ε_max_pl
        ε_z = ε_p[mask]; μ_z = μ_p[mask]; μ_az = μ_a[mask]
        plt_mu = plot(ε_z, μ_z;
            xlabel = L"\varepsilon = c^2",
            ylabel = L"\mu(\varepsilon)",
            color  = :steelblue,
            lw     = 0,
            markershape = :circle,
            ms     = 4,
            markerstrokewidth = 0,
            label  = L"\mu\ \mathrm{(computed)}",
            legend = :topleft,
            title  = "",
        )
        plot!(plt_mu, ε_z, μ_az;
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("\\nu_0 \\varepsilon^2,\\ \\nu_0 = %.2f", ν₀)))
        savefig(plt_mu, joinpath(figdir, "$(prefix)-mu.pdf"))
        println("    → $(prefix)-mu.pdf")

        # ── dN/dE plot ───────────────────────────────────────
        println("  Plotting dN/dE...")
        dNdE_num = Float64[]
        ε_dN     = Float64[]
        for j in 2:length(E_p)-1
            dE = E_p[j+1] - E_p[j-1]
            abs(dE) < 1e-14 && continue
            dN = N_p[j+1] - N_p[j-1]
            push!(dNdE_num, dN / dE)
            push!(ε_dN, ε_p[j])
        end

        plt_dN = plot(ε_dN, dNdE_num;
            xlabel = L"\varepsilon = c^2",
            ylabel = L"d\mathcal{N}/dE",
            color  = :steelblue,
            lw     = 0,
            markershape = :circle,
            ms     = 4,
            markerstrokewidth = 0,
            label  = L"d\mathcal{N}/dE\ \mathrm{(computed)}",
            legend = :topright,
            title  = "",
        )
        hline!(plt_dN, [dNdE_pred];
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("%.3f\\ \\mathrm{(predicted)}", dNdE_pred)))
        savefig(plt_dN, joinpath(figdir, "$(prefix)-dNdE.pdf"))
        println("    → $(prefix)-dNdE.pdf")
    else
        @warn "No μ values computed for $prefix — skipping μ/dN/dE figures."
    end
end

# ═══════════════════════════════════════════════════════════════
# PART B:  Transmission resonances  (V and -V)
# ═══════════════════════════════════════════════════════════════

a_fl = -1.0
b_fl =  1.0

# V(x) = 3sin(πx) on [-1,1]  (via antisymmetric_sine)
Vfun_pos = antisymmetric_sine(a_fl, b_fl, 3.0)
# -V(x)
Vfun_neg = x -> -Vfun_pos(x)

run_transmission_figures(Vfun_pos, a_fl, b_fl, "transmission", "V(x) = 3sin(πx)";
                          figdir=figdir, N_ode=N_ode, prof_colors=prof_colors)

run_transmission_figures(Vfun_neg, a_fl, b_fl, "transmission-neg", "-V(x) = -3sin(πx)";
                          figdir=figdir, N_ode=N_ode, prof_colors=prof_colors)

println("\n" * "="^70)
println("  DONE — figures saved to: $figdir")
println("="^70)
