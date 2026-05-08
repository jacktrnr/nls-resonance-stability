###############################################################
# transmission_figures.jl
#
# Corrected transmission resonance figures for main-Apr7.tex.
# Uses rescaled matching D̃(ν) with free-decay left BC for the
# eigenvalue μ(ε) = ε²ν, avoiding the degenerate Evans function.
#
# Joint tracking: the NLS branch is continued via BifurcationKit,
# and ν is tracked along the branch with predictor-corrector
# (pseudo-arclength–style ν continuation).
#
# Key corrections from the tex:
#   1. VoP Lemma sign fix (α → +α in Wronskian formula)
#   2. Ė from existence matching (not the tex's Ω/(2ℬ))
#   3. N₀ = 4γ (one effective soliton tail, not 8γ)
#
# Outputs (saved to Figures/):
#   transmission-potential.png
#   transmission-Ustar.pdf
#   transmission-NvsE.png
#   transmission-profiles.png
#   transmission-mu.pdf
#   transmission-dNdE.pdf
#   (and similarly with "transmission-neg" prefix for -V)
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
const N_ode = 3000
const prof_colors = [RGB(0.10, 0.35, 0.70), RGB(0.00, 0.60, 0.45), RGB(0.80, 0.20, 0.20)]

# ═══════════════════════════════════════════════════════════════
#   Rescaled matching D̃(ν)  —  free-decay left BC
# ═══════════════════════════════════════════════════════════════

"""
    compute_Dtilde_free(a, b, E, Vfun, c, ν, slope_sign; N_shoot=3000)

Rescaled matching function D̃(ν) = (w_in(b) − w_out(b)) / ε, where:
  μ = ε²ν,  ε = c²,  λ = √(κ² − μ)

Left BC (free decay):  φ(a) = 1, φ'(a) = λ
Right BC (PT ℓ=2 Jost):  w_out from Pöschl–Teller with soliton at x₀R

Returns D̃ (scalar) or NaN on failure.
"""
function compute_Dtilde_free(a, b, E, Vfun, c, ν, slope_sign; N_shoot=3000)
    ε = c^2;  ε < 1e-30 && return NaN
    μ = ε^2 * ν
    κv = sqrt(-E);  μ >= κv^2 && return NaN
    λ = sqrt(κv^2 - μ);  A = sqrt(-2E)

    # ── Integrate NLS interior ψ ──
    xi, ui, vi = integrate_support(a, b, E, Vfun; N=N_shoot, c=c, slope_sign=slope_sign)
    isempty(xi) && return NaN

    # ── Right soliton centre from derivative matching ──
    ub, vb = real(ui[end]), real(vi[end])
    (abs(ub) > A + 1e-10 || abs(ub) < 1e-15) && return NaN
    acosh_R = acosh(A / abs(ub))
    x0R = (vb * ub > 0) ? (b + acosh_R / κv) : (b - acosh_R / κv)
    uR  = tanh(κv * (b - x0R))

    # ── PT Jost w_out at x = b ──
    m  = λ / κv
    DR = m^2 - 1 + 3m * uR + 3uR^2
    abs(DR) < 1e-30 && return NaN
    wR_num = κv * (-m * DR + (1 - uR^2) * (3m + 6uR))
    w_out  = wR_num / DR

    # ── Interpolator for ψ ──
    ψ_at(x) = _linear_interp_scalar(xi, ui, x)

    # ── Integrate L₊ φ from a to b with left BC φ'/φ = λ ──
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


"""
    find_nu_root(a, b, E, Vfun, c, ν_guess, slope_sign; ν_window, N_pts, N_shoot)

Find the root of D̃(ν) closest to ν_guess by scan + bisection.
"""
function find_nu_root(a, b, E, Vfun, c, ν_guess, slope_sign;
                      ν_window=8.0, N_pts=500, N_shoot=3000)
    ν_lo = ν_guess - ν_window
    ν_hi = ν_guess + ν_window
    ν_range = range(ν_lo, ν_hi; length=N_pts)
    D_vals = [compute_Dtilde_free(a, b, E, Vfun, c, ν, slope_sign; N_shoot=N_shoot)
              for ν in ν_range]

    roots = Float64[]
    for i in 1:length(ν_range)-1
        D1, D2 = D_vals[i], D_vals[i+1]
        (isfinite(D1) && isfinite(D2) && D1 * D2 < 0) || continue
        lo, hi = ν_range[i], ν_range[i+1];  Dlo = D1
        for _ in 1:60
            mid = 0.5(lo + hi)
            Dm  = compute_Dtilde_free(a, b, E, Vfun, c, mid, slope_sign; N_shoot=N_shoot)
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
#   Main transmission figure generator
# ═══════════════════════════════════════════════════════════════

function run_transmission_figs(Vfun, a, b, prefix, Vlabel; figdir, N_ode)
    println("\n" * "="^70)
    println("  Transmission resonance:  $Vlabel")
    println("="^70)

    # ── 1. Find transmission resonance ──────────────────────────
    println("  Finding transmission resonance...")
    function _scan_γ(γ_lo, γ_hi, Vf, aa, bb, Nγ)
        γ_grid = range(γ_lo, γ_hi; length=Nγ)
        G_vals = Float64[]
        for γ_try in γ_grid
            sol = solve(ODEProblem(
                (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vf(x) + γ_try^2) * u[1]),
                [1.0, γ_try], (aa, bb)),
                Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false)
            push!(G_vals, sol.u[end][2] - γ_try * sol.u[end][1])
        end
        roots = Float64[]
        for j in 1:length(G_vals)-1
            G_vals[j] * G_vals[j+1] > 0 && continue
            lo, hi = γ_grid[j], γ_grid[j+1]
            for _ in 1:80
                mid = 0.5(lo + hi)
                sol = solve(ODEProblem(
                    (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vf(x) + mid^2) * u[1]),
                    [1.0, mid], (aa, bb)),
                    Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false)
                Gm = sol.u[end][2] - mid * sol.u[end][1]
                abs(hi - lo) < 1e-14 && break
                sign(Gm) == sign(G_vals[j]) ? (lo = mid) : (hi = mid)
            end
            push!(roots, 0.5(lo + hi))
        end
        return roots
    end

    γ_roots_pos = _scan_γ(0.01, 8.0, Vfun, a, b, 2000)
    γ_roots_neg = _scan_γ(-8.0, -0.01, Vfun, a, b, 2000)
    γ_roots = vcat(γ_roots_pos, γ_roots_neg)
    sort!(γ_roots; by=abs)
    @printf("  Found %d resonance(s): %s\n", length(γ_roots),
            join([@sprintf("%.6f", g) for g in γ_roots], ", "))
    isempty(γ_roots) && (@warn "No resonances — skipping $prefix."; return)

    γ_raw = γ_roots[1]   # smallest |γ|
    γ = abs(γ_raw)
    E_bif = -γ^2
    @printf("  γ_raw = %.8f → |γ| = %.8f,  E_bif = %.8f\n", γ_raw, γ, E_bif)

    # Remark 4.6: if γ < 0, reflect V
    reflected = γ_raw < 0
    if reflected
        println("  γ < 0 → reflecting V(x) → V(−x) for continuation")
        Vfun_cont = x -> Vfun(-x)
    else
        Vfun_cont = Vfun
    end

    # ── 2. Compute U★ and integrals ────────────────────────────
    # IMPORTANT: When reflected, use Vfun_cont (=V(−x)) and γ>0, since the
    # continuation branch lives on the reflected potential.  Otherwise, use
    # the original Vfun with γ_raw.
    V_for_us = reflected ? Vfun_cont : Vfun
    γ_for_us = reflected ? γ : γ_raw
    sol_us = solve(ODEProblem(
        (du, u, p, x) -> (du[1] = u[2]; du[2] = (V_for_us(x) + γ^2) * u[1]),
        [1.0, γ_for_us], (a, b)),
        Tsit5(); reltol=1e-13, abstol=1e-15,
        saveat=range(a, b; length=2001))
    x_us = sol_us.t
    U_us = [u[1] for u in sol_us.u]
    U_a  = U_us[1]     # ≈ 1 for transmission
    U_b  = U_us[end]
    h    = x_us[2] - x_us[1]

    I₂ = h * (0.5 * U_us[1]^2 + sum(U_us[2:end-1].^2) + 0.5 * U_us[end]^2)
    I₄ = h * (0.5 * U_us[1]^4 + sum(U_us[2:end-1].^4) + 0.5 * U_us[end]^4)

    # Paper formulas (unified April 9, 2026 — new Ω absorbs Ua⁴/(4γ))
    Ω = (2U_b^4 + U_a^4) / (4γ) - 2I₄          # eq 4.2 (new)
    ℬ = I₂ - (U_b^2 - U_a^2) / (2γ)             # paper ℬ (MINUS sign)
    Ω̃ = Ω - 3U_a^4 / (4γ)                        # convenience: Ω̃ = Ω - 3Ua⁴/(4γ)
    ν₀_tex = 3Ω / (4γ)                            # μ = ν₀ε², sign(μ) = sign(Ω)

    # Ė and dN/dE from paper formulas
    Ė_tex     = Ω̃ / (2ℬ)                          # Ė = Ω̃/(2ℬ)
    dNdE_tex  = 2ℬ^2 / Ω̃ - 2 / γ                 # dN/dE = 2ℬ²/Ω̃ - 2/γ

    # Cross-check: direct existence matching (should agree with Ė_tex)
    Ė_correct = -(I₄ + (1 - U_b^4) / (4γ)) / (I₂ + (1 - U_b^2) / (2γ))
    ν₀_corr = 3(2U_b^4 + U_a^4) / (16γ^2) - 3I₄ / (2γ)  # = 3Ω/(4γ)
    dNdε_corr = ℬ - 2Ė_correct / γ
    dNdE_corr = dNdε_corr / Ė_correct

    N_bif = 4γ   # one effective soliton tail

    @printf("  U_a = %.8f,  U_b = %.8f\n", U_a, U_b)
    @printf("  I₂ = %.8f,  I₄ = %.8f\n", I₂, I₄)
    @printf("  Ω = %.8f,  ℬ = %.8f,  Ω̃ = %.8f\n", Ω, ℬ, Ω̃)
    @printf("  Ė = %.6f  (cross-check: %.6f)\n", Ė_tex, Ė_correct)
    @printf("  ν₀ = %.6f  (cross-check: %.6f)\n", ν₀_tex, ν₀_corr)
    @printf("  dN/dE = %.6f  (cross-check: %.6f)\n", dNdE_tex, dNdE_corr)
    @printf("  Stability: Ω %s 0, dN/dE %s 0 → %s\n",
            Ω > 0 ? ">" : "<", dNdE_tex < 0 ? "<" : ">",
            (Ω > 0 && dNdE_tex < 0) ? "STABLE" : "UNSTABLE")

    # ── 3. Potential plot ──────────────────────────────────────
    println("  Plotting potential...")
    L = b - a
    xV = range(a - 0.15L, b + 0.15L; length=800)
    plt_V = plot(xV, Vfun.(xV);
        xlabel = L"x", ylabel = L"V(x)",
        color = :black, lw = 2.5, legend = false)
    vline!(plt_V, [a, b]; color = :gray, lw = 1.0, ls = :dash)
    savefig(plt_V, joinpath(figdir, "$(prefix)-potential.png"))
    println("    → $(prefix)-potential.png")

    # ── 4. U★ plot ─────────────────────────────────────────────
    println("  Plotting U★...")
    x_tail_L = range(a - 1.5, a; length=200)
    U_tail_L = U_a .* exp.(γ_for_us .* (x_tail_L .- a))
    x_tail_R = range(b, b + 1.5; length=200)
    U_tail_R = U_b .* exp.(γ_for_us .* (x_tail_R .- b))

    plt_U = plot(x_us, U_us;
        xlabel = L"x", ylabel = L"U_\star(x)",
        color = :steelblue, lw = 2.5, label = L"U_\star",
        legend = :topleft)
    plot!(plt_U, x_tail_L, U_tail_L;
        color = :steelblue, lw = 2.5, ls = :dash, label = "")
    plot!(plt_U, x_tail_R, U_tail_R;
        color = :steelblue, lw = 2.5, ls = :dash, label = "")
    vline!(plt_U, [a, b]; color = :gray, lw = 1.0, ls = :dot, label = "")
    ylims!(plt_U, (0, Inf))   # show U★ > 0
    savefig(plt_U, joinpath(figdir, "$(prefix)-Ustar.pdf"))
    println("    → $(prefix)-Ustar.pdf")

    # ── 5. Run NLS continuation (BifurcationKit) ───────────────
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
    @printf("  Found %d seed(s)\n", length(seeds))

    branches_all = Any[]
    for (si, seed) in enumerate(seeds)
        br_try = continue_single_seed(seed, a, b, Vfun_cont;
            N = N_ode, p_min = p_min, p_max = -1e-6,
            ds = 0.001, dsmin = 1e-7, dsmax = 0.001,
            max_steps = 2000, ζ_min = 1e-4)
        push!(branches_all, br_try)
        npts = isempty(br_try.branch) ? 0 : length(br_try.branch)
        @printf("    Seed %d/%d: %d points\n", si, length(seeds), npts)
    end

    # Select resonance branch: closest N to N_bif = 4γ near E_bif
    br_idx = nothing
    best_N_dist = Inf
    for (bi, br_try) in enumerate(branches_all)
        isempty(br_try.branch) && continue
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
        @printf("    Branch %d: N_near = %.4f (target %.4f)\n", bi, N_near, N_bif)
        if d < best_N_dist
            best_N_dist = d
            br_idx = bi
        end
    end

    br = isnothing(br_idx) ? nothing : branches_all[br_idx]
    if br === nothing || isempty(br.branch)
        @warn "No resonance branch found for $prefix."
        return
    end
    @printf("  Selected branch %d — %d points\n", br_idx, length(br.branch))

    # ── 6. Extract E, N, c along the branch ─────────────────────
    Es = Float64[];  Ns = Float64[];  cs = Float64[]
    for sol in br.branch
        sol.c < 1e-12 && continue
        sol.param > 0 && continue
        ss = get(sol, :slope_sign, +1)
        xs, us, vs = integrate_support(a, b, sol.param, Vfun_cont;
                                       N=N_ode, c=sol.c, slope_sign=ss)
        isempty(xs) && continue
        N_val = compute_norm(a, b, sol.param, xs, us, vs)
        isfinite(N_val) || continue
        push!(Es, sol.param)
        push!(Ns, N_val)
        push!(cs, sol.c)
    end
    @printf("  Branch: %d valid points, E ∈ [%.4f, %.4f]\n",
            length(Es), isempty(Es) ? NaN : minimum(Es), isempty(Es) ? NaN : maximum(Es))

    # ── 7. N vs E plot ──────────────────────────────────────────
    println("  Plotting N vs E...")
    Emin_plot = E_bif - 1.5
    E_range = range(Emin_plot, -0.01; length=500)
    N_sol   = 4 .* sqrt.(abs.(E_range))

    plt_NE = plot(E_range, N_sol;
        xlabel = L"E", ylabel = L"\mathcal{N}",
        color = :darkorange, ls = :dot, lw = 1.5,
        label = L"4\sqrt{|E|}",
        legend = :bottomleft)
    # Shaded stability band
    E_shade = range(Emin_plot, -0.01; length=500)
    plot!(plt_NE, E_shade, 4 .* sqrt.(abs.(E_shade));
        fillrange = N_bif, fillalpha = 0.15, fillcolor = :darkorange,
        color = :darkorange, lw = 0, label = "")
    hline!(plt_NE, [N_bif]; color = :darkorange, ls = :dash, lw = 1.5,
           label = latexstring("\\mathcal{N}_\\star = $(round(N_bif; digits=2))"))
    # Window 1 shaded region (opposite side of soliton from sandwich)
    if U_a > 1e-8   # only for transmission (half-line has U_a=0, no Window 1)
        slope_crit = -2/γ - 8γ * ℬ^2 / (3U_a^4)
        E_w1 = range(Emin_plot, E_bif; length=500)
        N_crit_line = N_bif .+ slope_crit .* (E_w1 .- E_bif)
        N_sol_w1 = 4 .* sqrt.(abs.(E_w1))
        # Window 1 is on the opposite side of soliton from sandwich.
        # For Ė < 0 (branch going left): above soliton; for Ė > 0: below soliton.
        if Ė_tex < 0
            # Window 1 above soliton: shade from critical line upward
            N_upper = max.(N_crit_line, N_sol_w1) .+ 0.5  # generous upper bound
            plot!(plt_NE, E_w1, N_crit_line;
                fillrange = N_upper, fillalpha = 0.10, fillcolor = :steelblue,
                color = :steelblue, lw = 0, label = "")
        else
            # Window 1 below soliton: shade from critical line downward
            N_lower = max.(N_crit_line .- 0.5, 0.0)
            plot!(plt_NE, E_w1, N_crit_line;
                fillrange = N_lower, fillalpha = 0.10, fillcolor = :steelblue,
                color = :steelblue, lw = 0, label = "")
        end
        # Draw the critical line itself
        plot!(plt_NE, E_w1, N_crit_line;
            color = :steelblue, ls = :dashdot, lw = 1.0, label = "")
    end
    # Branch
    plot!(plt_NE, Es, Ns; color = :steelblue, lw = 2.5, label = "branch")
    # Bifurcation point
    scatter!(plt_NE, [E_bif], [N_bif];
        marker = :circle, ms = 8, color = :limegreen,
        markerstrokewidth = 2, markerstrokecolor = :black, label = "")
    # Pick 3 profile points
    sp_bif = sortperm(abs.(Es .- E_bif))
    n_br = length(Es)
    prof_idxs = if n_br >= 3
        [sp_bif[max(1, round(Int, 0.02n_br))],
         sp_bif[round(Int, 0.15n_br)],
         sp_bif[round(Int, 0.40n_br)]]
    else
        collect(1:n_br)
    end
    for (i, idx) in enumerate(prof_idxs)
        scatter!(plt_NE, [Es[idx]], [Ns[idx]];
            marker = :circle, ms = 7,
            color = prof_colors[min(i, length(prof_colors))],
            markerstrokewidth = 1.5, markerstrokecolor = :black, label = "")
    end
    xlims!(plt_NE, (Emin_plot, 0.0))
    savefig(plt_NE, joinpath(figdir, "$(prefix)-NvsE.png"))
    println("    → $(prefix)-NvsE.png")

    # ── 8. Representative profiles ──────────────────────────────
    println("  Plotting profiles...")
    plt_prof = plot(;
        xlabel = L"x", ylabel = L"\psi_\varepsilon(x)",
        legend = :topright)
    for (i, idx) in enumerate(prof_idxs)
        dists = [abs(s.param - Es[idx]) + abs(s.c - cs[idx]) for s in br.branch]
        sol = br.branch[argmin(dists)]
        ss = get(sol, :slope_sign, +1)
        xs, us, vs = integrate_support(a, b, sol.param, Vfun_cont;
                                       N=N_ode, c=sol.c, slope_sign=ss)
        isempty(xs) && continue
        xg, ug = glue_full_solution(a, b, sol.param, xs, us, vs; Xmax=10.0)
        ε_val = sol.c^2
        xg_plot = reflected ? -real.(xg) : real.(xg)
        sp_x = sortperm(xg_plot)
        plot!(plt_prof, xg_plot[sp_x], real.(ug[sp_x]);
            color = prof_colors[min(i, length(prof_colors))], lw = 2.2,
            label = latexstring(@sprintf("\\varepsilon = %.3f", ε_val)))
    end
    vline!(plt_prof, [a, b]; color = :gray, lw = 1.0, ls = :dash, label = "")
    savefig(plt_prof, joinpath(figdir, "$(prefix)-profiles.png"))
    println("    → $(prefix)-profiles.png")

    # ── 8b. L₊ eigenfunction φ for representative branch points ──
    println("  Plotting L₊ eigenfunctions...")
    plt_eig = plot(;
        xlabel = L"x", ylabel = L"\varphi(x)",
        legend = :topright)
    for (i, idx) in enumerate(prof_idxs)
        dists = [abs(s.param - Es[idx]) + abs(s.c - cs[idx]) for s in br.branch]
        sol = br.branch[argmin(dists)]
        ss = get(sol, :slope_sign, +1)
        E_val = sol.param;  c_val = sol.c;  ε_val = c_val^2
        κv = sqrt(-E_val)

        # Integrate NLS ψ
        xi, ui, vi = integrate_support(a, b, E_val, Vfun_cont;
                                       N=N_ode, c=c_val, slope_sign=ss)
        isempty(xi) && continue

        # L₊ eigenfunction: φ'' = (V - 3ψ² - E)φ with free-decay BC
        ψ_at(x) = _linear_interp_scalar(xi, ui, x)
        λ_bc = κv  # free-decay left BC at μ≈0
        function ode_eig!(du, u, p, t)
            du[1] = u[2]
            du[2] = (Vfun_cont(t) - 3.0 * ψ_at(t)^2 - E_val) * u[1]
        end
        prob_e = ODEProblem(ode_eig!, [1.0, λ_bc], (Float64(a), Float64(b)))
        sol_e = solve(prob_e, Tsit5(); abstol=1e-13, reltol=1e-11,
                      saveat=range(Float64(a), Float64(b); length=500))
        isempty(sol_e.u) && continue
        φ_us = [u[1] for u in sol_e.u]
        x_eig = reflected ? -sol_e.t : sol_e.t
        sp_e = sortperm(x_eig)

        plot!(plt_eig, x_eig[sp_e], φ_us[sp_e];
            color = prof_colors[min(i, length(prof_colors))], lw = 2.2,
            label = latexstring(@sprintf("\\varepsilon = %.3f", ε_val)))
    end
    vline!(plt_eig, [a, b]; color = :gray, lw = 1.0, ls = :dash, label = "")
    savefig(plt_eig, joinpath(figdir, "$(prefix)-eigenfunctions.pdf"))
    println("    → $(prefix)-eigenfunctions.pdf")

    # ── 9. Eigenvalue tracking: D̃(ν) along the branch ──────────
    println("  Computing μ(ε) via rescaled matching D̃(ν)...")
    ε_mu  = Float64[]
    μ_num = Float64[]
    E_mu  = Float64[]
    N_mu  = Float64[]
    ν_all = Float64[]

    # Sort branch by increasing c
    valid = [(i, sol) for (i, sol) in enumerate(br.branch)
             if sol.c > 1e-12 && sol.param < 0]
    sort!(valid; by = t -> t[2].c)

    # Uniform sampling across the full branch, but denser at small c
    n_valid = length(valid)
    n_sample = min(100, n_valid)
    # Combine: 50 points in first 5% of branch + 50 uniformly across rest
    n_lo = min(50, n_valid)
    n_hi = n_sample - n_lo
    cutoff = max(2, round(Int, 0.05 * n_valid))
    lo_idxs = unique(round.(Int, range(1, cutoff; length=n_lo)))
    hi_idxs = unique(round.(Int, range(cutoff + 1, n_valid; length=max(n_hi, 1))))
    sample_idxs = sort(unique(vcat(lo_idxs, hi_idxs)))

    # Robust ν tracking: always scan a wide window centered on ν₀_corr,
    # find ALL D̃ roots, and pick the closest to the perturbation theory
    # prediction.  This avoids predictor drift causing root-jumping.
    prev_ν = Float64[]
    prev_c = Float64[]

    for (cnt, si) in enumerate(sample_idxs)
        si > n_valid && break
        i, sol = valid[si]
        ε  = sol.c^2
        ss = get(sol, :slope_sign, +1)

        # ── Wide scan centered on ν₀_corr; extend to cover predictor too ──
        ν_center = ν₀_corr
        ν_pred = if isempty(prev_ν)
            ν₀_corr
        elseif length(prev_ν) == 1
            prev_ν[end]
        else
            c1, c2 = prev_c[end-1], prev_c[end]
            ν1, ν2 = prev_ν[end-1], prev_ν[end]
            dc = sol.c - c2
            abs(c2 - c1) > 1e-14 ? ν2 + (ν2 - ν1) / (c2 - c1) * dc : ν2
        end
        # Window wide enough to cover both ν₀_corr and the predictor
        ν_window = max(8.0, 1.5 * abs(ν_pred - ν₀_corr) + 5.0)

        # ── Find root closest to ν₀_corr (robust against predictor drift) ──
        ν_ref = find_nu_root(a, b, sol.param, Vfun_cont, sol.c, ν_center, ss;
                             ν_window=ν_window, N_pts=500, N_shoot=max(N_ode, 3000))
        isfinite(ν_ref) || continue
        μ_val = ε^2 * ν_ref

        # Compute norm at this point
        xs_n, us_n, vs_n = integrate_support(a, b, sol.param, Vfun_cont;
                                              N=N_ode, c=sol.c, slope_sign=ss)
        isempty(xs_n) && continue
        N_val = compute_norm(a, b, sol.param, xs_n, us_n, vs_n)
        isfinite(N_val) || continue

        push!(ε_mu, ε)
        push!(μ_num, μ_val)
        push!(E_mu, sol.param)
        push!(N_mu, N_val)
        push!(ν_all, ν_ref)
        push!(prev_ν, ν_ref)
        push!(prev_c, sol.c)

        if cnt % 10 == 0 || cnt == 1 || cnt <= 5
            @printf("    [%d/%d] c=%.5e, ε=%.5e, μ=%+.5e, ν=%+.4f (pred=%+.4f), N=%.4f\n",
                    cnt, length(sample_idxs), sol.c, ε, μ_val, ν_ref, ν₀_corr, N_val)
        end
    end

    if !isempty(ε_mu)
        sp = sortperm(ε_mu)
        ε_p = ε_mu[sp]
        μ_p = μ_num[sp]
        E_p = E_mu[sp]
        N_p = N_mu[sp]
        ν_p = ν_all[sp]

        # ν₀ estimate: extrapolate from the 3 smallest-ε points
        # using Richardson-like extrapolation: ν(ε) ≈ ν₀ + ν₁ε + ...
        n_small = min(3, count(ε_p .< 0.05))
        if n_small == 0;  n_small = min(3, length(ν_p))  end
        ν₀_num = ν_p[1]  # best single estimate = smallest ε
        if n_small >= 2
            # Linear fit: ν = a + b*ε → ν₀ = a
            ε_fit = ε_p[1:n_small]
            ν_fit = ν_p[1:n_small]
            # Least squares: a + b*ε_i = ν_i
            ε_mean = sum(ε_fit) / n_small
            ν_mean = sum(ν_fit) / n_small
            num = sum((ε_fit .- ε_mean) .* (ν_fit .- ν_mean))
            den = sum((ε_fit .- ε_mean).^2)
            b_fit = abs(den) > 1e-30 ? num / den : 0.0
            ν₀_num = ν_mean - b_fit * ε_mean
        end
        @printf("\n  ν₀ estimate (extrapolated):             %.6f\n", ν₀_num)
        @printf("  ν₀ (smallest ε point):                  %.6f\n", ν_p[1])
        @printf("  ν₀ perturbation theory (corrected):     %.6f\n", ν₀_corr)
        @printf("  ν₀ tex (Ω formula):                     %.6f\n", ν₀_tex)

        # ── Save computed data to JLD2 for re-plotting ──────────
        datafile = joinpath(figdir, "$(prefix)-data.jld2")
        @save datafile ε_p μ_p E_p N_p ν_p Es Ns ν₀_corr ν₀_tex ν₀_num Ė_correct Ė_tex dNdE_corr dNdE_tex N_bif γ E_bif U_a U_b I₂ I₄ Ω Ω̃ ℬ x_us U_us
        println("  → Saved computed data to $(datafile)")

        # ── 10a. μ(ε) plot — full range ─────────────────────────
        println("  Plotting μ(ε)...")
        ε_max_pl = min(1.0, maximum(ε_p))
        mask = ε_p .<= ε_max_pl
        ε_z = ε_p[mask];  μ_z = μ_p[mask]

        # Prediction curve: μ = ν₀ε²
        ε_pred = range(0, ε_max_pl; length=200)
        μ_pred_corr = ν₀_corr .* ε_pred.^2

        plt_mu = scatter(ε_z, μ_z;
            xlabel = L"\varepsilon = c^2",
            ylabel = L"\mu(\varepsilon)",
            color  = :steelblue,
            ms = 4, markerstrokewidth = 0,
            label  = L"\mu\ \mathrm{(numerical)}",
            legend = :bottomleft,
        )
        plot!(plt_mu, ε_pred, μ_pred_corr;
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("\\nu_0 \\varepsilon^2,\\ \\nu_0 = %.2f", ν₀_corr)))
        savefig(plt_mu, joinpath(figdir, "$(prefix)-mu.pdf"))
        println("    → $(prefix)-mu.pdf")

        # ── 10a′. Multi-scale μ(ε) zoom-in plots ────────────────
        for (ε_cut, suffix) in [(0.005, "zoom0"), (0.02, "zoom1"), (0.05, "zoom2"),
                                (0.15, "zoom3"), (0.4, "zoom4")]
            mask_z = ε_p .<= ε_cut
            count(mask_z) < 3 && continue
            ε_zz = ε_p[mask_z]; μ_zz = μ_p[mask_z]
            ε_pr = range(0, ε_cut; length=200)
            μ_pr = ν₀_corr .* ε_pr.^2
            μ_tx = ν₀_tex .* ε_pr.^2
            plt_zz = scatter(ε_zz, μ_zz;
                xlabel = L"\varepsilon = c^2",
                ylabel = L"\mu(\varepsilon)",
                color  = :steelblue, ms = 5, markerstrokewidth = 0,
                label  = L"\mu\ \mathrm{(numerical)}",
                legend = :bottomleft)
            plot!(plt_zz, ε_pr, μ_pr;
                color = :firebrick3, lw = 2.5,
                label = latexstring(@sprintf("\\nu_0\\varepsilon^2,\\ \\nu_0 = %.2f", ν₀_corr)))
            plot!(plt_zz, ε_pr, μ_tx;
                color = :darkorange, lw = 2.0, ls = :dash,
                label = latexstring(@sprintf("\\nu_0^{\\mathrm{tex}}\\varepsilon^2,\\ \\nu_0 = %.2f", ν₀_tex)))
            savefig(plt_zz, joinpath(figdir, "$(prefix)-mu-$(suffix).pdf"))
            println("    → $(prefix)-mu-$(suffix).pdf")
        end

        # ── 10b. ν(ε) = μ/ε² plot — shows convergence to ν₀ ────
        println("  Plotting ν(ε) = μ/ε²...")
        mask_nu = ε_p .> 1e-10
        ε_nu = ε_p[mask_nu];  ν_nu = ν_p[mask_nu]

        plt_nu = scatter(ε_nu, ν_nu;
            xlabel = L"\varepsilon = c^2",
            ylabel = L"\nu(\varepsilon) = \mu/\varepsilon^2",
            color  = :steelblue,
            ms = 4, markerstrokewidth = 0,
            label  = L"\nu\ \mathrm{(numerical)}",
            legend = :bottomright,
        )
        hline!(plt_nu, [ν₀_corr];
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("\\nu_0 = %.2f\\ \\mathrm{(corrected)}", ν₀_corr)))
        hline!(plt_nu, [ν₀_tex];
            color = :darkorange, lw = 2.0, ls = :dash,
            label = latexstring(@sprintf("\\nu_0 = %.2f\\ \\mathrm{(tex)}", ν₀_tex)))
        savefig(plt_nu, joinpath(figdir, "$(prefix)-nu.pdf"))
        println("    → $(prefix)-nu.pdf")

        # ── 11. dN/dE plot ──────────────────────────────────────
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

        plt_dN = scatter(ε_dN, dNdE_num;
            xlabel = L"\varepsilon = c^2",
            ylabel = L"d\mathcal{N}/dE",
            color  = :steelblue,
            ms = 4, markerstrokewidth = 0,
            label  = L"d\mathcal{N}/dE\ \mathrm{(numerical)}",
            legend = :bottomright,
        )
        hline!(plt_dN, [dNdE_corr];
            color = :firebrick3, lw = 2.5, ls = :solid,
            label = latexstring(@sprintf("%.2f\\ \\mathrm{(corrected)}", dNdE_corr)))
        hline!(plt_dN, [dNdE_tex];
            color = :darkorange, lw = 2.0, ls = :dash,
            label = latexstring(@sprintf("%.2f\\ \\mathrm{(tex)}", dNdE_tex)))
        savefig(plt_dN, joinpath(figdir, "$(prefix)-dNdE.pdf"))
        println("    → $(prefix)-dNdE.pdf")
    else
        @warn "No μ values computed for $prefix — skipping μ/dN/dE figures."
    end
end

# ── Helper ───────────────────────────────────────────────────
function mean(x)
    isempty(x) && return NaN
    return sum(x) / length(x)
end

# ═══════════════════════════════════════════════════════════════
#   Run for V(x) = 3sin(πx) and −V(x) = −3sin(πx)
#   (only when this file is executed directly)
# ═══════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    a_fl = -1.0
    b_fl =  1.0

    Vfun_pos = antisymmetric_sine(a_fl, b_fl, 3.0)
    Vfun_neg = x -> -Vfun_pos(x)

    run_transmission_figs(Vfun_pos, a_fl, b_fl, "transmission", "V(x) = 3sin(πx)";
                          figdir=figdir, N_ode=N_ode)

    run_transmission_figs(Vfun_neg, a_fl, b_fl, "transmission-neg", "-V(x) = -3sin(πx)";
                          figdir=figdir, N_ode=N_ode)

    println("\n" * "="^70)
    println("  DONE — figures saved to: $figdir")
    println("="^70)
end
