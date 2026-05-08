###############################################
# dynamics_fullline.jl — NLS time evolution
###############################################
#
# Split-step integrator for:
#   iΨₜ = -Ψ'' + V(x)Ψ - |Ψ|²Ψ
#
# Two modes:
#   • Half-line [0, Xmax]:  Dirichlet at both ends, DST-I
#   • Full-line [-Xmax, Xmax]: Dirichlet at both ends, DST-I
#
# Complex absorbing potential (CAP) with sin² profile
# near both boundaries prevents spurious reflections.
#
# Loads precomputed profiles from JLD2 files produced
# by save_profiles.jl / run_complete.jl.
#
###############################################

using FFTW
using JLD2
using Printf
using Plots
using LaTeXStrings

# ═════════════════════════════════════════════
# SPLIT-STEP INTEGRATOR (DST-I, Dirichlet BCs)
# ═════════════════════════════════════════════

"""
    splitstep_evolve(ψ0, x, Vx, dt, Nt; save_every=10)

Evolve iΨₜ = -Ψ'' + V(x)Ψ - |Ψ|²Ψ on an interior grid `x`
with Dirichlet BCs at both endpoints using symmetric Strang
splitting with DST-I.

`ψ0`, `Vx` are vectors on the interior grid (no boundary zeros).
`Vx` may be complex-valued if a CAP is included.

Returns `(t_saves, ψ_saves)`.
"""
function splitstep_evolve(ψ0::Vector{ComplexF64}, x, Vx, dt, Nt;
                          save_every::Int=10)
    n = length(x)
    dx = x[2] - x[1]
    L = x[end] - x[1] + 2dx  # domain length including boundary zeros

    # DST-I eigenvalues: k_m = mπ/L, eigenvalue = k_m², m = 1..n
    k2 = [(m * π / L)^2 for m in 1:n]
    kinetic_full = exp.(-im .* k2 .* dt)

    ψ = copy(ψ0)
    t_saves = Float64[0.0]
    ψ_saves = Vector{ComplexF64}[copy(ψ)]

    for step in 1:Nt
        # Half-step: potential + nonlinear
        @. ψ *= exp(-im * (Vx - abs2(ψ)) * dt / 2)

        # Full-step: kinetic via DST-I
        ψ_hat = FFTW.r2r(real.(ψ), FFTW.RODFT00) .+
                im .* FFTW.r2r(imag.(ψ), FFTW.RODFT00)
        @. ψ_hat *= kinetic_full
        scale = 1.0 / (2 * (n + 1))
        ψ .= scale .* (FFTW.r2r(real.(ψ_hat), FFTW.RODFT00) .+
                        im .* FFTW.r2r(imag.(ψ_hat), FFTW.RODFT00))

        # Half-step: potential + nonlinear
        @. ψ *= exp(-im * (Vx - abs2(ψ)) * dt / 2)

        if step % save_every == 0 || step == Nt
            push!(t_saves, step * dt)
            push!(ψ_saves, copy(ψ))
        end
    end
    return t_saves, ψ_saves
end

# ═════════════════════════════════════════════
# COMPLEX ABSORBING POTENTIAL
# ═════════════════════════════════════════════

"""
    add_cap!(Vx_eff, x, width, strength; sides=:both)

Add a CAP W(x) ≥ 0 with sin² profile near the boundaries.
`sides` can be `:both`, `:left`, or `:right`.
For half-line [0, Xmax], use `sides=:right` to avoid absorbing
the soliton near x=0.
"""
function add_cap!(Vx_eff, x, width, strength; sides=:both)
    n = length(x)
    xL, xR = x[1] - (x[2]-x[1]), x[end] + (x[2]-x[1])
    for j in 1:n
        # Left absorber
        if (sides == :both || sides == :left) && x[j] < xL + width
            s = sin(π/2 * (xL + width - x[j]) / width)
            Vx_eff[j] -= im * strength * s^2
        end
        # Right absorber
        if (sides == :both || sides == :right) && x[j] > xR - width
            s = sin(π/2 * (x[j] - (xR - width)) / width)
            Vx_eff[j] -= im * strength * s^2
        end
    end
end

# ═════════════════════════════════════════════
# LOADING PROFILES FROM JLD2
# ═════════════════════════════════════════════

"""
    load_profile(jld2_path; idx=1)

Load a precomputed profile from a JLD2 file.
Returns `(x, ψ, ε², E_branch, N_branch, γ, E_bif)`.

`idx` selects which saved profile to use (1 = smallest ε, closest
to bifurcation point). `E_branch` and `N_branch` are the full
continuation branch arrays.
"""
function load_profile(jld2_path; idx=1)
    f = jldopen(jld2_path, "r")
    xs = f["prof_xs"]
    us = f["prof_us"]
    εs = f["prof_εs"]
    # Full branch data (continuation curve)
    Es = f["Es"]
    Ns = f["Ns"]
    γ = f["γ"]
    E_bif = f["E_bif"]
    close(f)
    return xs[idx], us[idx], εs[idx], Es, Ns, γ, E_bif
end

# ═════════════════════════════════════════════
# INTERPOLATION
# ═════════════════════════════════════════════

"""Linear interpolation of (x_data, y_data) onto x_query."""
function linear_interp(x_data, y_data, x_query)
    out = similar(x_query, eltype(y_data))
    for (i, xq) in enumerate(x_query)
        if xq <= x_data[1]
            out[i] = y_data[1]
        elseif xq >= x_data[end]
            out[i] = y_data[end]
        else
            j = searchsortedlast(x_data, xq)
            j = clamp(j, 1, length(x_data) - 1)
            t = (xq - x_data[j]) / (x_data[j+1] - x_data[j])
            out[i] = (1 - t) * y_data[j] + t * y_data[j+1]
        end
    end
    return out
end

"""
    interp_with_exp_tails(x_data, y_data, x_query, γ)

Linear interpolation inside [x_data[1], x_data[end]], and exponential decay
outside:  ψ(x) = ψ(x_end) · exp(-γ·(x - x_end))  for x > x_end,
          ψ(x) = ψ(x_start) · exp(-γ·(x_start - x))  for x < x_start.

This is the correct tail behavior for a bound state with decay rate γ, and
avoids injecting spurious mass when the simulation grid is much wider than
the stored profile.
"""
function interp_with_exp_tails(x_data, y_data, x_query, γ)
    out = similar(x_query, eltype(y_data))
    xa, xb = x_data[1], x_data[end]
    ya, yb = y_data[1], y_data[end]
    for (i, xq) in enumerate(x_query)
        if xq < xa
            out[i] = ya * exp(-γ * (xa - xq))
        elseif xq > xb
            out[i] = yb * exp(-γ * (xq - xb))
        else
            j = searchsortedlast(x_data, xq)
            j = clamp(j, 1, length(x_data) - 1)
            t = (xq - x_data[j]) / (x_data[j+1] - x_data[j])
            out[i] = (1 - t) * y_data[j] + t * y_data[j+1]
        end
    end
    return out
end

# ═════════════════════════════════════════════
# NEWTON REFINEMENT
# ═════════════════════════════════════════════

using LinearAlgebra, SparseArrays

"""
    build_laplacian(n, dx)

Sparse second-derivative matrix for Dirichlet BCs (n interior points).
"""
function build_laplacian(n, dx)
    e = ones(n)
    return spdiagm(-1 => e[1:end-1], 0 => -2e, 1 => e[1:end-1]) / dx^2
end

"""
    newton_refine(ψ_guess, E_guess, x_grid, Vx;
                  maxiter=50, tol=1e-10, fix_norm=true)

Newton-iterate to find (ψ, E) satisfying:
  -ψ'' + V·ψ - ψ³ - E·ψ = 0  (Dirichlet BCs)
with an additional constraint (either fixed norm or fixed E).

If `fix_norm=true`, treat E as unknown and impose ∫ψ²dx = N₀ (mass
of the initial guess). Otherwise, fix E and solve for ψ only.

Returns `(ψ_refined, E_refined, converged)`.
"""
function newton_refine(ψ_guess, E_guess, x_grid, Vx;
                       maxiter=80, tol=1e-10, fix_norm=true)
    n = length(x_grid)
    dx = x_grid[2] - x_grid[1]
    D2 = build_laplacian(n, dx)
    V_diag = Diagonal(Vx)
    N_target = dx * sum(ψ_guess.^2)

    ψ = copy(ψ_guess)
    E = E_guess

    # Helper: compute residual norm for given (ψ, E)
    function residual_norm(ψv, Ev)
        F = -D2 * ψv .+ Vx .* ψv .- ψv.^3 .- Ev .* ψv
        if fix_norm
            g = dx * sum(ψv.^2) - N_target
            return max(maximum(abs, F), abs(g))
        else
            return maximum(abs, F)
        end
    end

    for iter in 1:maxiter
        F = -D2 * ψ .+ Vx .* ψ .- ψ.^3 .- E .* ψ

        if fix_norm
            g = dx * sum(ψ.^2) - N_target
            res = max(maximum(abs, F), abs(g))
            if iter <= 5 || iter % 10 == 0
                @printf("    Newton iter %d: ||F||∞ = %.2e, |g| = %.2e\n",
                        iter, maximum(abs, F), abs(g))
            end
            res < tol && return (ψ, E, true)

            J = -D2 + V_diag - 3 * Diagonal(ψ.^2) - E * I

            # Schur complement: solve sparse J twice instead of dense augmented system
            # [J, -ψ; cᵀ, 0] [δψ; δE] = [-F; -g]  with c = 2dx·ψ
            c = 2dx .* ψ
            JinvF = J \ F
            Jinvψ = J \ ψ
            δE = (-g + dot(c, JinvF)) / dot(c, Jinvψ)
            δψ = -JinvF .+ δE .* Jinvψ

            # Backtracking line search
            α = 1.0
            for _ in 1:10
                rn = residual_norm(ψ .+ α .* δψ, E + α * δE)
                rn < res && break
                α *= 0.5
            end

            ψ .+= α .* δψ
            E += α * δE
        else
            res = maximum(abs, F)
            if iter <= 5 || iter % 10 == 0
                @printf("    Newton iter %d: ||F||∞ = %.2e\n", iter, res)
            end
            res < tol && return (ψ, E, true)

            J = -D2 + V_diag - 3 * Diagonal(ψ.^2) - E * I
            δψ = -(J \ F)

            α = 1.0
            for _ in 1:10
                rn = residual_norm(ψ .+ α .* δψ, E)
                rn < res && break
                α *= 0.5
            end

            ψ .+= α .* δψ
        end
    end

    @warn "Newton did not converge after $maxiter iterations"
    return (ψ, E, false)
end

# ═════════════════════════════════════════════
# DIAGNOSTICS
# ═════════════════════════════════════════════

"""Compute mass N(t) = ∫|Ψ|²dx."""
function compute_mass(ψ, dx)
    dx * sum(abs2, ψ)
end

"""Compute energy E(t) as Rayleigh quotient (kinetic + potential - nonlinear) / mass."""
function compute_energy_rayleigh(ψ, Vx_real, dx)
    n = length(ψ)
    # Kinetic via central differences (Dirichlet ghost points)
    ψe = [zero(eltype(ψ)); ψ; zero(eltype(ψ))]
    dψ = (ψe[3:end] .- ψe[1:end-2]) ./ (2dx)
    kin = dx * sum(abs2, dψ)
    pot = dx * sum(real(Vx_real[j]) * abs2(ψ[j]) for j in 1:n)
    nonl = dx * sum(abs2(ψ[j])^2 for j in 1:n)
    nrm = dx * sum(abs2, ψ)
    return (kin + pot - nonl) / nrm
end

"""Compute effective soliton norm N_eff(t) = 2·max|Ψ(x,t)|² / max_amplitude."""
function compute_soliton_norm(ψ)
    2 * maximum(abs, ψ)
end

# ═════════════════════════════════════════════
# MAIN SIMULATION DRIVER
# ═════════════════════════════════════════════

"""
    run_simulation(jld2_path, Vfun, label;
                   prof_idx=1, domain=:fullline,
                   Xmax=25.0, Ngrid=4096, Tmax=50.0, dt=1e-3,
                   δ_amp=0.03, pert_type=:gaussian,
                   save_every=50,
                   cap_width=5.0, cap_strength=5.0,
                   display_title="",
                   figdir="Figures")

Run a time evolution simulation for one example.

- `jld2_path`: path to JLD2 file with precomputed profiles
- `Vfun`: potential function V(x)
- `label`: string label for output files
- `prof_idx`: which profile to use (1 = smallest ε)
- `domain`: `:halfline` for [0,Xmax] or `:fullline` for [-Xmax,Xmax]
- `δ_amp`: perturbation amplitude (relative to max|ψ₀|)
- `pert_type`:
    `:gaussian` — add δ_amp · max|ψ₀| · exp(-(x-x_peak)²/2σ²) localized near soliton peak
    `:phase`    — multiply by exp(iδ_amp · exp(-(x-x_peak)²/2σ²)), localized phase kick
    `:scale`    — global rescaling ψ → (1+δ_amp)ψ
- `display_title`: title for figures (defaults to label)
"""
function run_simulation(jld2_path, Vfun, label;
                        prof_idx=1, domain=:fullline,
                        Xmax=25.0, Ngrid=4096,
                        Tmax=50.0, dt=1e-3,
                        δ_amp=0.03, pert_type=:gaussian,
                        save_every=50,
                        cap_width=5.0, cap_strength=5.0,
                        display_title="",
                        figdir="Figures",
                        E_guess=nothing,
                        skip_newton=false)

    println("="^60)
    println("  Simulation: $label")
    println("="^60)

    fig_title = isempty(display_title) ? label : display_title

    # --- Load profile ---
    x_prof, ψ_prof, ε2, E_branch, N_branch, γ, E_bif = load_profile(jld2_path; idx=prof_idx)
    println("  Profile loaded: ε² = $ε2, domain [$(x_prof[1]), $(x_prof[end])]")
    println("  γ = $γ, E_bif = $E_bif")
    println("  max|ψ| = $(maximum(abs, ψ_prof))")

    # --- Build uniform grid ---
    if domain == :halfline
        x_lo, x_hi = 0.0, Xmax
    else
        x_lo, x_hi = -Xmax, Xmax
    end
    dx = (x_hi - x_lo) / (Ngrid + 1)
    x_grid = [x_lo + j * dx for j in 1:Ngrid]

    # --- Interpolate profile onto grid (exp tails outside stored domain) ---
    ψ_bound = interp_with_exp_tails(x_prof, ψ_prof, x_grid, γ)

    # --- Potential on grid (real part) ---
    Vx_real = [Vfun(xi) for xi in x_grid]

    # --- (Optionally) Newton-refine the profile ---
    E_guess_used = (E_guess === nothing) ? E_bif : E_guess
    if skip_newton
        @printf("  Skipping Newton refinement; using raw interpolated profile, E_ref = %.6f\n", E_guess_used)
        E_ref = E_guess_used
        conv = true
    else
        println("  Newton-refining profile...")
        @printf("  Newton E_guess = %.6f (E_bif = %.6f)\n", E_guess_used, E_bif)
        ψ_ref, E_ref, conv = newton_refine(ψ_bound, E_guess_used, x_grid, Vx_real;
                                            maxiter=80, tol=1e-10, fix_norm=true)
        if conv
            @printf("  Newton converged: E = %.10f (was %.10f)\n", E_ref, E_guess_used)
            ψ_bound = ψ_ref
        else
            @warn "Newton did not converge — using asymptotic profile"
        end
    end

    # --- Effective potential with CAP ---
    Vx_eff = ComplexF64.(Vx_real)
    cap_sides = (domain == :halfline) ? :right : :both
    add_cap!(Vx_eff, x_grid, cap_width, cap_strength; sides=cap_sides)
    println("  CAP: width=$cap_width, strength=$cap_strength, sides=$cap_sides")

    # --- Build perturbation ---
    _, i_peak = findmax(abs.(ψ_bound))
    x_peak = x_grid[i_peak]
    σ_pert = 1.0 / (2γ)  # width ~ soliton scale

    if pert_type == :gaussian
        # Localized H^1 perturbation: even + odd Hermite-Gaussian components
        # Even part excites amplitude/mass modes; odd part excites translational modes
        amp = δ_amp * maximum(abs, ψ_bound)
        h = [begin
                 ξ = (x_grid[j] - x_peak) / σ_pert
                 amp * (exp(-ξ^2 / 2) + 0.5 * ξ * exp(-ξ^2 / 2))
             end for j in 1:Ngrid]
        ψ_init = ComplexF64.(ψ_bound .+ h)
        @printf("  IC: ψ₀ + even+odd bump (δ=%.3f, σ=%.3f, center=%.2f)\n",
                δ_amp, σ_pert, x_peak)
    elseif pert_type == :phase
        # Localized phase kick
        θ = [δ_amp * exp(-(x_grid[j] - x_peak)^2 / (2σ_pert^2)) for j in 1:Ngrid]
        ψ_init = ComplexF64.(ψ_bound .* exp.(im .* θ))
        @printf("  IC: ψ₀ · exp(iθ), phase kick (δ=%.3f, σ=%.3f)\n", δ_amp, σ_pert)
    elseif pert_type == :scale
        ψ_init = ComplexF64.((1.0 + δ_amp) .* ψ_bound)
        @printf("  IC: (1 + %.3f)·ψ₀ (global scale)\n", δ_amp)
    else
        error("Unknown pert_type: $pert_type")
    end
    N0 = compute_mass(ψ_init, dx)
    @printf("  Initial mass = %.6f\n", N0)
    @printf("  Grid: %d pts, dx = %.5f, domain [%.1f, %.1f]\n", Ngrid, dx, x_lo, x_hi)

    # --- Evolve ---
    Nt = round(Int, Tmax / dt)
    @printf("  Tmax = %.1f, dt = %.1e, %d steps\n", Tmax, dt, Nt)
    t_saves, ψ_saves = splitstep_evolve(ψ_init, x_grid, Vx_eff, dt, Nt;
                                         save_every=save_every)
    Nf = compute_mass(ψ_saves[end], dx)
    @printf("  Done: %d frames. Final mass = %.6f (drift = %.2e)\n",
            length(t_saves), Nf, abs(Nf - N0))

    # --- Compute diagnostics ---
    N_traj = [compute_mass(ψk, dx) for ψk in ψ_saves]
    E_traj = [compute_energy_rayleigh(ψk, Vx_real, dx) for ψk in ψ_saves]

    # --- Spacetime heatmap ---
    println("  Generating spacetime heatmap...")
    n_frames = length(t_saves)
    density_matrix = zeros(n_frames, Ngrid)
    for k in 1:n_frames
        density_matrix[k, :] .= abs2.(ψ_saves[k])
    end

    # Auto-zoom x-range: find columns with density > 1% of peak, pad by ±2
    peak_dens = maximum(density_matrix)
    col_active = vec(maximum(density_matrix; dims=1)) .> 0.01 * peak_dens
    j_first = findfirst(col_active)
    j_last  = findlast(col_active)
    x_lo_zoom = x_grid[j_first] - 2.0
    x_hi_zoom = x_grid[j_last]  + 2.0

    plt_st = heatmap(x_grid, t_saves, density_matrix;
        xlabel=L"x", ylabel=L"t",
        colorbar_title=L"|\Psi(x,t)|^2",
        color=:inferno, clims=(0, peak_dens),
        xlims=(x_lo_zoom, x_hi_zoom),
        size=(600, 400))

    st_path = joinpath(figdir, "$(label)-spacetime.pdf")
    savefig(plt_st, st_path)
    println("  Saved: $st_path")

    # --- (E, N) trajectory ---
    println("  Generating (E, N) trajectory...")

    # Zoom to trajectory neighborhood including bifurcation point
    all_E = vcat(E_traj, [E_bif])
    all_N = vcat(N_traj, [4γ])
    E_min, E_max = extrema(all_E)
    N_min, N_max = extrema(all_N)
    E_margin = max(0.3 * (E_max - E_min), 0.01 * abs(E_bif))
    N_margin = max(0.3 * (N_max - N_min), 0.01 * 4γ)
    E_view = (E_min - E_margin, E_max + E_margin)
    N_view = (N_min - N_margin, N_max + N_margin)

    plt_traj = plot(;
        xlabel=L"E", ylabel=L"\mathcal{N}",
        legend=:topright, size=(600, 400),
        xlims=E_view, ylims=N_view,
        title=fig_title)

    # Soliton curve N = 4√|E| (within view)
    E_sol_lo = min(E_view[1], -1e-4)
    E_sol = range(E_sol_lo, -1e-6, length=300)
    N_sol = [4 * sqrt(abs(e)) for e in E_sol]
    plot!(plt_traj, E_sol, N_sol; color=:orange, ls=:dash, lw=2, label=L"4\sqrt{|E|}")

    # Horizontal line N = 4γ (bifurcation mass)
    hline!(plt_traj, [4γ]; color=:gray, ls=:dot, lw=1, alpha=0.5, label=L"4\gamma")

    # Bifurcation branch
    plot!(plt_traj, E_branch, N_branch; color=:blue, lw=2, label="branch")

    # Bifurcation point
    scatter!(plt_traj, [E_bif], [4γ]; ms=6, color=:green,
             markerstrokewidth=1, label=L"(E_\star, \mathcal{N}_\star)")

    # Trajectory colored by time
    plot!(plt_traj, E_traj, N_traj;
        line_z=t_saves, color=:plasma, lw=1.5,
        label="", colorbar_title=L"t")
    scatter!(plt_traj, [E_traj[1]], [N_traj[1]];
        ms=5, color=:red, markerstrokewidth=0, label=L"t=0")

    traj_path = joinpath(figdir, "$(label)-trajectory.pdf")
    savefig(plt_traj, traj_path)
    println("  Saved: $traj_path")

    # --- Snapshot comparison: |ψ_bound|², |Ψ(x,0)|², |Ψ(x,T)|² ---
    println("  Generating snapshot comparison...")
    dens_bound = abs2.(ψ_bound)
    dens_init  = abs2.(ψ_saves[1])
    dens_final = abs2.(ψ_saves[end])

    plt_snap = plot(;
        xlabel=L"x", ylabel=L"|\Psi|^2",
        legend=:topright, size=(600, 400),
        xlims=(x_lo_zoom, x_hi_zoom),
        title=fig_title)
    plot!(plt_snap, x_grid, dens_bound;
        color=:gray, ls=:dash, lw=2, label="bound state")
    plot!(plt_snap, x_grid, dens_init;
        color=:blue, lw=1.5, label=L"t = 0")
    plot!(plt_snap, x_grid, dens_final;
        color=:red, lw=1.5, label=latexstring("t = $(round(Int, Tmax))"))

    snap_path = joinpath(figdir, "$(label)-snapshot.pdf")
    savefig(plt_snap, snap_path)
    println("  Saved: $snap_path")

    # --- Save simulation data to JLD2 ---
    println("  Saving simulation data...")
    data_path = joinpath(figdir, "$(label)-dynamics-data.jld2")
    jldsave(data_path;
        t_saves, x_grid, density_matrix,
        E_traj, N_traj,
        ψ_bound = ComplexF64.(ψ_bound),
        ψ_init  = ψ_saves[1],
        ψ_final = ψ_saves[end],
        E_bif, γ, E_branch, N_branch,
        E_ref = (conv ? E_ref : E_guess_used),
        label, Tmax, dt, δ_amp, Xmax, Ngrid,
        cap_width, cap_strength, domain=String(domain))
    println("  Saved: $data_path")

    return (; t_saves, ψ_saves, x_grid, E_traj, N_traj, ψ_bound, density_matrix)
end
