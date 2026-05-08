###############################################################
# resim_and_plot_prof3.jl
#
# Build Section 7 split diagnostics from a saved dynamics case.
# If a full-ψ cache exists, reuse it; otherwise rerun that case
# once to save ψ(t), then compute split trajectories and plots.
###############################################################

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "nls_bifurcation", "NLSBifurcation.jl"))
using .NLSBifurcation
include(joinpath(@__DIR__, "..", "src", "wells.jl"))
include(joinpath(@__DIR__, "..", "src", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))

using JLD2, Plots, LaTeXStrings, Printf

const BASEDIR = joinpath(@__DIR__, "..", "data")
const DESC = get(ENV, "RESIM_DESC", "fl-cos3half-A5")
const PROF_IDX = parse(Int, get(ENV, "RESIM_PROF_IDX", "3"))
const X_COMP = parse(Float64, get(ENV, "RESIM_XCOMP", "10.0"))
const SOL_W = parse(Float64, get(ENV, "RESIM_SOL_W", "15.0"))
const EXTEND_TIME = parse(Float64, get(ENV, "RESIM_EXTEND", "0.0"))
const XLIM_MIN_OVERRIDE = get(ENV, "RESIM_XLIM_MIN", "")
const XLIM_MAX_OVERRIDE = get(ENV, "RESIM_XLIM_MAX", "")

# ── Tunable parameters (override defaults from the saved dyn_path) ──
const PERT_TYPE = :scale   # :gaussian, :scale, :phase
const DELTA_AMP   = 0.03
const FORCE_RESIM = lowercase(get(ENV, "RESIM_FORCE", "false")) == "true"
const SPACETIME_MODE = get(ENV, "RESIM_SPACETIME_MODE", "asinh")
const PLOT_ONLY = lowercase(get(ENV, "RESIM_PLOT_ONLY", "false")) == "true"
const REBUILD_BRANCHES = lowercase(get(ENV, "RESIM_REBUILD_BRANCHES", "false")) == "true"
const SKIP_SPACETIME = lowercase(get(ENV, "RESIM_SKIP_SPACETIME", "false")) == "true"
const RHO0_FACTOR = parse(Float64, get(ENV, "RESIM_RHO0_FACTOR", "1e-7"))

# ═══════════════════════════════════════════════════════════════
# 1. Re-simulate with full ψ saving
# ═══════════════════════════════════════════════════════════════

jld_branch = joinpath(BASEDIR, DESC, "$DESC-data.jld2")
dyn_path   = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-dynamics-data.jld2")
resim_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-resim-psi.jld2")
split_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-split-data.jld2")

# Potential registry (must match run_survey_dynamics.jl)
function get_Vfun(desc)
    if desc == "fl-cos3half-A5"
        return x -> (-1 < x < 1) ? 5.0 * cos(1.5π * x) : 0.0
    elseif desc == "fl-sin-pi-A3"
        return x -> (-1 < x < 1) ? 3.0 * sin(π * x) : 0.0
    elseif desc == "fl-stable-gauss"
        return x -> begin
            (-1 < x < 1) || return 0.0
            6.0 * x * exp(-8.0 * x^2) * (1 - x^2)^2
        end
    elseif desc == "fl-cos7"
        return x -> (-1 < x < 1) ? 10.0 * cos(3.5π * x) : 0.0
    else
        error("Unknown DESC: $desc — add to get_Vfun")
    end
end

function format_time_label(t)
    t_round = round(t)
    if isapprox(t, t_round; atol=1e-9, rtol=0.0)
        return string(Int(t_round))
    end
    return @sprintf("%.1f", t)
end

function branch_display_labels(desc, branch_classes)
    labels = copy(branch_classes)
    if desc == "fl-sin-pi-A3"
        labels = [cls == "ground state branch" ? "E_0 branch" : cls for cls in labels]
    elseif desc == "fl-cos3half-A5"
        labels = [cls == "ground state branch" ? "E_1 branch" : cls for cls in labels]
    end
    return labels
end

Vfun = get_Vfun(DESC)
a, b = -1.0, 1.0

if isfile(resim_path) && !FORCE_RESIM
    println("Loading cached re-simulation: $resim_path")
    R = load(resim_path)
    t_saves = R["t_saves"]
    x_grid  = R["x_grid"]
    ψ_saves_mat = R["psi_saves"]  # matrix (n_frames, Ngrid)
    γ = R["γ"]
    E_bif = R["E_bif"]
    Vx_real = R["Vx_real"]

    if EXTEND_TIME > 0
        prev = load(dyn_path)
        dt = haskey(ENV, "RESIM_DT") ? parse(Float64, ENV["RESIM_DT"]) : prev["dt"]
        cap_width = prev["cap_width"]
        cap_strength = haskey(ENV, "RESIM_CAP") ? parse(Float64, ENV["RESIM_CAP"]) : prev["cap_strength"]
        ψ_current = ComplexF64.(ψ_saves_mat[end, :])
        Vx_eff = ComplexF64.(Vx_real)
        add_cap!(Vx_eff, x_grid, cap_width, cap_strength; sides=:both)

        Nt = round(Int, EXTEND_TIME / dt)
        save_every = max(1, Nt ÷ 300)
        @printf("Continuing cached ψ history by Tmax=%.1f with dt=%.1e (%d steps, save_every=%d)\n",
                EXTEND_TIME, dt, Nt, save_every)
        t_new, ψ_new = splitstep_evolve(ψ_current, x_grid, Vx_eff, dt, Nt;
                                        save_every=save_every)

        n_old = size(ψ_saves_mat, 1)
        n_add = length(t_new) - 1
        ψ_append = Matrix{ComplexF64}(undef, n_add, length(x_grid))
        for k in 2:length(t_new)
            ψ_append[k-1, :] .= ψ_new[k]
        end
        t_offset = t_saves[end]
        t_append = t_offset .+ t_new[2:end]

        ψ_saves_mat = vcat(ψ_saves_mat, ψ_append)
        t_saves = vcat(t_saves, t_append)
        @printf("  Extended cached history: %d -> %d frames, t_end = %.1f\n",
                n_old, size(ψ_saves_mat, 1), t_saves[end])
        jldsave(resim_path; t_saves, x_grid, psi_saves = ψ_saves_mat, γ, E_bif, Vx_real)
    end
else
    println("Re-simulating prof$PROF_IDX with full ψ save...")

    prev = load(dyn_path)
    Xmax       = haskey(ENV, "RESIM_XMAX") ? parse(Float64, ENV["RESIM_XMAX"]) : prev["Xmax"]
    Ngrid      = haskey(ENV, "RESIM_NGRID") ? parse(Int, ENV["RESIM_NGRID"]) : prev["Ngrid"]
    Tmax       = haskey(ENV, "RESIM_TMAX") ? parse(Float64, ENV["RESIM_TMAX"]) : prev["Tmax"]
    dt         = prev["dt"]
    δ_amp      = prev["δ_amp"]
    cap_width  = prev["cap_width"]
    cap_strength = haskey(ENV, "RESIM_CAP") ? parse(Float64, ENV["RESIM_CAP"]) : prev["cap_strength"]
    γ = prev["γ"]
    E_bif = prev["E_bif"]
    @printf("  Using Xmax=%.1f, Ngrid=%d, Tmax=%.1f, cap_strength=%.1f\n",
            Xmax, Ngrid, Tmax, cap_strength)

    # Rebuild grid and initial state exactly as run_simulation did
    x_lo, x_hi = -Xmax, Xmax
    dx = (x_hi - x_lo) / (Ngrid + 1)
    x_grid = [x_lo + j*dx for j in 1:Ngrid]

    # Load profile + Newton-refine
    xs_p, us_p, ε2, Es_branch, Ns_branch, γ_ret, E_bif_ret =
        load_profile(jld_branch; idx=PROF_IDX)
    ψ_bound = interp_with_exp_tails(xs_p, us_p, x_grid, γ)
    Vx_real = [Vfun(xi) for xi in x_grid]

    # E_guess from branch
    function E_at_mass(Es, Ns, N_target)
        k = argmin(abs.(Ns .- N_target))
        if 1 < k < length(Ns)
            for (j1, j2) in ((k-1, k), (k, k+1))
                n1, n2 = Ns[j1], Ns[j2]
                if (n1 - N_target)*(n2 - N_target) <= 0 && n1 != n2
                    t = (N_target - n1)/(n2 - n1)
                    return (1-t)*Es[j1] + t*Es[j2]
                end
            end
        end
        return Es[k]
    end
    function profile_mass(x, u)
        m = 0.0
        for j in 1:length(x)-1
            m += 0.5*(u[j]^2 + u[j+1]^2)*(x[j+1] - x[j])
        end
        return m
    end
    N_prof = profile_mass(xs_p, us_p)
    E_guess = E_at_mass(Es_branch, Ns_branch, N_prof)

    println("  E_guess = $E_guess, Newton refining...")
    ψ_ref, E_ref, conv = newton_refine(ψ_bound, E_guess, x_grid, Vx_real;
                                        maxiter=80, tol=1e-10, fix_norm=true)
    if conv
        println("  Newton converged: E = $E_ref")
        ψ_bound = ψ_ref
    else
        println("  Newton didn't converge (tol), using refined profile anyway")
        ψ_bound = ψ_ref
    end

    # CAP
    Vx_eff = ComplexF64.(Vx_real)
    add_cap!(Vx_eff, x_grid, cap_width, cap_strength; sides=:both)

    # Perturbation (gaussian even+odd, same as run_simulation)
    _, i_peak = findmax(abs.(ψ_bound))
    x_peak = x_grid[i_peak]
    σ_pert = 1.0/(2γ)
    amp = δ_amp * maximum(abs, ψ_bound)
    h = [amp*(exp(-((x_grid[j]-x_peak)/σ_pert)^2/2) +
              0.5*((x_grid[j]-x_peak)/σ_pert)*exp(-((x_grid[j]-x_peak)/σ_pert)^2/2))
         for j in 1:Ngrid]
    ψ_init = ComplexF64.(ψ_bound .+ h)
    println("  Initial mass = $(dx*sum(abs2, ψ_init))")

    # Evolve with save_every matching the previous run (same number of frames)
    Nt = round(Int, Tmax/dt)
    save_every = max(1, Nt ÷ 1500)
    @printf("  Tmax=%.1f, dt=%.1e, %d steps, save_every=%d\n", Tmax, dt, Nt, save_every)

    t_saves, ψ_saves = splitstep_evolve(ψ_init, x_grid, Vx_eff, dt, Nt;
                                         save_every=save_every)
    n_frames = length(t_saves)
    println("  Done: $n_frames frames")

    # Pack ψ_saves into matrix
    ψ_saves_mat = Matrix{ComplexF64}(undef, n_frames, Ngrid)
    for k in 1:n_frames
        ψ_saves_mat[k, :] .= ψ_saves[k]
    end

    println("  Saving: $resim_path")
    jldsave(resim_path; t_saves, x_grid, psi_saves = ψ_saves_mat, γ, E_bif, Vx_real)
end

n_frames = length(t_saves)
Ngrid = length(x_grid)
dx = x_grid[2] - x_grid[1]

function rayleigh_region(ψ, Vx, mask, dx)
    n = length(ψ)
    ψe = [zero(eltype(ψ)); ψ; zero(eltype(ψ))]
    dψ = (ψe[3:end] .- ψe[1:end-2]) ./ (2dx)
    kin = 0.0; pot = 0.0; nonl = 0.0; nrm = 0.0
    for j in 1:n
        mask[j] || continue
        kin  += abs2(dψ[j])
        pot  += Vx[j] * abs2(ψ[j])
        nonl += abs2(ψ[j])^2
        nrm  += abs2(ψ[j])
    end
    return (kin + pot - nonl) * dx / (nrm * dx + 1e-30), nrm * dx
end

# Bulk velocity on a region: v = 2P/N, P = ∫ Im(ψ̄ ∂_x ψ) dx
function bulk_velocity(ψ, mask, dx)
    n = length(ψ)
    ψe = [zero(eltype(ψ)); ψ; zero(eltype(ψ))]
    dψ = (ψe[3:end] .- ψe[1:end-2]) ./ (2dx)
    P = 0.0; N = 0.0
    for j in 1:n
        mask[j] || continue
        P += imag(conj(ψ[j]) * dψ[j])
        N += abs2(ψ[j])
    end
    N < 1e-12 && return 0.0
    return 2 * (P * dx) / (N * dx)
end

function soliton_peak_positions(ψ_saves_mat, x_grid, mask_out)
    n_frames = size(ψ_saves_mat, 1)
    x_peak_sol = fill(NaN, n_frames)
    dens_peak_sol = zeros(n_frames)
    for k in 1:n_frames
        ψk = @view ψ_saves_mat[k, :]
        max_out = 0.0
        j_peak = 0
        for j in eachindex(x_grid)
            mask_out[j] || continue
            a2 = abs2(ψk[j])
            if a2 > max_out
                max_out = a2
                j_peak = j
            end
        end
        if j_peak != 0
            x_peak_sol[k] = x_grid[j_peak]
            dens_peak_sol[k] = max_out
        end
    end
    return x_peak_sol, dens_peak_sol
end

"""Interpolate a branch (Es, Ns) at query E. Returns NaN if out of range."""
function branch_N_at(Es, Ns, e)
    (e < Es[1] - 1e-9 || e > Es[end] + 1e-9) && return NaN
    idx = searchsortedfirst(Es, e)
    if idx == 1
        return Ns[1]
    elseif idx > length(Es)
        return Ns[end]
    end
    t = (e - Es[idx-1]) / (Es[idx] - Es[idx-1])
    return (1-t)*Ns[idx-1] + t*Ns[idx]
end

"""Check if branches i and j agree in their overlap region (within tol)."""
function branches_overlap(Ei, Ni, Ej, Nj; tol=0.03, min_overlap=5)
    overlap_count = 0
    disagree = 0
    for (e, n) in zip(Ej, Nj)
        nj_on_i = branch_N_at(Ei, Ni, e)
        if !isnan(nj_on_i)
            overlap_count += 1
            if abs(nj_on_i - n) > tol
                disagree += 1
            end
        end
    end
    overlap_count < min_overlap && return false
    return disagree == 0
end

function classify_branch(Es, Ns, E_bif, N_bif; tolE=0.05, tolN=0.3)
    for (e, n) in zip(Es, Ns)
        if abs(e - E_bif) < tolE && abs(n - N_bif) < tolN
            return "transmission resonance branch"
        end
    end
    return "non-resonance branch"
end

function merge_and_classify_branches(branches_Es, branches_Ns, E_bif, γ)
    merged_Es = Vector{Vector{Float64}}()
    merged_Ns = Vector{Vector{Float64}}()
    merged_labels = String[]
    used = falses(length(branches_Es))
    for i in 1:length(branches_Es)
        used[i] && continue
        sp_i = sortperm(branches_Es[i])
        Ei = branches_Es[i][sp_i]
        Ni = branches_Ns[i][sp_i]
        group = [i]
        for j in (i+1):length(branches_Es)
            used[j] && continue
            sp_j = sortperm(branches_Es[j])
            Ej = branches_Es[j][sp_j]
            Nj = branches_Ns[j][sp_j]
            if branches_overlap(Ei, Ni, Ej, Nj) || branches_overlap(Ej, Nj, Ei, Ni)
                push!(group, j)
                used[j] = true
                all_E = vcat(Ei, Ej)
                all_N = vcat(Ni, Nj)
                sp_merge = sortperm(all_E)
                Ei = all_E[sp_merge]
                Ni = all_N[sp_merge]
                keep = trues(length(Ei))
                for k in 2:length(Ei)
                    if Ei[k] - Ei[k-1] < 1e-6
                        keep[k] = false
                    end
                end
                Ei = Ei[keep]
                Ni = Ni[keep]
            end
        end
        used[i] = true
        push!(merged_Es, Ei)
        push!(merged_Ns, Ni)
        push!(merged_labels, join(["seed $g" for g in group], "+"))
    end
    branch_classes = [classify_branch(Es, Ns, E_bif, 4γ)
                      for (Es, Ns) in zip(merged_Es, merged_Ns)]
    # Disambiguate co-existing non-resonance branches: the branch whose
    # rightmost (least-negative) E is most negative bifurcates from the
    # deepest linear bound state, i.e. the ground branch E_0.
    nr_idx = findall(==("non-resonance branch"), branch_classes)
    if length(nr_idx) >= 2
        rightmost_E = [maximum(merged_Es[i]) for i in nr_idx]
        sp = sortperm(rightmost_E)   # most-negative-max first
        branch_classes[nr_idx[sp[1]]] = "E_0 branch"
        for k in 2:length(nr_idx)
            branch_classes[nr_idx[sp[k]]] = "E_1 branch"
        end
    elseif length(nr_idx) == 1
        branch_classes[nr_idx[1]] = "ground state branch"
    end
    return merged_Es, merged_Ns, merged_labels, branch_classes
end

function compute_all_full_line_branches(Vfun, a, b, E_bif, γ)
    println("Rebuilding full-line branches from seed continuation...")
    seeds = find_all_seeds(a, b, Vfun;
        E_list=[E_bif - 0.5], N=3000, ζmax=8.0, nscan=5000,
        tolH=1e-10, slope_set=(+1, -1))
    seeds = deduplicate_seeds(seeds)
    @printf("  Found %d seed(s)\n", length(seeds))

    branches_Es = Vector{Vector{Float64}}()
    branches_Ns = Vector{Vector{Float64}}()
    for (si, seed) in enumerate(seeds)
        @printf("  Continuing seed %d/%d: E=%.4f, c=%.3e, slope=%+d\n",
                si, length(seeds), seed.p.E, seed.c, seed.slope_sign)
        br_try = continue_single_seed(seed, a, b, Vfun;
            N=3000, p_min=E_bif - 2.0, p_max=-1e-6,
            ds=0.001, dsmin=1e-7, dsmax=0.001, max_steps=2000, ζ_min=1e-4)
        isempty(br_try.branch) && continue

        Es = Float64[]
        Ns = Float64[]
        for sol in br_try.branch
            sol.c < 1e-12 && continue
            sol.param > 0 && continue
            ss = get(sol, :slope_sign, +1)
            xs, us, vs = integrate_support(a, b, sol.param, Vfun;
                N=3000, c=sol.c, slope_sign=ss)
            isempty(xs) && continue
            N_val = compute_norm(a, b, sol.param, xs, us, vs)
            isfinite(N_val) || continue
            push!(Es, sol.param)
            push!(Ns, N_val)
        end
        isempty(Es) && continue
        sp = sortperm(Es)
        push!(branches_Es, Es[sp])
        push!(branches_Ns, Ns[sp])
    end

    return merge_and_classify_branches(branches_Es, branches_Ns, E_bif, γ)
end

if PLOT_ONLY
    println("Loading cached split diagnostics: $split_path")
    split_prev = load(split_path)
    E_in = split_prev["E_in"]; N_in = split_prev["N_in"]
    E_out = split_prev["E_out"]; N_out = split_prev["N_out"]
    E_in_amp = split_prev["E_in_amp"]; E_out_amp = split_prev["E_out_amp"]
    v_out = split_prev["v_out"]; E_out_rest = split_prev["E_out_rest"]
    N_sol = split_prev["N_sol"]; E_sol_amp = split_prev["E_sol_amp"]
    v_sol = split_prev["v_sol"]; E_sol_rest = split_prev["E_sol_rest"]
    merged_Es = split_prev["merged_Es"]; merged_Ns = split_prev["merged_Ns"]
    merged_labels = haskey(split_prev, "merged_labels") ? split_prev["merged_labels"] : fill("branch", length(merged_Es))
    # Re-classify on load so PLOT_ONLY benefits from updated classification
    # (E_0 / E_1 disambiguation) even when the cached labels are stale.
    _, _, _, branch_classes = merge_and_classify_branches(merged_Es, merged_Ns, split_prev["E_bif"], split_prev["γ"])
else
    # ═══════════════════════════════════════════════════════════════
    # 2. Compute (N_in, E_in) on |x|<X_COMP and (N_out, E_out) on |x|>X_COMP
    # ═══════════════════════════════════════════════════════════════
    mask_in  = abs.(x_grid) .<  X_COMP
    mask_out = abs.(x_grid) .>= X_COMP

    println("Computing (E_in, N_in) and (E_out, N_out) trajectories...")
    E_in  = Float64[]; N_in  = Float64[]
    E_out = Float64[]; N_out = Float64[]
    E_in_amp  = Float64[]
    E_out_amp = Float64[]
    v_out     = Float64[]
    E_out_rest = Float64[]
    for k in 1:n_frames
        ψk = @view ψ_saves_mat[k, :]
        Ei, Ni = rayleigh_region(ψk, Vx_real, mask_in,  dx)
        Eo, No = rayleigh_region(ψk, Vx_real, mask_out, dx)
        push!(E_in, Ei); push!(N_in, Ni)
        push!(E_out, Eo); push!(N_out, No)

        max_in  = 0.0; max_out = 0.0
        for j in 1:Ngrid
            a2 = abs2(ψk[j])
            if mask_in[j]  && a2 > max_in;  max_in  = a2; end
            if mask_out[j] && a2 > max_out; max_out = a2; end
        end
        push!(E_in_amp,  -max_in  / 2)
        push!(E_out_amp, -max_out / 2)

        vk = bulk_velocity(ψk, mask_out, dx)
        push!(v_out, vk)
        push!(E_out_rest, Eo - vk^2 / 4)
    end

    N_sol = Float64[]
    E_sol_amp = Float64[]
    v_sol = Float64[]
    E_sol_rest = Float64[]
    for k in 1:n_frames
        ψk = @view ψ_saves_mat[k, :]
        max_out = 0.0; j_peak = 0
        for j in 1:Ngrid
            mask_out[j] || continue
            a2 = abs2(ψk[j])
            if a2 > max_out; max_out = a2; j_peak = j; end
        end
        if j_peak == 0 || max_out < 1e-6
            push!(N_sol, 0.0); push!(E_sol_amp, 0.0)
            push!(v_sol, 0.0); push!(E_sol_rest, 0.0)
            continue
        end
        local x_peak = x_grid[j_peak]
        mask_sol = (abs.(x_grid .- x_peak) .<= SOL_W) .& mask_out
        Esr, Nsr = rayleigh_region(ψk, Vx_real, mask_sol, dx)
        vsk = bulk_velocity(ψk, mask_sol, dx)
        push!(N_sol, Nsr)
        push!(E_sol_amp, -max_out / 2)
        push!(v_sol, vsk)
        push!(E_sol_rest, Esr - vsk^2 / 4)
    end
    @printf("  Isolated soliton: N_sol: %.3f → %.3f ; v_sol: %.3f → %.3f ; E_sol (rest): %.3f → %.3f ; E_sol (amp): %.3f → %.3f\n",
            N_sol[1], N_sol[end], v_sol[1], v_sol[end],
            E_sol_rest[1], E_sol_rest[end], E_sol_amp[1], E_sol_amp[end])
    @printf("  Clean-soliton check at t=T: 4√|E_rest| = %.3f  vs  N_sol = %.3f\n",
            4*sqrt(abs(E_sol_rest[end])), N_sol[end])
    @printf("  N_in:  %.3f → %.3f ; E_in  (Rayleigh): %.3f → %.3f ; E_in  (amp): %.3f → %.3f\n",
            N_in[1], N_in[end], E_in[1], E_in[end], E_in_amp[1], E_in_amp[end])
    @printf("  N_out: %.3f → %.3f ; E_out (Rayleigh): %.3f → %.3f ; E_out (amp): %.3f → %.3f\n",
            N_out[1], N_out[end], E_out[1], E_out[end], E_out_amp[1], E_out_amp[end])
    @printf("  v_out: %.3f → %.3f ; E_out (rest = Rayleigh - v²/4): %.3f → %.3f\n",
            v_out[1], v_out[end], E_out_rest[1], E_out_rest[end])

    # ═══════════════════════════════════════════════════════════════
    # 3. Load or rebuild branches, then dedupe/classify
    # ═══════════════════════════════════════════════════════════════

    if REBUILD_BRANCHES
        merged_Es, merged_Ns, merged_labels, branch_classes =
            compute_all_full_line_branches(Vfun, a, b, E_bif, γ)
    else
        if isfile(split_path)
            split_prev = load(split_path)
            branches_Es = haskey(split_prev, "branches_Es") ? split_prev["branches_Es"] : split_prev["merged_Es"]
            branches_Ns = haskey(split_prev, "branches_Ns") ? split_prev["branches_Ns"] : split_prev["merged_Ns"]
        else
            bd = load(jld_branch)
            sp = sortperm(bd["Es"])
            branches_Es = [bd["Es"][sp]]
            branches_Ns = [bd["Ns"][sp]]
            println("No split-data; bootstrapped from $jld_branch (single branch)")
        end
        println("Loaded $(length(branches_Es)) branches before dedup")
        merged_Es, merged_Ns, merged_labels, branch_classes =
            merge_and_classify_branches(branches_Es, branches_Ns, E_bif, γ)
    end

    println("After dedup: $(length(merged_Es)) branches")
    for (l, Es) in zip(merged_labels, merged_Es)
        @printf("  %s : %d pts, E ∈ [%.3f, %.3f]\n", l, length(Es),
                minimum(Es), maximum(Es))
    end
    for (l, c) in zip(merged_labels, branch_classes)
        println("  $l  →  $c")
    end
end

mask_in  = abs.(x_grid) .<  X_COMP
mask_out = abs.(x_grid) .>= X_COMP
x_peak_sol, dens_peak_sol = soliton_peak_positions(ψ_saves_mat, x_grid, mask_out)
split_valid = collect((N_sol .> 0.05) .& (dens_peak_sol .> 1e-4) .& (abs.(x_peak_sol) .> abs(b) + 2.0))

E_total = Float64[]
N_total = Float64[]
mask_all = trues(length(x_grid))
for k in 1:n_frames
    ψk = @view ψ_saves_mat[k, :]
    Et, Nt = rayleigh_region(ψk, Vx_real, mask_all, dx)
    push!(E_total, Et)
    push!(N_total, Nt)
end

# ═══════════════════════════════════════════════════════════════
# 4. Plot
# ═══════════════════════════════════════════════════════════════

println("Plotting...")

t_init_label = format_time_label(t_saves[1])
t_final_label = format_time_label(t_saves[end])
plot_branch_labels = branch_display_labels(DESC, branch_classes)
N_total = vec(sum(abs2.(ψ_saves_mat); dims=2)) .* dx

# ──────────────────────────────────────────────────────────────────────
# t_end clean split: soliton template + branch projection (no cutoffs)
# ──────────────────────────────────────────────────────────────────────

function fit_soliton_template(ψ, x_grid, mask_out, dx)
    n = length(x_grid)
    j_peak = 0; max_dens = 0.0
    for j in eachindex(x_grid)
        mask_out[j] || continue
        a2 = abs2(ψ[j])
        if a2 > max_dens; max_dens = a2; j_peak = j; end
    end
    j_peak == 0 && return (x0=NaN, γ=NaN, v=NaN, φ=NaN,
                            ψ_sol=zeros(ComplexF64, n), dens=zeros(n))
    x0 = x_grid[j_peak]
    # Standard NLS soliton: |ψ|²_peak = 2γ², so γ = sqrt(peak/2).
    γ_eff = sqrt(max(max_dens, 0.0) / 2)
    halfw = max(5.0, 3.0/max(γ_eff, 1e-3))
    P = 0.0; Nm = 0.0
    for j in 2:(n-1)
        mask_out[j] || continue
        abs(x_grid[j] - x0) <= halfw || continue
        dψ = (ψ[j+1] - ψ[j-1]) / (2dx)
        P += imag(conj(ψ[j]) * dψ)
        Nm += abs2(ψ[j])
    end
    v = Nm > 1e-12 ? 2*P/Nm : 0.0
    # Build a unit-amplitude moving sech template g(x) = sech(γ(x-x0)) e^{i v x / 2}.
    # Project ψ onto g over the OUTER region (mask_out). With the projection
    # coefficient C = ⟨g,ψ⟩_outer / ⟨g,g⟩_outer, ψ_sol := C·g satisfies
    # ⟨g, ψ-ψ_sol⟩_outer = 0, so on the outer region masses add exactly:
    # ∫_outer |ψ|² = ∫_outer |ψ_sol|² + ∫_outer |ψ-ψ_sol|².
    g_template = ComplexF64[sech(γ_eff*(xj - x0)) * exp(im*v*xj/2) for xj in x_grid]
    ip = sum(conj(g_template[j]) * ψ[j] for j in eachindex(x_grid) if mask_out[j]) * dx
    g2 = sum(abs2(g_template[j])         for j in eachindex(x_grid) if mask_out[j]) * dx
    C = g2 > 1e-30 ? ip/g2 : 2γ_eff*exp(im*angle(ψ[j_peak] * exp(-im*v*x0/2)))
    φ = angle(C)
    A_amp = abs(C)
    ψ_sol = C .* g_template
    dens = abs2.(ψ_sol)
    γ_fit = A_amp / 2     # effective γ via |ψ_sol|_max = A_amp = 2γ_fit
    return (x0=x0, γ=γ_fit, v=v, φ=φ, ψ_sol=ψ_sol, dens=dens)
end

function rayleigh_E_full(ψ, Vx, dx)
    n = length(ψ)
    kin = 0.0; pot = 0.0; nonl = 0.0; nrm = 0.0
    for j in 2:(n-1)
        dψ = (ψ[j+1] - ψ[j-1]) / (2dx)
        kin  += abs2(dψ)
        pot  += Vx[j] * abs2(ψ[j])
        nonl += abs2(ψ[j])^2
        nrm  += abs2(ψ[j])
    end
    nrm < 1e-30 && return (E=NaN, N=0.0)
    return (E=(kin + pot - nonl) * dx / (nrm * dx), N=nrm * dx)
end

function _interp_linear(xs, ys, xq)
    out = Vector{Float64}(undef, length(xq))
    for (k, x) in enumerate(xq)
        if x <= xs[1]
            out[k] = ys[1]
        elseif x >= xs[end]
            out[k] = ys[end]
        else
            j = searchsortedlast(xs, x)
            t = (x - xs[j]) / (xs[j+1] - xs[j])
            out[k] = (1-t)*ys[j] + t*ys[j+1]
        end
    end
    return out
end

function project_density_onto_branch(dens_resid, x_grid,
        branch_Es, branch_Ns, Vfun, a, b; N_int=2000)
    dx = x_grid[2] - x_grid[1]
    x_left = x_grid[1]; x_right = x_grid[end]
    best_E = NaN; best_N_branch = NaN; best_score = Inf; best_N_core = 0.0
    for (k, E) in enumerate(branch_Es)
        E < -1e-9 || continue
        xs, us, vs = integrate_support(a, b, E, Vfun; N=N_int, c=1e-3, slope_sign=+1)
        isempty(xs) && continue
        xfull, ψfull = glue_full_solution(a, b, E, xs, us, vs;
            x_left=x_left, x_right=x_right, NL=800, NR=800)
        isempty(xfull) && continue
        U = _interp_linear(Float64.(xfull), Float64.(real.(ψfull)), x_grid)
        Udens = U .^ 2
        # Strict shape match: ‖dens_resid − Udens‖² (no c² rescaling).
        # Forces the projection to find the branch whose mode actually matches
        # the residual mass and shape, instead of inflating a tiny-norm mode.
        score = 0.0; tot = 0.0
        for j in eachindex(x_grid)
            r = dens_resid[j] - Udens[j]
            score += r*r
            tot   += Udens[j]
        end
        score *= dx; tot *= dx
        if score < best_score
            best_score = score
            best_E = E
            best_N_branch = branch_Ns[k]
            best_N_core = tot
        end
    end
    return (E=best_E, N_branch=best_N_branch,
            N_core=best_N_core, score=best_score)
end

ψ_end = ComplexF64.(ψ_saves_mat[end, :])
mask_out_end = abs.(x_grid) .>= X_COMP
mask_in_end  = abs.(x_grid) .<  X_COMP
sol_fit = fit_soliton_template(ψ_end, x_grid, mask_out_end, dx)

# Core: cutoff Rayleigh on |x|<X_COMP at t_end (cached E_in/N_in).
E_core_end = E_in[end]; N_core_end = N_in[end]
N_total_init = sum(abs2, ψ_saves_mat[1, :]) * dx
N_total_end  = sum(abs2, ψ_end) * dx

# Outer-region mass split via the orthogonal projection in fit_soliton_template.
# Because ψ_sol = C·g with C = ⟨g,ψ⟩_outer/‖g‖²_outer, on the outer region
#     ∫_outer |ψ|² = ∫_outer |ψ_sol|² + ∫_outer |ψ-ψ_sol|²
# exactly (no cross-term), so N_core + N_sol + N_rad = N(T) by construction.
ψ_resid_end = ψ_end .- sol_fit.ψ_sol
N_inner_end_total = sum(abs2(ψ_end[j])         for j in eachindex(x_grid) if mask_in_end[j])  * dx
N_outer_end_total = sum(abs2(ψ_end[j])         for j in eachindex(x_grid) if mask_out_end[j]) * dx
N_sol_outer       = sum(abs2(sol_fit.ψ_sol[j]) for j in eachindex(x_grid) if mask_out_end[j]) * dx
N_outer_resid     = sum(abs2(ψ_resid_end[j])   for j in eachindex(x_grid) if mask_out_end[j]) * dx

N_sol_fit = N_sol_outer
# Place the soliton marker on the free-soliton curve N = 4√|E|, i.e., E = -(N/4)².
E_sol_fit = -(N_sol_fit / 4)^2
N_rad_fit = N_outer_resid

discrepancy = N_total_end - (N_core_end + N_sol_fit + N_rad_fit)
@printf("t=%.1f split: E_core=%.4f, N_core=%.4f, N_sol=%.4f, N_rad=%.4f, sum=%.4f (N_tot(T)=%.4f, discrepancy=%.2e)\n",
        t_saves[end], E_core_end, N_core_end, N_sol_fit, N_rad_fit,
        N_core_end + N_sol_fit + N_rad_fit, N_total_end, discrepancy)
@printf("  diagnostics: grid x∈[%.1f, %.1f]; outer (|x|≥%.1f): total=%.4f, soliton=%.4f, residual=%.4f\n",
        x_grid[1], x_grid[end], X_COMP, N_outer_end_total, N_sol_outer, N_outer_resid)

E_core_plot = [E_core_end]
N_core_plot = [N_core_end]
E_sol_plot  = [E_sol_fit]
N_sol_plot  = [N_sol_fit]
split_valid_plot = [isfinite(E_core_end) && N_core_end > 1e-6]

branch_stable = map(branch_classes) do cls
    !(cls in ("transmission resonance branch", "resonance", "E_1 branch"))
end

plt = split_trajectory_plot(;
    E_branches=merged_Es,
    N_branches=merged_Ns,
    branch_classes=branch_classes,
    branch_stable=branch_stable,
    branch_labels=plot_branch_labels,
    E_bif=E_bif,
    γ=γ,
    E_core=E_core_plot,
    N_core=N_core_plot,
    E_sol_rest=E_sol_plot,
    N_sol=N_sol_plot,
    E_total=E_total,
    N_total=N_total,
    split_valid=split_valid_plot,
    N_rad_final=N_rad_fit,
    core_label=latexstring("\\Psi_{\\mathrm{core}}(x,$t_final_label)"),
    soliton_label=latexstring("\\Psi_{\\mathrm{soliton}}(x,$t_final_label)"),
    total_label=latexstring("\\Psi(x,$t_init_label)"),
    rad_label=latexstring("\\mathcal{N}_{\\mathrm{rad}}($t_final_label)"),
)

out_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-trajectory-split.pdf")
savefig(plt, out_path)
println("Saved: $out_path")

if !PLOT_ONLY
    # Update split-data with the new trajectories
    jldsave(joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-split-data.jld2");
        t_saves, E_in, N_in, E_out, N_out,
        E_in_amp, E_out_amp, v_out, E_out_rest,
        N_sol, E_sol_amp, v_sol, E_sol_rest, SOL_W,
        x_peak_sol, dens_peak_sol, split_valid,
        E_total, N_total,
        merged_Es, merged_Ns, merged_labels, branch_classes,
        γ, E_bif, X_COMP)
    println("Updated split-data with E_in/E_out trajectories.")
end

# ═══════════════════════════════════════════════════════════════
# 5. Spacetime heatmap (no title, untitled per style spec)
# ═══════════════════════════════════════════════════════════════

if SKIP_SPACETIME
    println("Skipping spacetime regeneration (RESIM_SKIP_SPACETIME=true).")
else

density_matrix = Matrix{Float64}(undef, n_frames, Ngrid)
for k in 1:n_frames
    for j in 1:Ngrid
        density_matrix[k, j] = abs2(ψ_saves_mat[k, j])
    end
end
peak_dens = maximum(density_matrix)
active_frac = SPACETIME_MODE == "asinh" ? 0.001 : 0.01
col_active = vec(maximum(density_matrix; dims=1)) .> active_frac * peak_dens
j_first = findfirst(col_active); j_last = findlast(col_active)
x_lo_zoom = x_grid[j_first] - 2.0
x_hi_zoom = x_grid[j_last]  + 2.0
if !isempty(XLIM_MIN_OVERRIDE)
    x_lo_zoom = parse(Float64, XLIM_MIN_OVERRIDE)
end
if !isempty(XLIM_MAX_OVERRIDE)
    x_hi_zoom = parse(Float64, XLIM_MAX_OVERRIDE)
end
# Always symmetrize about x=0 so the V well (core) sits at the visual center.
x_rad = max(abs(x_lo_zoom), abs(x_hi_zoom))
x_lo_zoom = -x_rad
x_hi_zoom =  x_rad

ρ0 = max(1e-14, RHO0_FACTOR * peak_dens)

# (1) asinh-rescaled spacetime — emphasizes radiation
plt_st_asinh = heatmap(x_grid, t_saves, asinh.(density_matrix ./ ρ0);
    xlabel=L"x", ylabel=L"t",
    colorbar_title=L"\mathrm{asinh}(|\Psi|^2/\rho_0)",
    color=:inferno,
    xlims=(x_lo_zoom, x_hi_zoom),
    size=(900, 360),
    fontfamily="Computer Modern",
    tickfontsize=13, guidefontsize=16, colorbar_titlefontsize=14,
    margin=5Plots.mm,
    linewidth=0, linealpha=0, linecolor=:match)
st_asinh_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-spacetime.pdf")
savefig(plt_st_asinh, replace(st_asinh_path, ".pdf" => ".png"))
savefig(plt_st_asinh, st_asinh_path)
println("Saved: $st_asinh_path")

# (2) raw |ψ|² spacetime — same geometry, linear color scale
plt_st_raw = heatmap(x_grid, t_saves, density_matrix;
    xlabel=L"x", ylabel=L"t",
    colorbar_title=L"|\Psi(x,t)|^2",
    color=:inferno, clims=(0, peak_dens),
    xlims=(x_lo_zoom, x_hi_zoom),
    size=(900, 360),
    fontfamily="Computer Modern",
    tickfontsize=13, guidefontsize=16, colorbar_titlefontsize=14,
    margin=5Plots.mm,
    linewidth=0, linealpha=0, linecolor=:match)
st_raw_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-spacetime-raw.pdf")
savefig(plt_st_raw, replace(st_raw_path, ".pdf" => ".png"))
savefig(plt_st_raw, st_raw_path)
println("Saved: $st_raw_path")

end  # SKIP_SPACETIME
