###############################################################
# 04_custom_V_full_pipeline.jl     **headline example**
#
# Full pipeline on a user-supplied potential V(x) and initial perturbation δΨ₀:
#
#   (a) define V(x)
#   (b) find resonance pair (γ★, U★)
#   (c) continue the bifurcation branch ε ↦ (E(ε), ψ_ε)  via BifurcationKit
#   (d) plot the (E, N) diagram with Ω_★, dN/dE diagnostics
#   (e) pick a small ε, perturb the bifurcated state by δΨ₀
#   (f) evolve under NLS via split-step; plot spacetime + (E,N) trajectory
#
# Edit the highlighted section (1) to study any new compactly supported V and
# any starting perturbation.
#
# Usage:
#     julia --project=. examples/04_custom_V_full_pipeline.jl
#
# Expected runtime: ~5-10 minutes.
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "NLSResonanceStability.jl"))
using .NLSResonanceStability

using Plots, LaTeXStrings
using Printf, JLD2

mkpath(joinpath(@__DIR__, "output"))

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (1) USER INPUTS — change these to study a different problem      ║
# ╚══════════════════════════════════════════════════════════════════╝

# A compactly supported potential. Try anything; just keep supp V ⊂ [a, b].
const a, b = -1.0, 1.0
V(x) = 3.0 * sin(π * x) * (a ≤ x ≤ b)

# Continuation parameters.
const ε_max     = 0.04
const KIND      = :transmission     # :scattering for the half-line case
const N_BRANCH  = 50

# Perturbation δΨ₀ added to the bifurcated state at ε = ε_pick.
const ε_pick = 0.025
δΨ₀(x) = 0.005 * exp(-(x - 5.0)^2 / 1.0)   # small wavepacket downstream
const Tmax = 40.0
const dt   = 0.01

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (2) FIND RESONANCE                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
println("• Searching for the lowest-γ $(KIND) resonance ...")
γ★, U★, x_inner = compute_resonance(V; a = a, b = b, kind = KIND)
@printf "    γ★ = %.4f,    U★(b) = %.4f\n" γ★ U★[end]

Ω, B = NLSResonanceStability.compute_Omega_B(U★, x_inner, γ★, KIND)
@printf "    Ω★ = %+.4f,   ℬ★ = %+.4f\n" Ω B

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (3) BIFURCATION BRANCH                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
println("• Continuing the nonlinear branch via BifurcationKit ...")
branch = bifurcate_branch(V, γ★, U★;
                          x_grid   = x_inner,
                          a = a, b = b,
                          ε_max    = ε_max,
                          n_branch = N_BRANCH,
                          kind     = KIND)
println("    branch length:  $(length(branch.ε))")
println("    E range:        [$(round(minimum(branch.E); digits=4)),"
        * " $(round(maximum(branch.E); digits=4))]")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (4) (E,N) DIAGRAM                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
plt_branch = plot(branch.E, branch.N;
                  xlabel = L"E", ylabel = L"\mathcal{N}",
                  lw = 2, color = :navy, legend = false,
                  title = "Bifurcation branch", framestyle = :box, grid = false)
hline!(plt_branch, [4γ★]; color = :goldenrod, ls = :dash, lw = 1, label = "")
scatter!(plt_branch, [-γ★^2], [4γ★]; color = :black, marker = :circle, ms = 6,
         label = "")
annotate!(plt_branch, -γ★^2, 4γ★, text(@sprintf(" Ω★=%.2f, ℬ★=%.2f", Ω, B), :left, 9))

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (5) DYNAMICS FROM A PERTURBED BIFURCATED STATE                    ║
# ╚══════════════════════════════════════════════════════════════════╝
# Pick the branch entry closest to ε_pick.
i_pick = argmin(abs.(branch.ε .- ε_pick))
ψ_inner = branch.ψ[i_pick]
println("• Evolving from ε = $(round(branch.ε[i_pick]; digits=4)) for T = $(Tmax) ...")

# Embed the bifurcated state into a simulation grid that includes soliton tails.
const X_max = 60.0
const N     = 4096
x_sim = collect(range(-X_max, X_max; length = N))
ψ_sim = zeros(ComplexF64, N)
# Linear interpolate inner profile onto x_sim, set free-soliton tails outside [a,b].
for j in 1:N
    xj = x_sim[j]
    if a ≤ xj ≤ b
        # interp from inner grid
        k = searchsortedlast(x_inner, xj)
        k = clamp(k, 1, length(x_inner) - 1)
        t = (xj - x_inner[k]) / (x_inner[k+1] - x_inner[k])
        ψ_sim[j] = (1 - t) * ψ_inner[k] + t * ψ_inner[k+1]
    end
end
ψ_sim .+= δΨ₀.(x_sim)

Vx = V.(x_sim)
Nt = round(Int, Tmax / dt)
t_saves, ψ_saves = splitstep_evolve(ψ_sim, x_sim, Vx, dt, Nt;
                                    n_saves      = 300,
                                    cap_strength = 8.0,
                                    cap_width    = 6.0)

# ╔══════════════════════════════════════════════════════════════════╗
# ║ (6) PLOTS                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
ρ = abs2.(ψ_saves)
ρ0 = max(1e-12, 1e-7 * maximum(ρ))
plt_st = heatmap(x_sim, t_saves, asinh.(ρ ./ ρ0);
                 xlabel = L"x", ylabel = L"t", color = :inferno,
                 colorbar_title = L"\;\mathrm{asinh}(|\Psi|^2/\rho_0)",
                 framestyle = :box, grid = false,
                 xlims = (-X_max, X_max))

plt = plot(plt_branch, plt_st; layout = (1, 2), size = (1300, 460),
           plot_title = @sprintf("Custom V — branch + dynamics (ε = %.3f)", branch.ε[i_pick]),
           margin = 5Plots.mm)

out = joinpath(@__DIR__, "output", "04_custom_V_full_pipeline.pdf")
savefig(plt, out)
println("\nSaved: $out")
