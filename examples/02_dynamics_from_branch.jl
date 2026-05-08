###############################################################
# 02_dynamics_from_branch.jl
#
# Load a precomputed bifurcated state (from the shipped data/), perturb it,
# evolve under NLS, and plot the spacetime density and L² mass-vs-time.
#
# Uses `data/fl-stable-gauss/` (a stable transmission resonance per Fig 6.3 of
# the paper) so the dynamics are mild and run quickly.
#
# Usage:
#     julia --project=. examples/02_dynamics_from_branch.jl
#
# Expected runtime: ~1-2 minutes on a laptop.
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "NLSResonanceStability.jl"))
using .NLSResonanceStability

using JLD2
using Plots, LaTeXStrings
using Printf

mkpath(joinpath(@__DIR__, "output"))

# ── 1. Load the shipped bifurcation-branch data ───────────────────
const DATA = joinpath(@__DIR__, "..", "data", "fl-stable-gauss",
                      "fl-stable-gauss-data.jld2")
println("• Loading branch:  $DATA")
br = load(DATA)
@show keys(br)

# Pick one bifurcated state from the branch. The keys depend on how the data
# was saved; here we use a representative entry. Adjust if your data layout
# differs.
ε   = 0.02
γ★  = br["γ"]                    :: Float64
ψ_eps = br["psi_branch"][end]    :: Vector{Float64}   # rightmost on branch
x   = br["x_grid"]               :: Vector{Float64}
Vx  = br["Vx_real"]              :: Vector{Float64}

# Embed into the simulation grid (already provided here as `x`, `Vx`).
ψ0 = ComplexF64.(ψ_eps)
# ── 2. Perturb slightly ───────────────────────────────────────────
δ = 0.005 * exp.(-(x .- 1.5).^2 ./ 0.25)         # small Gaussian bump
ψ0 .+= δ

# ── 3. Evolve via split-step ──────────────────────────────────────
println("• Evolving under NLS for T = 20 ...")
dt    = 0.01
Tmax  = 20.0
Nt    = round(Int, Tmax / dt)
t_saves, ψ_saves = splitstep_evolve(ψ0, x, Vx, dt, Nt;
                                    n_saves      = 200,
                                    cap_strength = 5.0,
                                    cap_width    = 4.0)

# ── 4. Diagnostics + plots ────────────────────────────────────────
ρ = abs2.(ψ_saves)
mass = [sum(abs2, ψ_saves[i, :]) * (x[2] - x[1]) for i in 1:size(ψ_saves, 1)]

plt_st = heatmap(x, t_saves, ρ;
                 xlabel = L"x", ylabel = L"t",
                 title  = L"|\Psi(x,t)|^2",
                 color  = :inferno, framestyle = :box, grid = false,
                 xlims = (-30, 30))

plt_N  = plot(t_saves, mass;
              xlabel = L"t", ylabel = L"\mathcal{N}[\Psi(t)]",
              title = "L² mass (≤ initial; CAP absorbs outflow)",
              lw = 2, color = :navy, legend = false,
              framestyle = :box, grid = false)
hline!(plt_N, [mass[1]]; color = :gray, lw = 1, ls = :dash, label = "")

plt = plot(plt_st, plt_N; layout = (1, 2), size = (1100, 380),
           margin = 5Plots.mm)

out = joinpath(@__DIR__, "output", "02_dynamics.pdf")
savefig(plt, out)
println("\nSaved: $out")
