###############################################################
# regen_trans_stable_zoom.jl
# Plot deviation N - (N_bif + slope_sol*(E - E_bif)) from the
# soliton tangent so the slope discrepancies become full-scale.
###############################################################
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Plots, LaTeXStrings

const DATA = joinpath(@__DIR__, "Figures", "trans-stable-data.jld2")
const OUT  = joinpath(@__DIR__, "Figures", "paper", "trans-stable-NvsE-zoom.pdf")

d = load(DATA)
γ     = d["γ"];    E_bif = d["E_bif"];  N_bif = d["N_bif"]
U_a   = d["U_a"];  ℬ     = d["ℬ"];      dNdE  = d["dNdE_corr"]

slope_sol    = -2 / γ
slope_crit   = -2/γ - 8γ * ℬ^2 / (3 * U_a^4)
slope_branch = dNdE

# Zoom x half-width
Ez_hw = 2e-4
ΔE      = range(-Ez_hw, Ez_hw; length=500)
ΔE_ray  = range(-Ez_hw, 0.0;   length=300)

# Deviations from the soliton tangent: Δslope * ΔE
dev_sol    = zero(ΔE)
dev_crit   = (slope_crit   - slope_sol) .* ΔE
dev_branch = (slope_branch - slope_sol) .* ΔE_ray
# dN/dE = 0 line: in dev coords it has slope (0 - slope_sol) = -slope_sol = 2/γ
dev_dN0    = (-slope_sol) .* ΔE

Nmin = min(minimum(dev_crit), minimum(dev_branch))
Nmax = max(maximum(dev_crit), maximum(dev_branch))
pad  = 0.15 * (Nmax - Nmin)
Nmin -= pad; Nmax += pad

plt = plot(; xlabel=L"E - E_\star",
           ylabel=L"\mathcal{N} - \mathcal{N}_\star + (2/\gamma)(E - E_\star)",
           xlims=(-Ez_hw, Ez_hw), ylims=(Nmin, Nmax),
           legend=false, title="",
           fontfamily="Computer Modern",
           size=(520, 380), dpi=300,
           framestyle=:box, grid=false,
           legendfontsize=10, tickfontsize=11, guidefontsize=14,
           foreground_color_legend=nothing, background_color_legend=:white,
           margin=5Plots.mm)

# Sector shadings fanning out from origin (per Prop. geometry):
#   orange = gentle stable  (slope between 0 and -2/γ, i.e. between sol tangent and dN/dE=0 line)
#   green  = deep stable    (slope steeper than crit, i.e. beyond the Ω=0 tangent toward vertical)
#   white  = instability strips (sol↔crit band where Ω<0; dN/dE>0 wedge around vertical axis)
m_crit = slope_crit - slope_sol
m_dN0  = -slope_sol                      # dN/dE = 0 line has slope 2/γ in deviation coords
crit_L = m_crit * (-Ez_hw);   crit_R = m_crit * ( Ez_hw)
# dN/dE=0 line exits through top/bottom of the plot before reaching ±Ez_hw
dN0_topΔE =  Nmax / m_dN0;    dN0_botΔE =  Nmin / m_dN0

# ORANGE — gentle stable wedges (between sol and dN/dE=0 line)
plot!(plt, Shape([0.0, Ez_hw, Ez_hw, dN0_topΔE],
                 [0.0, 0.0,   Nmax,  Nmax]);
      fillcolor=:darkorange, fillalpha=0.13, linecolor=RGBA(0,0,0,0), label="")
plot!(plt, Shape([0.0, -Ez_hw, -Ez_hw, dN0_botΔE],
                 [0.0, 0.0,   Nmin,  Nmin]);
      fillcolor=:darkorange, fillalpha=0.13, linecolor=RGBA(0,0,0,0), label="")

# GREEN — deep stable wedges (beyond the Ω=0 tangent toward the vertical axis)
plot!(plt, Shape([0.0, -Ez_hw, -Ez_hw, 0.0],
                 [0.0, crit_L, Nmax,  Nmax]);
      fillcolor=:green, fillalpha=0.13, linecolor=RGBA(0,0,0,0), label="")
plot!(plt, Shape([0.0,  Ez_hw,  Ez_hw, 0.0],
                 [0.0,  crit_R, Nmin, Nmin]);
      fillcolor=:green, fillalpha=0.13, linecolor=RGBA(0,0,0,0), label="")

# Reference horizontal axes through origin
hline!(plt, [0]; color=:gray60, ls=:dot, lw=1, label="")
vline!(plt, [0]; color=:gray60, ls=:dot, lw=1, label="")

# Tangent / reference lines
plot!(plt, ΔE, dev_sol;    color=:darkorange, lw=2.2, ls=:dot,
      label=L"\mathrm{soliton}\ (-2/\gamma)")
plot!(plt, ΔE, dev_dN0;    color=:darkorange, lw=2.2, ls=:dash,
      label=L"d\mathcal{N}/dE=0")
plot!(plt, ΔE, dev_crit;   color=:green,      lw=1.4, ls=:solid,
      label=L"\Omega=0\ \mathrm{tangent}")
plot!(plt, ΔE_ray, dev_branch; color=:steelblue,  lw=3.0,
      label=L"\mathrm{branch}\ d\mathcal{N}/dE")

scatter!(plt, [0.0], [0.0]; marker=:circle, ms=6,
         color=:white, markerstrokewidth=1.5, markerstrokecolor=:black, label="")

savefig(plt, OUT)
println("Saved: $OUT")
println("  slopes:  sol=$(round(slope_sol,digits=3))  crit=$(round(slope_crit,digits=3))  branch=$(round(slope_branch,digits=3))")
