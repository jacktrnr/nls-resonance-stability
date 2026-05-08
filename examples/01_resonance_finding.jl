###############################################################
# 01_resonance_finding.jl
#
# Find a scattering resonance for a smooth bump potential on the half line and
# plot the resonance state U★(x).
#
# Usage:
#     julia --project=. examples/01_resonance_finding.jl
#
# Expected runtime: <30 s on a laptop.
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "NLSResonanceStability.jl"))
using .NLSResonanceStability

using Plots
using LaTeXStrings
using Printf

mkpath(joinpath(@__DIR__, "output"))

# ── 1. Choose a compactly supported potential ─────────────────────
# A smooth bump on [0, 1] with depth V0 = -7. `cosine_arch` is one of
# several pre-defined potentials in `src/wells.jl`; see the menu printed
# at module load.
const a, b = 0.0, 1.0       # half-line: V is supported in [a, b]
const V_arch = cosine_arch(a, b, -7.0)
V(x) = V_arch(x)

# ── 2. Find the lowest scattering resonance ───────────────────────
γ★, U★, x_grid = compute_resonance(V; a = a, b = b,
                                   kind = :scattering,
                                   γ_max = 4.0)

@printf "  γ★      = %.6f\n"  γ★
@printf "  U★(b)   = %.6f\n"  U★[end]
@printf "  E★      = -γ★² = %.6f\n"  -γ★^2
@printf "  N★      = 4γ★ = %.6f\n"   4γ★

# ── 3. Plot V(x) and U★(x) ────────────────────────────────────────
plt_V  = plot(x_grid, V.(x_grid);
              xlabel = L"x", ylabel = L"V(x)",
              lw = 2, color = :black, legend = false,
              title = "Potential", framestyle = :box, grid = false)

plt_U  = plot(x_grid, U★;
              xlabel = L"x", ylabel = L"U_\star(x)",
              lw = 2, color = :navy, legend = false,
              title = @sprintf("Scattering resonance, γ★ = %.4f", γ★),
              framestyle = :box, grid = false)
hline!(plt_U, [0]; color = :gray, lw = 0.5, label = "")

plt = plot(plt_V, plt_U; layout = (1, 2), size = (900, 360),
           margin = 5Plots.mm)

out = joinpath(@__DIR__, "output", "01_U_star.pdf")
savefig(plt, out)
println("\nSaved: $out")
