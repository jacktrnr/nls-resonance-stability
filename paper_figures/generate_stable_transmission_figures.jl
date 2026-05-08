###############################################################
# generate_stable_transmission_figures.jl
#
# Generates figures for the STABLE transmission resonance example:
#   V(x) = 6x·exp(-8x²)·(1-x²)²  on [-1,1]
#   (via scaled coordinate y = 2(x-a)/(b-a) - 1)
#
# This potential gives γ ≈ 0.104, Ω > 0, Window 1 stability.
#
# Uses run_transmission_figs() from transmission_figures.jl.
###############################################################

# transmission_figures.jl handles all includes, Pkg activation,
# plot defaults, and defines run_transmission_figs().
include(joinpath(@__DIR__, "transmission_figures.jl"))

# ── Potential definition (from tr_vcheck.jl spec 17) ──────────
# V(x) = 6y·exp(-8y²)·(1-y²)²  where y = 2(x-a)/(b-a) - 1
a_fl = -1.0
b_fl =  1.0

@inline _in_support(x, a, b) = (a < x < b)
@inline _scaled_coord(x, a, b) = 2.0 * (x - a) / (b - a) - 1.0

Vfun_stable = let a = a_fl, b = b_fl
    x -> begin
        _in_support(x, a, b) || return 0.0
        y = _scaled_coord(x, a, b)
        6.0 * y * exp(-8.0 * y^2) * (1 - y^2)^2
    end
end

run_transmission_figs(Vfun_stable, a_fl, b_fl, "trans-stable",
    "V(x) = 6x·exp(-8x²)·(1-x²)²";
    figdir=figdir, N_ode=5000)   # finer grid for small γ ≈ 0.104

println("\n" * "="^70)
println("  DONE — stable transmission figures saved to: $figdir")
println("="^70)
