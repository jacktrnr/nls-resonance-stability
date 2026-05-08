###############################################################
# 03_reproduce_fig62.jl
#
# End-to-end reproduction of Figure 6.2 of the paper (sin transmission
# resonance): the three panels showing U★(x), the (E,N) bifurcation diagram,
# and a few profiles ψ_ε(x) along the branch.
#
# This is a thin driver that runs the relevant section of
# `paper_figures/generate_paper_figures.jl`.
#
# Usage:
#     julia --project=. examples/03_reproduce_fig62.jl
#
# Outputs (written to examples/output/):
#   transmission-Ustar.pdf
#   transmission-NvsE.pdf
#   transmission-profiles.pdf
#
# Expected runtime: ~3-5 minutes (resonance scan + branch continuation).
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Override the output directory used by `generate_paper_figures.jl`.
ENV["NLS_FIGURES_DIR"] = joinpath(@__DIR__, "output")

# Hand off to the canonical figure script. It produces all three panels of
# Fig 6.2 (and others; see paper_figures/README.md for the full list).
include(joinpath(@__DIR__, "..", "paper_figures", "generate_paper_figures.jl"))
