###############################################################
# regenerate_resim.jl
#
# Rebuild the heavy `<example>-prof<k>-resim-psi.jld2` files from the
# accompanying `<example>-prof<k>-dynamics-data.jld2`. These hold the full
# Ψ(x,t) snapshot grid used by `resim_and_plot_prof3.jl` and `make_videos.py`,
# and are too large (~100-200 MB each) to ship in the git repo.
#
# Usage:
#     julia --project=. paper_figures/regenerate_resim.jl <example> [prof_idx]
#
# Examples:
#     julia --project=. paper_figures/regenerate_resim.jl fl-sin-pi-A3 3
#     julia --project=. paper_figures/regenerate_resim.jl fl-cos3half-A5 3
#
# Expected runtime: ~25-40 minutes per case on a laptop (2026 hardware).
# Memory: peaks around 1-2 GB (8192-point grid, 1500 saves, complex128).
###############################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "wells.jl"))
include(joinpath(@__DIR__, "..", "src", "dynamics.jl"))

using JLD2
using Printf

const BASEDIR = joinpath(@__DIR__, "..", "data")

if length(ARGS) < 1
    println("Usage: julia regenerate_resim.jl <example> [prof_idx]")
    println("Examples: fl-sin-pi-A3, fl-cos3half-A5, fl-stable-gauss")
    exit(1)
end

const DESC     = ARGS[1]
const PROF_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

dyn_path   = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-dynamics-data.jld2")
resim_path = joinpath(BASEDIR, DESC, "$DESC-prof$PROF_IDX-resim-psi.jld2")

isfile(dyn_path) || error("dynamics data not found: $dyn_path")

println("• Loading $dyn_path")
dyn = load(dyn_path)
ψ_init = dyn["ψ_init"]   :: Vector{ComplexF64}
x_grid = dyn["x_grid"]   :: Vector{Float64}
γ      = dyn["γ"]        :: Float64
E_bif  = dyn["E_bif"]    :: Float64
Vx     = dyn["Vx_real"]  :: Vector{Float64}
Tmax   = dyn["Tmax"]     :: Float64
dt     = dyn["dt"]       :: Float64
cap_strength = dyn["cap_strength"] :: Float64
cap_width    = dyn["cap_width"]    :: Float64

# Re-run the long-time evolution at higher temporal resolution.
# `splitstep_evolve` lives in src/dynamics.jl.
println("• Running split-step evolve to T=$(Tmax) ...")
t_saves, ψ_saves = splitstep_evolve(ψ_init, x_grid, Vx, dt, round(Int, Tmax/dt);
                                    n_saves = 1500,
                                    cap_strength = cap_strength,
                                    cap_width    = cap_width)

println("• Saving $resim_path")
jldsave(resim_path;
        t_saves     = t_saves,
        x_grid      = x_grid,
        psi_saves   = ψ_saves,
        γ           = γ,
        E_bif       = E_bif,
        Vx_real     = Vx)
println("• Done. Size = $(round(filesize(resim_path)/1e6; digits=1)) MB.")
