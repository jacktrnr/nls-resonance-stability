"""
    NLSResonanceStability

Companion code for *Sharp stability conditions of resonance-induced nonlinear
bound states of NLS/GP* (Turner & Weinstein, 2026).

This module groups the reusable numerical building blocks used in the paper:

  * `wells`         — compactly supported potential definitions
  * `resonances`    — quadratic-eigenvalue solver for SR/TR resonances on `[a,b]`
  * `dynamics`      — split-step Schrödinger evolver for the cubic NLS
  * `lplus`         — Pöschl–Teller Jost solutions and Lₖ Evans matching
  * `jl_evans`      — JL Evans function for the linearization JL = J·L
  * `plotting`      — shared helpers for (E,N) trajectory diagnostics
  * `continuation`  — `bifurcate_branch` end-to-end pipeline:
                      potential → resonance → BifurcationKit branch

A typical workflow is

```julia
using NLSResonanceStability

V(x) = 3 * sin(π * x) * (abs(x) ≤ 1)        # any L¹ potential supported in [-b, b]
γ⋆, U⋆ = compute_resonance(V; a=-1, b=1, kind=:transmission)
branch = bifurcate_branch(V, γ⋆, U⋆;        # full (ε, E, ψ_ε) family
                          ε_max = 0.05,
                          kind  = :transmission)
ψ_t   = splitstep_evolve(branch.ψ[end] .+ δ, V, t_grid)   # any custom Ψ₀
```

See `examples/` for runnable scripts.
"""
module NLSResonanceStability

using LinearAlgebra
using SparseArrays
using FFTW
using JLD2
using Printf
using LaTeXStrings
using OrdinaryDiffEq
using BifurcationKit

# Plots is heavy and only needed for figure-producing helpers; load lazily.
using Plots

include("wells.jl")
include("resonances.jl")
include("dynamics.jl")
include("lplus.jl")
include("jl_evans.jl")
include("plotting.jl")
include("continuation.jl")

# Re-export the most commonly used names. Internal/auxiliary functions stay
# unexported but accessible as `NLSResonanceStability.<name>`.
export
    # potentials
    square_well,
    cosine_arch,
    sine_arch,
    antisymmetric_sine,
    sech2_arch,
    smooth_tapered_well,
    smooth_double_arch,
    smooth_step,
    boundary_dipole,
    interior_dipole,
    # resonance finding
    compute_resonance,
    compute_resonances_fullline,
    # dynamics
    splitstep_evolve,
    compute_mass,
    compute_energy_rayleigh,
    compute_soliton_norm,
    # spectral diagnostics
    build_Lplus_evans_full,
    compute_Lplus_spectrum_evans_full,
    # continuation
    bifurcate_branch,
    Branch,
    # plotting helpers
    split_trajectory_plot

end # module
