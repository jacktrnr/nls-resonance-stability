module NLSBifurcation

using OrdinaryDiffEq
using OrdinaryDiffEq.SciMLBase: ReturnCode
using LinearAlgebra
using Printf
using Accessors
using BifurcationKit

# Core utilities
include("globals.jl")

# ODE integration and Hamiltonian residual
include("shooting.jl")

# Tail matching and solution gluing
include("glue.jl")

# Seed finding and BifurcationKit continuation
include("continuation.jl")
include("dirichlet_mode.jl")

# ─── Exports ───────────────────────────────────
export κ, c_from_ζ, q_from_ζ, clamp1, safe_div, sign_real, bisect_zero

export integrate_support, H_residual_ζ

export tail_shifts_from_ends, glue_full_solution,
       compute_norm, compute_H1_norm, can_glue

export find_seeds_at_E, find_all_seeds, print_seed_table,
       filter_seeds, deduplicate_seeds,
       continue_from_seeds, continue_single_seed,
       find_branches_at_fixed_E, find_branches_over_Es

export integrate_support_dirichlet, F_residual_dirichlet,
       find_seeds_at_E_dirichlet, find_all_seeds_dirichlet,
       deduplicate_dirichlet_seeds,
       continue_from_dirichlet_seeds, continue_single_seed_dirichlet,
       glue_dirichlet_solution, compute_norm_dirichlet

end # module
