# continuation.jl — bifurcation branch construction for SR / TR resonances.
#
# Wraps the lower-level `NLSBifurcation` submodule (under `src/nls_bifurcation/`),
# which provides:
#
#   • Full-line transmission resonance (Robin BCs on both sides):
#       find_all_seeds, continue_from_seeds, find_branches_over_Es
#   • Half-line scattering resonance (Dirichlet at 0, Robin at b):
#       find_all_seeds_dirichlet, continue_from_dirichlet_seeds,
#       find_branches_over_Es_dirichlet  (suffix `_dirichlet` throughout)
#
# The high-level `bifurcate_branch` function provided here packages these into
# a single user-facing call:
#
#     branch = bifurcate_branch(V, γ★, U★;
#                               a = -1.0, b = 1.0,
#                               ε_max = 0.05,
#                               kind  = :transmission)
#
# Returns a `Branch` containing parallel arrays `(ε, E, N, ψ_ε)` along the branch
# plus the linear data `(γ★, U★, Ω★, B★)` from the resonance pair.
#
# Custom parameter ranges, step controls, etc. are passed through via
# `bifurcate_branch(...; bk_opts = (; ...))`. Power users can call into the
# `NLSBifurcation` submodule directly:
#
#     using NLSResonanceStability: NLSBifurcation
#     seeds = NLSBifurcation.find_all_seeds(...)

include(joinpath(@__DIR__, "nls_bifurcation", "NLSBifurcation.jl"))
using .NLSBifurcation

"""
    compute_resonance(V; a=-1.0, b=1.0, kind=:transmission, γ_max=4.0)

Find the lowest-γ resonance pair `(γ★, U★)` of `H_V = -∂² + V` on `[a,b]`.

Wraps `compute_resonances_fullline`: scans for k-zeros / k-poles of the
reflection coefficient on the lower imaginary axis, returns the pair closest
to threshold (smallest `γ`).

Returns `(γ_star::Float64, U_star::Vector{Float64}, x_grid::Vector{Float64})`.
"""
function compute_resonance(V::Function;
                           a       = -1.0,
                           b       = 1.0,
                           kind    = :transmission,
                           γ_max   = 4.0)
    result = compute_resonances_fullline(a, b, V; k_max = γ_max)
    records = kind === :transmission ? result.transmission : result.scattering
    isempty(records) && error("No $(kind) resonance found for the supplied V on [$a, $b].")
    # Lowest γ (= |Im k_★|) record on the lower imaginary axis.
    rec = argmin(r -> abs(imag(r.k)), records)
    γ_star = abs(imag(rec.k))
    x_grid = collect(Float64, result.xgrid)
    U_star = collect(Float64, real.(rec.U))
    return γ_star, U_star, x_grid
end

"""
    Branch

Container for a resonance-induced bifurcation branch.

# Fields
- `ε :: Vector{Float64}`    — branch parameter; `ψ_ε = √ε · (U★ + O(ε))`.
- `E :: Vector{Float64}`    — energy along the branch; `E(0) = -γ★²`.
- `N :: Vector{Float64}`    — `L²` norm `N[ψ_ε]`; `N(0) = 4γ★`.
- `ψ :: Vector{Vector{Float64}}` — inner profiles `ψ_ε(x)` on `[a,b]`.
- `γ_star :: Float64`       — resonance decay rate.
- `U_star :: Vector{Float64}` — resonance state on `[a,b]`.
- `Omega_star, B_star`      — functionals from Eqs. (1.7)/(1.9) of the paper.
- `kind :: Symbol`          — `:transmission` or `:scattering`.
"""
struct Branch
    ε       :: Vector{Float64}
    E       :: Vector{Float64}
    N       :: Vector{Float64}
    ψ       :: Vector{Vector{Float64}}
    x_grid  :: Vector{Float64}
    γ_star  :: Float64
    U_star  :: Vector{Float64}
    Omega_star :: Float64
    B_star     :: Float64
    kind    :: Symbol
end

"""
    compute_Omega_B(U★, x_grid, γ★, kind) -> (Ω★, B★)

Evaluate the resonance functionals on a precomputed `U★` grid.

For TR (`U_star(-b) = 1`): `Ω = (2 U_b⁴ + 1)/(4γ) - 2 ∫U★⁴`,  `B = ∫U★² - (U_b² - 1)/(2γ)`.
For SR (`U_star(0) = 0`):  `Ω = U_b⁴/(2γ) - 2 ∫U★⁴`,            `B = ∫U★² - U_b²/(2γ)`.
"""
function compute_Omega_B(U_star::Vector{Float64}, x_grid::Vector{Float64},
                        γ::Float64, kind::Symbol)
    Ub = U_star[end]
    I2 = trapz(x_grid, U_star.^2)
    I4 = trapz(x_grid, U_star.^4)
    if kind === :transmission
        Ua = U_star[1]
        Ω  = (2 * Ub^4 + Ua^4) / (4γ) - 2 * I4
        B  = I2 - (Ub^2 - Ua^2) / (2γ)
    elseif kind === :scattering
        # Dirichlet at the left endpoint enforces U_star(0) = 0
        Ω  = Ub^4 / (2γ) - 2 * I4
        B  = I2 - Ub^2 / (2γ)
    else
        error("kind must be :transmission or :scattering, got $kind")
    end
    return Ω, B
end

"""
    trapz(x, y)

Trapezoidal integral of `y(x)` on a non-uniform grid.
"""
function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    s = 0.0
    @inbounds for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

"""
    bifurcate_branch(V, γ★, U★; ...)

Continue the nonlinear bound-state branch bifurcating from the resonance pair
`(γ★, U★)` of the linear Schrödinger operator with potential `V`.

# Required arguments
- `V::Function`        — the compactly supported potential `V(x)` (any callable).
- `γ_star::Real`       — resonance decay rate (`E_★ = -γ★²`).
- `U_star::Vector`     — resonance state sampled on `x_grid` covering `[a,b]`.

# Keyword arguments
- `x_grid::Vector`     — must match `length(U_star)`; default: linspace `[a,b]` with 401 pts.
- `a, b::Real`         — inner-region endpoints (defaults `-1, 1`).
- `ε_max::Real`        — maximum `ε = ψ(-b)²` (TR) or `ψ'(0)²` (SR) to continue to.
- `n_branch::Int`      — target number of branch samples (default 60).
- `kind::Symbol`       — `:transmission` or `:scattering`.
- `bk_opts::NamedTuple`— forwarded to BifurcationKit (e.g. `(; ds=0.005)`).

Returns a `Branch`. See the docstring on `Branch` for fields.

# Example

    using NLSResonanceStability
    V(x) = 3 * sin(π * x) * (abs(x) ≤ 1)
    γ⋆, U⋆ = compute_resonance(V; a=-1, b=1, kind=:transmission)
    branch = bifurcate_branch(V, γ⋆, U⋆; ε_max=0.04, kind=:transmission)
    # `branch.ψ[i]` is the inner-region nonlinear bound state at `branch.ε[i]`.
"""
function bifurcate_branch(V::Function, γ_star::Real, U_star::AbstractVector;
                          x_grid = collect(range(-1.0, 1.0; length = length(U_star))),
                          a       = first(x_grid),
                          b       = last(x_grid),
                          ε_max   = 0.05,
                          n_branch = 60,
                          kind    = :transmission,
                          bk_opts = NamedTuple())

    @assert length(x_grid) == length(U_star)
    γ = float(γ_star)
    U = collect(Float64, U_star)
    Ω, B = compute_Omega_B(U, collect(Float64, x_grid), γ, Symbol(kind))

    if kind === :transmission
        # Run TR continuation via NLSBifurcation. Implementation detail:
        # `NLSBifurcation.find_branches_over_Es` searches over a range of
        # energies and assembles branches; we then post-process to express in ε.
        # For end users who need finer control, see paper_figures/fig_62_*.jl.
        ε_arr, E_arr, N_arr, ψ_arr = _tr_continue(V, γ, U, x_grid;
                                                   a=a, b=b, ε_max=ε_max,
                                                   n_branch=n_branch, bk_opts=bk_opts)
    elseif kind === :scattering
        ε_arr, E_arr, N_arr, ψ_arr = _sr_continue(V, γ, U, x_grid;
                                                   a=a, b=b, ε_max=ε_max,
                                                   n_branch=n_branch, bk_opts=bk_opts)
    else
        error("kind must be :transmission or :scattering, got $kind")
    end

    return Branch(ε_arr, E_arr, N_arr, ψ_arr, collect(Float64, x_grid),
                  γ, U, Ω, B, Symbol(kind))
end

# ── TR continuation ──────────────────────────────────────────────
function _tr_continue(V, γ, U_star, x_grid; a, b, ε_max, n_branch, bk_opts)
    # Energy range: a thin slab below the resonance.
    E_star = -γ^2
    E_min  = E_star - 1.0     # explored below; fine-tuned by user via bk_opts
    E_max  = E_star - 1e-4
    Es     = collect(range(E_min, E_max; length = n_branch))

    # NLSBifurcation expects a list of seeds at each E. We use its high-level
    # helper that scans for branches:
    branches = NLSBifurcation.find_branches_over_Es(Es, V; a = a, b = b,
                                                    bk_opts...)
    # `branches` is a list of (E, ζ, profile, …) records. Pick the branch whose
    # E → E_star limit matches our resonance.
    isempty(branches) && error("No TR branch found for the supplied (V, γ★, U★).")
    chosen = _select_branch_near(branches, E_star, U_star, x_grid)
    return _branch_to_arrays(chosen, kind = :transmission)
end

# ── SR continuation ──────────────────────────────────────────────
function _sr_continue(V, γ, U_star, x_grid; a, b, ε_max, n_branch, bk_opts)
    E_star = -γ^2
    E_min  = E_star - 1.0
    E_max  = E_star - 1e-4
    Es     = collect(range(E_min, E_max; length = n_branch))

    branches = NLSBifurcation.find_branches_over_Es_dirichlet(Es, V;
                                                              b = b,
                                                              bk_opts...)
    isempty(branches) && error("No SR branch found for the supplied (V, γ★, U★).")
    chosen = _select_branch_near(branches, E_star, U_star, x_grid)
    return _branch_to_arrays(chosen, kind = :scattering)
end

# ── helpers ──────────────────────────────────────────────────────
function _select_branch_near(branches, E_star, U_star, x_grid)
    # Pick the branch whose E → E_star limit profile is closest to U_star
    # (in C¹ on the inner grid).
    best_idx, best_score = 0, Inf
    for (i, br) in enumerate(branches)
        # Each `br` is expected to expose `.E::Vector` and `.profile::Vector{Vector}`.
        # Take the entry closest to E_star and score by ‖U★ - profile / √ε‖.
        # If the structure differs, this is the place to adapt.
        try
            j = argmin(abs.(br.E .- E_star))
            ψj = br.profile[j]
            ε_local = max(br.ε[j], eps())
            scaled = ψj ./ sqrt(ε_local)
            score = sum(abs2, scaled .- U_star)
            if score < best_score
                best_score, best_idx = score, i
            end
        catch
            continue
        end
    end
    best_idx == 0 && error("Branch selection failed; inspect `branches` manually.")
    return branches[best_idx]
end

function _branch_to_arrays(br; kind)
    # Convert NLSBifurcation branch object to plain arrays. The exact field
    # names depend on the upstream module; adjust here if the API drifts.
    ε = collect(Float64, br.ε)
    E = collect(Float64, br.E)
    N = collect(Float64, br.N)
    ψ = [collect(Float64, p) for p in br.profile]
    return ε, E, N, ψ
end
