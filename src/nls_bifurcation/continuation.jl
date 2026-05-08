#####################
# continuation.jl   #
#####################

"""
Seed finding and BifurcationKit continuation for NLS homoclinics.

Workflow:
1. find_all_seeds(a, b, Vfun; E_list=...) — scan for seeds
2. Inspect / filter / deduplicate seeds
3. continue_from_seeds(seeds, a, b, Vfun; ...) — continue branches
"""

using Printf
using Accessors: @optic
using BifurcationKit: AutoSwitch, PALC, Natural, BifurcationProblem, ContinuationPar, NewtonPar, continuation
using OrdinaryDiffEq
using LinearAlgebra

# ============================================================================
# SEED FINDING
# ============================================================================

"""
    find_seeds_at_E(a, b, Vfun; E0=-1.0, N=1000, ζmax=6.0, nscan=3400,
                    tolH=1e-10, slope_set=(+1,-1))

Scan ζ ∈ (0, ζmax] for zeros of the Hamiltonian residual at x=b,
for fixed E = E0. Returns a vector of seed tuples

    (; ζ, c, slope_sign, p=(E=E0,))

suitable for continuation.
"""
function find_seeds_at_E(a, b, Vfun;
                         E0=-1.0, N=1000, ζmax=6.0, nscan=3400,
                         tolH=1e-10, slope_set=(+1, -1))
    seeds = []
    ζgrid = range(1e-5, ζmax; length=nscan)

    _golden_section_min_abs(f, lo, hi; maxit=40) = begin
        ϕ = (sqrt(5) - 1) / 2
        a0, b0 = lo, hi
        c0 = b0 - ϕ * (b0 - a0)
        d0 = a0 + ϕ * (b0 - a0)
        fc = abs(f(c0))
        fd = abs(f(d0))
        for _ in 1:maxit
            if fc < fd
                b0 = d0
                d0 = c0
                fd = fc
                c0 = b0 - ϕ * (b0 - a0)
                fc = abs(f(c0))
            else
                a0 = c0
                c0 = d0
                fc = fd
                d0 = a0 + ϕ * (b0 - a0)
                fd = abs(f(d0))
            end
        end
        return fc < fd ? c0 : d0
    end

    _newton_refine_ζ(f, ζ0, ζlo, ζhi; maxit=20, tol=tolH) = begin
        ζ = clamp(ζ0, ζlo, ζhi)
        for _ in 1:maxit
            fζ = f(ζ)
            isfinite(fζ) || break
            abs(fζ) <= tol && return ζ, fζ, true
            δζ = max(1e-8, 1e-5 * max(abs(ζ), 1.0))
            f_plus = f(min(ζ + δζ, ζhi))
            f_minus = f(max(ζ - δζ, ζlo))
            (isfinite(f_plus) && isfinite(f_minus)) || break
            df = (f_plus - f_minus) / (min(ζ + δζ, ζhi) - max(ζ - δζ, ζlo))
            abs(df) <= 1e-14 && break
            ζ_new = clamp(ζ - fζ / df, ζlo, ζhi)
            abs(ζ_new - ζ) <= tol * max(abs(ζ), 1.0) && return ζ_new, f(ζ_new), true
            ζ = ζ_new
        end
        fζ = f(ζ)
        return ζ, fζ, isfinite(fζ) && abs(fζ) <= 100tol
    end

    for slope_sign in slope_set
        for right_sign in (+1, -1)
            # Evaluate direct right-tail matching residual on a coarse grid
            R = [tail_residual_ζ(a, b, E0, Vfun;
                                 ζ=ζ, slope_sign=slope_sign, right_sign=right_sign, N=N) for ζ in ζgrid]
            absR = abs.(R)

            # Look for sign changes or near-zeros
            for i in 1:(length(ζgrid) - 1)
                ζ1, ζ2 = ζgrid[i], ζgrid[i+1]
                r1, r2 = R[i], R[i+1]

                if !isfinite(r1) || !isfinite(r2)
                    continue
                end

                take = false
                ζstar = NaN

                if abs(r1) ≤ tolH
                    ζstar = ζ1
                    take = true
                elseif abs(r2) ≤ tolH
                    ζstar = ζ2
                    take = true
                elseif r1 * r2 < 0
                    # refine by bisection
                    fζ = ζ -> tail_residual_ζ(a, b, E0, Vfun;
                                              ζ=ζ, slope_sign=slope_sign, right_sign=right_sign, N=N)
                    ζstar = bisect_zero(fζ, ζ1, ζ2; tol=tolH)
                    take = true
                end

                if take
                    cstar = c_from_ζ(ζstar, E0)
                    if all(!(abs(cstar - c_from_ζ(s.ζ, E0)) ≤ 1e-6 &&
                             s.slope_sign == slope_sign &&
                             get(s, :right_sign, right_sign) == right_sign) for s in seeds)
                        push!(seeds, (; ζ=ζstar,
                                       c=cstar,
                                       slope_sign=slope_sign,
                                       right_sign=right_sign,
                                       p=(E=E0,)))
                    end
                end
            end

            # Also catch shallow / tangent zeros by looking for local minima
            # of |residual| that do not produce a clean sign change.
            local_tol = max(1e-6, 1000tolH)
            for i in 2:(length(ζgrid) - 1)
                isfinite(absR[i]) || continue
                absR[i] <= absR[i-1] || continue
                absR[i] <= absR[i+1] || continue
                absR[i] <= local_tol || continue

                fζ = ζ -> tail_residual_ζ(a, b, E0, Vfun;
                                          ζ=ζ, slope_sign=slope_sign, right_sign=right_sign, N=N)
                ζcand = _golden_section_min_abs(fζ, ζgrid[i-1], ζgrid[i+1])
                ζref, rref, ok = _newton_refine_ζ(fζ, ζcand, ζgrid[i-1], ζgrid[i+1])
                ok || continue
                abs(rref) <= local_tol || continue

                cstar = c_from_ζ(ζref, E0)
                if all(!(abs(cstar - c_from_ζ(s.ζ, E0)) ≤ 1e-5 &&
                         s.slope_sign == slope_sign &&
                         get(s, :right_sign, right_sign) == right_sign) for s in seeds)
                    push!(seeds, (; ζ=ζref,
                                   c=cstar,
                                   slope_sign=slope_sign,
                                   right_sign=right_sign,
                                   p=(E=E0,)))
                end
            end
        end
    end

    return seeds
end

"""
    find_all_seeds(a, b, Vfun;
                   E_list::Vector{Float64},
                   N=1000, ζmax=20.0, nscan=3400,
                   tolH=1e-6, slope_set=(+1,-1))

Find all seeds across multiple energy values.

Returns a vector of named tuples, each containing:
- `ζ`: the ζ parameter value
- `c`: amplitude c = √(-2E) tanh(ζ)
- `slope_sign`: ±1
- `p`: parameter tuple (E=E,) for BifurcationKit
"""
function find_all_seeds(a, b, Vfun;
                        E_list::Vector{Float64},
                        N=1000,
                        ζmax=20.0,
                        nscan=3400,
                        tolH=1e-6,
                        slope_set=(+1, -1))

    all_seeds = []

    println("\n" * "="^70)
    println("SEED FINDING")
    println("="^70)

    for E0 in E_list
        println("\nSearching at E = $E0...")

        seeds_at_E = find_seeds_at_E(a, b, Vfun;
                                     E0=E0,
                                     N=N,
                                     ζmax=ζmax,
                                     nscan=nscan,
                                     tolH=tolH,
                                     slope_set=slope_set)

        println("  Found $(length(seeds_at_E)) seed(s)")
        for seed in seeds_at_E
            println("    ζ=$(round(seed.ζ, digits=4)), " *
                   "c=$(round(seed.c, digits=5)), " *
                   "slope=$(seed.slope_sign), " *
                   "right=$(get(seed, :right_sign, +1))")
        end

        append!(all_seeds, seeds_at_E)
    end

    println("\n" * "="^70)
    println("TOTAL: $(length(all_seeds)) seeds found")
    println("="^70)

    return all_seeds
end


"""
    print_seed_table(seeds; sort_by=:E)

Print a nicely formatted table of all seeds.

`sort_by` can be `:E`, `:c`, or `:ζ`
"""
function print_seed_table(seeds; sort_by=:E)
    if isempty(seeds)
        println("No seeds to display")
        return
    end

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    # Sort seeds
    sorted = if sort_by == :E
        sort(seeds, by=s->get_E(s))
    elseif sort_by == :c
        sort(seeds, by=s->s.c)
    elseif sort_by == :ζ
        sort(seeds, by=s->s.ζ)
    else
        seeds
    end

    println("\n" * "="^70)
    println("SEED TABLE (sorted by $sort_by)")
    println("="^70)
    println("Index | Energy (E) |  Amplitude (c) |    ζ      | Slope | Right")
    println("-"^70)

    for (i, seed) in enumerate(sorted)
        E_val = get_E(seed)
        slope_str = seed.slope_sign > 0 ? "+1" : "-1"
        right_str = get(seed, :right_sign, +1) > 0 ? "+1" : "-1"
        @printf("%5d | %10.5f | %14.6e | %9.5f | %5s | %5s\n",
                i, E_val, seed.c, seed.ζ, slope_str, right_str)
    end

    println("="^70)
end


"""
    filter_seeds(seeds;
                 E_min=-Inf, E_max=Inf,
                 c_min=0.0, c_max=Inf,
                 ζ_min=0.0, ζ_max=Inf,
                 slope_signs=nothing)

Filter seeds by various criteria.
"""
function filter_seeds(seeds;
                      E_min=-Inf, E_max=Inf,
                      c_min=0.0, c_max=Inf,
                      ζ_min=0.0, ζ_max=Inf,
                      slope_signs=nothing)

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    filtered = filter(seeds) do s
        # Energy filter
        E_val = get_E(s)
        (E_min ≤ E_val ≤ E_max) || return false

        # Amplitude filter
        (c_min ≤ s.c ≤ c_max) || return false

        # ζ filter
        (ζ_min ≤ s.ζ ≤ ζ_max) || return false

        # Slope filter
        if slope_signs !== nothing
            (s.slope_sign in slope_signs) || return false
        end

        return true
    end

    println("Filtered: $(length(seeds)) → $(length(filtered)) seeds")
    return filtered
end


"""
    deduplicate_seeds(seeds; E_tol=1e-3, c_tol=1e-4)

Remove duplicate seeds that are very close in (E, c) space.
"""
function deduplicate_seeds(seeds; E_tol=1e-3, c_tol=1e-4)
    isempty(seeds) && return seeds

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    unique_seeds = [seeds[1]]

    for seed in seeds[2:end]
        is_duplicate = false
        E_seed = get_E(seed)

        for existing in unique_seeds
            E_existing = get_E(existing)

            if abs(E_seed - E_existing) < E_tol &&
               abs(seed.c - existing.c) < c_tol &&
               seed.slope_sign == existing.slope_sign
                is_duplicate = true
                break
            end
        end

        if !is_duplicate
            push!(unique_seeds, seed)
        end
    end

    if length(unique_seeds) < length(seeds)
        println("Removed $(length(seeds) - length(unique_seeds)) duplicate(s)")
    end

    return unique_seeds
end


# ============================================================================
# BRANCH CONTINUATION
# ============================================================================

"""
    continue_from_seeds(seeds, a, b, Vfun;
                       N=7000, p_min=-10.0, p_max=-1e-12,
                       ds=0.01, dsmin=1e-4, dsmax=0.01,
                       max_steps=500, tol=1e-5, ζ_min=1e-3,
                       verbose=1)

Continue branches from a collection of seeds.

Returns vector of branch objects (from BifurcationKit).
"""
function continue_from_seeds(seeds, a, b, Vfun;
                            N=7000,
                            p_min=-10.0,
                            p_max=-1e-12,
                            ds=0.01,
                            dsmin=1e-4,
                            dsmax=0.01,
                            max_steps=500,
                            tol=1e-5,
                            ζ_min=1e-3,
                            verbose=1)

    branches = []

    println("\n" * "="^70)
    println("BRANCH CONTINUATION")
    println("="^70)
    println("Continuing $(length(seeds)) seed(s)...")

    for (idx, sol0) in enumerate(seeds)
        # Extract E value from p field
        E_val = try sol0.p.E catch; sol0.p end

        verbose > 0 && println("\n--- Seed $idx / $(length(seeds)) ---")
        verbose > 0 && println("E=$(E_val), c=$(sol0.c), slope=$(sol0.slope_sign), right=$(get(sol0, :right_sign, +1))")

        # Continuation parameter lens
        lensE = @optic _.E

        # Residual function
        F(ζ, p) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return @inbounds [tail_residual_ζ(a, b, E_val, Vfun;
                                              ζ=ζ[1],
                                              slope_sign=sol0.slope_sign,
                                              right_sign=get(sol0, :right_sign, +1),
                                              N=N)]
        end

        # Record function
        rec(ζ, p; k...) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return (; ζ=ζ[1],
                     c=c_from_ζ(ζ[1], E_val),
                     slope_sign=sol0.slope_sign,
                     right_sign=get(sol0, :right_sign, +1))
        end

        # Set up problem
        prob = BifurcationProblem(F, [sol0.ζ], sol0.p, lensE;
                                 record_from_solution=rec)

        # Continuation options
        opts = ContinuationPar(
            ds=ds,
            dsmin=dsmin,
            dsmax=dsmax,
            p_min=p_min,
            p_max=p_max,
            max_steps=max_steps,
            newton_options=NewtonPar(
                tol=tol,
                max_iterations=30,
            ),
            nev=1,
            detect_bifurcation=0,
        )

        # Stopping criterion
        last_br = Ref{Any}(nothing)

        finalise_solution = (z, tau, step, contResult; k...) -> begin
            last_br[] = contResult

            # Detect solver failures
            if any(isnan, z.u) || any(isinf, z.u)
                @warn "Stopping one side early: encountered NaN/Inf in solution."
                return false
            end

            ζ_val = z.u[1]

            # Allow first few steps
            if step ≤ 1
                return true
            end

            # Stop if |ζ| too small
            if abs(ζ_val) ≤ ζ_min
                @info "Reached ζ_min threshold — stopping this side only."
                return false
            end

            return true
        end


        # Run continuation with error handling
        br = try
            continuation(prob, PALC(), opts;
                        bothside=true,
                        verbosity=verbose,
                        finalise_solution=finalise_solution)
        catch e
            @warn "Continuation failed for seed $idx: $e"
            if last_br[] !== nothing
                verbose > 0 && println("Using partial branch")
                last_br[]
            else
                verbose > 0 && println("No partial branch available")
                nothing
            end
        end

        if br !== nothing
            n_pts = length(br.branch)
            E_range = if n_pts > 0
                Es = [sol.param for sol in br.branch]
                (minimum(Es), maximum(Es))
            else
                (NaN, NaN)
            end
            verbose > 0 && println("Branch completed: $n_pts points, E in $E_range")
        else
            verbose > 0 && println("Branch failed")
        end

        push!(branches, br === nothing ? (; branch=[]) : br)
    end

    # Summary
    n_success = sum(!isempty(br.branch) for br in branches)
    println("\n" * "="^70)
    println("CONTINUATION COMPLETE")
    println("Successful: $n_success / $(length(seeds))")
    println("="^70)

    return branches
end


"""
    continue_single_seed(seed, a, b, Vfun; kwargs...)

Continue a single seed (convenience wrapper).
"""
function continue_single_seed(seed, a, b, Vfun; kwargs...)
    branches = continue_from_seeds([seed], a, b, Vfun; kwargs...)
    return branches[1]
end


# ============================================================================
# CONVENIENCE: Combined workflow
# ============================================================================

"""
    find_branches_at_fixed_E(a, b, Vfun; E0=-1.0, ...)

Combines seed finding + continuation at one E.
"""
function find_branches_at_fixed_E(a, b, Vfun;
                                 E0=-1.0,
                                 N=1000,
                                 tolH=1e-10,
                                 ζmax=3.0,
                                 nscan=3400,
                                 p_min=-10.0,
                                 kwargs...)

    # Find seeds at this E
    seeds = find_all_seeds(a, b, Vfun;
                          E_list=[E0],
                          N=N,
                          ζmax=ζmax,
                          nscan=nscan,
                          tolH=tolH)

    # Continue them
    branches = continue_from_seeds(seeds, a, b, Vfun;
                                  N=N,
                                  p_min=p_min,
                                  kwargs...)

    return branches, seeds
end


"""
    find_branches_over_Es(a, b, Vfun; E0_list::Vector{Float64}, ...)

Combines seed finding + continuation over multiple E.
"""
function find_branches_over_Es(a, b, Vfun;
                              E0_list::Vector{Float64},
                              N=1000,
                              tolH=1e-8,
                              ζmax=8.0,
                              nscan=3400,
                              p_min=-10.0,
                              kwargs...)

    # Find all seeds
    seeds = find_all_seeds(a, b, Vfun;
                          E_list=E0_list,
                          N=N,
                          ζmax=ζmax,
                          nscan=nscan,
                          tolH=tolH)

    # Continue them all
    branches = continue_from_seeds(seeds, a, b, Vfun;
                                  N=N,
                                  p_min=p_min,
                                  kwargs...)

    return branches
end
