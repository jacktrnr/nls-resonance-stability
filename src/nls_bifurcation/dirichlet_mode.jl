########################
# dirichlet_mode.jl    #
########################

using Accessors: @optic
using BifurcationKit: BifurcationProblem, ContinuationPar, NewtonPar, PALC, continuation
using OrdinaryDiffEq

"""
    integrate_support_dirichlet(a, b, E, Vfun; N=1000, β=1e-3)

Integrate the stationary NLS/GP ODE on [a,b] with Dirichlet data

    u(a) = 0,   u'(a) = β.
"""
function integrate_support_dirichlet(a, b, E, Vfun; N=1000, β=1e-3)
    u0 = [0.0, abs(β)]

    function f!(du, u, p, x)
        du[1] = u[2]
        du[2] = (Vfun(x) - E) * u[1] - u[1]^3
    end

    prob = ODEProblem(f!, u0, (a, b))
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12,
                saveat=range(a, b; length=N+1))

    sol.retcode == ReturnCode.Success || return Float64[], Float64[], Float64[]
    U = reduce(hcat, sol.u)
    return sol.t, U[1, :], U[2, :]
end

"""
    F_residual_dirichlet(a, b, E, Vfun; β, N=1000)

Hamiltonian residual at x=b for the Dirichlet-at-a problem.
"""
function F_residual_dirichlet(a, b, E, Vfun; β, N=1000)
    x, u, v = integrate_support_dirichlet(a, b, E, Vfun; N=N, β=β)
    isempty(x) && return NaN

    ub, vb = u[end], v[end]
    H = 0.5 * vb^2 + 0.5 * E * ub^2 + 0.25 * ub^4

    A = sqrt(-2E)
    if abs(ub) > A + 1e-8
        return H / max(abs(ub), 1e-8) + 100.0 * (abs(ub) - A)^2
    end

    return H / max(abs(ub), 1e-8)
end

function bisect_F_dirichlet(a, b, E, Vfun, β_lo, β_hi; N=1000, tol=1e-10, maxiter=80)
    f_lo = F_residual_dirichlet(a, b, E, Vfun; β=β_lo, N=N)
    f_hi = F_residual_dirichlet(a, b, E, Vfun; β=β_hi, N=N)
    (!isfinite(f_lo) || !isfinite(f_hi) || f_lo * f_hi > 0) && return NaN

    lo, hi = β_lo, β_hi
    for _ in 1:maxiter
        mid = 0.5 * (lo + hi)
        f_mid = F_residual_dirichlet(a, b, E, Vfun; β=mid, N=N)
        if !isfinite(f_mid)
            return NaN
        end
        if abs(f_mid) <= tol || abs(hi - lo) <= tol
            return mid
        end
        if f_lo * f_mid < 0
            hi = mid
            f_hi = f_mid
        else
            lo = mid
            f_lo = f_mid
        end
    end
    return 0.5 * (lo + hi)
end

function find_seeds_at_E_dirichlet(a, b, Vfun;
                                   E0=-1.0, N=1000, βmax=10.0, nscan=3400,
                                   tolH=1e-10)
    seeds = []
    βgrid = range(1e-5, βmax; length=nscan)
    Fvals = [F_residual_dirichlet(a, b, E0, Vfun; β=β, N=N) for β in βgrid]

    for i in 1:(length(βgrid) - 1)
        β1, β2 = βgrid[i], βgrid[i + 1]
        f1, f2 = Fvals[i], Fvals[i + 1]
        (!isfinite(f1) || !isfinite(f2)) && continue

        take = false
        βstar = NaN
        if abs(f1) <= tolH
            βstar = β1
            take = true
        elseif abs(f2) <= tolH
            βstar = β2
            take = true
        elseif f1 * f2 < 0
            βstar = bisect_F_dirichlet(a, b, E0, Vfun, β1, β2; N=N, tol=tolH)
            take = isfinite(βstar)
        end

        if take
            is_dup = any(abs(seed.β - βstar) < 1e-5 &&
                         abs((try seed.p.E catch; seed.p end) - E0) < 1e-8 for seed in seeds)
            is_dup || push!(seeds, (; β=βstar, p=(E=E0,)))
        end
    end

    return seeds
end

function find_all_seeds_dirichlet(a, b, Vfun;
                                  E_list::Vector{Float64},
                                  N=1000, βmax=10.0, nscan=3400, tolH=1e-10)
    all_seeds = []

    println("\n" * "="^70)
    println("SEED FINDING (DIRICHLET AT a)")
    println("="^70)

    for E0 in E_list
        println("\nSearching at E = $E0...")
        seeds_at_E = find_seeds_at_E_dirichlet(a, b, Vfun;
                                               E0=E0, N=N, βmax=βmax,
                                               nscan=nscan, tolH=tolH)
        println("  Found $(length(seeds_at_E)) seed(s)")
        for seed in seeds_at_E
            println("    β=$(round(seed.β, digits=6))")
        end
        append!(all_seeds, seeds_at_E)
    end

    println("\n" * "="^70)
    println("TOTAL: $(length(all_seeds)) seeds found")
    println("="^70)
    return all_seeds
end

function deduplicate_dirichlet_seeds(seeds; E_tol=1e-3, β_tol=1e-4)
    isempty(seeds) && return seeds
    get_E(s) = try s.p.E catch; s.p end

    unique_seeds = [seeds[1]]
    for seed in seeds[2:end]
        E_seed = get_E(seed)
        is_dup = any(abs(E_seed - get_E(ex)) < E_tol && abs(seed.β - ex.β) < β_tol
                     for ex in unique_seeds)
        is_dup || push!(unique_seeds, seed)
    end

    if length(unique_seeds) < length(seeds)
        println("Removed $(length(seeds) - length(unique_seeds)) duplicate(s)")
    end
    return unique_seeds
end

function continue_from_dirichlet_seeds(seeds, a, b, Vfun;
                                       N=1000,
                                       p_min=-10.0,
                                       p_max=-1e-12,
                                       ds=0.01,
                                       dsmin=1e-4,
                                       dsmax=0.01,
                                       max_steps=500,
                                       tol=1e-8,
                                       β_min=1e-10,
                                       verbose=1)
    branches = []

    println("\n" * "="^70)
    println("BRANCH CONTINUATION (DIRICHLET AT a)")
    println("="^70)
    println("Continuing $(length(seeds)) seed(s)...")

    for (idx, seed) in enumerate(seeds)
        E_val = try seed.p.E catch; seed.p end
        verbose > 0 && println("\n--- Seed $idx / $(length(seeds)) ---")
        verbose > 0 && println("E=$(E_val), β=$(seed.β)")

        lensE = @optic _.E
        F(βv, p) = begin
            Ev = typeof(p) <: Number ? p : p.E
            [F_residual_dirichlet(a, b, Ev, Vfun; β=βv[1], N=N)]
        end
        rec(βv, p; k...) = begin
            Ev = typeof(p) <: Number ? p : p.E
            (; β=βv[1])
        end

        prob = BifurcationProblem(F, [seed.β], seed.p, lensE; record_from_solution=rec)
        opts = ContinuationPar(
            ds=ds, dsmin=dsmin, dsmax=dsmax,
            p_min=p_min, p_max=p_max,
            max_steps=max_steps,
            newton_options=NewtonPar(tol=tol, max_iterations=30),
            nev=1, detect_bifurcation=0,
        )

        last_br = Ref{Any}(nothing)
        finalise_solution = (z, tau, step, contResult; k...) -> begin
            last_br[] = contResult
            step <= 1 && return true
            return z.u[1] > β_min && isfinite(z.u[1])
        end

        br = try
            continuation(prob, PALC(), opts;
                         bothside=true, verbosity=verbose,
                         finalise_solution=finalise_solution)
        catch e
            @warn "Continuation failed for Dirichlet seed $idx: $e"
            isnothing(last_br[]) ? nothing : last_br[]
        end

        push!(branches, br === nothing ? (; branch=[]) : br)
    end

    n_success = sum(!isempty(br.branch) for br in branches)
    println("\n" * "="^70)
    println("CONTINUATION COMPLETE")
    println("Successful: $n_success / $(length(seeds))")
    println("="^70)
    return branches
end

function continue_single_seed_dirichlet(seed, a, b, Vfun; kwargs...)
    branches = continue_from_dirichlet_seeds([seed], a, b, Vfun; kwargs...)
    return branches[1]
end

function glue_dirichlet_solution(a, b, E, x, u, v; Xmax=50.0, NR=1200)
    if isempty(x) || E >= 0
        return Float64[], Float64[]
    end

    κv = sqrt(-E)
    A = sqrt(-2E)
    ub, vb = u[end], v[end]
    abs(ub) > A + 1e-10 && return Float64[], Float64[]

    s = sign_real(ub)
    abs(ub) < 1e-14 && (s = sign_real(vb))

    if abs(ub) < 1e-14
        x_shift = b + (vb > 0 ? 10.0 / κv : -10.0 / κv)
    else
        ratio = max(A / abs(ub), 1.0 + 1e-10)
        y0 = acosh(ratio)
        deriv_pos = -s * κv * A * sech(y0) * tanh(y0)
        deriv_neg = -s * κv * A * sech(-y0) * tanh(-y0)
        x_shift = abs(deriv_pos - vb) < abs(deriv_neg - vb) ? b - y0 / κv : b + y0 / κv
    end

    xR = range(b, Xmax; length=NR)
    ψR = s .* A .* sech.(κv .* (xR .- x_shift))
    xfull = vcat(x, xR[2:end])
    ψfull = vcat(u, ψR[2:end])
    return Float64.(real.(xfull)), Float64.(real.(ψfull))
end

function compute_norm_dirichlet(a, b, E, x, u, v)
    if isempty(x) || E >= 0
        return NaN
    end

    κv = κ(E)
    A = sqrt(-2E)
    ub, vb = u[end], v[end]
    abs(ub) > A + 1e-6 && return NaN

    dx = x[2] - x[1]
    N_bulk = dx * (sum(abs2, u) - 0.5 * (u[1]^2 + u[end]^2))
    arg = abs(ub) < 1e-14 ? (-1.0 + 1e-12) : clamp(-vb / (κv * ub), -1 + 1e-12, 1 - 1e-12)
    N_tail = (A^2 / κv) * (1 - arg)
    return real(N_bulk + N_tail)
end

