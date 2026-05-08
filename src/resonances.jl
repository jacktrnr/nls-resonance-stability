###############################################################
# resonances_fullline.jl
# Resonances of  H = -∂ₓ² + V(x)  on the full line ℝ
# with V compactly supported on [a, b].
#
# Two QEP problems, differing only in the left-boundary ghost:
#
#   TRANSMISSION (TR): same sign at both endpoints
#     u'(a) = +ik u(a)   u'(b) = +ik u(b)
#     → B[1,1] = +2/h,   B[M,M] = -2/h
#
#   SCATTERING (SR): opposite signs at the endpoints
#     u'(a) = -ik u(a)   u'(b) = +ik u(b)
#     → B[1,1] = -2/h,   B[M,M] = -2/h
#
# On the imaginary axis k = iσ, σ ∈ ℝ, these become:
#   TR: u'(a)=u'(b)=-σu   →  both +|k| when σ<0, both -|k| when σ>0
#   SR: u'(a)=+σu, u'(b)=-σu
#       σ>0  → bound states      u'(a)=+|k|u, u'(b)=-|k|u
#       σ<0  → scattering res.   u'(a)=-|k|u, u'(b)=+|k|u
#
# Grid: M = N+2 nodes, j = 1…M, x[j] = a + (j-1)·h, h = (b-a)/(M-1)
#
#   j=1     : x = a (left boundary)
#   j=2…M-1 : interior nodes (potential V active)
#   j=M     : x = b (right boundary)
#
# Ghost-point elimination at j=1 and j=M converts the resonance
# BCs into a QEP  (k²I + ik B − A) u = 0  (same structure as
# the half-line case, just with a larger system).
#
# A (M×M):
#   j=1  : A[1,1]=2/h², A[1,2]=-2/h²          (left ghost, V(a)=0)
#   j=2…M-1: standard tridiag + V
#   j=M  : A[M,M-1]=-2/h², A[M,M]=2/h²        (right ghost, V(b)=0)
#
# B (M×M, diagonal):
#   TR: B[1,1] = +2/h,  B[M,M] = -2/h
#   SR: B[1,1] = -2/h,  B[M,M] = -2/h
#
# Companion  C = [0, I; A, -iB]  (2M×2M)
# Eigenvalues of C are the resonances k.
#
# Classification of filtered eigenvalues:
#   TR companion, Im(k) < -im_tol                         → off-/lower-axis TR resonances
#   TR companion, |Re(k)| < axis_tol and |Im(k)| > im_tol → same-sign axis states
#   TR companion, |Im(k)| ≤ im_tol and |Re(k)| > axis_tol → real-axis TR candidates
#   SR companion, Im(k) < -im_tol                         → scattering resonances
#   SR companion, Im(k) > +im_tol and |Re(k)| < axis_tol → bound states (L² decay)
#
# Eigenfunctions normalized so U[1] = 1  (U★(a) = 1, matching Dec_2025).
###############################################################

using LinearAlgebra
using Printf
using OrdinaryDiffEq

function _shoot_full_profile(a, b, Vfun, k, left_sign, xgrid;
                             reltol=1e-11, abstol=1e-13)
    sol = solve(
        ODEProblem(
            (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) - k^2) * u[1]),
            ComplexF64[1.0 + 0.0im, im * k],
            (b, a)),
        Tsit5();
        reltol=reltol, abstol=abstol,
        saveat=reverse(xgrid))
    U = reverse(first.(sol.u))
    Up = reverse(last.(sol.u))
    return U, Up, Up[1] - left_sign * im * k * U[1]
end

function _full_shoot_residual(a, b, Vfun, k, left_sign;
                              reltol=1e-11, abstol=1e-13)
    sol = solve(
        ODEProblem(
            (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) - k^2) * u[1]),
            ComplexF64[1.0 + 0.0im, im * k],
            (b, a)),
        Tsit5();
        reltol=reltol, abstol=abstol, save_everystep=false)
    u_a, up_a = sol.u[end]
    return up_a - left_sign * im * k * u_a
end

function _jost_a_full(a, b, Vfun, k; reltol=1e-11, abstol=1e-13)
    sol = solve(
        ODEProblem(
            (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) - k^2) * u[1]),
            ComplexF64[1.0 + 0.0im, im * k],
            (b, a)),
        Tsit5();
        reltol=reltol, abstol=abstol, save_everystep=false)
    u_a, up_a = sol.u[end]
    return im * k * u_a - up_a
end

function _newton_refine_full_k(a, b, Vfun, k0, left_sign;
                               maxit=12, tol=1e-11,
                               reltol=1e-11, abstol=1e-13)
    k = ComplexF64(k0)
    ok = false
    for _ in 1:maxit
        F = _full_shoot_residual(a, b, Vfun, k, left_sign; reltol=reltol, abstol=abstol)
        if abs(F) <= tol
            ok = true
            break
        end
        δk = max(1e-8, 1e-6 * max(abs(k), 1.0))
        dF = (_full_shoot_residual(a, b, Vfun, k + δk, left_sign; reltol=reltol, abstol=abstol) -
              _full_shoot_residual(a, b, Vfun, k - δk, left_sign; reltol=reltol, abstol=abstol)) / (2δk)
        abs(dF) <= 1e-14 && break
        Δk = F / dF
        k -= Δk
        if abs(Δk) <= tol * max(abs(k), 1.0)
            ok = true
            break
        end
    end
    Fres = _full_shoot_residual(a, b, Vfun, k, left_sign; reltol=reltol, abstol=abstol)
    ok |= abs(Fres) <= 10tol
    return k, Fres, ok
end

_root_accepts(k, Fres, converged; residual_tol=1e-7) =
    isfinite(real(k)) && isfinite(imag(k)) &&
    isfinite(real(Fres)) && isfinite(imag(Fres)) &&
    abs(Fres) <= residual_tol * max(1.0, abs(k)) &&
    (converged || abs(Fres) <= 0.1 * residual_tol * max(1.0, abs(k)))

function _dedupe_resonance_records(records; tol=5e-5)
    out = typeof(records)(undef, 0)
    for r in sort(records; by = q -> (round(real(q.k), digits=8), round(imag(q.k), digits=8), abs(q.residual)))
        found = findfirst(q -> abs(q.k - r.k) <= tol, out)
        if isnothing(found)
            push!(out, r)
        else
            q = out[found]
            better = (r.converged && !q.converged) ||
                     (r.converged == q.converged && abs(r.residual) < abs(q.residual))
            better && (out[found] = r)
        end
    end
    return out
end

function _refine_full_seed_family(a, b, Vfun, k_seed, left_sign; target_half=:any)
    seeds = ComplexF64[k_seed]
    if target_half === :upper
        push!(seeds, complex(real(k_seed), abs(imag(k_seed))))
        push!(seeds, complex(0.0, abs(imag(k_seed))))
    elseif target_half === :lower
        push!(seeds, complex(real(k_seed), -abs(imag(k_seed))))
        push!(seeds, complex(0.0, -abs(imag(k_seed))))
    else
        push!(seeds, conj(k_seed))
    end
    best = nothing
    for s in seeds
        k_try, F_try, ok_try = _newton_refine_full_k(a, b, Vfun, s, left_sign)
        cand = (; k=k_try, residual=F_try, converged=ok_try, k_seed=s)
        if best === nothing
            best = cand
            continue
        end
        better = (cand.converged && !best.converged) ||
                 (cand.converged == best.converged && abs(cand.residual) < abs(best.residual))
        better && (best = cand)
    end
    return best
end

function _bound_state_scan(a, b, Vfun, xgrid, k_max;
                           κ_min=1e-6, N_scan=2400, tol=1e-11,
                           residual_tol=1e-7)
    κ_hi = max(κ_min + 1e-3, k_max)
    κ_grid = range(κ_min, κ_hi; length=N_scan)
    Fvals = Float64[]
    κvals = Float64[]

    for κ in κ_grid
        Fκ = _full_shoot_residual(a, b, Vfun, im * κ, -1)
        if isfinite(real(Fκ)) && isfinite(imag(Fκ)) && abs(imag(Fκ)) <= 1e-7 * max(1.0, abs(real(Fκ)))
            push!(κvals, κ)
            push!(Fvals, real(Fκ))
        end
    end

    roots = ComplexF64[]
    for j in 1:length(Fvals)-1
        f1 = Fvals[j]
        f2 = Fvals[j+1]
        κ1 = κvals[j]
        κ2 = κvals[j+1]
        if abs(f1) <= tol
            push!(roots, im * κ1)
            continue
        end
        signbit(f1) == signbit(f2) || begin
            lo, hi = κ1, κ2
            flo, fhi = f1, f2
            for _ in 1:70
                mid = 0.5 * (lo + hi)
                fmid = real(_full_shoot_residual(a, b, Vfun, im * mid, -1))
                if abs(fmid) <= tol || abs(hi - lo) <= tol * max(mid, 1.0)
                    lo = hi = mid
                    break
                end
                if signbit(flo) == signbit(fmid)
                    lo, flo = mid, fmid
                else
                    hi, fhi = mid, fmid
                end
            end
            push!(roots, im * (0.5 * (lo + hi)))
        end
    end

    out = NamedTuple[]
    for k0 in roots
        k, Fk, converged = _newton_refine_full_k(a, b, Vfun, k0, -1)
        _root_accepts(k, Fk, converged; residual_tol=residual_tol) || continue
        imag(k) > 0 || continue
        U, _, _ = _shoot_full_profile(a, b, Vfun, k, -1, xgrid)
        u1 = U[1]
        abs(u1) < 1e-10 * norm(U) && continue
        U ./= u1
        push!(out, (; k=k, γ=abs(imag(k)), U=U,
                     residual=Fk, converged=converged, k_seed=k0))
    end
    return _dedupe_resonance_records(out)
end

function _dedupe_seed_list(seeds; tol=5e-4)
    out = ComplexF64[]
    for k in sort(ComplexF64.(seeds); by = z -> (round(real(z), digits=6), round(imag(z), digits=6)))
        any(abs(k - q) <= tol for q in out) || push!(out, k)
    end
    return out
end

function _residual_minima_scan(a, b, Vfun, xgrid, k_max, left_sign;
                               re_max=k_max, im_lo=-k_max, im_hi=-1e-3,
                               N_re=90, N_im=70, residual_tol=1e-6,
                               max_candidates=120,
                               accept_tol=1e-7,
                               target_half=:lower)
    re_vals = collect(range(-re_max, re_max; length=N_re))
    im_vals = collect(range(im_lo, im_hi; length=N_im))
    Fmag = fill(Inf, N_im, N_re)

    for j in 1:N_im, i in 1:N_re
        k = re_vals[i] + im * im_vals[j]
        abs(k) > k_max && continue
        Fmag[j, i] = abs(_full_shoot_residual(a, b, Vfun, k, left_sign))
    end

    candidates = NamedTuple[]
    for j in 2:N_im-1, i in 2:N_re-1
        val = Fmag[j, i]
        isfinite(val) || continue
        is_local_min = true
        for jj in j-1:j+1, ii in i-1:i+1
            (jj == j && ii == i) && continue
            if Fmag[jj, ii] < val
                is_local_min = false
                break
            end
        end
        is_local_min || continue
        push!(candidates, (; k0=re_vals[i] + im * im_vals[j], val=val))
    end

    isempty(candidates) && return NamedTuple[]
    sort!(candidates; by = c -> c.val)
    cutoff = max(residual_tol, 10 * candidates[1].val)
    candidates = [c for c in candidates if c.val <= cutoff]
    length(candidates) > max_candidates && (candidates = candidates[1:max_candidates])

    out = NamedTuple[]
    for cand in candidates
        best = _refine_full_seed_family(a, b, Vfun, cand.k0, left_sign; target_half=target_half)
        k, Fk, converged = best.k, best.residual, best.converged
        _root_accepts(k, Fk, converged; residual_tol=accept_tol) || continue
        abs(k) <= k_max || continue
        if target_half === :lower
            imag(k) < 0 || continue
        elseif target_half === :upper
            imag(k) > 0 || continue
        end
        U, _, _ = _shoot_full_profile(a, b, Vfun, k, left_sign, xgrid)
        u1 = U[1]
        abs(u1) < 1e-10 * norm(U) && continue
        U ./= u1
        push!(out, (; k=k, γ=abs(imag(k)), U=U,
                     residual=Fk, converged=converged, k_seed=best.k_seed))
    end
    return _dedupe_resonance_records(out)
end

function _heatmap_jost_seed_scan(a, b, Vfun, xgrid, k_max;
                                 re_max=k_max, im_lo=-k_max, im_hi=-1e-3,
                                 N_re=220, N_im=140, max_candidates=240,
                                 loga_tol_window=1.0, accept_tol=1e-7,
                                 target_half=:any)
    re_vals = collect(range(-re_max, re_max; length=N_re))
    im_vals = collect(range(im_lo, im_hi; length=N_im))
    logA = fill(Inf, N_im, N_re)

    for j in 1:N_im, i in 1:N_re
        k = re_vals[i] + im * im_vals[j]
        abs(k) > k_max && continue
        logA[j, i] = log10(abs(_jost_a_full(a, b, Vfun, k)) + 1e-30)
    end

    candidates = NamedTuple[]
    for j in 2:N_im-1, i in 2:N_re-1
        val = logA[j, i]
        isfinite(val) || continue
        is_local_min = true
        for jj in j-1:j+1, ii in i-1:i+1
            (jj == j && ii == i) && continue
            if logA[jj, ii] < val
                is_local_min = false
                break
            end
        end
        is_local_min || continue
        push!(candidates, (; k0=re_vals[i] + im * im_vals[j], val=val))
    end

    isempty(candidates) && return NamedTuple[]
    sort!(candidates; by = c -> c.val)
    cutoff = candidates[1].val + loga_tol_window
    candidates = [c for c in candidates if c.val <= cutoff]
    length(candidates) > max_candidates && (candidates = candidates[1:max_candidates])

    out = NamedTuple[]
    for cand in candidates
        best = _refine_full_seed_family(a, b, Vfun, cand.k0, +1; target_half=target_half)
        k, Fk, converged = best.k, best.residual, best.converged
        _root_accepts(k, Fk, converged; residual_tol=accept_tol) || continue
        abs(k) <= k_max || continue
        if target_half === :lower
            imag(k) < 0 || continue
        elseif target_half === :upper
            imag(k) > 0 || continue
        end
        U, _, _ = _shoot_full_profile(a, b, Vfun, k, +1, xgrid)
        u1 = U[1]
        abs(u1) < 1e-10 * norm(U) && continue
        U ./= u1
        push!(out, (; k=k, γ=abs(imag(k)), U=U,
                     residual=Fk, converged=converged, k_seed=best.k_seed))
    end
    return _dedupe_resonance_records(out)
end

"""
    compute_resonances_fullline(a, b, Vfun; N=300, k_max=10.0, im_tol=1e-4,
                                axis_tol=max(1e-3, 10im_tol))

Find transmission resonances, scattering resonances, and bound states
of  H = -∂ₓ² + V(x)  on ℝ with V supported on (a, b).

Two 2M×2M companion EVPs are solved (TR and SR types).  V need not
be symmetric.

# Arguments
- `a`, `b`   : support endpoints
- `Vfun`     : potential V : ℝ → ℝ (zero outside (a,b))
- `N`        : number of *interior* FD nodes (matrix size 2(N+2)×2(N+2))
- `k_max`    : discard eigenvalues with |k| ≥ k_max
- `im_tol`   : threshold for Im(k) classification (default 1e-4)
- `axis_tol` : bound states must also satisfy |Re(k)| ≤ axis_tol

# Returns
NamedTuple with fields:
- `transmission` : TR resonances (Im k < 0, from TR companion)
- `transmission_axis` : TR states on/near iR (both signs, from TR companion)
- `transmission_real` : TR states on/near R (from TR companion)
- `scattering`   : SR resonances (Im k < 0, from SR companion)
- `bound_states` : L² eigenvalues on/near +iR (from SR companion)
- `xgrid`        : FD node positions (length M)

Each entry is a NamedTuple:
  `k`  — complex wavenumber
  `γ`  — Im part magnitude: |Im(k)| (real ≥ 0)
  `U`  — eigenfunction on xgrid, normalized U[1] = 1
"""
function compute_resonances_fullline(a, b, Vfun;
                                     N=300, k_max=10.0, im_tol=1e-4,
                                     axis_tol=max(1e-3, 10im_tol),
                                     bound_scan_N=2400,
                                     tr_scan_re_N=160,
                                     tr_scan_im_N=120,
                                     sr_scan_re_N=160,
                                     sr_scan_im_N=120,
                                     tr_scan_residual_tol=1e-3,
                                     sr_scan_residual_tol=1e-3,
                                     tr_scan_max_candidates=160,
                                     sr_scan_max_candidates=160,
                                     extra_matrix_Ns=(round(Int, 1.20N),),
                                     root_accept_tol=1e-7,
                                     use_heatmap_tr_seeds=false,
                                     heatmap_tr_re_max=k_max,
                                     heatmap_tr_im_lo=-k_max,
                                     heatmap_tr_im_hi=0.25,
                                     heatmap_tr_N_re=220,
                                     heatmap_tr_N_im=140,
                                     heatmap_tr_loga_tol_window=1.0,
                                     heatmap_tr_max_candidates=240)
    M  = N + 2                        # total nodes including both boundaries
    h  = (b - a) / (M - 1)
    xgrid = [a + (j-1)*h for j in 1:M]   # x[1]=a, …, x[M]=b

    # ── Build companion and solve, for both BC types ──────────────────────────
    function _solve_companion(N_seed, B_left_sign, B_right_sign)
        M_seed = N_seed + 2
        h_seed = (b - a) / (M_seed - 1)
        xgrid_seed = [a + (j-1)*h_seed for j in 1:M_seed]

        # Same A structure for both TR and SR (only B differs).
        A = zeros(ComplexF64, M_seed, M_seed)
        A[1, 1] =  2.0 / h_seed^2
        A[1, 2] = -2.0 / h_seed^2
        for j in 2:M_seed-1
            A[j, j-1] = -1.0 / h_seed^2
            A[j, j  ] =  2.0 / h_seed^2 + Vfun(xgrid_seed[j])
            A[j, j+1] = -1.0 / h_seed^2
        end
        A[M_seed, M_seed-1] = -2.0 / h_seed^2
        A[M_seed, M_seed  ] =  2.0 / h_seed^2

        B = zeros(ComplexF64, M_seed, M_seed)
        B[1, 1] = B_left_sign  * (2.0 / h_seed)
        B[M_seed, M_seed] = B_right_sign * (2.0 / h_seed)

        C = zeros(ComplexF64, 2M_seed, 2M_seed)
        for j in 1:M_seed
            C[j, M+j] = 1.0           # upper-right block: identity
        end
        C[M_seed+1:2M_seed, 1:M_seed] .=  A        # lower-left:  A
        C[M_seed+1:2M_seed, M_seed+1:2M_seed] .=  im .* B # lower-right: +iB

        F = eigen(C)
        return F.values, F.vectors
    end

    matrix_Ns = sort(unique(vcat([N], collect(extra_matrix_Ns))))
    matrix_Ns = [n for n in matrix_Ns if n >= 40]

    tr_seed_vals = ComplexF64[]
    sr_seed_vals = ComplexF64[]
    for N_seed in matrix_Ns
        ks_TR_seed, _ = _solve_companion(N_seed, +1, -1)
        ks_SR_seed, _ = _solve_companion(N_seed, -1, -1)
        append!(tr_seed_vals, ks_TR_seed)
        append!(sr_seed_vals, ks_SR_seed)
    end
    tr_seed_vals = _dedupe_seed_list(tr_seed_vals)
    sr_seed_vals = _dedupe_seed_list(sr_seed_vals)

    # ── Filter and build output entries ──────────────────────────────────────
    function _filter(ks, im_lo, im_hi, left_sign; target_half=:any)
        out = []
        for k_raw in ks
            k_seed = conj(k_raw)
            abs(k_seed) > k_max         && continue
            best = _refine_full_seed_family(a, b, Vfun, k_seed, left_sign; target_half=target_half)
            k, Fk, converged = best.k, best.residual, best.converged
            _root_accepts(k, Fk, converged; residual_tol=root_accept_tol) || continue
            abs(k) > k_max              && continue
            imag(k) < im_lo             && continue
            imag(k) > im_hi             && continue

            U, _, _ = _shoot_full_profile(a, b, Vfun, k, left_sign, xgrid)
            u1 = U[1]
            abs(u1) < 1e-10 * norm(U) && continue
            U ./= u1

            push!(out, (; k=k, γ=abs(imag(k)), U=U,
                         residual=Fk, converged=converged, k_seed=best.k_seed))
        end
        # Sort by |Im(k)|
        sort!(out; by = r -> imag(r.k))
        _dedupe_resonance_records(out)
    end

    tr_all = _filter(tr_seed_vals, -Inf, Inf, +1)
    append!(tr_all, _residual_minima_scan(a, b, Vfun, xgrid, k_max, +1;
                                          re_max=k_max,
                                          im_lo=-k_max,
                                          im_hi=-max(im_tol, 1e-3),
                                          N_re=tr_scan_re_N,
                                          N_im=tr_scan_im_N,
                                          residual_tol=tr_scan_residual_tol,
                                          max_candidates=tr_scan_max_candidates,
                                          accept_tol=root_accept_tol,
                                          target_half=:lower))
    if use_heatmap_tr_seeds
        append!(tr_all, _heatmap_jost_seed_scan(a, b, Vfun, xgrid, k_max;
                                                re_max=heatmap_tr_re_max,
                                                im_lo=heatmap_tr_im_lo,
                                                im_hi=heatmap_tr_im_hi,
                                                N_re=heatmap_tr_N_re,
                                                N_im=heatmap_tr_N_im,
                                                loga_tol_window=heatmap_tr_loga_tol_window,
                                                max_candidates=heatmap_tr_max_candidates,
                                                accept_tol=root_accept_tol,
                                                target_half=:any))
    end
    tr_all = _dedupe_resonance_records(tr_all)
    tr_res = NamedTuple[]
    tr_axis = []
    tr_real = []
    for r in tr_all
        if imag(r.k) < -im_tol
            push!(tr_res, r)
            continue
        end
        if abs(real(r.k)) <= axis_tol
            abs(imag(r.k)) > im_tol || continue
            push!(tr_axis, r)
        elseif abs(imag(r.k)) <= im_tol
            push!(tr_real, r)
        end
    end
    sort!(tr_axis; by = r -> imag(r.k))
    sort!(tr_real; by = r -> real(r.k))

    sr_res = _filter(sr_seed_vals, -Inf,      -im_tol, -1; target_half=:lower)
    append!(sr_res, _residual_minima_scan(a, b, Vfun, xgrid, k_max, -1;
                                          re_max=k_max,
                                          im_lo=-k_max,
                                          im_hi=-max(im_tol, 1e-3),
                                          N_re=sr_scan_re_N,
                                          N_im=sr_scan_im_N,
                                          residual_tol=sr_scan_residual_tol,
                                          max_candidates=sr_scan_max_candidates,
                                          accept_tol=root_accept_tol,
                                          target_half=:lower))
    sr_res = _dedupe_resonance_records(sr_res)
    bs = NamedTuple[]
    for r in _filter(sr_seed_vals, +im_tol, Inf, -1; target_half=:upper)
        abs(real(r.k)) <= axis_tol || continue
        push!(bs, r)
    end
    append!(bs, _bound_state_scan(a, b, Vfun, xgrid, k_max;
                                  κ_min=max(im_tol, 1e-6),
                                  N_scan=bound_scan_N,
                                  residual_tol=root_accept_tol))
    bs = _dedupe_resonance_records(bs; tol=max(axis_tol, 5e-5))
    sort!(bs; by = r -> imag(r.k))

    return (; transmission=tr_res, transmission_axis=tr_axis,
            transmission_real=tr_real,
            scattering=sr_res, bound_states=bs, xgrid=xgrid)
end


"""
    print_resonances_fullline(result)

Pretty-print the output of `compute_resonances_fullline`.
"""
function print_resonances_fullline(result)
    function _table(label, entries)
        println("  ── $label ($(length(entries))) ──")
        if isempty(entries)
            println("     (none found)")
            return
        end
        @printf("  %4s  %24s  %12s\n", "#", "k", "|Im(k)|")
        println("  " * "─"^44)
        for (i, r) in enumerate(entries)
            @printf("  %4d  %+10.5f %+10.5f i  %10.6f\n",
                    i, real(r.k), imag(r.k), r.γ)
        end
        println()
    end
    _table("Transmission resonances  k = -iγ  (TR)", result.transmission)
    _table("Transmission axis states k = ±iγ  (TR)", result.transmission_axis)
    _table("Transmission states      k ∈ R    (TR)", result.transmission_real)
    _table("Scattering   resonances           (SR)", result.scattering)
    _table("Bound states             k = +iγ  (BS)", result.bound_states)
end
