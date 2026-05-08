###############################################################
# trajectory_plot_utils.jl
#
# Shared styling/helpers for the Section 7 split (E,N) plots.
###############################################################

using Plots, LaTeXStrings

const TRAJ_COL_GS    = RGB(0.20, 0.55, 0.26)   # forest green
const TRAJ_COL_E1    = RGB(0.55, 0.30, 0.65)   # purple — excited E_1 branch
const TRAJ_COL_TR    = RGB(0.18, 0.36, 0.70)   # steel blue
const TRAJ_COL_SOL   = RGB(0.85, 0.50, 0.12)   # dark orange
const TRAJ_COL_CORE0 = RGB(0.80, 0.15, 0.15)   # red
const TRAJ_COL_CORET = RGB(0.45, 0.05, 0.05)   # deep red
const TRAJ_COL_RAD0  = RGB(0.00, 0.45, 0.55)   # teal
const TRAJ_COL_RADT  = RGB(0.00, 0.28, 0.38)   # dark teal
const TRAJ_COL_BIF   = :black
const TRAJ_COL_TOTAL = RGB(0.35, 0.35, 0.35)   # gray

function compound_marker_shape(outer::Symbol, inner::Symbol; Nc::Int=48)
    # Single closed polygon tracing outer outline + inner crosshair.
    # Outline start/end vertex is chosen to coincide with a cross arm tip
    # so the path closes without a visible connector "spoke."
    if outer == :circle && inner == :+
        θ = range(0, 2π, length=Nc+1)
        xs_o = collect(cos.(θ))
        ys_o = collect(sin.(θ))
    elseif outer == :circle && inner == :xcross
        θ = range(π/4, π/4 + 2π, length=Nc+1)
        xs_o = collect(cos.(θ))
        ys_o = collect(sin.(θ))
    elseif outer == :rect && inner == :+
        xs_o = [1.0,  1.0, -1.0, -1.0, 1.0, 1.0]
        ys_o = [0.0,  1.0,  1.0, -1.0,-1.0, 0.0]
    elseif outer == :rect && inner == :xcross
        xs_o = [1.0, -1.0, -1.0,  1.0, 1.0]
        ys_o = [1.0,  1.0, -1.0, -1.0, 1.0]
    elseif outer == :diamond && inner == :+
        xs_o = [1.0, 0.0, -1.0, 0.0, 1.0]
        ys_o = [0.0, 1.0,  0.0,-1.0, 0.0]
    elseif outer == :diamond && inner == :xcross
        xs_o = [0.5,  0.0, -0.5, -1.0, -0.5, 0.0,  0.5,  1.0,  0.5]
        ys_o = [0.5,  1.0,  0.5,  0.0, -0.5,-1.0, -0.5,  0.0,  0.5]
    else
        return Plots.Shape([1.0, -1.0], [0.0, 0.0])
    end
    if inner == :+
        s = max(abs(xs_o[1]), abs(ys_o[1]), 1e-12)  # = 1 in all + cases above
        # Trace from (xs_o[1], ys_o[1]) = (1,0) through + back to (1,0).
        cross_x = [0.0, 0.0, 0.0,  0.0, -s, 0.0, s]
        cross_y = [0.0, s,  -s,    0.0, 0.0, 0.0, 0.0]
    else
        d = xs_o[1]                                  # = 1/√2, 1, or 1/2
        # Trace from (d,d) through × back to (d,d).
        cross_x = [0.0, -d,   0.0, -d,   0.0, d,    0.0, d]
        cross_y = [0.0, -d,   0.0,  d,   0.0,-d,    0.0, d]
    end
    return Plots.Shape(vcat(xs_o, cross_x), vcat(ys_o, cross_y))
end

function endpoint_marker_visual!(plt, x, y; color, outer_shape, inner_shape)
    # Outer ring: white fill + colored stroke. Drawn early so branches sit on top.
    scatter!(plt, [x], [y];
        ms=13,
        color=:white,
        markerstrokecolor=color,
        markerstrokewidth=2.5,
        markershape=outer_shape,
        label="")
    # Inner crosshair glyph.
    scatter!(plt, [x], [y];
        ms=7,
        color=color,
        markerstrokecolor=color,
        markerstrokewidth=1.5,
        markershape=inner_shape,
        label="")
end

function endpoint_marker_legend!(plt; color, outer_shape, inner_shape, label)
    # Plots/GR cannot pack ring + crosshair into a single marker, so the
    # legend swatch shows the outer shape only. Use white fill + colored
    # stroke so the open shape matches the on-plot marker style.
    scatter!(plt, [-1e9], [-1e9];
        ms=13,
        color=:white,
        markerstrokecolor=color,
        markerstrokewidth=2.2,
        markershape=outer_shape,
        label=label)
end

function split_trajectory_plot(;
    E_branches,
    N_branches,
    branch_classes,
    branch_stable=fill(true, length(branch_classes)),
    branch_labels=String[],
    E_bif,
    γ,
    E_core,
    N_core,
    E_sol_rest=nothing,
    N_sol=nothing,
    E_total=nothing,
    N_total=nothing,
    split_valid=nothing,
    N_rad_final=nothing,
    core_label=L"\mathrm{core}",
    soliton_label=L"\mathrm{soliton}",
    total_label=L"\mathrm{total}",
    rad_label=L"\mathcal{N}_{\mathrm{rad}}",
    legend=:outerright,
    size=(900, 420),
)
    have_soliton = !(isnothing(E_sol_rest) || isnothing(N_sol))
    have_total = !(isnothing(E_total) || isnothing(N_total))
    have_split = !(isnothing(split_valid))
    vis_rad = have_soliton ? collect(N_sol .> 0.05) : Bool[]
    settled_rad = have_soliton ? collect(N_sol .> 0.3) : Bool[]
    valid_core = have_split ? collect(split_valid) : trues(length(E_core))
    valid_rad = if have_split && have_soliton
        collect(split_valid .& vis_rad)
    else
        vis_rad
    end

    viz_E = if have_soliton && any(settled_rad)
        vcat(E_core[valid_core], E_sol_rest[settled_rad], have_total ? E_total : Float64[], [E_bif])
    else
        vcat(E_core[valid_core], have_total ? E_total : Float64[], [E_bif])
    end
    viz_N = if have_soliton && any(settled_rad)
        vcat(N_core[valid_core], N_sol[settled_rad], have_total ? N_total : Float64[], [4γ])
    else
        vcat(N_core[valid_core], have_total ? N_total : Float64[], [4γ])
    end
    viz_E_safe = filter(isfinite, viz_E)
    viz_N_safe = filter(isfinite, viz_N)

    # Ensure every branch has its rightmost (least-negative) E visible — its
    # endpoint, which marks where the linear bound state lives, must be in frame.
    branch_right_E = isempty(E_branches) ? Float64[] :
                     [maximum(filter(isfinite, Es); init=-Inf) for Es in E_branches]
    branch_right_E = filter(e -> isfinite(e) && e < 0, branch_right_E)
    E_min_branches = isempty(branch_right_E) ? Inf : minimum(branch_right_E)

    E_min = min(minimum(viz_E_safe), E_bif, E_min_branches) - 0.05
    E_max = 0.0
    N_min = -0.05
    # Include each branch's peak N within the visible E range, so y-axis fits all branches.
    branch_top_N = Float64[]
    for (Es, Ns) in zip(E_branches, N_branches)
        keep = (Es .>= E_min) .& (Es .<= E_max)
        any(keep) && push!(branch_top_N, maximum(Ns[keep]))
    end
    N_max_branches = isempty(branch_top_N) ? -Inf : maximum(branch_top_N)
    N_max = max(maximum(viz_N_safe), 4γ, N_max_branches) + 0.4

    plt = plot(;
        xlabel=L"E", ylabel=L"\mathcal{N}",
        xlims=(E_min, E_max), ylims=(N_min, N_max),
        legend=legend, title="",
        fontfamily="Computer Modern",
        size=size, dpi=300,
        framestyle=:box, grid=false,
        legendfontsize=13, tickfontsize=13, guidefontsize=16,
        foreground_color_legend=nothing, background_color_legend=:white,
        margin=5Plots.mm)

    # Markers drawn first so the branches sit on top of them.
    if have_total
        endpoint_marker_visual!(plt, E_total[1], N_total[1];
            color=TRAJ_COL_TOTAL, outer_shape=:circle, inner_shape=:+)
    end
    if any(valid_core)
        last_core = findlast(valid_core)
        endpoint_marker_visual!(plt, E_core[last_core], N_core[last_core];
            color=TRAJ_COL_CORET, outer_shape=:rect, inner_shape=:+)
    end
    if have_soliton && any(valid_rad)
        last_vis = findlast(valid_rad)
        endpoint_marker_visual!(plt, E_sol_rest[last_vis], N_sol[last_vis];
            color=TRAJ_COL_RADT, outer_shape=:diamond, inner_shape=:xcross)
    end

    E_sol_curve = range(min(E_min, -1e-4), -1e-6, length=400)
    plot!(plt, E_sol_curve, [4sqrt(abs(e)) for e in E_sol_curve];
          color=TRAJ_COL_SOL, ls=:dash, lw=2, label=L"\mathcal{N}[\mathcal{S}_E]")

    class_color = Dict(
        "ground state branch" => TRAJ_COL_GS,
        "transmission resonance branch" => TRAJ_COL_TR,
        "resonance" => TRAJ_COL_TR,
        "E_0 branch" => TRAJ_COL_GS,
        "E_1 branch" => TRAJ_COL_E1,
        "branch" => :purple,
    )
    class_short = Dict(
        "ground state branch" => "ground state",
        "transmission resonance branch" => L"\mathrm{TR\ branch}",
        "resonance" => L"\mathrm{TR\ branch}",
        "E_0 branch" => L"E_0\ \mathrm{branch}",
        "E_1 branch" => L"E_1\ \mathrm{branch}",
        "branch" => "branch",
    )
    seen_classes = Set{String}()
    plot_labels = isempty(branch_labels) ? branch_classes : branch_labels
    for (Es, Ns, cls, lbl_key, stable) in zip(E_branches, N_branches, branch_classes, plot_labels, branch_stable)
        col = get(class_color, cls, :purple)
        keep = (Es .>= E_min - 0.3) .& (Es .<= E_max + 0.1)
        count(keep) >= 1 || continue
        if lbl_key in seen_classes
            lbl = ""
        else
            base = get(class_short, lbl_key, lbl_key)
            suffix = stable ? "\\ \\mathrm{(stable)}" : "\\ \\mathrm{(unstable)}"
            if base isa LaTeXString
                inner = string(base)[2:end-1]
                lbl = latexstring(inner * suffix)
            else
                lbl = latexstring("\\mathrm{" * string(base) * "}" * suffix)
            end
        end
        push!(seen_classes, lbl_key)
        plot!(plt, Es[keep], Ns[keep]; color=col, lw=4.5, label=lbl)
    end

    if !(isnothing(N_rad_final)) && isfinite(N_rad_final) && N_rad_final > 1e-8
        hline!(plt, [N_rad_final];
            color=:gray35, ls=:dot, lw=1.8, alpha=0.95,
            label=rad_label)
    end

    # Solid y=0 axis line drawn on top of markers/branches so it isn't covered.
    hline!(plt, [0.0]; color=:black, lw=1.0, alpha=0.9, label="")

    # Off-screen labeled scatters for the legend.
    if have_total
        endpoint_marker_legend!(plt; color=TRAJ_COL_TOTAL,
            outer_shape=:circle, inner_shape=:+, label=total_label)
    end
    if any(valid_core)
        endpoint_marker_legend!(plt; color=TRAJ_COL_CORET,
            outer_shape=:rect, inner_shape=:+, label=core_label)
    end
    if have_soliton && any(valid_rad)
        endpoint_marker_legend!(plt; color=TRAJ_COL_RADT,
            outer_shape=:diamond, inner_shape=:xcross, label=soliton_label)
    end

    return plt
end
