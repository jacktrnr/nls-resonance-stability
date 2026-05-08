# Data

Small, regenerable artifacts needed to reproduce the paper figures without
re-running the full continuation/dynamics pipeline. Each subdirectory
corresponds to one example in the paper.

## Inventory

| Directory          | Role in paper                  | Size |
|--------------------|--------------------------------|------|
| `fl-sin-pi-A3/`    | Figs 6.2 and 7.1 (sin TR; unstable) | ~49 MB |
| `fl-cos3half-A5/`  | Fig 7.2 (cos TR; sign-changing $U_\star$, undetermined $E_1$) | ~49 MB |
| `fl-stable-gauss/` | Fig 6.3 (Gaussian TR; stable, steep-slope regime) | ~48 MB |
| `hl-smooth-V-10/`  | Fig 6.1 (smooth half-line SR; stable) | ~48 MB |
| `*-data.jld2`      | Top-level: precomputed bifurcation branches `psi_branch`, `E_branch` for the three full-line examples  | ~1 MB total |

## What's in each `.jld2`

  * `<desc>-data.jld2`             — bifurcation-branch arrays $(E, \mathcal N, \psi)$, the resonance pair $(\gamma_\star, U_\star)$, the potential `Vx_real`, plus stability functionals.
  * `<desc>-prof3-dynamics-data.jld2` — initial state and the dynamics-grid metadata (`Tmax`, `dt`, `cap_strength`, `cap_width`, `Xmax`, `Ngrid`) used by the long-time simulations of Section 7.
  * `<desc>-prof3-split-data.jld2` — soliton/core/radiation diagnostic split at the final time of the long-time run.

## Heavy resim-psi files (NOT in this directory)

The full $\Psi(x,t)$ snapshot grids `<desc>-prof3-resim-psi.jld2`
(~100-200 MB each) are needed by `paper_figures/resim_and_plot_prof3.jl`
and `paper_figures/make_videos.py`. They are excluded via `.gitignore`;
regenerate locally with

```bash
julia --project=. paper_figures/regenerate_resim.jl fl-sin-pi-A3 3
julia --project=. paper_figures/regenerate_resim.jl fl-cos3half-A5 3
```

(~25–40 minutes per case). The output files land in this directory under the
matching `<desc>/` subfolder.
