# Paper figure scripts

Every figure and video in the paper is produced by one of the scripts in this
directory. Outputs default to `Figures/` (relative to the package root); you
can override the location by setting the `NLS_FIGURES_DIR` environment
variable.

## Figure → script map

| Figure | Description | Script |
|---|---|---|
| 6.0 | Schematic of stability sectors | `tikz_schematic.tex` (compile with the paper) |
| 6.1 | Smooth half-line scattering example  ($U_\star$, $\mathcal N$ vs $E$, profiles, $\mu$ vs $\varepsilon^2$) | `generate_paper_figures.jl` (smooth-* outputs) |
| 6.2 | Sine transmission example (unstable) | `generate_paper_figures.jl` (transmission-* outputs) |
| 6.3 | Gaussian transmission example (stable, steep-slope) | `generate_stable_transmission_figures.jl` + `regen_trans_stable_zoom.jl` |
| 6.4 | $\mu(\varepsilon)$ verification (smooth + sine cases) | `generate_paper_figures.jl` (smooth-mu-zoom, transmission-mu-zoom) |
| 7.1 | Long-time dynamics for sin TR (`fl-sin-pi-A3`) | `resim_and_plot_prof3.jl` (DESC=fl-sin-pi-A3) |
| 7.2 | Long-time dynamics for cos TR (`fl-cos3half-A5`) | `resim_and_plot_prof3.jl` (DESC=fl-cos3half-A5) |
| Videos | `videos/*.mp4` accompanying Figs 7.1–7.2 | `make_videos.py` (Python; needs ffmpeg) |

## Heavy data

`resim_and_plot_prof3.jl` and `make_videos.py` need
`<example>-prof<k>-resim-psi.jld2` files (~100–200 MB each). They are **not**
shipped in this repo; regenerate them with:

```bash
julia --project=. paper_figures/regenerate_resim.jl fl-sin-pi-A3 3
julia --project=. paper_figures/regenerate_resim.jl fl-cos3half-A5 3
```

(~25–40 minutes per case on a recent laptop). Outputs land in
`data/<example>/` and are picked up automatically by the figure scripts.

## Running

```bash
# Figs 6.1, 6.2, 6.4 (and additional diagnostic outputs):
julia --project=. paper_figures/generate_paper_figures.jl

# Fig 6.3 (stable transmission, steep-slope regime):
julia --project=. paper_figures/generate_stable_transmission_figures.jl
julia --project=. paper_figures/regen_trans_stable_zoom.jl

# Fig 7.1 (after fetching/regenerating fl-sin-pi-A3 resim-psi):
DESC=fl-sin-pi-A3 PROF_IDX=3 julia --project=. paper_figures/resim_and_plot_prof3.jl

# Fig 7.2:
DESC=fl-cos3half-A5 PROF_IDX=3 julia --project=. paper_figures/resim_and_plot_prof3.jl

# Videos (after regenerating both resim-psi files):
python3 paper_figures/make_videos.py            # both cases
python3 paper_figures/make_videos.py fl-sin-pi-A3   # one case
```
