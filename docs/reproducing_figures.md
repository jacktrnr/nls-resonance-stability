# Reproducing the paper figures

This document walks through reproducing every figure and video in
*Sharp stability conditions of resonance-induced nonlinear bound states*
(Turner & Weinstein, 2026). All commands assume you are at the package root
(`nls-resonance-stability/`) and have already run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Outputs default to `Figures/` (in the package root) unless overridden via
the `NLS_FIGURES_DIR` environment variable.

---

## Figure 6.0 (schematic of stability sectors)

This is a TikZ figure compiled with the paper LaTeX source; see
`paper_figures/tikz_schematic.tex`. No Julia run needed.

---

## Figures 6.1, 6.2, and 6.4 (smooth scattering, sin transmission, μ verification)

```bash
julia --project=. paper_figures/generate_paper_figures.jl
```

Outputs:
- `Figures/smooth-Ustar.pdf`, `smooth-NvsE.pdf`, `smooth-profiles.pdf`, `smooth-mu-zoom.pdf` (Fig 6.1 + 6.4 left)
- `Figures/transmission-Ustar.pdf`, `transmission-NvsE.pdf`, `transmission-profiles.pdf`, `transmission-mu-zoom.pdf` (Fig 6.2 + 6.4 right)

Runtime: ~5–10 minutes (resonance scan + branch continuation for both cases).

---

## Figure 6.3 (stable Gaussian transmission, steep-slope regime)

```bash
julia --project=. paper_figures/generate_stable_transmission_figures.jl
julia --project=. paper_figures/regen_trans_stable_zoom.jl
```

Outputs `Figures/trans-stable-V.pdf`, `trans-stable-Ustar.pdf`,
`trans-stable-profiles.pdf`, `trans-stable-NvsE.pdf`,
`trans-stable-NvsE-zoom.pdf`. Runtime: ~5 minutes.

---

## Figure 7.1 (long-time dynamics: sin TR)

Step 1 — regenerate the heavy snapshot grid:

```bash
julia --project=. paper_figures/regenerate_resim.jl fl-sin-pi-A3 3
```

(~25–40 minutes; produces `data/fl-sin-pi-A3/fl-sin-pi-A3-prof3-resim-psi.jld2`.)

Step 2 — produce the figure panels:

```bash
DESC=fl-sin-pi-A3 PROF_IDX=3 julia --project=. paper_figures/resim_and_plot_prof3.jl
```

Outputs land in `data/fl-sin-pi-A3/`:
- `fl-sin-pi-A3-prof3-spacetime-raw.pdf` (Fig 7.1a)
- `fl-sin-pi-A3-prof3-spacetime.pdf`     (Fig 7.1b, asinh-rescaled)
- `fl-sin-pi-A3-prof3-trajectory-split.pdf` (Fig 7.1c)

---

## Figure 7.2 (long-time dynamics: cos TR)

Same as 7.1, with `DESC=fl-cos3half-A5`:

```bash
julia --project=. paper_figures/regenerate_resim.jl fl-cos3half-A5 3
DESC=fl-cos3half-A5 PROF_IDX=3 julia --project=. paper_figures/resim_and_plot_prof3.jl
```

---

## Videos accompanying Figs 7.1, 7.2

After both `*-resim-psi.jld2` files exist:

```bash
python3 paper_figures/make_videos.py
```

Produces `videos/fl-sin-pi-A3-dynamics.mp4` and
`videos/fl-cos3half-A5-dynamics.mp4`. Each video is ~50 s at 30 fps,
~1.8 MB. Runtime: ~10 minutes per video.

To re-render only one case:

```bash
python3 paper_figures/make_videos.py fl-sin-pi-A3
```
