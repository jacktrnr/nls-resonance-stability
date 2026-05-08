# nls-resonance-stability

Companion code for the paper

> **Sharp stability conditions of resonance-induced nonlinear bound states**
> J. C. Turner & M. I. Weinstein, 2026.
> *(arXiv:XXXX.XXXXX)*

A Julia toolkit for studying the focusing 1D cubic NLS / Gross–Pitaevskii equation
$$ i\partial_t\Psi = -\partial_x^2\Psi + V(x)\Psi - |\Psi|^2\Psi $$
in the presence of a compactly supported linear potential $V$, focusing on the
nonlinear bound states that bifurcate from **scattering** and **transmission
resonances** of $H_V = -\partial_x^2 + V$.

---

## Demos

Long-time evolution of the two unstable transmission-resonance bound states
shown in Section 7 of the paper. Top: $|\Psi(x,t)|^2$ profile (with
$\mathrm{Re}\,\Psi$, $\mathrm{Im}\,\Psi$, and $|\Psi_0|^2$ as faint references).
Bottom: rescaled spacetime density $\mathrm{asinh}(|\Psi|^2/\rho_0)$.

<video src="https://raw.githubusercontent.com/jacktrnr/nls-resonance-stability/main/videos/fl-sin-pi-A3-dynamics.mp4" autoplay loop muted playsinline width="100%"></video>

*Figure 7.1 dynamics: $V(x) = 3\sin(\pi x)$ on $[-1,1]$. The bifurcated state ejects a soliton and the inner core relaxes onto the stable $E_0$ branch.*

<video src="https://raw.githubusercontent.com/jacktrnr/nls-resonance-stability/main/videos/fl-cos3half-A5-dynamics.mp4" autoplay loop muted playsinline width="100%"></video>

*Figure 7.2 dynamics: $V(x) = 5\cos(3\pi x/2)$ on $[-1,1]$. Sign-changing core; the inner state settles on the coexisting $E_1$ branch with slow radiation.*

---

## Repository layout

```
nls-resonance-stability/
├── src/                 — reusable library
│   ├── NLSResonanceStability.jl  (top-level module)
│   ├── wells.jl                  (potentials)
│   ├── resonances.jl             (linear resonance solver)
│   ├── continuation.jl           (BifurcationKit branch wrapper)
│   ├── dynamics.jl               (split-step NLS evolver)
│   ├── lplus.jl                  (L₊ Evans matching, Pöschl–Teller)
│   ├── jl_evans.jl               (J𝓛 Evans for full linearization)
│   ├── plotting.jl               (trajectory diagram styling)
│   └── nls_bifurcation/          (lower-level BifurcationKit driver)
├── examples/            — short, didactic demos
│   ├── 01_resonance_finding.jl
│   ├── 02_dynamics_from_branch.jl
│   ├── 03_reproduce_fig62.jl
│   └── 04_custom_V_full_pipeline.jl   ← the headline example
├── paper_figures/       — scripts that reproduce every figure in the paper
├── data/                — small data needed to reproduce the figures (~190 MB)
├── videos/              — pre-rendered MP4s of the unstable dynamics
└── docs/                — derivations and reproduction guide
```

---

## Quick start

```bash
git clone https://github.com/<user>/nls-resonance-stability.git
cd nls-resonance-stability
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/01_resonance_finding.jl
```

The first run will precompile dependencies (~5–10 minutes one-off). Subsequent
runs of any example take ≪1 minute up to a few minutes; expected runtimes are
in each script's header.

---

## Headline example

`examples/04_custom_V_full_pipeline.jl` runs the complete pipeline for an
arbitrary user-supplied potential $V(x)$ and initial perturbation $\delta\Psi_0$:

  1. Find the resonance pair $(\gamma_\star, U_\star)$.
  2. Continue the bifurcation branch $\varepsilon \mapsto (E(\varepsilon),\psi_\varepsilon)$.
  3. Plot the $(E,\mathcal{N})$ diagram with stability diagnostics
     ($\Omega_\star, d\mathcal{N}/dE$).
  4. Pick a small $\varepsilon$, perturb $\psi_\varepsilon$ by $\delta\Psi_0$, evolve
     under NLS via split-step.
  5. Plot the spacetime density.

Edit the user-input block near the top of the file to study any new
compactly supported $V$.

---

## Reproducing paper figures

See `paper_figures/README.md` for the figure-to-script map. Four scripts
generate every figure in the paper:

| Figure        | Script                                      |
|---------------|---------------------------------------------|
| 6.0 (schematic) | `paper_figures/tikz_schematic.tex`        |
| 6.1, 6.2, 6.4 | `paper_figures/generate_paper_figures.jl`   |
| 6.3           | `paper_figures/generate_stable_transmission_figures.jl` |
| 7.1, 7.2      | `paper_figures/resim_and_plot_prof3.jl`     |
| Videos        | `paper_figures/make_videos.py`              |

Some figures (7.1, 7.2 and the videos) require heavy `*-resim-psi.jld2` files
(~100–200 MB each) **not** shipped in this repo. Regenerate them locally:

```bash
julia --project=. paper_figures/regenerate_resim.jl fl-sin-pi-A3 3
julia --project=. paper_figures/regenerate_resim.jl fl-cos3half-A5 3
```

Each takes ~25–40 minutes on a recent laptop. Outputs land in `data/<example>/`
and are picked up automatically by the figure scripts and the video renderer.

---

## Citing

If you use this code, please cite

```bibtex
@article{TurnerWeinstein2026Stability,
  author  = {Turner, Jackson C. and Weinstein, Michael I.},
  title   = {Sharp stability conditions of resonance-induced nonlinear bound states},
  journal = {…},
  year    = {2026},
  doi     = {…},
}
```

A `CITATION.cff` is included for GitHub's "Cite this repository" widget.

---

## License

MIT. See [LICENSE](LICENSE).
