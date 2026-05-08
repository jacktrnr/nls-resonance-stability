# Examples

Each script is self-contained and writes its outputs into `examples/output/`.

| Script | What it shows | Runtime |
|---|---|---|
| `01_resonance_finding.jl` | Find a scattering resonance for a smooth bump on $[0,1]$ and plot $V$ and $U_\star$. | <30 s |
| `02_dynamics_from_branch.jl` | Load a precomputed bifurcated state and evolve under NLS; plot the spacetime density and mass trace. | 1–2 min |
| `03_reproduce_fig62.jl` | Run the canonical pipeline that produces Figure 6.2 of the paper (sin TR resonance: U★, $\mathcal N$ vs $E$, profiles). | 3–5 min |
| `04_custom_V_full_pipeline.jl` | **Headline.** Drop in any compactly supported $V(x)$ and any initial perturbation $\delta\Psi_0$. The script finds the resonance, continues the bifurcation branch, then evolves under NLS and plots both the $(E,\mathcal N)$ diagram and the spacetime density. | 5–10 min |

Run any of them with

```bash
julia --project=. examples/04_custom_V_full_pipeline.jl
```

The first invocation will trigger Julia precompilation (one-off ~5–10 min).
