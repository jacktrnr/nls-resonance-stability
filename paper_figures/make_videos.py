#!/usr/bin/env python3
"""make_videos.py — render publication-quality MP4s of the wave dynamics
for the two transmission-resonance examples in the paper.

Reads the same resim-psi.jld2 files used by `resim_and_plot_prof3.jl`,
pre-renders the spacetime heatmap once, then animates by updating only
the top-panel curves and the current-time line.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------- styling ----------
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "savefig.dpi":      120,
})

CASES = {
    "fl-sin-pi-A3": dict(
        title = r"$V(x) = 3\sin(\pi x),\;\; x\in[-1,1]$",
        path  = "dynamics-figures/fl-sin-pi-A3/fl-sin-pi-A3-prof3-resim-psi.jld2",
        xrad  = 40.0,
    ),
    "fl-cos3half-A5": dict(
        title = r"$V(x) = 5\cos(3\pi x/2),\;\; x\in[-1,1]$",
        path  = "dynamics-figures/fl-cos3half-A5/fl-cos3half-A5-prof3-resim-psi.jld2",
        xrad  = 80.0,
    ),
}


def load_jld2(path):
    """Read the fields we need from a JLD2 (HDF5) file."""
    with h5py.File(path, "r") as f:
        x = f["x_grid"][...].astype(np.float64)
        t = f["t_saves"][...].astype(np.float64)
        # psi_saves is stored as a compound (re,im) array
        ps = f["psi_saves"][...]
        if ps.dtype.names and "re" in ps.dtype.names:
            psi = ps["re"].astype(np.float64) + 1j * ps["im"].astype(np.float64)
        else:
            psi = ps.astype(np.complex128)
    return x, t, psi


def render(label, fps=30, stride=1, outdir="videos", verbose=True):
    cfg   = CASES[label]
    path  = cfg["path"]
    title = cfg["title"]
    xrad  = cfg["xrad"]

    if verbose:
        print(f"[{label}] loading {path}")
    x, t, psi = load_jld2(path)
    # JLD2 sometimes stores the matrix with axes (Nx, Nt) instead of (Nt, Nx).
    # Detect by length.
    if psi.shape[0] == len(x):
        psi = psi.T
    assert psi.shape == (len(t), len(x)), (psi.shape, len(t), len(x))

    rho   = np.abs(psi) ** 2
    rmax  = rho.max()
    rho0  = 1e-7 * rmax
    rho_a = np.arcsinh(rho / rho0)
    ramax = rho_a.max()

    mask = (x >= -xrad) & (x <= xrad)
    xw   = x[mask]
    rhoW = rho[:, mask]
    psiW = psi[:, mask]
    rho_aW = rho_a[:, mask]
    rho_init = rhoW[0]
    rho_max  = rhoW.max()
    amp_max  = max(np.abs(psiW.real).max(), np.abs(psiW.imag).max())
    ymin = -1.1 * amp_max
    ymax =  1.1 * max(rho_max, amp_max)

    # ---------- figure (static elements built once) ----------
    fig = plt.figure(figsize=(12.8, 7.2))
    fig.suptitle(title, fontsize=14)

    gs = fig.add_gridspec(
        2, 2,
        height_ratios = [0.40, 0.60],
        width_ratios  = [1.0, 0.022],
        hspace = 0.06,
        wspace = 0.02,
        left = 0.06, right = 0.95, top = 0.92, bottom = 0.07,
    )
    ax_top  = fig.add_subplot(gs[0, 0])
    ax_bot  = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_cbar = fig.add_subplot(gs[1, 1])

    # ---- Top static ----
    ax_top.axhline(0, color="black", lw=0.4, alpha=0.4)
    ax_top.axvspan(-1, 1, color="goldenrod", alpha=0.10)
    line_re, = ax_top.plot(xw, psiW[0].real,
                           color=(0.55, 0.55, 0.55, 0.40), lw=0.8,
                           label=r"$\mathrm{Re}\,\Psi$")
    line_im, = ax_top.plot(xw, psiW[0].imag,
                           color=(0.85, 0.55, 0.10, 0.40), lw=0.8,
                           label=r"$\mathrm{Im}\,\Psi$")
    ax_top.plot(xw, rho_init, color="gray", lw=1.0, ls=":", alpha=0.7,
                label=r"$|\Psi_0|^2$")
    line_amp, = ax_top.plot(xw, rhoW[0],
                            color=(0.05, 0.10, 0.45), lw=1.8,
                            label=r"$|\Psi(x,t)|^2$")
    ax_top.set_xlim(-xrad, xrad)
    ax_top.set_ylim(ymin, ymax)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r"")
    ax_top.legend(loc="upper right", frameon=False, fontsize=9)
    title_artist = ax_top.set_title("", fontsize=11)

    # ---- Bottom: pre-rendered heatmap ----
    im = ax_bot.imshow(
        rho_aW,
        origin = "lower",
        aspect = "auto",
        extent = [-xrad, xrad, t[0], t[-1]],
        cmap   = "inferno",
        vmin   = 0,
        vmax   = ramax,
        interpolation = "nearest",
    )
    ax_bot.set_xlabel(r"$x$")
    ax_bot.set_ylabel(r"$t$")
    ax_bot.axvline(-1, color="white", lw=0.6, ls="--", alpha=0.7)
    ax_bot.axvline( 1, color="white", lw=0.6, ls="--", alpha=0.7)
    hline = ax_bot.axhline(t[0], color=(0.20, 0.95, 0.95, 0.85), lw=1.3)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(r"$\mathrm{asinh}(|\Psi|^2/\rho_0)$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ---------- animation ----------
    frames = range(0, len(t), stride)

    def update(fi):
        ti = t[fi]
        line_re.set_ydata(psiW[fi].real)
        line_im.set_ydata(psiW[fi].imag)
        line_amp.set_ydata(rhoW[fi])
        hline.set_ydata([ti, ti])
        title_artist.set_text(f"$t = {ti:6.2f}$")
        return line_re, line_im, line_amp, hline, title_artist

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{label}-dynamics.mp4")
    writer = FFMpegWriter(fps=fps, bitrate=4000,
                          codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])

    if verbose:
        print(f"[{label}] rendering {len(frames)} frames at {fps} fps -> {out}")
    t0 = time.time()
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=writer, dpi=120)
    plt.close(fig)
    if verbose:
        size = os.path.getsize(out) / 1e6
        print(f"[{label}] done in {time.time()-t0:.1f}s  ({size:.2f} MB)")
    return out


def render_static_frame(label, out="frame-test.png", frac=0.55):
    """Save a single-frame PNG for layout sanity-check."""
    cfg  = CASES[label]
    x, t, psi = load_jld2(cfg["path"])
    if psi.shape[0] == len(x):
        psi = psi.T
    rho   = np.abs(psi) ** 2
    rmax  = rho.max()
    rho0  = 1e-7 * rmax
    rho_a = np.arcsinh(rho / rho0)
    ramax = rho_a.max()
    xrad  = cfg["xrad"]
    mask  = (x >= -xrad) & (x <= xrad)
    xw    = x[mask]
    rhoW  = rho[:, mask]; psiW = psi[:, mask]; rho_aW = rho_a[:, mask]
    rho_init = rhoW[0]
    rho_max  = rhoW.max()
    amp_max  = max(np.abs(psiW.real).max(), np.abs(psiW.imag).max())
    ymin = -1.1 * amp_max; ymax = 1.1 * max(rho_max, amp_max)

    fi = int(len(t) * frac)
    ti = t[fi]
    fig = plt.figure(figsize=(12.8, 7.2))
    fig.suptitle(cfg["title"], fontsize=14)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.40, 0.60],
                          width_ratios=[1.0, 0.022], hspace=0.06, wspace=0.02,
                          left=0.06, right=0.95, top=0.92, bottom=0.07)
    ax_top  = fig.add_subplot(gs[0, 0])
    ax_bot  = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_cbar = fig.add_subplot(gs[1, 1])

    ax_top.axhline(0, color="black", lw=0.4, alpha=0.4)
    ax_top.axvspan(-1, 1, color="goldenrod", alpha=0.10)
    ax_top.plot(xw, psiW[fi].real, color=(0.55,0.55,0.55,0.40), lw=0.8, label=r"$\mathrm{Re}\,\Psi$")
    ax_top.plot(xw, psiW[fi].imag, color=(0.85,0.55,0.10,0.40), lw=0.8, label=r"$\mathrm{Im}\,\Psi$")
    ax_top.plot(xw, rho_init, color="gray", lw=1.0, ls=":", alpha=0.7, label=r"$|\Psi_0|^2$")
    ax_top.plot(xw, rhoW[fi], color=(0.05,0.10,0.45), lw=1.8, label=r"$|\Psi(x,t)|^2$")
    ax_top.set_xlim(-xrad, xrad); ax_top.set_ylim(ymin, ymax)
    ax_top.set_xticklabels([])
    ax_top.legend(loc="upper right", frameon=False, fontsize=9)
    ax_top.set_title(f"$t = {ti:6.2f}$", fontsize=11)

    im = ax_bot.imshow(rho_aW, origin="lower", aspect="auto",
                       extent=[-xrad, xrad, t[0], t[-1]],
                       cmap="inferno", vmin=0, vmax=ramax, interpolation="nearest")
    ax_bot.set_xlabel(r"$x$"); ax_bot.set_ylabel(r"$t$")
    ax_bot.axvline(-1, color="white", lw=0.6, ls="--", alpha=0.7)
    ax_bot.axvline( 1, color="white", lw=0.6, ls="--", alpha=0.7)
    ax_bot.axhline(ti, color=(0.20,0.95,0.95,0.85), lw=1.3)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(r"$\mathrm{asinh}(|\Psi|^2/\rho_0)$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=list(CASES.keys()),
                    help="case labels to render (default: all)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test", action="store_true",
                    help="render single frame PNG instead of full video")
    args = ap.parse_args()
    for c in args.cases:
        if args.test:
            render_static_frame(c, out=f"frame-test-{c}.png")
        else:
            render(c, fps=args.fps, stride=args.stride)
