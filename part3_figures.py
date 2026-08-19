"""
COMP3710 Lab 1, Part 3 --- figure generation for the demonstration.

Run:   python part3_figures.py [--scale small|full] [--out DIR]

Every figure is designed to answer one specific demo question:

  part3_fern.png          the deliverable itself
  part3_maps.png          "explain how the fractal is formed"
  part3_burnin.png        "why is running N chains in parallel legitimate?"
  part3_boxcount.png      "substantial analysis" -- dimension, validated
  part3_probs.png         attractor vs invariant measure
  part3_deterministic.png the same attractor with no randomness at all
  part3_variants.png      the maps *are* the image (fractal compression)
  part3_scaling.png       "justify that it uses the GPU in a reasonable way"

Author: s4984244
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # compute nodes have no display

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

import part3_ifs as P

# --------------------------------------------------------------------------- #

FERN_CMAP = LinearSegmentedColormap.from_list(
    "fern", ["#07120b", "#0d3b1e", "#1c7a37", "#4fbf5c", "#c9f2a0"])
MAP_COLOURS = ["#e6a532", "#3fa34d", "#4a90d9", "#c8503f"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "DejaVu Sans",
})


def tone_map(hist: torch.Tensor, gamma: float = 0.42) -> np.ndarray:
    """Visit counts -> [0, 1] brightness.

    The histogram spans ~4 orders of magnitude (the stem is visited 100x less
    often than the trunk), so a linear map would show a bare stalk and nothing
    else.  A gamma curve on the log of the counts keeps the sparse regions
    visible without blowing out the dense ones.
    """
    h = hist.float()
    h = torch.log1p(h)
    h = h / h.max().clamp(min=1e-9)
    return (h ** gamma).cpu().numpy()


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 1. the fern itself
# --------------------------------------------------------------------------- #


def fig_fern(out, n_points, resolution, results):
    hist, extent, info = P.chaos_game(
        P.FERN, n_points=n_points, n_iter=140, resolution=resolution)
    results["hero_render"] = info

    # measure the dimension here so it can be annotated onto the deliverable
    # itself -- the lab sheet asks for the analysis to be "part of the output"
    dim = P.measure_dimension(P.FERN, n_iter=100, n_grid=4096,
                              n_points=max(3_000_000,
                                           min(n_points, 20_000_000)))
    aff = P.FERN.affinity_dimension()
    results["hero_dimension"] = {"measured": dim["dimension"],
                                 "r2": dim["r2"], "affinity_bound": aff}

    fig, ax = plt.subplots(figsize=(6, 11))
    ax.imshow(tone_map(hist), cmap=FERN_CMAP,
              extent=extent, origin="upper", interpolation="nearest")
    ax.set_facecolor("#07120b")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Barnsley fern -- parallel chaos game in PyTorch\n"
        f"{info['n_points']:,} chains x {info['n_iter']} steps  "
        f"= {info['points_plotted']:,} points  |  "
        f"{info['seconds']:.2f} s on {info['device']}  |  "
        f"burn-in {info['burn_in']} steps", fontsize=9, color="#c9f2a0")
    for s in ax.spines.values():
        s.set_color("#1c7a37")

    # -- inset: the box-counting fit, on the deliverable itself ------------ #
    ins = ax.inset_axes([0.04, 0.545, 0.30, 0.225])
    eps, counts, ok = dim["eps"], dim["counts"], dim["mask"]
    x, y = np.log(1 / eps), np.log(counts)
    ins.plot(x, y, "o", ms=2.5, color="#2f6b40")
    ins.plot(x[ok], y[ok], "o", ms=3.2, color="#c9f2a0")
    xf = np.linspace(x[ok].min(), x[ok].max(), 10)
    ins.plot(xf, dim["dimension"] * xf + dim["intercept"], "-", lw=1.4,
             color="#e6a532")
    ins.set_facecolor("#0b1c12")
    ins.set_xlabel(r"$\log(1/\epsilon)$", fontsize=7, color="#9fd4a8")
    ins.set_ylabel(r"$\log N(\epsilon)$", fontsize=7, color="#9fd4a8")
    ins.tick_params(labelsize=6, colors="#9fd4a8", length=2)
    for s in ins.spines.values():
        s.set_color("#2f6b40")
    ins.grid(alpha=0.18, color="#9fd4a8")

    ax.text(0.04, 0.965,
            "box-counting dimension\n"
            f"$D_B$ = {dim['dimension']:.3f}   ($R^2$ = {dim['r2']:.4f})\n"
            f"Falconer bound  {aff:.3f}\n"
            "no closed form: $f_3,f_4$ are self-affine",
            transform=ax.transAxes, fontsize=8.5, color="#c9f2a0",
            va="top", linespacing=1.55)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "part3_fern.png"), dpi=150,
                facecolor="#07120b")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. how the fractal is formed
# --------------------------------------------------------------------------- #


def fig_maps(out, n_points):
    pts, lab = P.last_map_labels(P.FERN, n_points=n_points)
    titles = ["$f_1$  stem\n$\\det A=0$, rank 1",
              "$f_2$  main frond\nsimilarity, $s=0.851$",
              "$f_3$  left leaflet\nself-affine",
              "$f_4$  right leaflet\nself-affine, $\\det<0$"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 7.5))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05,
               c=[MAP_COLOURS[i] for i in lab], linewidths=0)
    ax.set_title("the fern, coloured by\nwhich map placed the point")
    for i in range(4):
        m = lab == i
        a = axes[i + 1]
        a.scatter(pts[:, 0], pts[:, 1], s=0.04, c="#dcdcdc", linewidths=0)
        a.scatter(pts[m, 0], pts[m, 1], s=0.05, c=MAP_COLOURS[i], linewidths=0)
        a.set_title(f"{titles[i]}\n$p={P.FERN.p[i]:.2f}$, "
                    f"{100 * m.mean():.1f}% of points")
    for a in axes:
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor("white")
    fig.suptitle(
        "Hutchinson operator:  fern  =  $f_1$(fern) $\\cup$ $f_2$(fern) "
        "$\\cup$ $f_3$(fern) $\\cup$ $f_4$(fern)\n"
        "$f_2$ reproduces the entire fern one notch smaller -- that is the "
        "self-similarity, and it is why the leaflets repeat forever",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "part3_maps.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. why parallel chains are legitimate
# --------------------------------------------------------------------------- #


def fig_burnin(out, results):
    steps = [0, 1, 3, 8, 20, 50]
    snaps, sep, typ = P.transient_snapshots(P.FERN, steps=steps,
                                            n_points=15_000)
    pixel = results.get("hero_render", {}).get("pixel_size", 0.005)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[1] * 6 + [1.9], wspace=0.3)
    for j, k in enumerate(steps):
        ax = fig.add_subplot(gs[0, j])
        s = snaps[k]
        ax.scatter(s[:, 0], s[:, 1], s=0.6, c="#1c7a37", linewidths=0)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$k$ = {k}", fontsize=10)

    ax = fig.add_subplot(gs[0, 6])
    L = float(P.FERN.lipschitz().max())
    k = np.arange(len(sep))
    ax.semilogy(k, np.maximum(sep, 1e-18), "o-", ms=3, lw=1.2,
                color="#c8503f", label="measured  $\\max|x_k-y_k|$")
    ax.semilogy(k, sep[0] * L ** k, "--", lw=1.2, color="#4a90d9",
                label=f"worst case  $L^k$,  $L={L:.3f}$")
    ax.semilogy(k, sep[0] * typ ** k, ":", lw=1.4, color="#e6a532",
                label=f"typical  $e^{{k\\sum p_i\\log\\sigma_i}}$ = "
                      f"${typ:.3f}^k$")
    ax.axhline(pixel, color="#444", lw=1, ls="-.",
               label=f"one pixel ({pixel:.1e})")
    ax.axvline(P.FERN.burn_in_for(pixel, 11.1), color="#444", lw=1, alpha=.5)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("separation of the two ensembles")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Forgetting the initial condition.  Left: one ensemble, from a "
        "uniform random cloud, after $k$ steps.\n"
        "Right: two ensembles from *different* clouds driven by the *same* "
        "map sequence -- the max separation tracks the worst case $L^k$, "
        "which is why $L$ (not the typical rate) sizes the burn-in.\n"
        f"Discarding {P.FERN.burn_in_for(pixel, 11.1)} steps puts every chain "
        "within one pixel of the attractor, so $N$ short parallel chains "
        "replace one long sequential chain.", fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(os.path.join(out, "part3_burnin.png"), dpi=130)
    plt.close(fig)
    results["burn_in"] = {
        "lipschitz_L": L, "typical_contraction": typ,
        "separation_by_step": [float(s) for s in sep],
        "burn_in_used": P.FERN.burn_in_for(pixel, 11.1)}


# --------------------------------------------------------------------------- #
# 4. box-counting dimension, validated first
# --------------------------------------------------------------------------- #


def fig_boxcount(out, n_points, n_grid, results):
    systems = P.VALIDATION_SET + [P.FERN]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    table = []

    for ax, ifs in zip(axes, systems):
        r = P.measure_dimension(ifs, n_points=n_points, n_iter=80,
                                n_grid=n_grid, factor=2)
        eps, counts, ok = r["eps"], r["counts"], r["mask"]
        x, y = np.log(1 / eps), np.log(counts)
        ax.plot(x, y, "o", ms=4, color="#bbbbbb", label="all scales")
        ax.plot(x[ok], y[ok], "o", ms=5, color="#1c7a37", label="fit window")
        xf = np.linspace(x[ok].min(), x[ok].max(), 10)
        ax.plot(xf, r["dimension"] * xf + r["intercept"], "-", lw=1.6,
                color="#c8503f",
                label=f"slope $D_B$ = {r['dimension']:.3f}")

        exact = r["exact"]
        theo = exact if exact is not None else ifs.affinity_dimension()
        lbl = ("exact  " if exact is not None else "affinity dim  ")
        ax.plot(xf, theo * xf + (y[ok].mean() - theo * x[ok].mean()), "--",
                lw=1.2, color="#4a90d9", label=f"{lbl}{theo:.3f}")

        ax.set_xlabel(r"$\log(1/\epsilon)$")
        ax.set_ylabel(r"$\log N(\epsilon)$")
        ax.set_title(f"{ifs.name}\n$R^2$ = {r['r2']:.5f} over "
                     f"{r['decades']:.2f} decades")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        row = {"set": ifs.name, "measured": round(r["dimension"], 4),
               "exact": None if exact is None else round(exact, 4),
               "affinity_dim": round(ifs.affinity_dimension(), 4),
               "r2": round(r["r2"], 6), "scales_used": r["n_used"],
               "decades": round(r["decades"], 2),
               "min_points_per_box": round(r["pts_per_box_min"], 1)}
        if exact is not None:
            row["rel_error_pct"] = round(100 * r["rel_error"], 2)
        table.append(row)

    fig.suptitle(
        "Box-counting dimension.  The three left panels are controls with a "
        "known answer -- the estimator is validated on them before it is "
        "trusted on the fern (right).\n"
        "Grey points are deliberately excluded: at coarse $\\epsilon$ every "
        "box is occupied so the slope is forced to 2, and at fine $\\epsilon$ "
        "the boxes start isolating individual samples so the slope collapses "
        "towards 0.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(os.path.join(out, "part3_boxcount.png"), dpi=130)
    plt.close(fig)
    results["dimension_table"] = table

    # -- is the fit window cherry-picked?  sweep the thresholds ------------- #
    sens = []
    r = P.measure_dimension(P.FERN, n_points=n_points, n_iter=80,
                            n_grid=n_grid, factor=2)
    for lo in (0.005, 0.01, 0.02, 0.05):
        for hi in (0.15, 0.25, 0.5):
            f = P.fit_dimension(r["eps"], r["counts"], n_points, n_grid,
                                lo_frac=lo, hi_frac=hi)
            sens.append({"lo_frac": lo, "hi_frac": hi,
                         "dimension": round(f["dimension"], 4),
                         "scales": f["n_used"], "r2": round(f["r2"], 5)})
    results["window_sensitivity_fern"] = sens

    # -- the grid-alignment experiment ------------------------------------- #
    align = []
    for ifs in (P.CANTOR_DUST, P.SIERPINSKI):
        for fac, ng in ((2, 4096), (3, 3 ** 7)):
            r = P.measure_dimension(ifs, n_points=n_points, n_iter=80,
                                    n_grid=ng, factor=fac)
            align.append({"set": ifs.name, "grid_factor": fac,
                          "measured": round(r["dimension"], 4),
                          "exact": round(r["exact"], 4),
                          "rel_error_pct": round(100 * r["rel_error"], 2)})
    results["grid_alignment"] = align


# --------------------------------------------------------------------------- #
# 5. probabilities change the measure, not the attractor
# --------------------------------------------------------------------------- #


def fig_probs(out, n_points, results):
    dets = np.abs(P.FERN.det())
    variants = [
        (P.FERN, "Barnsley's weights\n(0.01, 0.85, 0.07, 0.07)"),
        (P.FERN.with_probabilities(np.maximum(dets, 1e-3), "det"),
         "$\\propto |\\det A_i|$\n(area-proportional)"),
        (P.FERN.with_probabilities(np.ones(4), "uniform"),
         "uniform (0.25 each)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 8))
    rows = []
    for ax, (ifs, title) in zip(axes, variants):
        hist, extent, info = P.chaos_game(
            ifs, n_points=n_points, n_iter=120, resolution=700,
            bounds=(-2.35, 2.85, -0.15, 10.2))
        ax.imshow(tone_map(hist), cmap=FERN_CMAP, extent=extent,
                  origin="upper", interpolation="nearest")
        d = P.measure_dimension(ifs, n_points=min(n_points, 3_000_000),
                                n_iter=80, n_grid=2048)
        ax.set_title(f"{title}\n$D_B$ = {d['dimension']:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        rows.append({"weights": ifs.name, "measured_dim": round(
            d["dimension"], 4), "occupancy": round(info["occupancy"], 4)})
    fig.suptitle(
        "The probabilities are importance sampling, not part of the fractal.\n"
        "In theory all $p_i>0$ give the same *support*, so the same attractor "
        "and the same dimension.  In practice the right-hand panel measures\n"
        "$D_B\\approx1.27$, not $1.80$ -- with $p_1=0.25$ a quarter of all "
        "steps collapse the point back onto the stem ($\\det A_1=0$), the\n"
        "invariant measure concentrates there, and a finite sample never "
        "reaches the fine leaflets.  Raising the iteration count does not fix\n"
        "it (300 steps gives 1.297): this is the stationary measure, not a "
        "transient.  Barnsley's weights are chosen so that points arrive at a\n"
        "rate matching each region's area, $p_i\\propto|\\det A_i|$ -- and "
        "$p_1=0.01$ rather than $0$ because the stem has zero area yet must "
        "still be drawn.\n"
        "The probability-free check is in part3_net.png.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(os.path.join(out, "part3_probs.png"), dpi=130)
    plt.close(fig)
    results["probability_study"] = rows


# --------------------------------------------------------------------------- #
# 5b. probability-free measurement: the adaptive deterministic net
# --------------------------------------------------------------------------- #


def fig_net(out, eps_list, n_grid, results):
    rows, nets = [], {}
    for eps in eps_list:
        try:
            pts, info = P.deterministic_net(P.FERN, eps=eps)
        except MemoryError as exc:
            print(f"   [net] eps={eps:g} skipped: {exc}")
            continue
        e, c = P.box_count(pts, n_grid=n_grid)
        ok = e >= eps          # the net carries no information below eps
        fit = P.fit_dimension(e[ok], c[ok], n_points=info["n_points"],
                              n_grid=n_grid)
        rows.append({"eps": eps, "depth": info["depth"],
                     "peak_live": info["peak_live"],
                     "n_points": info["n_points"],
                     "dimension": round(fit["dimension"], 4),
                     "r2": round(fit["r2"], 5), "scales": fit["n_used"]})
        if len(nets) < 4:
            nets[eps] = pts[torch.randperm(
                pts.shape[0], device=pts.device)[:400_000]].cpu().numpy()
        del pts

    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, len(nets) + 1,
                          width_ratios=[1] * len(nets) + [2.0], wspace=0.3)
    for j, (eps, pts) in enumerate(sorted(nets.items(), reverse=True)):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(pts[:, 0], pts[:, 1], s=0.08, c="#1c7a37", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$\\epsilon$ = {eps:g}\n{len(pts):,} shown", fontsize=9)

    ax = fig.add_subplot(gs[0, len(nets)])
    if rows:
        e = np.array([r["eps"] for r in rows])
        d = np.array([r["dimension"] for r in rows])
        ax.semilogx(e, d, "o-", color="#1c7a37", label="$D_B$ of the net")
        aff = P.FERN.affinity_dimension()
        ax.axhline(aff, ls="--", color="#4a90d9",
                   label=f"Falconer affinity dim = {aff:.4f} (upper bound)")
        cg = results.get("dimension_table", [{}])[-1].get("measured")
        if cg:
            ax.axhline(cg, ls=":", color="#c8503f",
                       label=f"chaos game, Barnsley $p$ = {cg:.3f}")
        ax.set_xlabel(r"net resolution $\epsilon$")
        ax.set_ylabel("$D_B$")
        ax.invert_xaxis()
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3, which="both")
    results["deterministic_net"] = rows

    fig.suptitle(
        "A probability-free measurement.  Expand *every* map at every level; "
        "retire a branch once its accumulated contraction\n"
        "puts it within $\\epsilon$ of the attractor.  The result is a "
        "guaranteed $\\epsilon$-cover, not a random draw, so it does not\n"
        "depend on $p$ -- this is the instrument that settles what "
        "part3_probs.png raised.  Full expansion to depth 48 would need\n"
        "$4^{48}$ nodes; adaptive pruning needs $O(\\epsilon^{-D})\\approx"
        "10^6$, and every level is one batched einsum.  Refining the net "
        "raises $D_B$ towards the bound.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(os.path.join(out, "part3_net.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. deterministic IFS -- no randomness at all
# --------------------------------------------------------------------------- #


def fig_deterministic(out):
    dev = _dev()
    bounds = (-0.15, 1.15, -0.15, 1.15)
    H = W = 512
    ys = torch.linspace(bounds[3], bounds[2], H, device=dev)
    xs = torch.linspace(bounds[0], bounds[1], W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    starts = {
        "a filled square": ((gx > 0.05) & (gx < 0.95) &
                            (gy > 0.05) & (gy < 0.95)).float(),
        "a ring": (((gx - .5) ** 2 + (gy - .45) ** 2 < .16) &
                   ((gx - .5) ** 2 + (gy - .45) ** 2 > .06)).float(),
        "a single blob": (((gx - .8) ** 2 + (gy - .8) ** 2) < .01).float(),
    }
    show = [0, 1, 2, 4, 8]
    fig, axes = plt.subplots(len(starts), len(show),
                             figsize=(2.3 * len(show), 2.3 * len(starts)))
    for i, (name, img0) in enumerate(starts.items()):
        frames = P.hutchinson(P.SIERPINSKI, img0, steps=max(show),
                              bounds=bounds, device=dev)
        for j, k in enumerate(show):
            ax = axes[i, j]
            ax.imshow(frames[k].cpu().numpy(), cmap="Greens", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"$W^{{{k}}}(S_0)$")
            if j == 0:
                ax.set_ylabel(f"$S_0$ = {name}", fontsize=9)
    fig.suptitle(
        "The same attractor with no randomness at all: iterate the Hutchinson\n"
        "operator $W(S)=\\bigcup_i f_i(S)$ on a binary image, as one\n"
        "`grid_sample` per map followed by an element-wise max.\n"
        "$W$ contracts compact sets under the Hausdorff metric, so by Banach's\n"
        "fixed point theorem *every* starting set reaches the same limit --\n"
        "which is what the three rows show.  Sierpinski rather than the fern:\n"
        "the pull-back needs invertible $A_i$, and the stem map has "
        "$\\det A_1=0$.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(os.path.join(out, "part3_deterministic.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. the 24 coefficients *are* the image
# --------------------------------------------------------------------------- #


def fig_variants(out, n_points):
    rng = np.random.default_rng(11)
    specs = [("original", P.FERN.A.copy(), P.FERN.b.copy())]
    tweaks = [("stem thicker\n$A_1[0,0]:0\\to0.03$", (0, 0, 0), 0.03),
              ("fronds tighter\n$A_2[0,1]:0.04\\to0.10$", (1, 0, 1), 0.10),
              ("leaflets larger\n$A_3[1,1]:0.22\\to0.32$", (2, 1, 1), 0.32),
              ("droops\n$A_2[0,0]:0.85\\to0.80$", (1, 0, 0), 0.80)]
    for name, (k, i, j), val in tweaks:
        A = P.FERN.A.copy()
        A[k, i, j] = val
        specs.append((name, A, P.FERN.b.copy()))
    for t in range(2):
        A = P.FERN.A.copy()
        A += rng.normal(0, 0.02, A.shape)
        A[0] = P.FERN.A[0]
        specs.append((f"random jitter\n$\\sigma=0.02$ (seed {t})", A,
                      P.FERN.b.copy()))

    fig, axes = plt.subplots(1, len(specs), figsize=(2.1 * len(specs), 7.5))
    for ax, (name, A, b) in zip(axes, specs):
        ifs = P.IFS(name, A, b, P.FERN.p)
        try:
            if ifs.lipschitz().max() >= 1.0:
                raise ValueError("not contractive")
            hist, extent, _ = P.chaos_game(ifs, n_points=n_points, n_iter=110,
                                           resolution=420)
            ax.imshow(tone_map(hist), cmap=FERN_CMAP, extent=extent,
                      origin="upper", interpolation="nearest")
        except Exception as exc:  # a non-contractive system has no attractor
            ax.text(0.5, 0.5, f"diverges\n({exc})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "A whole plant is stored in 24 numbers.  Perturbing one coefficient "
        "changes the entire fractal coherently at every scale, because that "
        "coefficient is applied at every scale.\n"
        "This is the idea behind fractal image compression: do not store the "
        "picture, store the contraction whose fixed point is the picture.",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(out, "part3_variants.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 8. does it actually use the GPU well?
# --------------------------------------------------------------------------- #


def fig_scaling(out, sizes, results):
    dev = _dev()
    rows = []
    for n in sizes:
        _, _, info = P.chaos_game(P.FERN, n_points=n, n_iter=60,
                                  resolution=1000, device=dev,
                                  bounds=(-2.35, 2.85, -0.15, 10.2))
        rows.append((n, info["seconds"], info["points_per_second"]))
    n = np.array([r[0] for r in rows], dtype=float)
    t = np.array([r[1] for r in rows])
    thr = np.array([r[2] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.loglog(n, t, "o-", color="#1c7a37")
    a1.loglog(n, t[-1] * n / n[-1], "--", color="#888", label="ideal $O(N)$")
    a1.set_xlabel("chains $N$")
    a1.set_ylabel("wall time (s), 60 iterations")
    a1.set_title(f"cost is linear once the device is saturated\n({dev})")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, which="both")

    a2.semilogx(n, thr / 1e6, "o-", color="#c8503f")
    a2.set_xlabel("chains $N$")
    a2.set_ylabel("million points plotted / second")
    a2.set_title("throughput plateaus = device fully occupied\n"
                 "small $N$ is dominated by kernel launch overhead")
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "part3_scaling.png"), dpi=130)
    plt.close(fig)
    results["scaling"] = [{"n_chains": int(a), "seconds": round(b, 4),
                           "points_per_s": round(c)} for a, b, c in rows]


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["small", "full"], default="full",
                    help="small = laptop sanity check, full = A100 run")
    ap.add_argument("--out", default="out")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="figure names to skip, e.g. --skip variants scaling")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    full = args.scale == "full"
    cfg = dict(
        hero_points=20_000_000 if full else 400_000,
        hero_res=2400 if full else 700,
        map_points=1_500_000 if full else 250_000,
        dim_points=20_000_000 if full else 1_500_000,
        dim_grid=8192 if full else 2048,
        prob_points=4_000_000 if full else 400_000,
        var_points=1_500_000 if full else 200_000,
        scaling=[10 ** k for k in range(4, 9)] if full
        else [10 ** k for k in range(3, 7)],
        net_eps=[0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125] if full
        else [0.08, 0.04, 0.02, 0.01, 0.005],
    )

    dev = _dev()
    results = {"device": str(dev), "torch": torch.__version__,
               "scale": args.scale, "config": cfg,
               "gpu": torch.cuda.get_device_name(0)
               if dev.type == "cuda" else None}
    print(f"torch {torch.__version__} | device {dev}"
          + (f" ({results['gpu']})" if results["gpu"] else ""))
    print(P.describe(P.FERN))
    print(f"  Falconer affinity dimension: "
          f"{P.FERN.affinity_dimension():.4f}   (upper bound; Moran N/A)\n")
    results["fern_structure"] = P.describe(P.FERN)
    results["fern_affinity_dimension"] = P.FERN.affinity_dimension()

    jobs = [
        ("fern", lambda: fig_fern(args.out, cfg["hero_points"],
                                  cfg["hero_res"], results)),
        ("maps", lambda: fig_maps(args.out, cfg["map_points"])),
        ("burnin", lambda: fig_burnin(args.out, results)),
        ("boxcount", lambda: fig_boxcount(args.out, cfg["dim_points"],
                                          cfg["dim_grid"], results)),
        ("probs", lambda: fig_probs(args.out, cfg["prob_points"], results)),
        ("net", lambda: fig_net(args.out, cfg["net_eps"], cfg["dim_grid"],
                                results)),
        ("deterministic", lambda: fig_deterministic(args.out)),
        ("variants", lambda: fig_variants(args.out, cfg["var_points"])),
        ("scaling", lambda: fig_scaling(args.out, cfg["scaling"], results)),
    ]
    for name, fn in jobs:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        t0 = time.perf_counter()
        fn()
        print(f"[ok]   part3_{name}.png   ({time.perf_counter() - t0:.1f} s)",
              flush=True)

    path = os.path.join(args.out, "part3_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {path}")

    if "dimension_table" in results:
        print("\nBox-counting dimension:")
        for r in results["dimension_table"]:
            ex = r["exact"]
            print(f"  {r['set']:22s} D = {r['measured']:.4f}   "
                  + (f"exact {ex:.4f}   error {r['rel_error_pct']:.2f}%"
                     if ex is not None else
                     f"no closed form; affinity dim "
                     f"{r['affinity_dim']:.4f}")
                  + f"   R2 = {r['r2']:.5f}")


if __name__ == "__main__":
    main()
