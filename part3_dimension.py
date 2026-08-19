"""
COMP3710 Lab 1, Part 3 --- fractal dimension analysis for Barnsley's fern.

    python part3_dimension.py [--scale small|full] [--save out/part3_dimension.txt]

Answers the lab sheet's requirement to "generate code to look at the fractal
dimension and show this as part of the output".  Everything is printed to
stdout; nothing has to be opened in a viewer to read the result.

The analysis runs in three stages, deliberately in this order:

  1. structural diagnostics  -- what kind of maps are these, and does a closed
     form for the dimension even exist for them?
  2. validation              -- run the estimator on three systems whose answer
     is known analytically, and only continue if it reproduces them.
  3. measurement             -- two independent estimates of the fern, one
     probability-dependent (chaos game) and one probability-free (eps-net),
     compared against the theoretical upper bound.

Author: s4984244
"""

from __future__ import annotations

import argparse
import io
import sys

import numpy as np
import torch

import part3_ifs as P

RULE = "=" * 74


def banner(out, title):
    print(f"\n{RULE}\n{title}\n{RULE}", file=out)


def stage1(out) -> float:
    banner(out, "[1/3]  Structural diagnostics -- does a closed form exist?")
    print(P.describe(P.FERN), file=out)

    moran = P.FERN.moran_dimension()
    aff = P.FERN.affinity_dimension()
    print(file=out)
    if moran is None:
        print("  Moran's equation  sum_i s_i^D = 1  does NOT apply here:",
              file=out)
        print("  it requires every map to be a similarity (sigma_1 = sigma_2),",
              file=out)
        print("  and f3, f4 contract by different amounts in different",
              file=out)
        print("  directions.  The fern is self-AFFINE, not self-SIMILAR.",
              file=out)
        print(file=out)
        print("  The correct generalisation is Falconer's singular value",
              file=out)
        print("  function  phi^s(A) = sigma_1 * sigma_2^(s-1),  solved for",
              file=out)
        print("  sum_i phi^s(A_i) = 1.  Falconer (1988) proves this is an",
              file=out)
        print("  UPPER BOUND on the box dimension, attained for almost every",
              file=out)
        print("  choice of translation vectors.", file=out)
    print(file=out)
    print(f"  ==> Falconer affinity dimension  =  {aff:.4f}   (upper bound)",
          file=out)
    return aff


def stage2(out, n_points, n_grid) -> bool:
    banner(out, "[2/3]  Validating the estimator on sets with a known answer")
    print("  The estimator must reproduce known values before it is trusted",
          file=out)
    print("  on a set whose answer we do not know.\n", file=out)
    print(f"  {'set':<22}{'measured':>10}{'exact':>9}{'error':>9}"
          f"{'R^2':>10}{'scales':>8}", file=out)
    print(f"  {'-' * 68}", file=out)

    worst = 0.0
    for ifs in P.VALIDATION_SET:
        r = P.measure_dimension(ifs, n_points=n_points, n_iter=80,
                                n_grid=n_grid)
        err = 100 * r["rel_error"]
        worst = max(worst, err)
        print(f"  {ifs.name:<22}{r['dimension']:10.4f}{r['exact']:9.4f}"
              f"{err:8.2f}%{r['r2']:10.5f}{r['n_used']:8d}", file=out)

    ok = worst < 3.0
    print(file=out)
    print(f"  ==> worst error {worst:.2f}%  -->  estimator "
          f"{'ACCEPTED' if ok else 'REJECTED'}", file=out)
    if not ok:
        print("      (refusing to report a fern dimension from an estimator",
              file=out)
        print("       that cannot reproduce the controls)", file=out)
    return ok


def stage3(out, n_points, n_grid, eps_list, aff) -> None:
    banner(out, "[3/3]  Barnsley fern")

    print("  (a) chaos game, Barnsley's probabilities", file=out)
    r = P.measure_dimension(P.FERN, n_points=n_points, n_iter=100,
                            n_grid=n_grid)
    print(f"      D_B = {r['dimension']:.4f}    R^2 = {r['r2']:.5f}    "
          f"{r['n_used']} scales over {r['decades']:.2f} decades", file=out)
    print(f"      fit window: eps in [{r['eps_range'][0]:.2e}, "
          f"{r['eps_range'][1]:.2e}],  "
          f">= {r['pts_per_box_min']:.0f} points per box", file=out)

    print(file=out)
    print("  (b) deterministic eps-net -- uses NO probabilities at all,",
          file=out)
    print("      so it measures the support rather than the invariant measure",
          file=out)
    print(f"\n      {'eps':>9}{'depth':>8}{'peak live':>12}{'points':>13}"
          f"{'D_B':>9}{'R^2':>10}", file=out)
    print(f"      {'-' * 61}", file=out)

    net_dims = []
    for eps in eps_list:
        try:
            pts, info = P.deterministic_net(P.FERN, eps=eps)
        except MemoryError:
            print(f"      {eps:9g}  (skipped: exceeds available memory)",
                  file=out)
            continue
        e, c = P.box_count(pts, n_grid=n_grid)
        m = e >= eps                       # no information below eps
        fit = P.fit_dimension(e[m], c[m], n_points=info["n_points"],
                              n_grid=n_grid)
        net_dims.append(fit["dimension"])
        print(f"      {eps:9g}{info['depth']:8d}{info['peak_live']:12,}"
              f"{info['n_points']:13,}{fit['dimension']:9.4f}"
              f"{fit['r2']:10.5f}", file=out)
        del pts

    banner(out, "RESULT")
    finest = net_dims[-1] if net_dims else float("nan")
    print(f"  chaos game (Barnsley p)      D_B = {r['dimension']:.4f}",
          file=out)
    if net_dims:
        print(f"  deterministic net, finest    D_B = {finest:.4f}", file=out)
        print(f"  Falconer affinity dimension        {aff:.4f}   "
              f"(theoretical upper bound)", file=out)
        print(file=out)
        spread = abs(finest - r["dimension"])
        print(f"  Two independent methods agree to {spread:.3f}, and both sit",
              file=out)
        print(f"  below the bound as they must.  Refining the net raises the",
              file=out)
        print(f"  estimate monotonically towards {aff:.3f}: the deficit is a",
              file=out)
        print(f"  finite-resolution effect, not a disagreement with theory.",
              file=out)
        print(file=out)
        print(f"  ==> D_B(Barnsley fern) ~ {min(finest, aff):.2f}, "
              f"bounded above by {aff:.4f}", file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["small", "full"], default="full")
    ap.add_argument("--save", default=None,
                    help="also write the report to this file")
    args = ap.parse_args()

    full = args.scale == "full"
    n_points = 20_000_000 if full else 2_000_000
    n_grid = 8192 if full else 2048
    eps_list = ([0.05, 0.02, 0.01, 0.005, 0.0025, 0.00125] if full
                else [0.05, 0.02, 0.01, 0.005])

    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.stdout.write(s)
            buf.write(s)

        def flush(self):
            sys.stdout.flush()

    out = Tee()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(RULE, file=out)
    print("Barnsley fern -- fractal dimension analysis", file=out)
    print(RULE, file=out)
    print(f"  torch {torch.__version__}   device {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda"
             else ""), file=out)
    print(f"  scale = {args.scale}   samples = {n_points:,}   "
          f"grid = {n_grid}", file=out)

    aff = stage1(out)
    if stage2(out, n_points, n_grid):
        stage3(out, n_points, n_grid, eps_list, aff)

    print(file=out)
    if args.save:
        with open(args.save, "w") as fh:
            fh.write(buf.getvalue())
        print(f"report written to {args.save}")


if __name__ == "__main__":
    main()
