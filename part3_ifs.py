"""
COMP3710 Lab 1, Part 3 --- Iterated Function Systems on the GPU.

Core library: IFS definitions, a fully vectorised ("N chains at once") chaos
game, density rendering, and a GPU box-counting dimension estimator.

The headline fractal is Barnsley's fern.  It is an *affine* IFS, which makes it
structurally very different from the escape-time fractals of Parts 1--2:

  * Mandelbrot/Julia:  a fixed grid of pixels, iterate z <- z^2 + c, ask "did
    this pixel escape?".  The unknown is a *per-pixel scalar*.
  * IFS / chaos game:  no grid at all.  We iterate *points* under a randomly
    chosen contraction, and the fractal is the attractor those points fall onto.
    The image is a *histogram* of where the points went.

Author: s4984244
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# IFS definition
# --------------------------------------------------------------------------- #


@dataclass
class IFS:
    """An affine iterated function system  f_i(v) = A_i v + b_i,  i = 1..K.

    Attributes
    ----------
    name : str
        Human readable label, used for figure titles / filenames.
    A : (K, 2, 2) float array -- the linear parts.
    b : (K, 2)    float array -- the translations.
    p : (K,)      float array -- probability of selecting each map in the chaos
        game.  These do *not* change the attractor (see `exact_dimension`
        docstring); they only change the invariant measure, i.e. the brightness.
    exact_dim : float | None
        Closed-form box-counting dimension, when one exists.  Only available for
        systems whose maps are genuine *similarities* satisfying the open set
        condition, in which case Moran's equation  sum_i s_i^D = 1  applies.
    """

    name: str
    A: np.ndarray
    b: np.ndarray
    p: np.ndarray
    exact_dim: float | None = None
    note: str = ""
    _torch_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64).reshape(-1, 2, 2)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1, 2)
        self.p = np.asarray(self.p, dtype=np.float64).reshape(-1)
        assert len(self.A) == len(self.b) == len(self.p), "K mismatch"
        self.p = self.p / self.p.sum()

    @property
    def K(self) -> int:
        return len(self.A)

    # -- structural diagnostics -------------------------------------------- #

    def singular_values(self) -> np.ndarray:
        """(K, 2) singular values of each A_i.

        These are the contraction factors along the two principal directions.
        If the two are equal the map is a *similarity* (shape preserved);
        if they differ the map is *self-affine* (shape sheared), and no simple
        closed form for the dimension exists.
        """
        return np.linalg.svd(self.A, compute_uv=False)

    def lipschitz(self) -> np.ndarray:
        """(K,) Lipschitz constant of each map = largest singular value."""
        return self.singular_values()[:, 0]

    def is_similarity(self, tol: float = 1e-9) -> np.ndarray:
        s = self.singular_values()
        return np.abs(s[:, 0] - s[:, 1]) < tol

    def det(self) -> np.ndarray:
        """(K,) determinants.  |det| is the area contraction factor."""
        return np.linalg.det(self.A)

    def moran_dimension(self) -> float | None:
        """Solve Moran's equation  sum_i s_i^D = 1  for D.

        Valid only when every map is a similarity and the open set condition
        holds.  Returns None otherwise, which is the honest answer for the fern.
        """
        if not self.is_similarity(tol=1e-6).all():
            return None
        s = self.lipschitz()
        if np.any(s <= 0):  # a degenerate (rank-deficient) map
            return None
        lo, hi = 0.0, 10.0
        for _ in range(200):  # bisection; sum_i s_i^D is strictly decreasing
            mid = 0.5 * (lo + hi)
            if (s**mid).sum() > 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def burn_in_for(self, pixel_size: float, diameter: float) -> int:
        """Iterations needed for the transient to fall below one pixel.

        After k steps the point is within  diameter * L^k  of the attractor,
        where L = max_i Lipschitz(f_i).  Solve  diameter * L^k < pixel_size.

        This is what makes the *parallel* chaos game legitimate: every one of
        the N chains forgets its (arbitrary) starting point after the same
        small number of steps, so N short chains are as good as one long chain.
        """
        L = float(self.lipschitz().max())
        return int(np.ceil(np.log(pixel_size / diameter) / np.log(L)))

    # -- torch plumbing ----------------------------------------------------- #

    def tensors(self, device, dtype):
        key = (str(device), str(dtype))
        if key not in self._torch_cache:
            A = torch.tensor(self.A, device=device, dtype=dtype)
            b = torch.tensor(self.b, device=device, dtype=dtype)
            cdf = torch.tensor(np.cumsum(self.p), device=device, dtype=dtype)
            cdf[-1] = 1.0  # guard against float round-off in the last bin
            self._torch_cache[key] = (A, b, cdf)
        return self._torch_cache[key]

    def with_probabilities(self, p, tag: str) -> "IFS":
        return IFS(f"{self.name} [{tag}]", self.A.copy(), self.b.copy(), p,
                   self.exact_dim, self.note)


# --------------------------------------------------------------------------- #
# The zoo
# --------------------------------------------------------------------------- #

#: Barnsley's fern (Barnsley, *Fractals Everywhere*, 1988).  Four affine maps:
#:   f1  stem            -- SINGULAR (det = 0): collapses the plane onto a line
#:   f2  main frond      -- a similarity, ratio 0.851, plus a ~2.6 deg rotation
#:   f3  left  leaflet   -- self-affine (two different contraction factors)
#:   f4  right leaflet   -- self-affine, and orientation reversing (det < 0)
FERN = IFS(
    name="Barnsley fern",
    A=[[[0.00, 0.00], [0.00, 0.16]],
       [[0.85, 0.04], [-0.04, 0.85]],
       [[0.20, -0.26], [0.23, 0.22]],
       [[-0.15, 0.28], [0.26, 0.24]]],
    b=[[0.00, 0.00],
       [0.00, 1.60],
       [0.00, 1.60],
       [0.00, 0.44]],
    p=[0.01, 0.85, 0.07, 0.07],
    exact_dim=None,  # maps are self-affine, not similarities -> no closed form
    note="f1 is rank deficient (det=0); f3, f4 are not similarities.",
)

#: Validation target 1: Sierpinski triangle.  Three similarities of ratio 1/2
#: satisfying the open set condition  =>  D = log 3 / log 2 exactly.
SIERPINSKI = IFS(
    name="Sierpinski triangle",
    A=[[[0.5, 0.0], [0.0, 0.5]]] * 3,
    b=[[0.0, 0.0], [0.5, 0.0], [0.25, 0.5 * np.sqrt(3) / 2]],
    p=[1 / 3, 1 / 3, 1 / 3],
    exact_dim=float(np.log(3) / np.log(2)),
)

#: Validation target 2: the unit square, built as an IFS of four half-scale
#: copies.  A space-filling set: D = log 4 / log 2 = 2 exactly.
SQUARE = IFS(
    name="Filled square",
    A=[[[0.5, 0.0], [0.0, 0.5]]] * 4,
    b=[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
    p=[0.25] * 4,
    exact_dim=2.0,
)

#: Validation target 3: 2-D Cantor dust.  Four one-third-scale copies at the
#: corners  =>  D = log 4 / log 3 ~= 1.2619.  A totally disconnected set, so it
#: probes the *low* end of the dimension range.
CANTOR_DUST = IFS(
    name="Cantor dust",
    A=[[[1 / 3, 0.0], [0.0, 1 / 3]]] * 4,
    b=[[0.0, 0.0], [2 / 3, 0.0], [0.0, 2 / 3], [2 / 3, 2 / 3]],
    p=[0.25] * 4,
    exact_dim=float(np.log(4) / np.log(3)),
)

VALIDATION_SET = [CANTOR_DUST, SIERPINSKI, SQUARE]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _now():
    import time
    return time.perf_counter()


def describe(ifs: IFS) -> str:
    """One-screen structural summary -- handy to read out during the demo."""
    s = ifs.singular_values()
    d = ifs.det()
    lines = [f"{ifs.name}: K = {ifs.K} affine maps"]
    lines.append(f"  {'map':>4} {'p':>7} {'sigma1':>8} {'sigma2':>8} "
                 f"{'det':>9}  kind")
    for i in range(ifs.K):
        kind = "similarity" if abs(s[i, 0] - s[i, 1]) < 1e-6 else "self-affine"
        if s[i, 1] < 1e-12:
            kind = "SINGULAR (rank 1)"
        elif d[i] < 0:
            kind += ", flips orientation"
        lines.append(f"  f{i + 1:<3} {ifs.p[i]:7.3f} {s[i, 0]:8.4f} "
                     f"{s[i, 1]:8.4f} {d[i]:9.4f}  {kind}")
    lines.append(f"  Lipschitz constant L = max sigma1 = {s[:, 0].max():.4f}")
    m = ifs.moran_dimension()
    lines.append(f"  Moran similarity dimension: "
                 f"{m:.4f}" if m is not None else
                 "  Moran similarity dimension: N/A (not all maps are "
                 "similarities)")
    return "\n".join(lines)
