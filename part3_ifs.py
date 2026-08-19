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

    def affinity_dimension(self) -> float:
        """Falconer's affinity dimension, from the singular value function.

        Moran's equation needs similarities.  When the maps are self-affine
        (different contraction in different directions, as in f3 and f4 of the
        fern) the right generalisation replaces s_i^D by Falconer's singular
        value function

            phi^s(A) = sigma_1^s                      for 0 <= s <= 1
            phi^s(A) = sigma_1 * sigma_2^(s-1)        for 1 <  s <= 2

        and solves  sum_i phi^s(A_i) = 1.  Falconer (1988) proved this is an
        *upper bound* for the box dimension of the attractor, attained for
        almost every choice of translations.  So it is not an exact answer for
        this particular fern -- but it is a principled number to compare the
        measurement against, and it reduces to Moran's equation when every map
        is a similarity.
        """
        s1, s2 = self.singular_values().T

        def phi_sum(s: float) -> float:
            if s <= 1.0:
                return float(np.sum(s1**s))
            return float(np.sum(s1 * np.where(s2 > 0, s2, 0.0) ** (s - 1)))

        lo, hi = 0.0, 2.0
        if phi_sum(2.0) > 1.0:      # the attractor has positive area
            return 2.0
        for _ in range(200):        # phi_sum is strictly decreasing in s
            mid = 0.5 * (lo + hi)
            if phi_sum(mid) > 1.0:
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
# The chaos game --- vectorised over N independent chains
# --------------------------------------------------------------------------- #


def chaos_game(
    ifs: IFS,
    n_points: int = 1_000_000,
    n_iter: int = 100,
    burn_in: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: int = 2000,
    device=None,
    dtype=torch.float32,
    seed: int = 0,
    return_last: bool = False,
):
    """Run `n_points` chaos-game chains in lock-step and histogram the orbit.

    The classical chaos game is a *sequential* algorithm: one point, iterated
    millions of times.  That is the worst possible shape for a GPU.  The fix
    rests on a theorem, not on a hack:

        The IFS attractor A is the unique fixed point of the Hutchinson
        operator, which is a contraction on the compact subsets of R^2 under
        the Hausdorff metric (Banach fixed point theorem).  Consequently the
        chaos game converges to A from *any* starting point, and the limiting
        empirical measure is the same for almost every realisation.

    So instead of one chain of length T we run N chains of length T/N -- here,
    N = 10^6..10^8 chains of length ~100.  Every step is then a single batched
    tensor expression over all N points at once, with no Python-level loop over
    points and no data-dependent branching.

    Returns
    -------
    hist : (resolution_y, resolution_x) int64 tensor -- visit counts.
    extent : (x0, x1, y0, y1) tuple for `imshow`.
    info : dict of diagnostics (timing, burn-in, points plotted, ...).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)

    A, b, cdf = ifs.tensors(device, dtype)

    # -- 1. calibrate the viewport with a cheap pilot run ------------------- #
    if bounds is None:
        bounds = _estimate_bounds(ifs, device, dtype, gen)
    x0, x1, y0, y1 = bounds
    span = max(x1 - x0, y1 - y0)

    W = resolution
    H = max(1, int(round(resolution * (y1 - y0) / (x1 - x0))))
    pixel = (x1 - x0) / W

    if burn_in is None:
        burn_in = max(1, ifs.burn_in_for(pixel_size=pixel, diameter=span))
    n_iter = max(n_iter, burn_in + 1)

    # -- 2. random initial points; they will be forgotten after burn-in ----- #
    pts = torch.rand((n_points, 2), generator=gen, device=device, dtype=dtype)
    pts[:, 0] = pts[:, 0] * (x1 - x0) + x0
    pts[:, 1] = pts[:, 1] * (y1 - y0) + y0

    hist = torch.zeros(H * W, device=device, dtype=torch.int64)
    sx = W / (x1 - x0)
    sy = H / (y1 - y0)

    _sync(device)
    t0 = _now()

    for step in range(n_iter):
        # --- pick one map per point, in parallel -------------------------- #
        # searchsorted over the CDF is the vectorised form of "roll a die":
        # O(log K) per point, no host round-trip, no per-point branch.
        u = torch.rand(n_points, generator=gen, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, u).clamp_(max=ifs.K - 1)

        # --- apply it: one einsum == n_points matrix-vector products ------- #
        pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]

        # --- accumulate, but only once the transient has died out ---------- #
        if step >= burn_in:
            ix = ((pts[:, 0] - x0) * sx).long()
            iy = ((y1 - pts[:, 1]) * sy).long()  # flip y: row 0 is the top
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            hist += torch.bincount(iy[ok] * W + ix[ok], minlength=H * W)

    _sync(device)
    elapsed = _now() - t0

    plotted = int((n_iter - burn_in) * n_points)
    info = {
        # report the actual card rather than just "cuda": the figure title is
        # the evidence that this ran on the A100 and not on a laptop CPU
        "device": (torch.cuda.get_device_name(device)
                   if device.type == "cuda" else str(device)),
        "dtype": str(dtype).replace("torch.", ""),
        "n_points": n_points,
        "n_iter": n_iter,
        "burn_in": burn_in,
        "points_plotted": plotted,
        "map_applications": n_iter * n_points,
        "seconds": elapsed,
        "points_per_second": plotted / max(elapsed, 1e-9),
        "pixel_size": pixel,
        "resolution": (H, W),
        "occupancy": float((hist > 0).float().mean()),
    }
    out = (hist.reshape(H, W), (x0, x1, y0, y1), info)
    return (*out, pts) if return_last else out


def _estimate_bounds(ifs, device, dtype, gen, n=200_000, iters=60, pad=0.02):
    """Cheap pilot run to find the attractor's bounding box."""
    A, b, cdf = ifs.tensors(device, dtype)
    pts = torch.zeros((n, 2), device=device, dtype=dtype)
    for _ in range(iters):
        u = torch.rand(n, generator=gen, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, u).clamp_(max=ifs.K - 1)
        pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]
    lo = pts.min(0).values.tolist()
    hi = pts.max(0).values.tolist()
    mx = (hi[0] - lo[0]) * pad
    my = (hi[1] - lo[1]) * pad
    return (lo[0] - mx, hi[0] + mx, lo[1] - my, hi[1] + my)


def last_map_labels(ifs, n_points=400_000, burn_in=None, device=None,
                    dtype=torch.float32, seed=1):
    """Return points on the attractor together with *which* map produced them.

    Colouring by this label is the clearest way to answer the demo question
    "how is the fractal formed?": each f_i paints one recognisable piece, and
    the union of the four pieces is the whole fern again -- self-similarity
    made visible.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    A, b, cdf = ifs.tensors(device, dtype)
    if burn_in is None:
        burn_in = 60
    pts = torch.rand((n_points, 2), generator=gen, device=device, dtype=dtype)
    idx = None
    for _ in range(burn_in + 1):
        u = torch.rand(n_points, generator=gen, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, u).clamp_(max=ifs.K - 1)
        pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]
    return pts.cpu().numpy(), idx.cpu().numpy()


def transient_snapshots(ifs, steps, n_points=20_000, device=None,
                        dtype=torch.float64, seed=2, spread=6.0):
    """Watch the initial condition being forgotten.

    Two ensembles are launched from *different* random starting clouds but are
    then driven by the *same* sequence of randomly chosen maps.  After k steps
    the pair (x_k, y_k) satisfies

        |x_k - y_k| = |f_{i_k} o ... o f_{i_1}(x_0) - (same)(y_0)|
                    <= (prod_j sigma_1(A_{i_j})) |x_0 - y_0|
                    <= L^k |x_0 - y_0|,      L = max_i sigma_1(A_i)

    so this measures exactly the quantity the burn-in argument is about, with
    no reference sample and therefore no sampling floor -- it decays all the
    way to machine epsilon.  (Measuring distance to a finite point-cloud
    "attractor" instead would bottom out at the cloud's nearest-neighbour
    spacing, which is an artefact of the yardstick, not of the algorithm.)

    Returns
    -------
    snaps : {k: (n_points, 2) array} -- ensemble A at the requested steps.
    sep   : (K+1,) array -- max |x_k - y_k| at every step.
    typical : float -- exp(sum_i p_i log sigma_1(A_i)), the *typical* per-step
        contraction, which is faster than the worst case L.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    A, b, cdf = ifs.tensors(device, dtype)

    a_pts = (torch.rand((n_points, 2), generator=gen, device=device,
                        dtype=dtype) - 0.5) * spread
    b_pts = (torch.rand((n_points, 2), generator=gen, device=device,
                        dtype=dtype) - 0.5) * spread

    snaps, sep = {}, []
    for k in range(max(steps) + 1):
        if k in steps:
            snaps[k] = a_pts.cpu().numpy().copy()
        sep.append(float((a_pts - b_pts).norm(dim=1).max()))
        u = torch.rand(n_points, generator=gen, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, u).clamp_(max=ifs.K - 1)
        a_pts = torch.einsum("nij,nj->ni", A[idx], a_pts) + b[idx]
        b_pts = torch.einsum("nij,nj->ni", A[idx], b_pts) + b[idx]

    s1 = ifs.lipschitz()
    typical = float(np.exp(np.sum(ifs.p * np.log(np.maximum(s1, 1e-300)))))
    return snaps, np.array(sep), typical


# --------------------------------------------------------------------------- #
# Box-counting dimension, on the GPU
# --------------------------------------------------------------------------- #


def deterministic_net(ifs: IFS, eps: float, diameter: float = 11.1,
                      device=None, dtype=torch.float32,
                      max_live: int = 20_000_000, max_depth: int = 400):
    """A probability-free eps-net of the attractor, by adaptive expansion.

    The chaos game needs probabilities, and the probabilities decide where the
    samples land.  That makes it a poor instrument for answering "is the
    *support* the same?".  This routine removes them entirely.

    Start from one point and apply *all* K maps at every level, carrying for
    each node the product of Lipschitz constants along its word,
    r = prod_j sigma_1(A_{i_j}).  That node is within  r * diameter  of the
    attractor, so once  r * diameter < eps  it is retired: it is already a
    valid representative at resolution eps, and expanding it further would only
    add detail finer than one pixel.  Live nodes keep splitting.

    The output covers the attractor to within eps *by construction* -- a
    genuine cover, not a random draw -- so box counting it at scales >= eps is
    unbiased with respect to the choice of p.  Naive full expansion to the same
    depth would need K^d nodes (4^46 for the fern at pixel resolution);
    adaptive pruning needs only O(eps^-D), about 10^6 here.

    Every level is one batched einsum over all live nodes at once.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A, b, _ = ifs.tensors(device, dtype)
    sig = torch.tensor(ifs.lipschitz(), device=device, dtype=dtype)

    live = torch.zeros((1, 2), device=device, dtype=dtype)
    r = torch.full((1,), diameter, device=device, dtype=dtype)
    retired, depth, peak = [], 0, 1

    while live.shape[0] and depth < max_depth:
        depth += 1
        # expand: (K, M, 2) -> (K*M, 2); one einsum covers every map at once
        nxt = (torch.einsum("kij,mj->kmi", A, live)
               + b[:, None, :]).reshape(-1, 2)
        nr = (r[None, :] * sig[:, None]).reshape(-1)

        done = nr < eps
        if bool(done.any()):
            retired.append(nxt[done].clone())
        live, r = nxt[~done], nr[~done]
        peak = max(peak, live.shape[0])
        if live.shape[0] > max_live:
            raise MemoryError(f"live set hit {live.shape[0]:,} at depth "
                              f"{depth}; increase eps (now {eps:g})")

    pts = torch.cat(retired) if retired else live
    return pts, {"depth": depth, "eps": eps, "peak_live": peak,
                 "n_points": int(pts.shape[0])}


def box_count(points, n_grid: int = 8192, bounds=None, device=None,
              factor: int = 2):
    """Box counts N(eps) for eps = side/n_grid * factor^j, j = 0, 1, 2, ...

    Implementation note (this is the part that is *not* a for-loop):
    rasterise once into an (n_grid, n_grid) boolean occupancy grid, then get
    every coarser scale for free by reshaping into blocks and reducing with
    `any` over the block axes.  Coarsening by `factor` is a single tensor op,
    so the whole multi-scale sweep costs about as much as one pass.

    A *square* grid is used deliberately: the box must have the same side in x
    and y or "eps" is not well defined.

    `factor` matters more than it looks.  A set built from thirds (the Cantor
    dust) sampled on a grid of halves is measured at scales that never line up
    with its own construction, and the estimate is biased high by several
    percent.  Setting factor = 3 (with n_grid a power of 3) removes that
    mismatch.  See `part3_figures.py::fig_boxcount` for the demonstration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # `points` may already be a tensor on the compute device -- going through
    # np.asarray() would raise "can't convert cuda:0 device type tensor to
    # numpy", and would round-trip through host memory even when it works.
    if torch.is_tensor(points):
        pts = points.to(device=device, dtype=torch.float64)
    else:
        pts = torch.as_tensor(np.asarray(points), device=device,
                              dtype=torch.float64)

    if bounds is None:
        lo = pts.min(0).values
        hi = pts.max(0).values
    else:
        lo = torch.tensor(bounds[:2], device=device, dtype=torch.float64)
        hi = torch.tensor(bounds[2:], device=device, dtype=torch.float64)
    side = float((hi - lo).max()) * 1.000001  # square, and strictly covering

    ij = ((pts - lo) / side * n_grid).long().clamp_(0, n_grid - 1)
    occ = torch.zeros(n_grid * n_grid, device=device, dtype=torch.bool)
    occ[ij[:, 1] * n_grid + ij[:, 0]] = True
    occ = occ.reshape(n_grid, n_grid)

    eps, counts = [], []
    g, b = n_grid, 1
    while g >= factor:
        eps.append(side * b / n_grid)
        counts.append(int(occ.sum()))
        occ = occ.reshape(g // factor, factor,
                          g // factor, factor).any(3).any(1)
        g //= factor
        b *= factor
    eps.append(side * b / n_grid)
    counts.append(int(occ.sum()))
    return np.array(eps), np.array(counts, dtype=np.int64)


def fit_dimension(eps, counts, n_points, n_grid, lo_frac=0.01, hi_frac=0.25):
    """Least-squares slope of log N vs log(1/eps), over the *valid* window.

    Both ends of the curve are systematically wrong and must be excluded:

      * coarse end -- once every box is occupied, N(eps) saturates at
        (side/eps)^2 and the local slope is forced to 2, whatever the set is.
      * fine end -- with a finite sample of P points, N(eps) can never exceed
        P; boxes eventually isolate individual points and the slope collapses
        towards 0.  This is a *sampling* artefact, not a property of the set.

    We keep the window where  N < hi_frac * (side/eps)^2  (not saturated) and
    N < lo_frac * P (far from the sample-size ceiling), and report R^2 so the
    quality of the fit can be judged rather than assumed.
    """
    total_boxes = (eps.max() / eps) ** 2
    ok = (counts > 8) & (counts < hi_frac * total_boxes) & \
         (counts < lo_frac * n_points)
    if ok.sum() < 3:  # fall back: drop two decades at each end
        ok = np.zeros_like(counts, dtype=bool)
        ok[2:-2] = True

    x = np.log(1.0 / eps[ok])
    y = np.log(counts[ok])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    lo_eps, hi_eps = float(eps[ok].min()), float(eps[ok].max())
    return {"dimension": float(slope), "intercept": float(intercept),
            "r2": float(r2), "mask": ok, "n_used": int(ok.sum()),
            "eps_range": (lo_eps, hi_eps),
            "decades": float(np.log10(hi_eps / lo_eps)),
            "pts_per_box_min": float(n_points / counts[ok].max())}


def measure_dimension(ifs, n_points=2_000_000, n_iter=60, n_grid=4096,
                      device=None, dtype=torch.float32, seed=7, factor=2):
    """End-to-end: sample the attractor, box-count it, fit, compare to theory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    A, b, cdf = ifs.tensors(device, dtype)
    pts = torch.rand((n_points, 2), generator=gen, device=device, dtype=dtype)
    for _ in range(n_iter):
        u = torch.rand(n_points, generator=gen, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, u).clamp_(max=ifs.K - 1)
        pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]

    eps, counts = box_count(pts, n_grid=n_grid, device=device, factor=factor)
    fit = fit_dimension(eps, counts, n_points=n_points, n_grid=n_grid)
    exact = ifs.exact_dim if ifs.exact_dim is not None else ifs.moran_dimension()
    fit.update({"eps": eps, "counts": counts, "exact": exact, "factor": factor,
                "name": ifs.name, "n_points": n_points, "n_grid": n_grid})
    if exact is not None:
        fit["abs_error"] = abs(fit["dimension"] - exact)
        fit["rel_error"] = fit["abs_error"] / exact
    return fit


# --------------------------------------------------------------------------- #
# Deterministic IFS (Hutchinson operator) --- the same attractor, no randomness
# --------------------------------------------------------------------------- #


def hutchinson(ifs: IFS, img: torch.Tensor, steps: int, bounds, device=None):
    """Apply the Hutchinson operator  W(S) = union_i f_i(S)  to a binary image.

    Implemented as a *pull-back*: a destination pixel p belongs to f_i(S) iff
    f_i^{-1}(p) lies in S.  So one step is K calls to `grid_sample` followed by
    an element-wise max over the K results -- again, pure batched tensor work.

    This requires every A_i to be invertible.  The fern's stem map f1 has
    det = 0 (it projects the plane onto a segment), so the pull-back form does
    not apply to the fern -- which is precisely why the chaos game is the more
    robust of the two algorithms.  Demonstrated here on Sierpinski instead.

    Returns the sequence of images, illustrating Banach's theorem: *any*
    starting image converges to the same attractor.
    """
    import torch.nn.functional as F

    if device is None:
        device = img.device
    x0, x1, y0, y1 = bounds
    H, W = img.shape
    Ainv = torch.tensor(np.linalg.inv(ifs.A), device=device, dtype=torch.float32)
    bt = torch.tensor(ifs.b, device=device, dtype=torch.float32)

    # world coordinates of every destination pixel
    ys = torch.linspace(y1, y0, H, device=device)
    xs = torch.linspace(x0, x1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    world = torch.stack([gx, gy], dim=-1).reshape(-1, 2)          # (H*W, 2)

    # source = f_i^{-1}(dest) = A_i^{-1} (dest - b_i), for all i at once.
    # -> (K, H*W, 2), computed in a single batched einsum.
    shift = torch.einsum("kij,kj->ki", Ainv, bt)                  # (K, 2)
    src = torch.einsum("kij,nj->kni", Ainv, world) - shift[:, None, :]

    # normalise to grid_sample's [-1, 1] convention
    gxn = (src[..., 0] - x0) / (x1 - x0) * 2 - 1
    gyn = (y1 - src[..., 1]) / (y1 - y0) * 2 - 1
    grid = torch.stack([gxn, gyn], dim=-1).reshape(ifs.K, H, W, 2)

    frames = [img.clone()]
    cur = img
    for _ in range(steps):
        batch = cur[None, None].expand(ifs.K, 1, H, W)
        warped = F.grid_sample(batch, grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)
        cur = warped.squeeze(1).amax(dim=0)
        frames.append(cur.clone())
    return frames


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
