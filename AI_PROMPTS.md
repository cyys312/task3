# AI interaction log — Lab 1 Part 3 (Barnsley fern)

Required by Lab 1 §3.3 Important Notes: *"If you use AI models to generate
Fractal code and plots etc. You must document all of your prompts and the
outputs/reasoning of the model."*

**Scope.** This file covers **Part 3 only** — the Barnsley fern / IFS work in
this repository. Parts 1 and 2 (Gaussian, Gabor, Mandelbrot, Julia) are logged
separately and nothing about them appears here.

## Evidence

- **Model:** Claude (Anthropic)
- **Session date:** 2026-08-19 to 2026-08-20
- **Full transcript:** not published; available on request

> This file is the record itself. The prompts below are reproduced exactly as
> typed (Chinese, unedited) with an English gloss, in the order they were sent.
> The commit trail at the end maps three of them onto the repository history.

---

## Part A — building and correcting the implementation

The opening prompt set the task and the constraints:

> **"开始重新做第三个任务，网课我上完了已经"**
> *(Start Part 3; I've already finished the online course.)*

followed by three choices I made when asked: **the IFS fern** (over Clifford
attractors and the Apollonian gasket), an existing repository, and **Rangpur's
A100** as the target device.

Producing a chaos-game implementation that runs and draws a fern took very few
prompts — this is a heavily-represented task online and the model is largely
reproducing a known pattern. Everything below started with **me or the data
finding something wrong in the output**, not with the model volunteering a
correction.

| # | What was observed | Diagnosis | Fix | Evidence |
|---|---|---|---|---|
| A1 | The transient-decay curve flattened at ~0.06 and stopped falling, instead of decaying geometrically as theory predicts. | The metric was "distance from each point to a 40 k-point reference cloud". That bottoms out at the reference cloud's own nearest-neighbour spacing — an artefact of the yardstick, not of the algorithm. | Replaced it with the separation between two ensembles started from different clouds but driven by the *same* map sequence. Measures forgetting of the initial condition directly, has no sampling floor, decays to machine precision. | `part3_burnin.png` |
| A2 | Box counting gave 1.39 for the Cantor dust against an exact 1.2619 — a 10.3 % error, far worse than the other controls. | Hypothesised a scale-alignment problem: the dust is built from thirds but the estimator coarsened by halves, so measurement scales never line up with construction scales. | Added a `factor` parameter to `box_count`. Re-measuring on a base-3 grid dropped the error to **1.0 %**. | notes §4.4 |
| A3 | Concern that the fit window on the log-log plot had been chosen to flatter the answer. | — | Swept 12 combinations of the two window thresholds. Loosening the coarse-end cut biases *every* control **towards 2**, exactly as the saturation argument predicts — confirming the window is theory-driven, not fitted. | `part3_results.json` → `window_sensitivity_fern` |
| A4 | **A claim in the figure caption was falsified by the measurement.** Both the model and I had asserted that the selection probabilities cannot affect the dimension. Uniform weights measured `D_B` = 1.27, not 1.80. | First hypothesis (insufficient burn-in) was wrong — 300 iterations still gave 1.297, so it is the stationary measure, not a transient. Correct explanation: `f₁` has `det = 0` and collapses points onto the stem; at `p₁ = 0.25` the invariant measure concentrates there and a finite sample never reaches the fine leaflets. | Wrote a **third algorithm** specifically to settle it: a deterministic adaptive ε-net using no probabilities at all. It returns 1.806, agreeing with the Barnsley-weight result. Rewrote the caption, which had been wrong. | `part3_probs.png`, `part3_net.png` |
| A5 | Needed a theoretical anchor for the fern, which has no Moran solution. | `f₃` and `f₄` are self-affine (σ₁ ≠ σ₂), so Moran's equation does not apply and `moran_dimension()` correctly returns `None`. | Implemented **Falconer's affinity dimension** from the singular value function. Verified it reduces to Moran's equation on the three similarity controls before trusting its 1.8433 for the fern. | notes §5 |

---

## Part B — working through the mechanism

The lab requires me to explain at the demo *how the fractal is formed*. The
prompts below are the sequence in which I built that understanding. Several
were **wrong hypotheses that the model corrected with a measurement** — those
are recorded as they happened, because the corrections are the useful part.

| # | Prompt (verbatim) | English gloss | Outcome |
|---|---|---|---|
| B1 | 因为我是用ai做的吗所以我得做这些，但是你得给我讲解下混沌游戏和这个叶子是什么东西 | Is it because I used AI that I have to do these? Also explain the chaos game and what this fern is. | Confirmed the documentation and analysis requirements are conditional on AI use, but that §3.4.1 requires explaining the formation regardless. Started from the halfway-rule chaos game on a triangle → Sierpinski. |
| B2 | 往他的方向走一半他的方向我没搞明白 | "Move halfway toward it" — I don't understand the direction part. | Resolved: the jump is just the **midpoint** of the current point and the chosen vertex; no angles are involved. Worked through four steps with explicit coordinates. |
| B3 | 2.0625，2.75 | *(my own computation of P₅)* | Correct. Confirmed I could apply the rule myself before generalising to matrix form. |
| B4 | 0，0 | *(my answer for what f₁ does to a point)* | **Half right.** `x' = 0` was correct; `y' = 0.16y`, not 0. Led to the key fact that `f₁` is **rank-deficient** (`det A₁ = 0`), collapses the plane onto a segment, and is **not invertible** — which reappears in A4 and in why the deterministic pull-back algorithm cannot be used on the fern. |
| B5 | 就是缩小加旋转 | It's just scaling plus rotation. | Correct as a description of the *operation*, but missed the *consequence*: `f₂(F)` is not a component of the fern, it is **a complete fern one size down**. That is the source of the infinite recursion. |
| B6 | 为什么 | Why? *(why is the result independent of the starting point and the dice)* | Got the two-part answer: contraction ⟹ any two orbits driven by the same maps converge at rate `L^k` (L = 0.851), so the start is forgotten in 46 steps; and the attractor is the set of all reachable map-sequences, which is fixed. This is also the justification for the parallel implementation. |
| B7 | 有限集合 | A finite set. | **Wrong, and worth recording.** The Sierpinski triangle is infinite (dimension 1.585). Also, the deterministic figure applies *all* maps every step, so there is no sequence argument available. The correct answer is the same contraction argument lifted to **sets** under the **Hausdorff metric** — Banach's fixed point theorem. |
| B8 | 不同点开始，是不是只是行状一样，绝对大小不同，就比如0，1和0，100生成的图像 | Do different starting points give the same shape but different absolute size — e.g. (0,1) vs (0,100)? | **Wrong, and tested.** Ran four starting points including (0, 10000) and (−5000, −5000): the bounding boxes agree **to six decimal places**. Size and position are fixed by the translations `b`, not by the start. Cost of a distant start is logarithmic — 10 000× further costs only 57 extra steps. |
| B9 | 就是其他参数一样，起点小他就在小的叶子上，起点大就在大的叶子上 | Same parameters — a small start lands on a small leaflet, a large start on a large one? | **Wrong, and tested.** Traced one dice sequence from two starts: at step 1 the far start is at y = 8501, i.e. **outside the fern entirely**, not on a large leaflet. Produced the correct hierarchy: the **most recent** map decides which leaflet; the **starting point** is buried under k contractions and only affects sub-pixel detail. My guess had the hierarchy inverted. |
| B10 | 起点收缩到符合f几就去当对应的部分对吧，太大了在最终的图像外不符合f几就不显示 | The start contracts until it matches some fᵢ and becomes that part; if it's too big it falls outside the final image and isn't displayed? | **Wrong on both counts, and tested.** (i) The map is chosen **blind** by `torch.rand` — the point's position is never inspected, there is no matching step. (ii) Nothing is filtered automatically: rendering with `burn_in = 0` **does** draw the early points as a haze around the fern (`burnin_effect.png`). Burn-in is a deliberate discard of 46 steps, computed from the Lipschitz constant. The real reason "last map = f₃ ⟹ left leaflet" holds is the **image** property `f₃(F) = left leaflet`, which requires the point to already be on `F` — so the colouring figure and burn-in are the same fact. |
| B11 | 不会吧，但是这个概率有什么用呢 | It won't [grow the leaflet] — but then what are the probabilities for? | Correct. Led to the **support vs invariant measure** distinction, and to the resolution of A4: the probabilities are **importance sampling** (`pᵢ ∝ |det Aᵢ|`), not part of the fractal; `p₁ = 0.01 ≠ 0` because the stem has zero area but must still be drawn. |
| B12 | 没看懂 | Didn't follow. *(on einsum index notation)* | The abstract notation `"nij,nj->ni"` did not land. Resolved by dropping the notation and running the same computation two ways — an explicit `for` loop and `einsum` — on three 2×2 matrices with printed numbers, verifying they are identical. Generalisation only afterwards. |

---

## Conclusion

**Was the model useful?** For *producing a paradigm*, yes and quickly — the
generated code drew a fern almost immediately. For *questioning one*, much
less. It passed "does it draw a fern" on the first attempt; it did not pass
"is the burn-in justified" (A1), "is the dimension estimator validated" (A2),
"is this fit window honest" (A3), or "is the claim in this caption true" (A4).

A4 is the clearest single case: the statement *"probabilities do not affect
the dimension"* was asserted confidently by both of us, is true of the support
and false of any finite sample, and was falsified only because a measurement
disagreed with the caption. The instrument that settled it — the
probability-free ε-net — had to be written from scratch afterwards.

Part B is the other half of the same point. Five of my own hypotheses (B4, B7,
B8, B9, B10) were wrong, and in each case the correction came from running the
thing and looking at numbers rather than from argument. The starting-point
questions (B8–B10) in particular produced the hierarchy result — most recent
map sets the coarse position, starting point only perturbs sub-pixel detail —
which is not something I would have got from reading the code.

## Commit trail

The code was written in a single AI-assisted session. The repository history
splits it into the order the work actually happened in, one working state per
commit — it does not pretend to span more time than it did. Three commits
correspond directly to rows of the table above:

| commit message | row |
|---|---|
| `Measure the transient as ensemble separation, not distance to a finite cloud` | **A1** |
| `Support non-dyadic grid factors; cuts Cantor dust error from 10.3% to 1.0%` | **A2** |
| `Add probability-free deterministic epsilon-net; resolves the uniform-weights anomaly` | **A4** |

`git log --oneline` therefore reads as a summary of this file.
