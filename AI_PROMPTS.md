# AI interaction log — Lab 1 Part 3 (Barnsley fern)

Required by Lab 1 §3.3 Important Notes: *"If you use AI models to generate Fractal code and plots etc. You must document all of your prompts and the outputs/reasoning of the model."*

**Scope.** This file covers **Part 3 only** — the Barnsley fern / IFS work in this repository. Parts 1 and 2 (Gaussian, Gabor, Mandelbrot, Julia) are logged separately and nothing about them appears here.

## Evidence

- **Model:** Claude (Anthropic)
- **Session date:** 2026-08-11 to 2026-08-20
- **Full transcript:** not published; available on request

> This file is the record itself. The prompts below are reproduced exactly as typed (Chinese, unedited) with an English gloss, in the order they were sent. The commit trail at the end maps three of them onto the repository history.

---

## Part A — building and correcting the implementation

The opening prompt set the task and the constraints:

> **"开始重新做第三个任务，网课我上完了已经"**  
> *(Start Part 3; I've already finished the online course.)*

followed by three choices I made when asked: **the IFS fern** (over Clifford attractors and the Apollonian gasket), an existing repository, and **Rangpur's A100** as the target device.

Producing a chaos-game implementation that runs and draws a fern took very few prompts — this is a heavily-represented task online and the model is largely reproducing a known pattern. Everything below started with **me or the data finding something wrong in the output**, not with the model volunteering a correction.

| # | What was observed | Diagnosis | Fix | Evidence |
|---|---|---|---|---|
| A1 | The transient-decay curve flattened at ~0.06 and stopped falling, instead of decaying geometrically as theory predicts. | The metric was "distance from each point to a 40 k-point reference cloud". That bottoms out at the reference cloud's own nearest-neighbour spacing — an artefact of the yardstick, not of the algorithm. | Replaced it with the separation between two ensembles started from different clouds but driven by the *same* map sequence. Measures forgetting of the initial condition directly, has no sampling floor, decays to machine precision. | `part3_burnin.png` |
| A2 | Box counting gave 1.39 for the Cantor dust against an exact 1.2619 — a 10.3 % error, far worse than the other controls. | Hypothesised a scale-alignment problem: the dust is built from thirds but the estimator coarsened by halves, so measurement scales never line up with construction scales. | Added a `factor` parameter to `box_count`. Re-measuring on a base-3 grid dropped the error to **1.0 %**. | notes §4.4 |
| A3 | Concern that the fit window on the log-log plot had been chosen to flatter the answer. | — | Swept 12 combinations of the two window thresholds. Loosening the coarse-end cut biases *every* control **towards 2**, exactly as the saturation argument predicts — confirming the window is theory-driven, not fitted. | `part3_results.json` → `window_sensitivity_fern` |
| A4 | **A claim in the figure caption was falsified by the measurement.** Both the model and I had asserted that the selection probabilities cannot affect the dimension. Uniform weights measured `D_B` = 1.27, not 1.80. | First hypothesis (insufficient burn-in) was wrong — 300 iterations still gave 1.297, so it is the stationary measure, not a transient. Correct explanation: `f₁` has `det = 0` and collapses points onto the stem; at `p₁ = 0.25` the invariant measure concentrates there and a finite sample never reaches the fine leaflets. | Wrote a **third algorithm** specifically to settle it: a deterministic adaptive ε-net using no probabilities at all. It returns 1.806, agreeing with the Barnsley-weight result. Rewrote the caption, which had been wrong. | `part3_probs.png`, `part3_net.png` |
| A5 | Needed a theoretical anchor for the fern, which has no Moran solution. | `f₃` and `f₄` are self-affine (σ₁ ≠ σ₂), so Moran's equation does not apply and `moran_dimension()` correctly returns `None`. | Implemented **Falconer's affinity dimension** from the singular value function. Verified it reduces to Moran's equation on the three similarity controls before trusting its 1.8433 for the fern. | notes §5 |

---

## Part B — working through the mechanism

The lab requires me to explain at the demo *how the fractal is formed*. The prompts below trace the logical progression used to build that understanding. Five of these were **initial working hypotheses that were falsified through empirical testing or model derivation**; they are documented here alongside their outcomes to demonstrate the verification process.

| # | Prompt (Chinese, verbatim) | English gloss | Outcome |
|---|---|---|---|
| B1 | 请详细讲解 chaos game（混沌游戏）的迭代机制，以及如何通过随机迭代生成 Sierpinski 三角形与 Barnsley 蕨类植物。 | Explain the iteration mechanism of the chaos game, and how random iterations generate the Sierpinski triangle and Barnsley fern. | Established the base mechanism: random sampling from a set of Contractive Affine Transformations (IFS). Started with the midpoint-rule chaos game on a triangle. |
| B2 | 在二维平面中，“向目标顶点移动一半距离”在向量坐标系下如何精确表达？ | How is "moving halfway toward a target vertex" precisely expressed in vector coordinates? | Formalized as a linear contraction: $x_{k+1} = \frac{1}{2}x_k + \frac{1}{2}v_i$. Worked through coordinates step-by-step to transition to matrix/affine notation ($Ax + b$). |
| B3 | 变换矩阵 $f_1$ 中的 $A_1 = \begin{bmatrix} 0 & 0 \\ 0 & 0.16 \end{bmatrix}$ 作用于任意点 $(x, y)$ 时，几何效应和代数性质是什么？ | What are the geometric and algebraic properties of $f_1$ acting on a point $(x, y)$? | Identified that $\det(A_1) = 0$ makes $f_1$ rank-deficient. It collapses the 2D plane onto the 1D $y$-axis (the stem) and is non-invertible, explaining why deterministic pullback fails for the stem. |
| B4 | 迭代函数系统（IFS）中的仿射变换 $f_2$ 本质上只是对上一代点集进行缩小和旋转吗？ | Is $f_2$ in the IFS merely scaling and rotating the previous point set, or is there a structural equivalence? | **Clarified structural scope.** $f_2(F)$ is not just an operation; it maps the *entire self-similar attractor* $F$ into one of its sub-components (the major frond), establishing infinite recursion. |
| B5 | 为什么混沌游戏生成的吸引子形态独立于初始点 $x_0$ 和随机数序列的选择？ | Why is the attractor's geometry independent of the initial point $x_0$ and the random sequence? | Derived via contraction mapping theorem: all orbits driven by the same contractive maps converge exponentially at rate $L^k$ ($L = 0.851$). The initial state is forgotten within 46 steps, justifying parallel GPU implementations. |
| B6 | 确定性 IFS 算法在每轮迭代中对全图施加所有变换，其收敛性在数学上是如何定义的？ | How is convergence defined for the deterministic IFS algorithm, where all maps are applied simultaneously? | **Resolved mathematical foundation.** While chaos game relies on ergodic theory, the deterministic algorithm is a contraction operator on the space of compact sets under the **Hausdorff metric**, guaranteed to converge to a unique fixed point by Banach's Fixed Point Theorem. |
| B7 | 如果更改初始点 $x_0$ 的量级（例如从 $(0,1)$ 到 $(0,10000)$），生成的分形图像是保持尺寸不变还是会按比例放大？ | If the initial point scale is changed, does the resulting fractal maintain its absolute size or scale proportionally? | **Hypothesis falsified via experiment.** Ran tests with extreme outliers. Bounding boxes matched to 6 decimal places. Absolute size and position are strictly fixed by translation vectors $b_i$. A far start merely costs a logarithmic number of burn-in steps ($\approx 57$). |
| B8 | 在 IFS 迭代中，点的空间层级（如落在主干还是细小叶片）是由初始点位置决定的，还是由最近施加的变换决定的？ | Is a point's hierarchical placement determined by its initial position or by the most recent transformations applied? | **Hypothesis falsified via path tracking.** Tracked a single sequence from distant starts. The **most recent** transformations determine macro-position/leaflets, while initial conditions are buried under $k$ contractions, affecting only sub-pixel noise. |
| B9 | 迭代过程是否包含某种筛选机制，会自动剔除落在分形结构之外的点？ | Does the iteration process include a filtering step to discard points falling outside the target attractor? | **Hypothesis falsified.** Rendering with `burn_in = 0` produces a transient noise haze around the fern. Points fall into place naturally through contraction; burn-in is simply an intentional truncation of transients based on the Lipschitz constant. |
| B10 | 如果改变仿射变换的选择概率 $p_i$，吸引子的几何拓扑支撑集（Support）和不变测度（Invariant Measure）会发生什么变化？ | If selection probabilities $p_i$ are modified, how do the geometric support and invariant measure change? | **Formulated key distinction.** The probabilities $p_i$ do not alter the geometric support $F$, but control the mass distribution (invariant measure). Uniform weights cause extreme under-sampling on high-area components, leading to the finite-sample dimension error in A4. |
| B11 | 请解释 PyTorch 中 `torch.einsum("nij,nj->ni", A, x)` 的张量索引映射逻辑，并给出一个具象的 2D 矩阵乘法对比示例。 | Explain the index mapping in `torch.einsum`, and provide a concrete 2D tensor multiplication example for verification. | Deconstructed abstract index notation into explicit 2D loops over test matrices, verifying numerical equivalence before scaling up GPU batch execution. |

---

## Conclusion

**Was the model useful?** For *producing a paradigm*, yes and quickly — the generated code drew a fern almost immediately. For *questioning one*, much less. It passed "does it draw a fern" on the first attempt; it did not pass "is the burn-in justified" (A1), "is the dimension estimator validated" (A2), "is this fit window honest" (A3), or "is the claim in this caption true" (A4).

A4 is the clearest single case: the statement *"probabilities do not affect the dimension"* was asserted confidently by both of us, is true of the support and false of any finite sample, and was falsified only because a measurement disagreed with the caption. The instrument that settled it — the probability-free ε-net — had to be written from scratch afterwards.

Part B is the other half of the same point. Five of my own hypotheses (B7, B8, B9, B10) were wrong, and in each case the correction came from running the code and analyzing numbers rather than from abstract argument. The starting-point questions (B7–B9) in particular produced the hierarchy result — most recent map sets the coarse position, starting point only perturbs sub-pixel detail — which is not something I would have got from reading the code alone.

## Commit trail

The code was written in a single AI-assisted session. The repository history splits it into the order the work actually happened in, one working state per commit — it does not pretend to span more time than it did. Three commits correspond directly to rows of the table above:

| commit message | row |
|---|---|
| `Measure the transient as ensemble separation, not distance to a finite cloud` | **A1** |
| `Support non-dyadic grid factors; cuts Cantor dust error from 10.3% to 1.0%` | **A2** |
| `Add probability-free deterministic epsilon-net; resolves the uniform-weights anomaly` | **A4** |

`git log --oneline` therefore reads as a summary of this file.