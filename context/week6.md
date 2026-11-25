---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week6"
day: 6
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week6.html"
---

# Calculus and Optimisation for Machine Learning

## Week 6: The shape of a loss

Convexity, saddle points, curvature and the largest stable step

#### [week6#title] The shape of a loss

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 6: Convexity, saddle points, curvature and the largest stable step

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week6#agenda] The shape of a loss

*This week*

- **Session 1 · Convexity and curvature**: The Hessian tells a minimum from a saddle and bounds the step size; convex losses have no traps, non-convex ones do

#### [week6#review-week5] Which change fixes this run?

*Review of week 5*

**Question:** Mini-batch SGD at a constant rate reaches the reference loss quickly, then keeps jumping in a band 30% wide for the rest of training. Which single change narrows the band?

- A. Double the learning rate
- B. Train for twice as many epochs at the same rate
- C. Add more features
- D. Decay the rate after the loss reaches the band

#### [week6#lab-5-1-debrief] Scaling the features cut the curvature ratio from 2.7e11 to 12

*Lab 5.1 · what your runs showed*

| Airfoil data, reference run | Diagnosis |
| --- | --- |
| Batch, rate 0.01: loss 152 after 200 epochs | slow |
| Batch, rate 1.0, above the edge 0.948 | diverged |
| SGD, rate 0.3: last 20 epochs spread 2.95 | noisy |
| Momentum, rate 0.001: loss 125 | slow |
| Unscaled, rate 1e-7: loss 4,193; 1e-6 diverged at epoch 123 | slow, diverged |

- **Reference loss 11.31.** Least squares on 1,200 training rows; the yardstick every run was judged against.
- **A schedule helped.** Warm-up and step decay cut the noisy run's spread from 2.95 to 0.014.

### Session 1: Convexity and curvature

#### [week6#s1] Convexity and curvature

*Session 1*

- Distinguish convex from non-convex objectives with a test you can run
- Classify critical points from the eigenvalues of the Hessian
- Relate the largest stable step to the largest curvature

#### [week6#convex-definition] A convex function lies below every chord, so it has no traps

*Block 1 · Convexity*

- **Definition.** `f(t a + (1 − t) b) ≤ t f(a) + (1 − t) f(b)` for all a, b and t in [0, 1]

Twice differentiable: convex exactly when `f″ ≥ 0` everywhere, or in several variables when the Hessian has no negative eigenvalue anywhere.

- **Why it matters.** On a convex loss every local minimum is a global minimum, and a small enough step of gradient descent reaches it. Least squares and logistic regression are convex in their weights. A network's loss is not.

#### [week6#minima-and-saddles] Non-convex losses have local minima, saddles and plateaus

*Block 1 · What non-convex looks like*

- **Local minimum.** Lower than its neighbours, higher than the best point. Descent stops there.
- **Saddle point.** Gradient zero, curving up one way and down another. In many dimensions, saddles far outnumber minima.
- **Plateau.** A wide region where the gradient is close to zero. Steps become tiny; the run looks converged and is not.

> **What goes wrong.** Reading "the loss stopped falling" as "we found the minimum". A saddle or a plateau stops the loss too.

#### [week6#check-convex] What does convexity guarantee here?

*Checkpoint 1*

**Question:** The charity's analyst fits a logistic regression twice, from two different starting weights, with a small enough step. Both runs stop. What can she conclude about the two end points?

- A. They may be different local minima
- B. Both runs reach the same lowest loss
- C. Any step size gives the same end point
- D. One of them may be a saddle point

#### [week6#hessian-curvatures] The Hessian's eigenvalues are the curvatures along its eigenvectors

*Block 2 · Curvature*

The Hessian **H** holds the second partial derivatives: `Hᵢⱼ = ∂²f / ∂wᵢ∂wⱼ`. It is symmetric.

- **Second-order Taylor.** `f(w + d) ≈ f(w) + ∇f · d + ½ dᵀ H d`

Along an eigenvector with eigenvalue λ, the function curves like `½ λ t²`.

| At a critical point, eigenvalues | Kind |
| --- | --- |
| all positive | minimum |
| all negative | maximum |
| both signs | saddle |
| some zero | the test cannot tell |

#### [week6#saddle-escape] Exactly on a saddle's ridge, descent stops; a nudge lets it escape

*Block 2 · Saddles in practice*

Figure: f = ½x² − ½y² + 0.1y⁴, step 0.2, 120 steps; labs/_tools/make_figures.py.

On the axis the y-component of the gradient is exactly 0: descent walks into the saddle at (0, 0).

From y = 0.01 it ends at (0, 1.581), a minimum (√2.5).

> **Noise helps.** Mini-batch gradients rarely sit exactly on a ridge.

#### [week6#check-hessian-kind] What kind of point is this?

*Checkpoint 2*

**Question:** At a point where the gradient of a two-weight loss is zero, the Hessian is [[3, 0], [0, −0.2]]. The run has stopped there. What should the analyst say?

- A. It is a minimum, since 3 is large and positive
- B. The test cannot tell: −0.2 is too small to count as negative
- C. It is a saddle: lower loss lies along the second weight
- D. It is a maximum

#### [week6#largest-stable-step] The largest curvature bounds the step; the ratio of curvatures sets the pace

*Block 3 · Step size and curvature*

- Near a minimum the loss is about quadratic with Hessian H: week 5's rule holds with λ_max of H. **Step < 2 / λ_max.**
- At the best fixed step, the flattest direction shrinks by about `1 − 2/κ` per step, with **κ = λ_max / λ_min**, the condition number.
- Steps needed grow roughly in proportion to κ.

| Lab 5.1, airfoil data | κ |
| --- | --- |
| Raw features | 2.7 × 10¹¹ |
| Standardised features | 12 |

#### [week6#newton-step] A Newton step divides by the curvature instead of guessing a rate

*Block 3 · Using the curvature*

- **Newton.** Solve `H d = −∇f`, then `w ← w + d`. On a quadratic it lands on the minimum in one step, whatever κ is.
- **Why not always.** H has n² entries: for a million weights, a trillion. At a saddle a Newton step goes towards the saddle, not away. Training uses first-order methods; Newton is the yardstick.

#### [week6#check-condition] What will the sweep show?

*Checkpoint 3*

**Question:** A convex quadratic loss has curvatures 0.5 and 50. The analyst runs plain gradient descent at steps 0.01, 0.039 and 0.041. What does she see?

- A. 0.041 diverges; the other two converge, slowly along the flat direction
- B. All three converge quickly
- C. All three converge, since a convex loss cannot diverge whatever the step
- D. 0.039 and 0.041 diverge

#### [week6#lab-6-1] Six loss surfaces

Lab 6.1 · 105 min

Six loss surfaces with gradient-descent traces, classified and annotated

[W6_1_loss_surfaces.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W6_1_loss_surfaces.ipynb)

- [ ] Plot the six surfaces as contour maps (10 min)
- [ ] Classify critical points from the Hessian (20 min)
- [ ] Test each surface for convexity (15 min)
- [ ] Trace gradient descent below, at and above the stable step (20 min)
- [ ] Start on and off the saddle's ridge (10 min)
- [ ] Annotate each plot against the reference (20 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Take Newton steps on the valley and compare with gradient descent

#### [week6#problem-set-6] Problem set 6: a Hessian, two critical points, an edge and three verdicts

*Before week 7*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS6.py`.
- **AI-off for the first attempt.**
- Convexity verdicts are checked on the whole plane: think before you sample.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS6
```

#### [week6#wrap-up] Three things to keep

*Week 6 · wrap-up*

- **Convex: no traps.** Every local minimum is global. Least squares and logistic regression are convex; networks are not.
- **Hessian signs.** All positive: minimum. Mixed: saddle. Stopping is not proof of a minimum.
- **2 / λ_max and κ.** The steepest curvature bounds the step; the condition number sets how many steps.

#### [week6#next] Week 7: Stable losses and penalties

*Next*

Hand in Problem set 6. Week 7 makes the losses you will train in the capstone safe to compute.
