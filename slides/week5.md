---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week5"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week5.html"
---

# Calculus and Optimisation for Machine Learning

## Week 5: Gradient descent

Step size, stochastic gradients, momentum and reading loss curves

#### [week5#title] Gradient descent

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 5: Step size, stochastic gradients, momentum and reading loss curves

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week5#agenda] Gradient descent

*This week*

- **Session 1 · Descent and loss curves**: Gradient descent steps against the gradient; the step size, the batch and momentum decide whether a run converges, crawls or diverges

#### [week5#review-week4] What does a matrix–vector node send back to x?

*Review of week 4*

**Question:** A node computes y = A x with A of shape 4 × 6, and receives the upstream gradient g = ∂L/∂y. What does its backward pass add to x's gradient?

- A. A g
- B. g xᵀ
- C. The 4 × 6 Jacobian A
- D. Aᵀ g

#### [week5#lab-4-1-debrief] The engine agreed with the framework to about 1e-13

*Lab 4.1 · what your runs showed*

| Reference engine, solution run | Result |
| --- | --- |
| Test suite | 10 of 10 |
| Against central differences, 50 random expressions, worst | 3e-9 |
| Against PyTorch, Lab 3.1 loss | 6e-16 |
| Against PyTorch, ten random expressions, worst | 3e-13 |

- **One = instead of +=.** Overwriting in multiplication broke 6 of the 10 tests: every graph where a node reaches the output along more than one path.

### Session 1: Descent and loss curves

#### [week5#s1] Descent and loss curves

*Session 1*

- Apply batch, mini-batch stochastic and momentum gradient descent
- Describe what adaptive step methods, warm-up and decay change
- Diagnose divergence, oscillation and slow convergence from a loss curve

#### [week5#descent-step] Step against the gradient, by a rate you choose

*Block 1 · Gradient descent*

- **The update.** `w ← w − η ∇L(w)`

From week 2: −∇L is the steepest way down, and the local line promises a fall of about `η |∇L|²`.

The promise holds only while the step is small enough for the line to be right. The rate η decides that.

- **What goes wrong.** A rate that is too small wastes thousands of steps; one that is too large overshoots the valley and climbs out of it. Neither shows in the code: both show in the loss curve.

#### [week5#stable-rate] On a quadratic, each direction shrinks by |1 − η λ| per step

*Block 1 · How large is too large*

- For `L = ½ Σ λᵢ uᵢ²`, one step multiplies each `uᵢ` by `1 − η λᵢ`.
- **Stable** when every |1 − η λᵢ| < 1, that is **η < 2 / λ_max**.
- **Slow** when η λ_min is tiny: the flattest direction barely moves.
- Near the edge, 1 − η λ_max is close to −1: the steep direction flips sign every step.

Figure: 0.5 (u² + 10 v²), λ_max = 10, edge at η = 0.2; labs/_tools/make_figures.py.

#### [week5#check-stable-rate] Which rates will the analyst see diverge?

*Checkpoint 1*

**Question:** A least-squares loss has curvatures (eigenvalues of AᵀA / n) from 0.05 to 4. The analyst sweeps the rate over 0.01, 0.1, 0.4 and 0.6 with full-batch gradient descent. Which runs diverge?

- A. 0.1, 0.4 and 0.6
- B. 0.4 and 0.6
- C. 0.6 only
- D. None of them

#### [week5#stochastic-gradients] A mini-batch gradient is cheap, right on average, and noisy

*Block 2 · Stochastic and mini-batch gradients*

- The loss is a mean over n rows, so its gradient is a mean too.
- A random batch of b rows gives an **unbiased** estimate at **b / n** of the cost.
- Many cheap steps per pass over the data (an **epoch**), instead of one exact step.
- The noise does not vanish at the minimum: with a fixed rate, the weights keep moving.

- **Batch size as a dial.** b = 1 is stochastic gradient descent in its first form; b = n is full batch. Larger b: less noise per step, more cost per step. The lab uses b = 32.

#### [week5#momentum-path] Momentum keeps a velocity, so the flat direction speeds up

*Block 2 · Momentum*

- **Heavy ball.** `v ← β v + ∇L(w)`
  `w ← w − η v`

Steps that agree add up, to about `η / (1 − β)` in a steady slope; steps that flip sign cancel. With β = 0.9 it averages about the last ten gradients.

Figure: Same quadratic, same rate 0.04, 25 steps: loss 6.5 without momentum, 0.095 with β = 0.8.

#### [week5#adaptive-and-schedules] Adaptive methods scale each coordinate; schedules change the rate over time

*Block 2 · Adaptive steps and schedules*

- **Adaptive step (Adam, for example).** Keep running averages of g and g². Step each coordinate by `η · m̂ / (√v̂ + ε)`: about η per coordinate whatever the gradient's scale. Common defaults: β₁ = 0.9, β₂ = 0.999.
- **Warm-up, then decay.** **Warm-up**: start small for the first steps, when gradients are large and erratic. **Decay**: shrink the rate later, to narrow the noise band. Step, cosine and linear decays are all in use.

> **Awareness level.** Which adaptive optimiser is in common use changes; AdamW, scaling rules and newer methods are SPC-DLR-302's subject.

#### [week5#check-momentum] What does momentum change here?

*Checkpoint 2*

**Question:** Plain gradient descent at its largest stable rate zigzags across a narrow valley and crawls along it. The analyst adds momentum with β = 0.9 and keeps the rate. What should she expect first?

- A. Faster progress along the valley, possibly with overshoot; retune the rate
- B. Faster progress along the valley, and nothing else
- C. A lower final loss than the least-squares solution
- D. No change unless the batch size changes

#### [week5#reading-loss-curves] Each failure has a shape you can name from the curve

*Block 3 · Reading loss curves*

| The curve | Diagnosis | Likely cause | First fix |
| --- | --- | --- | --- |
| Grows, or turns into NaN | diverged | η above 2 / λ_max | divide η by 3 to 10 |
| Falls fast, then crawls far above the best | slow | η too small for λ_min; unscaled features | scale features; raise η; momentum |
| Falls, then a wide band that never narrows | noisy | small batch at a large constant η | decay the rate; larger batch |
| Saw-tooth on the way down | near the edge | η close to 2 / λ_max | lower η a little |
| Smooth fall to the reference | converged |  | keep it, and log it |

#### [week5#log-every-run] Log every run, not only the best one

*Block 3 · Reporting*

```
# file: out/experiments.csv (shape)
variant,learning_rate,seed,epochs_run,final_loss,diagnosis
batch,0.5,0,200,...,converged
batch,2.0,0,57,inf,diverged
```

- A sweep that reports only its best rate hides how fragile the method is.
- Every run gets its rate, its seed, its result and a diagnosis in words.
- The capstone is scored on this: every run, seed and failure, not the best result alone.

#### [week5#check-diagnose-curve] Name this run

*Checkpoint 3*

**Question:** On a dataset of household energy use, a run on raw, unscaled features falls from 900 to 520 in ten epochs, then sits near 500 for 190 more. The least-squares loss is 3. Any larger rate diverges. What is the diagnosis?

- A. Converged
- B. Slow: the rate fits the steepest direction and starves the flattest
- C. Noisy: use a larger batch
- D. Diverged: lower the rate

#### [week5#lab-5-1] Learning-rate sweeps

Lab 5.1 · 105 min

Learning-rate sweeps for three gradient descent variants on public regression data

[W5_1_lr_sweeps.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W5_1_lr_sweeps.ipynb)

- [ ] Load the public data, split it and find the reference loss (15 min)
- [ ] Sweep the learning rate for batch gradient descent (20 min)
- [ ] Sweep it for mini-batch stochastic gradient descent (15 min)
- [ ] Sweep it for momentum (15 min)
- [ ] Run on unscaled features and diagnose (15 min)
- [ ] Write the experiment log with a diagnosis per run (15 min)
- [ ] Add warm-up and step decay to the noisy run (10 min)
- [ ] ✱ Stretch: Add an adaptive step method and compare its sweep

#### [week5#problem-set-5] Problem set 5: descent, a stable rate, momentum and four diagnoses

*Before week 6*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS5.py`.
- **AI-off for the first attempt.**
- The checker runs your functions on its own small problems.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS5
```

#### [week5#wrap-up] Three things to keep

*Week 5 · wrap-up*

- **2 / λ_max.** The steepest curvature sets the largest stable rate; the flattest sets how slow it is.
- **Noise and velocity.** Mini-batches trade exactness for cheap steps; momentum averages steps; schedules shrink the noise band.
- **Curves are evidence.** Diagnose from the loss curve on a log scale, and log every run with its rate and seed.

#### [week5#next] Week 6: The shape of a loss

*Next*

Hand in Problem set 5. Keep your experiment log: week 6 explains its curvatures with the Hessian.
