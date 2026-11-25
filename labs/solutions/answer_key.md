# Labs answer key

The results of the reference solutions, from a full run (not quick mode) of the notebooks in this
folder. Every number here was printed by a solution notebook or by `labs/_tools/make_reference.py`;
rerun them to check. Lab 5.1 uses the downloaded UCI file: a run on the synthetic stand-in gives
other numbers. Where an answer leads to a decision, the reference keeps the duties separate: the
data analyst proposes and prices options, the finance officer approves the budget, and the
programme lead decides how districts are weighted.

Worked answers to the problem sets and the capstone reference run are not here: the trainer keeps
them outside the repository.

## Lab 1.1 · Three ways to a derivative

- Hand derivatives: p′(0.6) = 0.9975, E′(20) = 2.0, q′(3) = 1.4171; SymPy agrees to about 4e-16.
- Best step: forward difference 6.8e-8 measured (theory √ε ≈ 1.5e-8); central difference 1e-5
  measured (theory ε^(1/3) ≈ 6.1e-6).
- The tangent line stays within 1% up to a change of 5 thousand; the quadratic up to 10.

## Lab 2.1 · Gradients and a Jacobian, checked

- Partial derivatives of completion probability: 0.8556 per unit of attendance share, 0.0642 per
  weekly hour; steepest direction 4° from the attendance axis, slope 0.858.
- Moving 5 thousand from D1 to D4: predicted −4.93 enrolments, actual −5.61; moving 20: predicted
  −19.71, actual −33.45. The linear prediction is a local one.
- The wrong component in last term's notes is the one for w_h (index 1). The analyst reports the
  error and the corrected gradient; retuning the model is approved by the programme lead.

## Lab 3.1 · Backpropagation by hand (AI-off)

- Forward: z = 0.65 and 0.70, p = 0.657 and 0.6682, losses 0.4201 and 1.1032, mean L = 0.7616.
- Backward: ∂L/∂w₁ = −0.0207, ∂L/∂w₂ = −0.0361, ∂L/∂b = 0.1626; agrees with central differences
  to a relative error of about 1e-4 (the hand values are rounded to four decimals).
- Forward mode needs 3 passes for the three parameters; reverse mode needs 1.

## Lab 4.1 · A scalar reverse-mode engine

- The reference engine passes 10 of 10 tests; worst finite-difference relative error about 3e-9
  over 50 random expressions; agrees with PyTorch autograd to about 6e-16 on the lab 3.1 loss and
  3e-13 over ten random expressions.
- Overwriting gradients (`=` instead of `+=`) fails six tests: add_mul, constants_both_sides,
  same_node_twice, random_expressions, two_learner_loss and zero_grad.

## Lab 5.1 · Learning-rate sweeps (Airfoil Self-Noise)

- Reference least-squares loss 11.31 on the training rows (12.39 held out); starting loss 7,813.8.
- Curvatures of the scaled problem 0.174 to 2.109, so the batch edge is 2 / 2.109 = 0.948.
- Diagnoses: batch 0.01 slow, 0.03 to 0.9 converged, 1.0 diverged; SGD 0.003 to 0.1 converged,
  0.3 noisy (spread 2.95 over the last 20 epochs; 0.014 with warm-up and decay); momentum 0.001
  slow, the rest converged; unscaled batch 1e-7 slow (loss 4,193), 1e-6 diverged (epoch 123).
- Curvature ratio: about 2.7e11 unscaled against 12 after standardising.
- Adam (stretch): rate 0.01 still far off (1,532.65), 0.1 near the reference (11.33), 1.0 at 13.36.

## Lab 6.1 · Six loss surfaces

| Surface | Hessian eigenvalues at the critical point | Convex | Trace ended |
|---|---|---|---|
| bowl | 1, 2 | yes | minimum, global |
| narrow | 1, 25 | yes | diverged (step 0.085 > 0.08) |
| valley | 0.388, 51.612 at (1, 1) | no | minimum, global |
| saddle | −2, 2 at the origin; 2, 4 at (0, 1.414) | no | stuck at the saddle |
| bumpy | | no | minimum, not global |
| plateau | | no | did not move (gradient about 3e-7) |

- Narrow after 60 steps: 0.0096 at step 0.05, 0.599 at 0.079, 242 at 0.081.
- From (1, 0) descent stops at (0, 0); from (1, 0.001) it reaches (0, 1.4142).
- Newton (stretch) reaches (1, 1) on the valley in 7 steps.

## Lab 7.1 · Stable softmax and a penalty

- Naive cross-entropy on the planted rows: nan, nan and inf. Stable mean cross-entropy: 22.00.
- Gradient check of P − Y: 9e-10 on the logits, 4e-10 on the weights.
- Plain training, 2,000 steps: weight norm 3.70 at step 50 and 16.22 at the end, still growing;
  held-out loss 0.071, accuracy 1.000.

| λ | ‖W‖ | training loss | held-out loss | held-out accuracy | held-out loss, re-test |
|---|---|---|---|---|---|
| 0 | 16.216 | 0.0940 | 0.0712 | 1.000 | 0.7000 |
| 0.001 | 11.099 | 0.1357 | 0.1126 | 1.000 | 0.5411 |
| 0.01 | 4.904 | 0.2872 | 0.2730 | 0.980 | 0.4621 |
| 0.1 | 1.701 | 0.5824 | 0.5856 | 0.880 | 0.6530 |
| 1 | 0.391 | 0.9223 | 0.9265 | 0.787 | 0.9425 |

- On the noise-free first results no penalty improved the held-out loss (lowest at λ = 0). On the
  re-test, where 10 of the 150 held-out learners land one level away, λ = 0.01 gives the lowest
  held-out loss (0.462 against 0.700 without a penalty). Either way the penalty bounded the
  weights. Smallest Hessian eigenvalue (stretch): about −1e-12 plain, 0.0100 with λ = 0.01.

## Lab 8.1 · A budget allocation two ways

- Two districts with 40 thousand: x₁ = 23.214, x₂ = 16.786, λ = 1.8065 enrolments per thousand.
- Five districts, 120 thousand: D1 46.571, D2 37.029, D3 34.429, D4 1.971, D5 0; 254.46
  enrolments; λ = 1.0606. One more thousand adds 1.0574.
- Projected gradient ascent reaches the same allocation within 1e-3 after 431 steps at step 1, 83 at
  5 and 17 at 20; at step 50 it never settles (gap 3.8 after 500 steps).
- Floor of 10 thousand: D1 39.682, D2 31.057, D3 29.261, D4 10, D5 10; total enrolments 254.46 →
  246.72; enrolments of learners furthest behind 56.11 → 57.97; λ = 1.2077; the floor costs 0.488
  enrolments per thousand in D4 and 0.741 in D5.
- The recommendation states both options with these figures and keeps the roles apart: the data
  analyst proposes and prices, the finance officer approves the budget and checks the arithmetic,
  the programme lead decides the weight given to districts furthest behind.
- Need-weighted allocation (stretch): 29.67, 30.02, 36.62, 15.41, 8.28; λ = 0.8103 in weighted
  units.
