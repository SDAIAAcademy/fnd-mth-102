# Pre-work · A school calculus refresher (80 minutes)

For anyone whose school calculus is rusty. Do it before week 1, on paper, without an assistant.
It is not graded; the answers are at the end so you can check yourself. If you finish part A in
under 15 minutes with every answer right, skip to part D.

## A · Slopes and rates (15 minutes)

The slope of the straight line through two points of a curve, `(f(b) - f(a)) / (b - a)`, is an
average rate of change. The derivative `f'(x)` is what that slope approaches as the two points
close in on `x`.

1. For `f(x) = x²`, compute the slope between `x = 3` and `x = 3.1`, then between `3` and `3.01`.
   What number do the slopes approach?
2. A learner's practice minutes grow from 40 in week 2 to 58 in week 5. What is the average rate
   of change, in minutes per week?

## B · The rules (30 minutes)

| Function | Derivative |
|---|---|
| `c` (a constant) | `0` |
| `xⁿ` | `n xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `ln x` | `1 / x` |
| `c · f(x)` | `c · f'(x)` |
| `f(x) + g(x)` | `f'(x) + g'(x)` |
| `f(x) · g(x)` (product rule) | `f'(x) g(x) + f(x) g'(x)` |
| `f(g(x))` (chain rule) | `f'(g(x)) · g'(x)` |

3. `d/dx (4x³ - 2x + 7)`
4. `d/dx (x² eˣ)`
5. `d/dx e^(3x)`
6. `d/dx ln(1 + x²)`
7. `d/dx (2x + 1)⁵`

## C · Reading a derivative (20 minutes)

8. `f(x) = x³ - 3x`. Where is `f'(x) = 0`? Which of those points is a local minimum?
9. The second derivative `f''(x)` is the derivative of `f'(x)`. For `f(x) = x³ - 3x`, what sign
   does `f''` have at each point of question 8, and what does that sign tell you?
10. Without computing: is the slope of `ln x` larger at `x = 1` or at `x = 10`?

## D · Python warm-up (15 minutes)

11. Write a function `slope(f, a, b)` that returns `(f(b) - f(a)) / (b - a)`. Use it with
    `f = lambda x: x ** 2`, `a = 3` and `b = 3 + 1e-6`. How close is the result to the answer of
    question 1?

## Answers

1. 6.1 and 6.01; they approach 6, which is `f'(3) = 2 · 3`.
2. (58 - 40) / (5 - 2) = 6 minutes per week.
3. `12x² - 2`
4. `2x eˣ + x² eˣ = x eˣ (2 + x)`
5. `3 e^(3x)`
6. `2x / (1 + x²)`
7. `10 (2x + 1)⁴`
8. `f'(x) = 3x² - 3 = 0` at `x = -1` and `x = 1`; the local minimum is at `x = 1`.
9. `f''(x) = 6x`: negative at `-1` (the curve bends down: a local maximum), positive at `1` (it
   bends up: a local minimum).
10. At `x = 1`: the slope `1 / x` shrinks as `x` grows.
11. About 6.000001: the slope of a very short chord is close to the derivative. Week 1 shows how
    short is too short.
