# Final Model Summary

## Clean vs Complex
- `LR (BASE) A`, `LR (BASE) A+C`: clean, low-collinearity baselines.
- `PLS` models: compressed, higher-signal representations.

## Signal Density Spectrum
- `A`: minimal signal.
- `A+C`: structured signal.
- `F` / `D+E+F`: concentrated signal.
- `A+B+C+D+E+F`: full signal.

## Linear vs Non-Linear
- `LR`: linear baseline.
- `MLP4`: non-linear capacity.

## Model Roles
- `LR (BASE) A`: stability anchor / sanity check.
- `LR (BASE) A+C`: best clean linear model.
- `LR (PLS) F`: minimal high-signal compressed.
- `LR (PLS) D+E+F`: stronger compressed signal.
- `LR (PLS) A+B+C+D+E+F`: maximum signal (likely best LR).
- `MLP4 (PLS) C+D+E+F`: non-linear + structured signal.

## What Outcomes Will Tell You
- `A+B+C+D+E+F (PLS)` wins: signal distributed → full features best.
- `D+E+F ≈ A+B+C+D+E+F`: redundancy → compact model sufficient.
- `A+C (BASE)` competitive: simpler representation sufficient → PLS not essential.
- `MLP4 (PLS)` wins: non-linear interactions matter.
- `MLP4 ≈ LR (PLS)`: mostly linear in latent space.

## Verdict
Run these six models. This setup isolates representation vs model complexity and signal density cleanly, maximizing interpretability and decision power.

## Evaluation Tip
Beyond scores, check rank stability vs CV, unexpected collapses, and whether MLP behaves differently — those deltas are most informative.
