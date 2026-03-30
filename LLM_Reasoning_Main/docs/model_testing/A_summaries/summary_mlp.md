# MLP Summary

## Pipeline Evolution Note
- Raw features -> noisy
- Add reasoning -> improves
- Too many features -> overfit
- Pruning -> stabilizes
- MLP (unregularized) -> overfits
- MLP (regularized) -> collapses

## MLP-Only Results Summary
- MLP32: strongest but clearly overfits (larger train/CV gap and higher variance).
- MLP4: moderate performance with less overfit than MLP32.
- MLP2: weakest but most stable (little to no overfit).
- PLS helps MLPs find signal relative to Base (more consistent gains and stability across combos).

## Quoted Results (CV +/- std, Full)
Base (aggressive pruning):
- MLP32 shows clear overfit: A+E = 0.278+/-0.053 (CV) vs 0.411 (Full), A+D = 0.264+/-0.054 vs 0.400.
- MLP4 is less overfit: A+E = 0.297+/-0.069 vs 0.309, HQ = 0.225+/-0.040 vs 0.255.
- MLP2 is stable: HQ = 0.240+/-0.038 vs 0.243, A+E = 0.279+/-0.074 vs 0.278.

PLS (n_components=6) lifts signal vs Base:
- MLP32: HQ 0.230+/-0.032 -> 0.256+/-0.050, A+B+C+D+E+F 0.299+/-0.064 -> 0.343+/-0.064.
- MLP4: A 0.266+/-0.050 -> 0.275+/-0.028, F 0.260+/-0.069 -> 0.292+/-0.054.
- MLP2: A 0.239+/-0.039 -> 0.272+/-0.052, F 0.238+/-0.037 -> 0.266+/-0.035.
