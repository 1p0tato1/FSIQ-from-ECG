# Results summary

Generated automatically on 2026-11-18.
Protocol: nested cross-validation, outer 5-fold × 10 repeats, inner 5-fold, search budget 50, seed 42. All preprocessing is fitted inside each training fold.

## Best model per target

| target | model | MAE | RMSE | R² | baseline MAE | baseline R² |
|---|---|---|---|---|---|---|
| QIT | ElasticNet | 10.39 ± 1.09 | 12.80 ± 1.36 | -0.045 ± 0.057 | 10.35 | -0.039 |

- **QIT**: no model reaches a mean R² above zero, i.e. none explains more variance than always predicting the training mean.

## Limitations

- Small sample: 189 children, 101 candidate predictors (48 kept by the Lasso screen when fitted on every child).
- Single cohort: all estimates come from cross-validation on the same recruitment; there is no external validation.
- Fold-to-fold spread is large relative to differences between models, so rankings are not decisive.
- Mean imputation and linear screening may discard non-linear or missingness-related signal.

## Next steps

- Elastic Net with a Bayesian treatment of the hyper-parameters / coefficients.
- Add clinical covariates beyond the HRV summary features.
- Work from the raw ECG signal instead of pre-computed HRV indices.
- Extend the evaluation to QIV, QIP and NCL (set `TARGETS` in `.env`).
