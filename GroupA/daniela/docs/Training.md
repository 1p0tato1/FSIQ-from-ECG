## Scope
- **Data source:** cleaned + merged table produced by `data_preprocessing.py` / `data_merge.py` / `data_features.py` (`data/clean_data_shifted/merged_dataset_features.csv`, recommended).
- **Target:** `Note composite_WPPSI` (configurable via `TARGET_COLUMN`; alternatives: `NS Verb._WPPSI`, `NS Perf._WPPSI`, `Rang percentile_WPPSI`).
- **Features:** all HRV/clinical columns except ID/metadata keys and all WPPSI columns (to prevent target leakage).

## Preprocessing (already applied upstream)
The merged dataset arrives pre-cleaned. Steps already applied (see this document's Data cleaning section):
1. WPPSI empty-row removal + subject matching by NUMBER with DOB consistency check.
2. Column-shift repair via `fix_shifted_rows()` (`SHIFT=2`, `MISSING="-"`) in `clean_data_shifted/`.
3. `Passage == "M0"` filtering (birth recordings only).
4. Empty-cell subject removal across both HRV files.
5. Removal of WPPSI-only subjects.
6. Linear + nonlinear HRV merge on `Monogramme` + `DDN` + `Passage`; then HRV + WPPSI merge on `Monogramme` + `DDN`.
7. Drop fully-empty feature columns; drop constant + highly correlated features (`|r| > 0.95`).

Inside the notebook:
- Rows missing the target are dropped (`df.dropna(subset=[TARGET_COLUMN])`).
- ID/metadata keys (`Numéro`, `Monogramme`, `DDN`, `Sexe`, `Passage`, `Date`, `Recorder`, etc.) are excluded from features.
- **All WPPSI columns** (not just the target) are excluded from `feature_cols` to prevent target leakage. Previously only `TARGET_COLUMN` was excluded, which allowed WPPSI sub-scores to leak into the feature matrix and produced inflated results (MAE 0.7496 / R² 0.9942).

## Model selection & training
Because the sample is small (n ≈ 190) relative to the number of features (~60), regularized models are preferred and hyperparameters are tuned with nested cross-validation, matching the rapport standard (`Plan for the Week.md`; `dataset.md` §5.3.3):

- **Outer CV:** 5-fold `KFold` (shuffle, random_state=42) — unbiased performance estimation.
- **Inner CV:** 4-fold `KFold` + `GridSearchCV` — hyperparameter tuning only, scoring by `neg_mean_absolute_error`.
- **Pipeline:** `SimpleImputer(strategy="mean")` → `StandardScaler()` → model, fit on training folds only.
- Selection criterion: best model by outer-CV MAE. The selected model is refit on all data via `GridSearchCV` and saved as `regression_model_{TARGET_COLUMN}.pkl`.

## Evaluation
- Honest out-of-sample performance via `cross_val_predict` on the outer CV folds (no leakage).
- Metrics reported: MAE, MSE, RMSE, R².
- **R² > 0:** model is better than predicting the mean.
- **R² = 0:** same as the mean.
- **R² < 0:** worse than guessing the mean.

## Visualization
Cell 7 produces Predicted vs Actual IQ and Residual plots using out-of-fold (outer CV) predictions, for diagnosing model performance.

## Save / Load
- Save: `joblib.dump(search_full.best_estimator_, f"regression_model_{TARGET_COLUMN}.pkl")`
- Load: `joblib.load(f"regression_model_{TARGET_COLUMN}.pkl")` then `predict(X_new)`.
---
# Results 
1. [[Regularized Models + Auto shifting]]
2. [[Regularized Models + Manual shifting]]
3. [[Other Models + Manual shifting]]
4. [[Other models + Auto shifting]]
5. 