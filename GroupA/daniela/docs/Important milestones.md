---
tags:
  - documentation
---
---
# Data cleaning
Data cleaned in two ways
## First cleaned it following these steps
1. WPPSI results file: remove the rows with empty cells; remove the same  
   subjects from the two HRV files (linear and non-linear).
2. HRV files: keep only the rows whose "Passage" column equals "M0". 
3. HRV files: if a row has empty cells in EITHER HRV file, the corresponding  
   subject is removed from BOTH HRV files and from the WPPSI file.
4. WPPSI file: remove the subjects that do not appear in any HRV file.
## Then realized about the shift of the columns
1. WPPSI results file: remove the rows with empty cells; remove the same  
   subjects from the two HRV files (linear and non-linear).
2. ==🟡HRV files: repair the rows whose values are shifted to the left==  
   ==🟡(the last cells are empty because the values moved left by SHIFT columns).==
3. HRV files: keep only the rows whose "Passage" column equals "M0". 
4. HRV files: if a row has empty cells in EITHER HRV file, the corresponding  
   subject is removed from BOTH HRV files and from the WPPSI file.
5. WPPSI file: remove the subjects that do not appear in any HRV file.
## But saved both ways in case the data shifting altered the data itself
[[WS26-27/BIP Proyect/FSIQ-from-ECG/data/clean_data_m_shifted/HRV_Lineaire_Full.csv|HRV_Lineaire_Full]] was shifted manually after cleaning the data with  [[data_preprocessing_m.py]]. While [[WS26-27/BIP Proyect/FSIQ-from-ECG/data/clean_data_shifted/HRV_Lineaire_Full.csv|HRV_Lineaire_Full]] was shifted through the python file [[data_preprocessing.py]].

Documentation about [[data_preprocessing_m.py]] in [[Data Preprocessing and manual shifting]].

Documentation about [[data_preprocessing.py]] in [[Data Preprocessing and automatic shifting]].

## Key differences between the two files

| Aspect | [[data_preprocessing_m.py]] (manual shifting) | [[data_preprocessing.py]] (automatic shifting) |
|---|---|---|
| Repairs shifted HRV columns | **No** — the left shift is left untouched by the script | **Yes** — via `fix_shifted_rows()`, re-inserts `SHIFT=2` missing cells (`-`) at the best-fitting column |
| Order of shift repair | N/A (none) | Done **before** filtering by `Passage == "M0"` |
| Output directory | `data/clean_data_m_shifted/` | `data/clean_data_shifted/` |
| Shift-related config | None | `SHIFT = 2`, `MISSING = "-"` |
| Extra functions | None | `fix_shifted_rows`, `cell_kind`, `score` |
| Column shift repair | Performed **manually** on the cleaned CSVs afterwards | Performed **automatically** in-script |

**When to use which:** use [[data_preprocessing_m.py]] + [[Data Preprocessing and manual shifting]] when you want the cleaning without any automatic column repair (the shift was fixed by hand on the saved CSVs). Use [[data_preprocessing.py]] + [[Data Preprocessing and automatic shifting]] when you want the shift repaired automatically by the script.

Both files share the same cleaning steps (WPPSI empty-row removal, subject matching by NUMBER with DOB consistency check, `Passage == "M0"` filtering, empty-cell subject removal across both HRV files, and removal of WPPSI-only subjects) and neither modifies the input files in place.

---
# Data merge
On [[data_merge.py]] data is going to be merged in one table easy to use when creating the machine learning methods.

Detailed info in [[Data Merge]].

---
# Clean Features

On [[data_feature.py]] the features get a basic clean.

Detailed info in [[Feature Cleaning]].

---
# training.ipynb

Regression Method 1 — WPPSI-III IQ from Neonatal ECG/HRV. Trains regularized regression models to predict WPPSI-III IQ scores (Full-Scale `Note composite_WPPSI`, Verbal `NS Verb._WPPSI`, Performance `NS Perf._WPPSI`, …) from neonatal linear + nonlinear HRV features (AuBE cohort, M0 recordings only).

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
- **Models compared:** `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`.
- **Pipeline:** `SimpleImputer(strategy="mean")` → `StandardScaler()` → model, fit on training folds only.
- Selection criterion: best model by outer-CV MAE. The selected model is refit on all data via `GridSearchCV` and saved as `regression_model_{TARGET_COLUMN}.pkl`.

## Evaluation
- Honest out-of-sample performance via `cross_val_predict` on the outer CV folds (no leakage).
- Metrics reported: MAE, MSE, RMSE, R².
- **R² > 0:** model is better than predicting the mean.
- **R² = 0:** same as the mean.
- **R² < 0:** worse than guessing the mean.

### Latest results (BayesianRidge, nested 5×4 CV)
| Model | MAE | ±MAE | RMSE | R² | ±R² |
|---|---|---|---|---|---|
| Ridge | 12.0631 | 1.1584 | 15.1008 | −0.1338 | 0.0360 |
| Lasso | 11.7211 | 0.7123 | 14.4901 | −0.0468 | 0.0530 |
| ElasticNet | 11.9192 | 1.0399 | 14.8654 | −0.0993 | 0.0333 |
| BayesianRidge | 11.6422 | 0.7271 | 14.3503 | −0.0256 | 0.0226 |

Out-of-fold predictions (outer 5-fold CV): MAE 11.6422, MSE 207.0812, RMSE 14.3903, R² −0.0086.
These indicate HRV indices alone do not yet provide useful individual prediction of the WPPSI Full-Scale IQ (R² near zero / negative).

## Visualization
Cell 7 produces Predicted vs Actual IQ and Residual plots using out-of-fold (outer CV) predictions, for diagnosing model performance.

## Save / Load
- Save: `joblib.dump(search_full.best_estimator_, f"regression_model_{TARGET_COLUMN}.pkl")`
- Load: `joblib.load(f"regression_model_{TARGET_COLUMN}.pkl")` then `predict(X_new)`.

## Next steps
1. Confirm `TARGET_COLUMN` matches the cleaned merged dataset (case-sensitive).
2. Run all cells: load data → preprocess → train model → evaluate → visualize → save.
3. Try additional models (`RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`) and compare.
4. Add nested cross-validation for hyperparameter tuning (already implemented) and consider feature selection.
5. Run `test_for_model.py` from the project root for the standardized cross-group comparison.
6. Report MAE and R² per `dataset.md` §9.
7. Per `GroupA/Plan for the Week.md` (Friday): review results, error analysis, compare to baseline (mean predictor), prepare tutor meeting, write `results/SUMMARY.md`.

---
# Results Documentation
1.  [[Regularized Models + Auto shifting]]
2. [[Regularized Models + Manual shifting]]
3. 