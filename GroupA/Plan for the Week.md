# Monday (Done)
## Main goal
Set up repository, development environment, git workflow, and start data inspection.

### Steps
- [x] Initialize repository, create `.gitignore` excluding `data/` and `.venv` (health data protection)
- [x] Set up Python virtual environment and install `requirements.txt`
- [x] Each member creates a personal branch (`groupA/firstname-topic`); no direct push to `main`
- [x] Git workflow introduction for team
- [x] Started data inspection & cleanup (loading CSVs, checking structure)

---

# Tuesday
## Main goal
**Finish data preprocessing, merging, and feature engineering**, produce final modeling dataset.

### Steps
- [x] Filter both HRV files to `Passage == "M0"` (birth recordings only)
- [x] Drop WPPSI rows missing target IQ (QIT, QIV, QIP, NCL)
- [x] Merge linear + nonlinear HRV on `Monogramme` + `DDN` + `Passage`
- [x] Merge HRV + WPPSI on `Monogramme` + `DDN`
- [x] Drop fully-empty feature columns; impute remaining NaN with column mean
- [ ] **Feature selection / engineering:**
    - [ ] Correlation analysis & removal of highly correlated features (|r| > 0.95)
    - [ ] Variance threshold (remove near-zero variance)
    - [ ] Optional: Lasso/Elastic Net coefficient screening
    - [ ] Optional: Domain-driven grouping (temporal, frequential, geometric, nonlinear)
- [ ] Standardize features with `StandardScaler` fit on training data only
- [ ] Save cleaned dataset locally (not committed to repo)
- [ ] Document preprocessing pipeline (match [[WS26-27/BIP Proyect/personal-work/datasets/dataset|dataset]] §5.2, §7)

---

# Wednesday
## Main goal
**Implement all regression models** with proper hyperparameter spaces.

### Models to implement (minimum)
| Model             | Key hyperparameters to tune                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Elastic Net**   | `alpha` (logspace -4 to 1), `l1_ratio` [0.1, 0.3, 0.5, 0.7, 0.9]                                                                                                  |
| **Random Forest** | `n_estimators` [200, 500], `max_depth` [3, 5, 7, None], `min_samples_leaf` [5, 10, 20], `max_features` ["sqrt", "log2"]                                           |
| **LightGBM**      | `n_estimators` [100, 300], `learning_rate` [0.01, 0.05, 0.1], `max_depth` [3, 5, -1], `min_child_samples` [10, 20], `reg_alpha`, `reg_lambda`                     |
| **XGBoost**       | `n_estimators` [100, 300], `learning_rate` [0.01, 0.05, 0.1], `max_depth` [3, 5, 7], `min_child_weight` [1, 5], `reg_alpha`, `reg_lambda`, `subsample` [0.8, 1.0] |
| **SVR**           | `C` [0.1, 1, 10, 100], `epsilon` [0.01, 0.1, 1], `kernel` ["rbf", "linear"], `gamma` ["scale", "auto"]                                                            |

### Steps
- [ ] Create `models/` module with each model class + hyperparameter grid
- [ ] Implement `fit()`, `predict()`, `get_params()`, `set_params()` interface
- [ ] Quick sanity check: fit on small subset, verify predictions work
- [ ] Save trained models + scaler with `joblib` (for reuse Thursday)

---

# Thursday
## Main goal
**Run standardized evaluation across all models & all IQ targets** with repeated k-fold CV.

### Evaluation Protocol
- **CV Strategy:** **Repeated 10-fold CV** (10 repeats × 10 folds = 100 fits per model)  
  *or* **10× Repeated 5-fold CV** (10 repeats × 5 folds = 50 fits) — *choose based on compute time*
- **Nested structure:** Inner CV for hyperparameter tuning, outer CV for performance estimation
- **Targets:** QIT (primary), QIV, QIP, NCL
- **Metrics per fold:** MAE, MSE, RMSE, R²
- **Aggregation:** Mean ± SD across outer folds × repeats

### Steps
- [ ] Implement `evaluation.py` with:
    - [ ] `RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)` or `n_splits=5, n_repeats=10`
    - [ ] Inner CV: `GridSearchCV` / `RandomizedSearchCV` (5-fold)
    - [ ] Outer CV: `cross_validate` with multiple metrics
    - [ ] Pipeline: `StandardScaler` → Model (prevents data leakage)
- [ ] Register all 5+ models in `METHOD_REGISTRY`
- [ ] Register all 4 targets in `TARGET_COLUMNS`
- [ ] Run evaluation for **each model × each target**
- [ ] Save results to `results/model_comparison.csv` with columns:
  `model, target, mae_mean, mae_std, mse_mean, mse_std, rmse_mean, rmse_std, r2_mean, r2_std`
- [ ] Generate comparison plots:
    - [ ] Bar plot: MAE per model (grouped by target)
    - [ ] Bar plot: R² per model (with baseline R²=0 line)
    - [ ] Box plots of fold-level scores
- [ ] Identify best model per target (lowest MAE, highest R² > 0)

---

# Friday
## Main goal
**Feedback, error correction, tutor meeting prep, and deliverables.**

### Steps
- [ ] Review all results: any failures, convergence warnings, extreme variance?
- [ ] Error analysis on best model: residual plots, predicted vs actual, outlier inspection
- [ ] Compare to baseline (mean predictor): confirm R² > 0 or document honestly
- [ ] **Tutor meeting prep:**
    - [ ] 1-slide summary: problem, data, methods, best result, limitation
    - [ ] Key finding: "Elastic Net / RF / XGBoost achieves MAE=X, R²=Y"
    - [ ] Main limitation: n≈180, p≈100, no external validation
    - [ ] Next steps: Elastic Net + Bayesian, clinical features, raw ECG
- [ ] Open pull requests to `main`, code review, merge approved branches
- [ ] Write short summary in `results/SUMMARY.md`:
    - [ ] Best model per target + metrics
    - [ ] Limitations (small sample, high dimensionality, no external cohort)
    - [ ] Next steps
- [ ] Verify no health data files committed to repository
- [ ] Archive cleaned dataset + trained models (local only)